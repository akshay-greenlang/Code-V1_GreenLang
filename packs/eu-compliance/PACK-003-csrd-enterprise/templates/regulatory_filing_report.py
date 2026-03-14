"""
RegulatoryFilingReportTemplate - Filing status and history for CSRD Enterprise Pack.

This module implements the regulatory filing report template with filing
calendar, submission history, validation results, version comparison,
acknowledgment tracking, compliance coverage matrix, and filing provenance chain.

Example:
    >>> template = RegulatoryFilingReportTemplate()
    >>> data = {"filing_calendar": [...], "submission_history": [...]}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class RegulatoryFilingReportTemplate:
    """
    Regulatory filing status and history template.

    Renders filing calendars, submission history, validation results,
    version comparisons, acknowledgment tracking, compliance coverage
    matrices, and filing provenance chains.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    VALIDATION_SEVERITIES = ["error", "warning", "info"]

    FILING_STATUSES = [
        "draft", "pending_review", "submitted", "accepted",
        "rejected", "revision_required",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryFilingReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render filing report as Markdown.

        Args:
            data: Report data with filing_calendar, submission_history,
                  validation_results, version_comparison, acknowledgments,
                  compliance_coverage, provenance_chain.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_filing_calendar(data))
        sections.append(self._render_md_submission_history(data))
        sections.append(self._render_md_validation_results(data))
        sections.append(self._render_md_version_comparison(data))
        sections.append(self._render_md_acknowledgments(data))
        sections.append(self._render_md_compliance_coverage(data))
        sections.append(self._render_md_provenance_chain(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render filing report as self-contained HTML.

        Args:
            data: Report data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_filing_calendar(data))
        body_parts.append(self._render_html_submission_history(data))
        body_parts.append(self._render_html_validation_results(data))
        body_parts.append(self._render_html_version_comparison(data))
        body_parts.append(self._render_html_acknowledgments(data))
        body_parts.append(self._render_html_compliance_coverage(data))
        body_parts.append(self._render_html_provenance_chain(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Regulatory Filing Report</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"report-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render filing report as structured JSON.

        Args:
            data: Report data dict.

        Returns:
            Structured dict with all report sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "regulatory_filing_report",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "filing_calendar": data.get("filing_calendar", []),
            "submission_history": data.get("submission_history", []),
            "validation_results": data.get("validation_results", {}),
            "version_comparison": data.get("version_comparison", {}),
            "acknowledgments": data.get("acknowledgments", []),
            "compliance_coverage": data.get("compliance_coverage", []),
            "provenance_chain": data.get("provenance_chain", []),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        title = data.get("title", "Regulatory Filing Report")
        ts = self._format_date(self.generated_at)
        return f"# {title}\n\n**Generated:** {ts}\n\n---"

    def _render_md_filing_calendar(self, data: Dict[str, Any]) -> str:
        """Render filing calendar with deadlines and countdowns."""
        calendar: List[Dict[str, Any]] = data.get("filing_calendar", [])
        if not calendar:
            return "## Filing Calendar\n\n_No upcoming filings._"

        lines = [
            "## Filing Calendar (Upcoming Deadlines)",
            "",
            "| Filing | Authority | Deadline | Days Left | Status | Priority |",
            "|--------|----------|----------|-----------|--------|----------|",
        ]
        for f in calendar:
            name = f.get("filing_name", "-")
            authority = f.get("authority", "-")
            deadline = f.get("deadline", "-")
            days_left = f.get("days_left", 0)
            status = f.get("status", "-")
            priority = f.get("priority", "normal").upper()
            urgency = "URGENT" if days_left <= 7 else ""
            lines.append(
                f"| {name} | {authority} | {deadline} | "
                f"{days_left} {urgency} | {status} | {priority} |"
            )

        return "\n".join(lines)

    def _render_md_submission_history(self, data: Dict[str, Any]) -> str:
        """Render submission history table."""
        history: List[Dict[str, Any]] = data.get("submission_history", [])
        if not history:
            return "## Submission History\n\n_No submissions recorded._"

        lines = [
            "## Submission History",
            "",
            "| Date | Target | Format | Status | Reference |",
            "|------|--------|--------|--------|-----------|",
        ]
        for h in history:
            date = h.get("submission_date", "-")
            target = h.get("target_authority", "-")
            fmt = h.get("format", "-")
            status = h.get("status", "-")
            ref = h.get("reference_number", "-")
            lines.append(f"| {date} | {target} | {fmt} | {status} | {ref} |")

        return "\n".join(lines)

    def _render_md_validation_results(self, data: Dict[str, Any]) -> str:
        """Render validation results detail."""
        validation = data.get("validation_results", {})
        if not validation:
            return "## Validation Results\n\n_No validation results available._"

        summary = validation.get("summary", {})
        items: List[Dict[str, Any]] = validation.get("items", [])

        lines = [
            "## Validation Results",
            "",
            f"**Total Checks:** {summary.get('total_checks', 0)}",
            f"**Errors:** {summary.get('errors', 0)}",
            f"**Warnings:** {summary.get('warnings', 0)}",
            f"**Info:** {summary.get('info', 0)}",
            "",
        ]

        if items:
            lines.extend([
                "| # | Severity | Category | Rule | Message |",
                "|---|----------|----------|------|---------|",
            ])
            for idx, item in enumerate(items, 1):
                severity = item.get("severity", "info").upper()
                category = item.get("category", "-")
                rule = item.get("rule_id", "-")
                message = item.get("message", "-")
                lines.append(
                    f"| {idx} | {severity} | {category} | {rule} | {message} |"
                )

        return "\n".join(lines)

    def _render_md_version_comparison(self, data: Dict[str, Any]) -> str:
        """Render version comparison diff."""
        comparison = data.get("version_comparison", {})
        if not comparison:
            return "## Version Comparison\n\n_No version comparison available._"

        v1 = comparison.get("version_a", "-")
        v2 = comparison.get("version_b", "-")
        changes: List[Dict[str, Any]] = comparison.get("changes", [])

        lines = [
            "## Version Comparison",
            "",
            f"**Comparing:** {v1} vs {v2}",
            f"**Total Changes:** {len(changes)}",
            "",
            "| Section | Change Type | Old Value | New Value |",
            "|---------|------------|-----------|-----------|",
        ]
        for c in changes:
            section = c.get("section", "-")
            change_type = c.get("change_type", "-")
            old_val = c.get("old_value", "-")
            new_val = c.get("new_value", "-")
            lines.append(f"| {section} | {change_type} | {old_val} | {new_val} |")

        return "\n".join(lines)

    def _render_md_acknowledgments(self, data: Dict[str, Any]) -> str:
        """Render acknowledgment tracker."""
        acks: List[Dict[str, Any]] = data.get("acknowledgments", [])
        if not acks:
            return "## Acknowledgment Tracker\n\n_No acknowledgments recorded._"

        lines = [
            "## Acknowledgment Tracker",
            "",
            "| Authority | Filing | Status | Received Date | Reference |",
            "|----------|--------|--------|--------------|-----------|",
        ]
        for a in acks:
            authority = a.get("authority", "-")
            filing = a.get("filing_name", "-")
            status = a.get("status", "-")
            received = a.get("received_date", "-")
            ref = a.get("reference", "-")
            lines.append(
                f"| {authority} | {filing} | {status} | {received} | {ref} |"
            )

        return "\n".join(lines)

    def _render_md_compliance_coverage(self, data: Dict[str, Any]) -> str:
        """Render compliance coverage matrix."""
        coverage: List[Dict[str, Any]] = data.get("compliance_coverage", [])
        if not coverage:
            return "## Compliance Coverage\n\n_No coverage data available._"

        lines = [
            "## Compliance Coverage Matrix",
            "",
            "| Requirement | Disclosure Filed | Status | Completeness |",
            "|------------|-----------------|--------|-------------|",
        ]
        for c in coverage:
            req = c.get("requirement", "-")
            disclosure = c.get("disclosure_ref", "-")
            status = c.get("status", "-")
            completeness = self._format_percentage(c.get("completeness_pct", 0))
            lines.append(
                f"| {req} | {disclosure} | {status} | {completeness} |"
            )

        return "\n".join(lines)

    def _render_md_provenance_chain(self, data: Dict[str, Any]) -> str:
        """Render filing provenance chain."""
        chain: List[Dict[str, Any]] = data.get("provenance_chain", [])
        if not chain:
            return "## Filing Provenance\n\n_No provenance data available._"

        lines = [
            "## Filing Provenance Chain",
            "",
            "_Data -> Report -> Package -> Submission_",
            "",
        ]
        for idx, step in enumerate(chain, 1):
            stage = step.get("stage", "-")
            artifact = step.get("artifact", "-")
            sha256 = step.get("sha256_hash", "-")
            timestamp = step.get("timestamp", "-")
            actor = step.get("actor", "-")
            lines.extend([
                f"### Step {idx}: {stage}",
                f"- **Artifact:** {artifact}",
                f"- **SHA-256:** `{sha256}`",
                f"- **Timestamp:** {timestamp}",
                f"- **Actor:** {actor}",
                "",
            ])

        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Filing Report generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for regulatory filing report."""
        return """
:root {
    --primary: #1e40af; --primary-light: #dbeafe; --success: #057a55;
    --warning: #e3a008; --danger: #e02424; --info: #1c64f2;
    --bg: #f1f5f9; --card-bg: #fff; --text: #1e293b;
    --text-muted: #64748b; --border: #e2e8f0;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.report-container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.report-header { background: linear-gradient(135deg, #1e40af, #6366f1);
    color: #fff; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }
.report-header h1 { font-size: 26px; }
.report-header .subtitle { opacity: 0.85; margin-top: 4px; font-size: 14px; }
.section { margin-bottom: 24px; background: var(--card-bg); border-radius: 10px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section-title { font-size: 18px; font-weight: 600; color: var(--primary);
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
th { background: var(--primary-light); color: var(--primary); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:hover { background: #f8fafc; }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; }
.status-badge.accepted, .status-badge.submitted { background: #d1fae5; color: #057a55; }
.status-badge.pending_review, .status-badge.draft { background: #fef9c3; color: #92400e; }
.status-badge.rejected { background: #fde8e8; color: #e02424; }
.status-badge.revision_required { background: #feecdc; color: #d97706; }
.severity-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; }
.severity-badge.error { background: #fde8e8; color: #e02424; }
.severity-badge.warning { background: #fef9c3; color: #92400e; }
.severity-badge.info { background: #dbeafe; color: #1e40af; }
.countdown { font-weight: 700; }
.countdown.urgent { color: var(--danger); }
.countdown.soon { color: var(--warning); }
.countdown.normal { color: var(--success); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 16px; }
.summary-card { border: 1px solid var(--border); border-radius: 8px;
    padding: 14px; text-align: center; }
.summary-card .sc-value { font-size: 24px; font-weight: 700; }
.summary-card .sc-label { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.summary-card.errors .sc-value { color: var(--danger); }
.summary-card.warnings .sc-value { color: var(--warning); }
.summary-card.passed .sc-value { color: var(--success); }
.diff-added { background: #d1fae5; color: #057a55; }
.diff-removed { background: #fde8e8; color: #e02424; }
.diff-modified { background: #fef9c3; color: #92400e; }
.provenance-step { display: flex; margin-bottom: 16px; }
.provenance-number { width: 36px; height: 36px; background: var(--primary);
    color: #fff; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 14px; flex-shrink: 0;
    margin-right: 14px; }
.provenance-content { flex: 1; border-left: 2px solid var(--primary-light);
    padding-left: 14px; padding-bottom: 16px; }
.provenance-content .stage-name { font-weight: 600; font-size: 14px;
    color: var(--primary); }
.provenance-content .artifact { font-size: 13px; margin-top: 4px; }
.provenance-content .hash { font-family: monospace; font-size: 11px;
    color: var(--text-muted); margin-top: 2px; word-break: break-all; }
.provenance-content .meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.coverage-cell { text-align: center; }
.coverage-cell.full { color: var(--success); font-weight: 600; }
.coverage-cell.partial { color: var(--warning); font-weight: 600; }
.coverage-cell.missing { color: var(--danger); font-weight: 600; }
.priority-badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 700; text-transform: uppercase; }
.priority-badge.high { background: #fde8e8; color: #e02424; }
.priority-badge.medium { background: #fef9c3; color: #92400e; }
.priority-badge.normal { background: #d1fae5; color: #057a55; }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = self._escape_html(data.get("title", "Regulatory Filing Report"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"report-header\">\n"
            f"  <h1>{title}</h1>\n"
            f"  <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_filing_calendar(self, data: Dict[str, Any]) -> str:
        """Render HTML filing calendar with countdown indicators."""
        calendar: List[Dict[str, Any]] = data.get("filing_calendar", [])
        if not calendar:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Filing Calendar</h2>\n"
                "  <p>No upcoming filings.</p>\n</div>"
            )

        rows = ""
        for f in calendar:
            name = self._escape_html(f.get("filing_name", "-"))
            authority = self._escape_html(f.get("authority", "-"))
            deadline = f.get("deadline", "-")
            days_left = f.get("days_left", 0)
            status = f.get("status", "draft").lower().replace(" ", "_")
            priority = f.get("priority", "normal").lower()
            urgency_cls = (
                "urgent" if days_left <= 7
                else "soon" if days_left <= 30
                else "normal"
            )
            rows += (
                f"<tr><td><strong>{name}</strong></td>"
                f"<td>{authority}</td><td>{deadline}</td>"
                f"<td><span class=\"countdown {urgency_cls}\">"
                f"{days_left} days</span></td>"
                f"<td><span class=\"status-badge {status}\">{status}</span></td>"
                f"<td><span class=\"priority-badge {priority}\">"
                f"{priority}</span></td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Filing Calendar</h2>\n"
            "  <table><thead><tr>"
            "<th>Filing</th><th>Authority</th><th>Deadline</th>"
            "<th>Countdown</th><th>Status</th><th>Priority</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_submission_history(self, data: Dict[str, Any]) -> str:
        """Render HTML submission history."""
        history: List[Dict[str, Any]] = data.get("submission_history", [])
        if not history:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Submission History</h2>\n"
                "  <p>No submissions recorded.</p>\n</div>"
            )

        rows = ""
        for h in history:
            status = h.get("status", "draft").lower().replace(" ", "_")
            rows += (
                f"<tr><td>{h.get('submission_date', '-')}</td>"
                f"<td>{self._escape_html(h.get('target_authority', '-'))}</td>"
                f"<td>{h.get('format', '-')}</td>"
                f"<td><span class=\"status-badge {status}\">{status}</span></td>"
                f"<td><code>{h.get('reference_number', '-')}</code></td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Submission History</h2>\n"
            "  <table><thead><tr>"
            "<th>Date</th><th>Target</th><th>Format</th>"
            "<th>Status</th><th>Reference</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_validation_results(self, data: Dict[str, Any]) -> str:
        """Render HTML validation results."""
        validation = data.get("validation_results", {})
        if not validation:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Validation Results</h2>\n"
                "  <p>No validation results available.</p>\n</div>"
            )

        summary = validation.get("summary", {})
        summary_html = (
            "<div class=\"summary-grid\">\n"
            f"  <div class=\"summary-card\"><div class=\"sc-value\">"
            f"{summary.get('total_checks', 0)}</div>"
            f"<div class=\"sc-label\">Total Checks</div></div>\n"
            f"  <div class=\"summary-card errors\"><div class=\"sc-value\">"
            f"{summary.get('errors', 0)}</div>"
            f"<div class=\"sc-label\">Errors</div></div>\n"
            f"  <div class=\"summary-card warnings\"><div class=\"sc-value\">"
            f"{summary.get('warnings', 0)}</div>"
            f"<div class=\"sc-label\">Warnings</div></div>\n"
            f"  <div class=\"summary-card passed\"><div class=\"sc-value\">"
            f"{summary.get('info', 0)}</div>"
            f"<div class=\"sc-label\">Info</div></div>\n"
            "</div>\n"
        )

        items = validation.get("items", [])
        rows = ""
        for idx, item in enumerate(items, 1):
            severity = item.get("severity", "info")
            rows += (
                f"<tr><td>{idx}</td>"
                f"<td><span class=\"severity-badge {severity}\">"
                f"{severity}</span></td>"
                f"<td>{self._escape_html(item.get('category', '-'))}</td>"
                f"<td><code>{item.get('rule_id', '-')}</code></td>"
                f"<td>{self._escape_html(item.get('message', '-'))}</td></tr>\n"
            )

        table_html = ""
        if rows:
            table_html = (
                "<table><thead><tr>"
                "<th>#</th><th>Severity</th><th>Category</th>"
                "<th>Rule</th><th>Message</th>"
                "</tr></thead>\n"
                f"<tbody>{rows}</tbody></table>"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Validation Results</h2>\n"
            f"  {summary_html}\n{table_html}\n"
            "</div>"
        )

    def _render_html_version_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML version comparison diff."""
        comparison = data.get("version_comparison", {})
        if not comparison:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Version Comparison</h2>\n"
                "  <p>No version comparison available.</p>\n</div>"
            )

        v1 = self._escape_html(comparison.get("version_a", "-"))
        v2 = self._escape_html(comparison.get("version_b", "-"))
        changes: List[Dict[str, Any]] = comparison.get("changes", [])

        rows = ""
        for c in changes:
            change_type = c.get("change_type", "modified")
            diff_cls = f"diff-{change_type}" if change_type in ("added", "removed", "modified") else ""
            rows += (
                f"<tr class=\"{diff_cls}\">"
                f"<td>{self._escape_html(c.get('section', '-'))}</td>"
                f"<td>{change_type}</td>"
                f"<td>{self._escape_html(str(c.get('old_value', '-')))}</td>"
                f"<td>{self._escape_html(str(c.get('new_value', '-')))}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Version Comparison</h2>\n"
            f"  <p>Comparing: <strong>{v1}</strong> vs <strong>{v2}</strong> "
            f"({len(changes)} changes)</p>\n"
            "  <table><thead><tr>"
            "<th>Section</th><th>Change</th><th>Old Value</th><th>New Value</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_acknowledgments(self, data: Dict[str, Any]) -> str:
        """Render HTML acknowledgment tracker."""
        acks: List[Dict[str, Any]] = data.get("acknowledgments", [])
        if not acks:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Acknowledgment Tracker</h2>\n"
                "  <p>No acknowledgments recorded.</p>\n</div>"
            )

        rows = ""
        for a in acks:
            status = a.get("status", "pending").lower().replace(" ", "_")
            rows += (
                f"<tr><td>{self._escape_html(a.get('authority', '-'))}</td>"
                f"<td>{self._escape_html(a.get('filing_name', '-'))}</td>"
                f"<td><span class=\"status-badge {status}\">{status}</span></td>"
                f"<td>{a.get('received_date', '-')}</td>"
                f"<td><code>{a.get('reference', '-')}</code></td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Acknowledgment Tracker</h2>\n"
            "  <table><thead><tr>"
            "<th>Authority</th><th>Filing</th><th>Status</th>"
            "<th>Received</th><th>Reference</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_compliance_coverage(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance coverage matrix."""
        coverage: List[Dict[str, Any]] = data.get("compliance_coverage", [])
        if not coverage:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Compliance Coverage</h2>\n"
                "  <p>No coverage data available.</p>\n</div>"
            )

        rows = ""
        for c in coverage:
            completeness = c.get("completeness_pct", 0)
            cov_cls = (
                "full" if completeness >= 100
                else "partial" if completeness >= 50
                else "missing"
            )
            rows += (
                f"<tr><td>{self._escape_html(c.get('requirement', '-'))}</td>"
                f"<td>{self._escape_html(c.get('disclosure_ref', '-'))}</td>"
                f"<td><span class=\"status-badge "
                f"{'accepted' if completeness >= 100 else 'pending_review'}\">"
                f"{c.get('status', '-')}</span></td>"
                f"<td class=\"coverage-cell {cov_cls}\">"
                f"{self._format_percentage(completeness)}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Compliance Coverage Matrix</h2>\n"
            "  <table><thead><tr>"
            "<th>Requirement</th><th>Disclosure</th><th>Status</th>"
            "<th>Completeness</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_provenance_chain(self, data: Dict[str, Any]) -> str:
        """Render HTML filing provenance chain."""
        chain: List[Dict[str, Any]] = data.get("provenance_chain", [])
        if not chain:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Filing Provenance</h2>\n"
                "  <p>No provenance data available.</p>\n</div>"
            )

        steps = ""
        for idx, step in enumerate(chain, 1):
            stage = self._escape_html(step.get("stage", "-"))
            artifact = self._escape_html(step.get("artifact", "-"))
            sha = step.get("sha256_hash", "-")
            timestamp = step.get("timestamp", "-")
            actor = self._escape_html(step.get("actor", "-"))
            steps += (
                f"<div class=\"provenance-step\">\n"
                f"  <div class=\"provenance-number\">{idx}</div>\n"
                f"  <div class=\"provenance-content\">\n"
                f"    <div class=\"stage-name\">{stage}</div>\n"
                f"    <div class=\"artifact\">Artifact: {artifact}</div>\n"
                f"    <div class=\"hash\">SHA-256: {sha}</div>\n"
                f"    <div class=\"meta\">{timestamp} | {actor}</div>\n"
                f"  </div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Filing Provenance Chain</h2>\n"
            "  <p style=\"font-size:13px;color:#64748b;margin-bottom:16px\">"
            "Data &rarr; Report &rarr; Package &rarr; Submission</p>\n"
            f"  {steps}\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Filing Report generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format numeric value with thousands separator."""
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage."""
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string."""
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
