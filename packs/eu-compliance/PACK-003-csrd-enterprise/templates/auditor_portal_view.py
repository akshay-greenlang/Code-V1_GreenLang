"""
AuditorPortalViewTemplate - Auditor workspace template for CSRD Enterprise Pack.

This module implements the auditor portal view with engagement overview,
evidence browser, finding tracker, comment threads, document viewer,
assurance opinion form, progress tracker, and export capabilities.

Example:
    >>> template = AuditorPortalViewTemplate()
    >>> data = {"engagement": {...}, "findings": [...], "evidence": [...]}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AuditorPortalViewTemplate:
    """
    Auditor workspace template for CSRD assurance engagements.

    Provides a structured view for auditors including engagement overview,
    evidence browsing by ESRS category, finding tracking, threaded
    discussions, and assurance opinion management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    ESRS_CATEGORIES: Dict[str, List[str]] = {
        "E": ["E1 - Climate Change", "E2 - Pollution", "E3 - Water & Marine",
              "E4 - Biodiversity", "E5 - Resource Use"],
        "S": ["S1 - Own Workforce", "S2 - Value Chain Workers",
              "S3 - Affected Communities", "S4 - Consumers"],
        "G": ["G1 - Business Conduct"],
    }

    SEVERITY_LEVELS = ["critical", "major", "moderate", "minor", "observation"]

    FINDING_STATUSES = ["open", "in_progress", "resolved", "closed", "deferred"]

    ASSURANCE_LEVELS = ["limited", "reasonable"]

    OPINION_TYPES = ["unqualified", "qualified", "adverse", "disclaimer"]

    ENGAGEMENT_PHASES = [
        "Planning", "Risk Assessment", "Evidence Gathering",
        "Testing", "Review", "Reporting", "Sign-off",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AuditorPortalViewTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render auditor portal view as Markdown.

        Args:
            data: Portal data with engagement, evidence, findings,
                  comments, documents, opinion, and progress.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_engagement_overview(data))
        sections.append(self._render_md_evidence_browser(data))
        sections.append(self._render_md_finding_tracker(data))
        sections.append(self._render_md_comment_threads(data))
        sections.append(self._render_md_document_viewer(data))
        sections.append(self._render_md_assurance_opinion(data))
        sections.append(self._render_md_progress_tracker(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render auditor portal view as self-contained HTML.

        Args:
            data: Portal data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_engagement_overview(data))
        body_parts.append(self._render_html_evidence_browser(data))
        body_parts.append(self._render_html_finding_tracker(data))
        body_parts.append(self._render_html_comment_threads(data))
        body_parts.append(self._render_html_document_viewer(data))
        body_parts.append(self._render_html_assurance_opinion(data))
        body_parts.append(self._render_html_progress_tracker(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Auditor Portal</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"portal-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render auditor portal view as structured JSON.

        Args:
            data: Portal data dict.

        Returns:
            Structured dict with all portal sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "auditor_portal_view",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "engagement": self._build_json_engagement(data),
            "evidence": self._build_json_evidence(data),
            "findings": self._build_json_findings(data),
            "comments": self._build_json_comments(data),
            "documents": self._build_json_documents(data),
            "assurance_opinion": self._build_json_opinion(data),
            "progress": self._build_json_progress(data),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        title = data.get("title", "Auditor Portal")
        ts = self._format_date(self.generated_at)
        return f"# {title}\n\n**Generated:** {ts}\n\n---"

    def _render_md_engagement_overview(self, data: Dict[str, Any]) -> str:
        """Render engagement overview section."""
        eng = data.get("engagement", {})
        if not eng:
            return "## Engagement Overview\n\n_No engagement data available._"

        lines = [
            "## Engagement Overview",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Engagement ID | {eng.get('engagement_id', '-')} |",
            f"| Client | {eng.get('client_name', '-')} |",
            f"| Scope | {eng.get('scope', '-')} |",
            f"| Reporting Period | {eng.get('reporting_period', '-')} |",
            f"| Assurance Level | {eng.get('assurance_level', '-')} |",
            f"| Lead Auditor | {eng.get('lead_auditor', '-')} |",
            f"| Team Members | {', '.join(eng.get('team_members', []))} |",
            f"| Start Date | {eng.get('start_date', '-')} |",
            f"| Target Completion | {eng.get('target_completion', '-')} |",
            f"| Hours Spent | {self._format_number(eng.get('hours_spent', 0), 1)} |",
            f"| Hours Budget | {self._format_number(eng.get('hours_budget', 0), 1)} |",
        ]
        return "\n".join(lines)

    def _render_md_evidence_browser(self, data: Dict[str, Any]) -> str:
        """Render evidence browser with category tree."""
        evidence: List[Dict[str, Any]] = data.get("evidence", [])
        if not evidence:
            return "## Evidence Browser\n\n_No evidence items available._"

        lines = ["## Evidence Browser", ""]

        for pillar, standards in self.ESRS_CATEGORIES.items():
            lines.append(f"### {pillar} - {'Environmental' if pillar == 'E' else 'Social' if pillar == 'S' else 'Governance'}")
            lines.append("")
            for standard in standards:
                std_code = standard.split(" - ")[0]
                std_evidence = [
                    e for e in evidence if e.get("standard", "").startswith(std_code)
                ]
                lines.append(f"#### {standard} ({len(std_evidence)} items)")
                if std_evidence:
                    lines.append("")
                    lines.append("| # | Document | Type | Status | Uploaded |")
                    lines.append("|---|----------|------|--------|----------|")
                    for idx, item in enumerate(std_evidence, 1):
                        name = item.get("name", "-")
                        doc_type = item.get("type", "-")
                        status = item.get("status", "-")
                        uploaded = item.get("uploaded_at", "-")
                        lines.append(
                            f"| {idx} | {name} | {doc_type} | {status} | {uploaded} |"
                        )
                lines.append("")

        return "\n".join(lines)

    def _render_md_finding_tracker(self, data: Dict[str, Any]) -> str:
        """Render finding tracker table."""
        findings: List[Dict[str, Any]] = data.get("findings", [])
        if not findings:
            return "## Finding Tracker\n\n_No findings recorded._"

        lines = [
            "## Finding Tracker",
            "",
            f"**Total Findings:** {len(findings)}",
            "",
            "| Finding ID | Category | Severity | Status | Response | Due Date |",
            "|-----------|----------|----------|--------|----------|----------|",
        ]
        for f in findings:
            fid = f.get("finding_id", "-")
            category = f.get("category", "-")
            severity = f.get("severity", "-").upper()
            status = f.get("status", "-")
            response = f.get("response_summary", "-")
            due = f.get("due_date", "-")
            lines.append(
                f"| {fid} | {category} | {severity} | {status} | {response} | {due} |"
            )

        return "\n".join(lines)

    def _render_md_comment_threads(self, data: Dict[str, Any]) -> str:
        """Render comment threads."""
        comments: List[Dict[str, Any]] = data.get("comments", [])
        if not comments:
            return "## Discussion Threads\n\n_No comments yet._"

        lines = ["## Discussion Threads", ""]
        for thread in comments:
            finding_id = thread.get("finding_id", "-")
            lines.append(f"### Thread: {finding_id}")
            lines.append("")
            for msg in thread.get("messages", []):
                author = msg.get("author", "Unknown")
                timestamp = msg.get("timestamp", "")
                text = msg.get("text", "")
                role = msg.get("role", "")
                lines.append(f"> **{author}** ({role}) - {timestamp}")
                lines.append(f"> {text}")
                lines.append("")

        return "\n".join(lines)

    def _render_md_document_viewer(self, data: Dict[str, Any]) -> str:
        """Render document viewer listing."""
        documents: List[Dict[str, Any]] = data.get("documents", [])
        if not documents:
            return "## Linked Documents\n\n_No documents linked._"

        lines = [
            "## Linked Documents",
            "",
            "| # | Document | Type | Size | Finding | Uploaded |",
            "|---|----------|------|------|---------|----------|",
        ]
        for idx, doc in enumerate(documents, 1):
            name = doc.get("name", "-")
            doc_type = doc.get("type", "-")
            size = doc.get("size", "-")
            finding = doc.get("finding_id", "-")
            uploaded = doc.get("uploaded_at", "-")
            lines.append(
                f"| {idx} | {name} | {doc_type} | {size} | {finding} | {uploaded} |"
            )

        return "\n".join(lines)

    def _render_md_assurance_opinion(self, data: Dict[str, Any]) -> str:
        """Render assurance opinion form summary."""
        opinion = data.get("assurance_opinion", {})
        if not opinion:
            return "## Assurance Opinion\n\n_Opinion not yet formed._"

        lines = [
            "## Assurance Opinion",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Assurance Level | {opinion.get('assurance_level', '-')} |",
            f"| Opinion Type | {opinion.get('opinion_type', '-')} |",
            f"| Qualified Areas | {', '.join(opinion.get('qualified_areas', []))} |",
            f"| Key Observations | {opinion.get('key_observations', '-')} |",
            f"| Emphasis of Matter | {opinion.get('emphasis_of_matter', '-')} |",
            f"| Signing Partner | {opinion.get('signing_partner', '-')} |",
            f"| Opinion Date | {opinion.get('opinion_date', '-')} |",
        ]
        return "\n".join(lines)

    def _render_md_progress_tracker(self, data: Dict[str, Any]) -> str:
        """Render progress tracker."""
        progress = data.get("progress", {})
        if not progress:
            return "## Progress Tracker\n\n_No progress data available._"

        phases: List[Dict[str, Any]] = progress.get("phases", [])
        pending_items = progress.get("pending_items", [])

        lines = [
            "## Progress Tracker",
            "",
            "### Engagement Phases",
            "",
            "| Phase | Status | Completion | Due Date |",
            "|-------|--------|-----------|----------|",
        ]
        for p in phases:
            name = p.get("name", "-")
            status = p.get("status", "-")
            completion = self._format_percentage(p.get("completion_pct", 0))
            due = p.get("due_date", "-")
            lines.append(f"| {name} | {status} | {completion} | {due} |")

        if pending_items:
            lines.extend([
                "",
                "### Pending Items",
                "",
            ])
            for item in pending_items:
                lines.append(f"- {item.get('description', '-')} (Due: {item.get('due_date', '-')})")

        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Auditor Portal generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for auditor portal."""
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
.portal-container { max-width: 1300px; margin: 0 auto; padding: 24px; }
.portal-header { background: linear-gradient(135deg, #1e40af, #312e81);
    color: #fff; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }
.portal-header h1 { font-size: 26px; }
.portal-header .subtitle { opacity: 0.85; margin-top: 4px; font-size: 14px; }
.section { margin-bottom: 24px; background: var(--card-bg); border-radius: 10px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section-title { font-size: 18px; font-weight: 600; color: var(--primary);
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
.overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; margin-bottom: 16px; }
.overview-item { border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
.overview-item .ov-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.5px; }
.overview-item .ov-value { font-size: 18px; font-weight: 600; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
th { background: var(--primary-light); color: var(--primary); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:hover { background: #f8fafc; }
.severity-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; }
.severity-badge.critical { background: #fde8e8; color: #e02424; }
.severity-badge.major { background: #feecdc; color: #d97706; }
.severity-badge.moderate { background: #fef9c3; color: #92400e; }
.severity-badge.minor { background: #d1fae5; color: #057a55; }
.severity-badge.observation { background: #dbeafe; color: #1e40af; }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; }
.status-badge.open { background: #fde8e8; color: #e02424; }
.status-badge.in_progress { background: #fef9c3; color: #92400e; }
.status-badge.resolved { background: #d1fae5; color: #057a55; }
.status-badge.closed { background: #e2e8f0; color: #64748b; }
.status-badge.deferred { background: #dbeafe; color: #1e40af; }
.category-tree { list-style: none; }
.category-tree .pillar { font-weight: 600; font-size: 15px; padding: 8px 0;
    border-bottom: 1px solid var(--border); color: var(--primary); cursor: pointer; }
.category-tree .standard { padding: 6px 0 6px 20px; font-size: 13px; }
.category-tree .evidence-count { background: var(--primary-light); color: var(--primary);
    padding: 1px 8px; border-radius: 10px; font-size: 11px; margin-left: 6px; }
.comment-thread { border-left: 3px solid var(--primary); padding-left: 16px;
    margin-bottom: 16px; }
.comment-thread .thread-header { font-weight: 600; font-size: 14px;
    color: var(--primary); margin-bottom: 8px; }
.comment-msg { background: #f8fafc; border-radius: 8px; padding: 12px;
    margin-bottom: 8px; }
.comment-msg .msg-author { font-weight: 600; font-size: 13px; }
.comment-msg .msg-role { color: var(--text-muted); font-size: 11px; margin-left: 6px; }
.comment-msg .msg-time { color: var(--text-muted); font-size: 11px; float: right; }
.comment-msg .msg-text { margin-top: 6px; font-size: 13px; line-height: 1.5; }
.comment-msg.reply { margin-left: 20px; border-left: 2px solid var(--border); }
.opinion-form { border: 2px solid var(--primary); border-radius: 10px; padding: 20px; }
.opinion-field { margin-bottom: 12px; }
.opinion-field .field-label { font-size: 12px; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.opinion-field .field-value { font-size: 15px; font-weight: 500; }
.progress-bar-container { display: flex; align-items: center; margin-bottom: 8px; }
.progress-label { width: 140px; font-size: 13px; font-weight: 500; }
.progress-bar { flex: 1; height: 20px; background: #e2e8f0; border-radius: 10px;
    overflow: hidden; margin: 0 10px; }
.progress-fill { height: 100%; border-radius: 10px; transition: width 0.3s; }
.progress-fill.completed { background: var(--success); }
.progress-fill.in-progress { background: var(--info); }
.progress-fill.pending { background: #d1d5db; }
.progress-pct { width: 50px; text-align: right; font-size: 12px; font-weight: 600; }
.pending-list { list-style: none; margin-top: 16px; }
.pending-list li { padding: 8px 12px; border-left: 3px solid var(--warning);
    margin-bottom: 6px; background: #fffbeb; border-radius: 0 6px 6px 0; font-size: 13px; }
.pending-list .due-date { color: var(--text-muted); font-size: 11px; }
.doc-list { list-style: none; }
.doc-item { display: flex; align-items: center; padding: 10px 0;
    border-bottom: 1px solid var(--border); }
.doc-item .doc-icon { width: 36px; height: 36px; background: var(--primary-light);
    color: var(--primary); border-radius: 6px; display: flex; align-items: center;
    justify-content: center; font-size: 14px; font-weight: 700; margin-right: 12px; }
.doc-item .doc-info { flex: 1; }
.doc-item .doc-name { font-weight: 500; font-size: 13px; }
.doc-item .doc-meta { font-size: 11px; color: var(--text-muted); }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = self._escape_html(data.get("title", "Auditor Portal"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"portal-header\">\n"
            f"  <h1>{title}</h1>\n"
            f"  <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_engagement_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML engagement overview cards."""
        eng = data.get("engagement", {})
        if not eng:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Engagement Overview</h2>\n"
                "  <p>No engagement data available.</p>\n</div>"
            )

        fields = [
            ("Engagement ID", eng.get("engagement_id", "-")),
            ("Client", eng.get("client_name", "-")),
            ("Scope", eng.get("scope", "-")),
            ("Reporting Period", eng.get("reporting_period", "-")),
            ("Assurance Level", eng.get("assurance_level", "-")),
            ("Lead Auditor", eng.get("lead_auditor", "-")),
            ("Start Date", eng.get("start_date", "-")),
            ("Target Completion", eng.get("target_completion", "-")),
            ("Hours Spent", f"{self._format_number(eng.get('hours_spent', 0), 1)} / {self._format_number(eng.get('hours_budget', 0), 1)}"),
            ("Team", ", ".join(eng.get("team_members", []))),
        ]

        cards = ""
        for label, value in fields:
            cards += (
                f"<div class=\"overview-item\">\n"
                f"  <div class=\"ov-label\">{label}</div>\n"
                f"  <div class=\"ov-value\">{self._escape_html(str(value))}</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Engagement Overview</h2>\n"
            f"  <div class=\"overview-grid\">{cards}</div>\n"
            "</div>"
        )

    def _render_html_evidence_browser(self, data: Dict[str, Any]) -> str:
        """Render HTML evidence browser with ESRS category tree."""
        evidence: List[Dict[str, Any]] = data.get("evidence", [])

        tree_html = "<ul class=\"category-tree\">\n"
        for pillar, standards in self.ESRS_CATEGORIES.items():
            pillar_name = (
                "Environmental" if pillar == "E"
                else "Social" if pillar == "S"
                else "Governance"
            )
            pillar_evidence = [
                e for e in evidence if e.get("standard", "").startswith(pillar)
            ]
            tree_html += (
                f"<li class=\"pillar\">{pillar} - {pillar_name}"
                f"<span class=\"evidence-count\">{len(pillar_evidence)}</span></li>\n"
            )
            for standard in standards:
                std_code = standard.split(" - ")[0]
                std_evidence = [
                    e for e in evidence
                    if e.get("standard", "").startswith(std_code)
                ]
                tree_html += (
                    f"<li class=\"standard\">{standard}"
                    f"<span class=\"evidence-count\">{len(std_evidence)}</span></li>\n"
                )

                if std_evidence:
                    tree_html += "<table><thead><tr><th>Document</th><th>Type</th><th>Status</th><th>Uploaded</th></tr></thead><tbody>\n"
                    for item in std_evidence:
                        name = self._escape_html(item.get("name", "-"))
                        doc_type = item.get("type", "-")
                        status = item.get("status", "pending")
                        uploaded = item.get("uploaded_at", "-")
                        tree_html += (
                            f"<tr><td>{name}</td><td>{doc_type}</td>"
                            f"<td><span class=\"status-badge {status}\">{status}</span></td>"
                            f"<td>{uploaded}</td></tr>\n"
                        )
                    tree_html += "</tbody></table>\n"

        tree_html += "</ul>\n"

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Evidence Browser</h2>\n"
            f"  {tree_html}\n"
            "</div>"
        )

    def _render_html_finding_tracker(self, data: Dict[str, Any]) -> str:
        """Render HTML finding tracker table."""
        findings: List[Dict[str, Any]] = data.get("findings", [])
        if not findings:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Finding Tracker</h2>\n"
                "  <p>No findings recorded.</p>\n</div>"
            )

        summary = self._build_findings_summary(findings)
        summary_html = (
            "<div class=\"overview-grid\">\n"
            f"  <div class=\"overview-item\"><div class=\"ov-label\">Total</div>"
            f"<div class=\"ov-value\">{summary['total']}</div></div>\n"
            f"  <div class=\"overview-item\"><div class=\"ov-label\">Open</div>"
            f"<div class=\"ov-value\">{summary['open']}</div></div>\n"
            f"  <div class=\"overview-item\"><div class=\"ov-label\">Critical/Major</div>"
            f"<div class=\"ov-value\">{summary['critical_major']}</div></div>\n"
            f"  <div class=\"overview-item\"><div class=\"ov-label\">Resolved</div>"
            f"<div class=\"ov-value\">{summary['resolved']}</div></div>\n"
            "</div>\n"
        )

        rows = ""
        for f in findings:
            severity = f.get("severity", "minor")
            status = f.get("status", "open")
            rows += (
                f"<tr><td>{self._escape_html(f.get('finding_id', '-'))}</td>"
                f"<td>{self._escape_html(f.get('category', '-'))}</td>"
                f"<td><span class=\"severity-badge {severity}\">{severity}</span></td>"
                f"<td><span class=\"status-badge {status}\">{status}</span></td>"
                f"<td>{self._escape_html(f.get('response_summary', '-'))}</td>"
                f"<td>{f.get('due_date', '-')}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Finding Tracker</h2>\n"
            f"  {summary_html}\n"
            "  <table><thead><tr>"
            "<th>Finding ID</th><th>Category</th><th>Severity</th>"
            "<th>Status</th><th>Response</th><th>Due Date</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_comment_threads(self, data: Dict[str, Any]) -> str:
        """Render HTML comment threads."""
        comments: List[Dict[str, Any]] = data.get("comments", [])
        if not comments:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Discussion Threads</h2>\n"
                "  <p>No comments yet.</p>\n</div>"
            )

        threads_html = ""
        for thread in comments:
            finding_id = self._escape_html(thread.get("finding_id", "-"))
            msgs = ""
            for idx, msg in enumerate(thread.get("messages", [])):
                reply_cls = " reply" if idx > 0 else ""
                author = self._escape_html(msg.get("author", "Unknown"))
                role = self._escape_html(msg.get("role", ""))
                timestamp = msg.get("timestamp", "")
                text = self._escape_html(msg.get("text", ""))
                msgs += (
                    f"<div class=\"comment-msg{reply_cls}\">\n"
                    f"  <span class=\"msg-author\">{author}</span>"
                    f"  <span class=\"msg-role\">{role}</span>"
                    f"  <span class=\"msg-time\">{timestamp}</span>\n"
                    f"  <div class=\"msg-text\">{text}</div>\n"
                    f"</div>\n"
                )

            threads_html += (
                f"<div class=\"comment-thread\">\n"
                f"  <div class=\"thread-header\">Finding: {finding_id}</div>\n"
                f"  {msgs}\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Discussion Threads</h2>\n"
            f"  {threads_html}\n"
            "</div>"
        )

    def _render_html_document_viewer(self, data: Dict[str, Any]) -> str:
        """Render HTML document viewer listing."""
        documents: List[Dict[str, Any]] = data.get("documents", [])
        if not documents:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Linked Documents</h2>\n"
                "  <p>No documents linked.</p>\n</div>"
            )

        items = ""
        for doc in documents:
            name = self._escape_html(doc.get("name", "-"))
            doc_type = doc.get("type", "PDF")
            ext = doc_type[:3].upper()
            size = doc.get("size", "-")
            finding = doc.get("finding_id", "-")
            uploaded = doc.get("uploaded_at", "-")
            items += (
                f"<div class=\"doc-item\">\n"
                f"  <div class=\"doc-icon\">{ext}</div>\n"
                f"  <div class=\"doc-info\">\n"
                f"    <div class=\"doc-name\">{name}</div>\n"
                f"    <div class=\"doc-meta\">{doc_type} | {size} | Finding: {finding} | {uploaded}</div>\n"
                f"  </div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Linked Documents</h2>\n"
            f"  <div class=\"doc-list\">{items}</div>\n"
            "</div>"
        )

    def _render_html_assurance_opinion(self, data: Dict[str, Any]) -> str:
        """Render HTML assurance opinion form."""
        opinion = data.get("assurance_opinion", {})
        if not opinion:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Assurance Opinion</h2>\n"
                "  <p>Opinion not yet formed.</p>\n</div>"
            )

        fields = [
            ("Assurance Level", opinion.get("assurance_level", "-")),
            ("Opinion Type", opinion.get("opinion_type", "-")),
            ("Qualified Areas", ", ".join(opinion.get("qualified_areas", []))),
            ("Key Observations", opinion.get("key_observations", "-")),
            ("Emphasis of Matter", opinion.get("emphasis_of_matter", "-")),
            ("Signing Partner", opinion.get("signing_partner", "-")),
            ("Opinion Date", opinion.get("opinion_date", "-")),
        ]

        form_html = ""
        for label, value in fields:
            form_html += (
                f"<div class=\"opinion-field\">\n"
                f"  <div class=\"field-label\">{label}</div>\n"
                f"  <div class=\"field-value\">{self._escape_html(str(value))}</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Assurance Opinion</h2>\n"
            f"  <div class=\"opinion-form\">{form_html}</div>\n"
            "</div>"
        )

    def _render_html_progress_tracker(self, data: Dict[str, Any]) -> str:
        """Render HTML progress tracker with bars."""
        progress = data.get("progress", {})
        if not progress:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Progress Tracker</h2>\n"
                "  <p>No progress data available.</p>\n</div>"
            )

        phases = progress.get("phases", [])
        bars_html = ""
        for p in phases:
            name = self._escape_html(p.get("name", "-"))
            completion = p.get("completion_pct", 0)
            status = p.get("status", "pending").replace(" ", "-").lower()
            fill_cls = (
                "completed" if completion >= 100
                else "in-progress" if completion > 0
                else "pending"
            )
            bars_html += (
                f"<div class=\"progress-bar-container\">\n"
                f"  <div class=\"progress-label\">{name}</div>\n"
                f"  <div class=\"progress-bar\">"
                f"<div class=\"progress-fill {fill_cls}\" "
                f"style=\"width:{completion:.0f}%\"></div></div>\n"
                f"  <div class=\"progress-pct\">"
                f"{self._format_percentage(completion)}</div>\n"
                f"</div>\n"
            )

        pending_items = progress.get("pending_items", [])
        pending_html = ""
        if pending_items:
            items = ""
            for item in pending_items:
                desc = self._escape_html(item.get("description", "-"))
                due = item.get("due_date", "-")
                items += (
                    f"<li>{desc} <span class=\"due-date\">Due: {due}</span></li>\n"
                )
            pending_html = (
                f"<h3 style=\"margin-top:16px;font-size:15px;color:#1e40af\">"
                f"Pending Items</h3>\n"
                f"<ul class=\"pending-list\">{items}</ul>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Progress Tracker</h2>\n"
            f"  {bars_html}\n"
            f"  {pending_html}\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Auditor Portal generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _build_json_engagement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON engagement section."""
        return data.get("engagement", {})

    def _build_json_evidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON evidence section grouped by category."""
        evidence: List[Dict[str, Any]] = data.get("evidence", [])
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for e in evidence:
            std = e.get("standard", "unknown")
            grouped.setdefault(std, []).append(e)
        return {
            "total_items": len(evidence),
            "by_standard": grouped,
        }

    def _build_json_findings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON findings section."""
        findings = data.get("findings", [])
        return {
            "total": len(findings),
            "summary": self._build_findings_summary(findings),
            "items": findings,
        }

    def _build_json_comments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON comments section."""
        return data.get("comments", [])

    def _build_json_documents(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON documents section."""
        return data.get("documents", [])

    def _build_json_opinion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON assurance opinion section."""
        return data.get("assurance_opinion", {})

    def _build_json_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON progress section."""
        return data.get("progress", {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_findings_summary(findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Build a summary count of findings by status and severity.

        Args:
            findings: List of finding dicts.

        Returns:
            Summary counts dict.
        """
        total = len(findings)
        open_count = sum(1 for f in findings if f.get("status") == "open")
        critical_major = sum(
            1 for f in findings
            if f.get("severity") in ("critical", "major")
        )
        resolved = sum(
            1 for f in findings
            if f.get("status") in ("resolved", "closed")
        )
        return {
            "total": total,
            "open": open_count,
            "critical_major": critical_major,
            "resolved": resolved,
        }

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash.

        Args:
            content: Content to hash.

        Returns:
            Hexadecimal SHA-256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format numeric value with thousands separator.

        Args:
            value: Numeric value.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage.

        Args:
            value: Numeric value.

        Returns:
            Percentage string.
        """
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string.

        Args:
            dt: Datetime object.

        Returns:
            Formatted date string.
        """
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Raw text.

        Returns:
            HTML-safe string.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
