# -*- coding: utf-8 -*-
"""
ComplianceDashboardTemplate - Multi-Framework Compliance Dashboard for PACK-041.

Generates a multi-framework compliance dashboard covering framework readiness
scores (0-100 bar chart data), per-framework gap analysis tables, critical
gaps requiring immediate action, remediation action plan with priority,
submission timeline and deadlines, and framework-specific notes.

Sections:
    1. Framework Readiness Scores
    2. Per-Framework Gap Analysis
    3. Critical Gaps
    4. Remediation Action Plan
    5. Submission Timeline
    6. Framework-Specific Notes

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Regulatory References:
    - GHG Protocol Corporate Standard
    - ESRS E1 Climate Change
    - CDP Climate Change Questionnaire
    - TCFD Recommendations
    - SBTi Criteria v5.1
    - ISO 14064-1:2018

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "41.0.0"


def _score_color(score: float) -> str:
    """Return CSS class for score value."""
    if score >= 80:
        return "score-high"
    if score >= 50:
        return "score-medium"
    return "score-low"


def _score_label(score: float) -> str:
    """Return text label for score."""
    if score >= 90:
        return "Ready"
    if score >= 70:
        return "Near Ready"
    if score >= 50:
        return "In Progress"
    return "Significant Gaps"


class ComplianceDashboardTemplate:
    """
    Multi-framework compliance dashboard template.

    Renders compliance dashboards with framework readiness scores, gap
    analysis, critical gaps, remediation plans, submission timelines, and
    framework notes. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ComplianceDashboardTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ComplianceDashboardTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render compliance dashboard as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_readiness_scores(data),
            self._md_gap_analysis(data),
            self._md_critical_gaps(data),
            self._md_remediation_plan(data),
            self._md_submission_timeline(data),
            self._md_framework_notes(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render compliance dashboard as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_readiness_scores(data),
            self._html_gap_analysis(data),
            self._html_critical_gaps(data),
            self._html_remediation_plan(data),
            self._html_submission_timeline(data),
            self._html_framework_notes(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render compliance dashboard as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "compliance_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "readiness_scores": data.get("readiness_scores", []),
            "gap_analysis": data.get("gap_analysis", []),
            "critical_gaps": data.get("critical_gaps", []),
            "remediation_plan": data.get("remediation_plan", []),
            "submission_timeline": data.get("submission_timeline", []),
            "framework_notes": data.get("framework_notes", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Compliance Dashboard - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_readiness_scores(self, data: Dict[str, Any]) -> str:
        """Render Markdown framework readiness scores."""
        scores = data.get("readiness_scores", [])
        if not scores:
            return "## 1. Framework Readiness Scores\n\nNo readiness data available."
        lines = [
            "## 1. Framework Readiness Scores",
            "",
            "| Framework | Score | Status | Met | Total | Bar |",
            "|-----------|-------|--------|-----|-------|-----|",
        ]
        for fw in sorted(scores, key=lambda x: x.get("score", 0), reverse=True):
            name = fw.get("framework", "")
            score = fw.get("score", 0)
            status = _score_label(score)
            met = fw.get("requirements_met", 0)
            total = fw.get("requirements_total", 0)
            bar = "#" * int(score / 5)  # Simple text bar
            lines.append(f"| {name} | {score:.0f}% | {status} | {met} | {total} | {bar} |")
        avg = sum(fw.get("score", 0) for fw in scores) / len(scores) if scores else 0
        lines.append(f"\n**Average Readiness:** {avg:.0f}%")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown per-framework gap analysis."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return "## 2. Per-Framework Gap Analysis\n\nNo gap analysis performed."
        lines = ["## 2. Per-Framework Gap Analysis"]
        for fw in gaps:
            framework = fw.get("framework", "")
            requirements = fw.get("requirements", [])
            lines.extend([
                "",
                f"### {framework}",
                "",
                "| Requirement | Status | Priority | Gap Description |",
                "|-------------|--------|---------|-----------------|",
            ])
            for req in requirements:
                name = req.get("requirement", "")
                status = req.get("status", "-")
                priority = req.get("priority", "-")
                gap = req.get("gap_description", "-")
                lines.append(f"| {name} | **{status}** | {priority} | {gap} |")
        return "\n".join(lines)

    def _md_critical_gaps(self, data: Dict[str, Any]) -> str:
        """Render Markdown critical gaps."""
        critical = data.get("critical_gaps", [])
        if not critical:
            return "## 3. Critical Gaps\n\nNo critical gaps identified."
        lines = [
            "## 3. Critical Gaps Requiring Immediate Action",
            "",
            "| Framework | Gap | Impact | Days to Deadline | Required Action |",
            "|-----------|-----|--------|-----------------|----------------|",
        ]
        for gap in critical:
            fw = gap.get("framework", "")
            desc = gap.get("gap_description", "")
            impact = gap.get("impact", "-")
            days = gap.get("days_to_deadline", "-")
            action = gap.get("required_action", "-")
            lines.append(f"| {fw} | {desc} | {impact} | {days} | {action} |")
        return "\n".join(lines)

    def _md_remediation_plan(self, data: Dict[str, Any]) -> str:
        """Render Markdown remediation action plan."""
        plan = data.get("remediation_plan", [])
        if not plan:
            return "## 4. Remediation Action Plan\n\nNo remediation actions planned."
        lines = [
            "## 4. Remediation Action Plan",
            "",
            "| Priority | Action | Framework(s) | Owner | Target Date | Status | Effort |",
            "|----------|--------|-------------|-------|------------|--------|--------|",
        ]
        for action in plan:
            priority = action.get("priority", "-")
            desc = action.get("description", "")
            frameworks = ", ".join(action.get("frameworks", []))
            owner = action.get("owner", "-")
            target = action.get("target_date", "-")
            status = action.get("status", "-")
            effort = action.get("effort_estimate", "-")
            lines.append(f"| {priority} | {desc} | {frameworks} | {owner} | {target} | {status} | {effort} |")
        return "\n".join(lines)

    def _md_submission_timeline(self, data: Dict[str, Any]) -> str:
        """Render Markdown submission timeline."""
        timeline = data.get("submission_timeline", [])
        if not timeline:
            return "## 5. Submission Timeline\n\nNo submission deadlines configured."
        lines = [
            "## 5. Submission Timeline & Deadlines",
            "",
            "| Framework | Submission Type | Deadline | Status | Days Remaining | Notes |",
            "|-----------|---------------|---------|--------|---------------|-------|",
        ]
        for entry in sorted(timeline, key=lambda x: x.get("deadline", "9999")):
            fw = entry.get("framework", "")
            sub_type = entry.get("submission_type", "-")
            deadline = entry.get("deadline", "-")
            status = entry.get("status", "-")
            days = entry.get("days_remaining", "-")
            notes = entry.get("notes", "-")
            lines.append(f"| {fw} | {sub_type} | {deadline} | **{status}** | {days} | {notes} |")
        return "\n".join(lines)

    def _md_framework_notes(self, data: Dict[str, Any]) -> str:
        """Render Markdown framework-specific notes."""
        notes = data.get("framework_notes", [])
        if not notes:
            return ""
        lines = ["## 6. Framework-Specific Notes", ""]
        for note in notes:
            fw = note.get("framework", "")
            content = note.get("note", "")
            lines.append(f"**{fw}:** {content}\n")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-041 Scope 1-2 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Compliance Dashboard - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #e9c46a;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".score-high{color:#2a9d8f;font-weight:700;}\n"
            ".score-medium{color:#e9c46a;font-weight:700;}\n"
            ".score-low{color:#e63946;font-weight:700;}\n"
            ".bar-bg{background:#e8e8e8;border-radius:4px;height:20px;width:200px;display:inline-block;}\n"
            ".bar-fill{border-radius:4px;height:20px;display:inline-block;}\n"
            ".bar-green{background:#2a9d8f;}\n"
            ".bar-yellow{background:#e9c46a;}\n"
            ".bar-red{background:#e63946;}\n"
            ".critical{background:#fff0f0;border-left:4px solid #e63946;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Compliance Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n<hr>\n</div>"
        )

    def _html_readiness_scores(self, data: Dict[str, Any]) -> str:
        """Render HTML readiness scores with bar charts."""
        scores = data.get("readiness_scores", [])
        if not scores:
            return ""
        rows = ""
        for fw in sorted(scores, key=lambda x: x.get("score", 0), reverse=True):
            name = fw.get("framework", "")
            score = fw.get("score", 0)
            css = _score_color(score)
            bar_color = "bar-green" if score >= 80 else ("bar-yellow" if score >= 50 else "bar-red")
            bar_width = int(score * 2)
            status = _score_label(score)
            rows += (
                f'<tr><td>{name}</td><td class="{css}">{score:.0f}%</td>'
                f'<td>{status}</td><td><div class="bar-bg">'
                f'<div class="bar-fill {bar_color}" style="width:{bar_width}px"></div>'
                f"</div></td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. Framework Readiness</h2>\n'
            "<table><thead><tr><th>Framework</th><th>Score</th>"
            "<th>Status</th><th>Progress</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return ""
        parts = ['<div class="section">\n<h2>2. Gap Analysis</h2>']
        for fw in gaps:
            framework = fw.get("framework", "")
            requirements = fw.get("requirements", [])
            rows = ""
            for req in requirements:
                status = req.get("status", "-")
                css = "score-high" if status == "Met" else ("score-medium" if status == "Partial" else "score-low")
                rows += (
                    f'<tr><td>{req.get("requirement", "")}</td>'
                    f'<td class="{css}">{status}</td>'
                    f'<td>{req.get("priority", "-")}</td>'
                    f'<td>{req.get("gap_description", "-")}</td></tr>\n'
                )
            parts.append(
                f"<h3>{framework}</h3>\n"
                "<table><thead><tr><th>Requirement</th><th>Status</th>"
                f"<th>Priority</th><th>Gap</th></tr></thead>\n<tbody>{rows}</tbody></table>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _html_critical_gaps(self, data: Dict[str, Any]) -> str:
        """Render HTML critical gaps."""
        critical = data.get("critical_gaps", [])
        if not critical:
            return ""
        rows = ""
        for gap in critical:
            rows += (
                f'<tr class="critical"><td>{gap.get("framework", "")}</td>'
                f'<td>{gap.get("gap_description", "")}</td>'
                f'<td>{gap.get("impact", "-")}</td>'
                f'<td>{gap.get("days_to_deadline", "-")}</td>'
                f'<td>{gap.get("required_action", "-")}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Critical Gaps</h2>\n'
            "<table><thead><tr><th>Framework</th><th>Gap</th><th>Impact</th>"
            "<th>Days Left</th><th>Action</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_remediation_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML remediation plan."""
        plan = data.get("remediation_plan", [])
        if not plan:
            return ""
        rows = ""
        for action in plan:
            frameworks = ", ".join(action.get("frameworks", []))
            rows += (
                f"<tr><td>{action.get('priority', '-')}</td>"
                f"<td>{action.get('description', '')}</td>"
                f"<td>{frameworks}</td><td>{action.get('owner', '-')}</td>"
                f"<td>{action.get('target_date', '-')}</td>"
                f"<td>{action.get('status', '-')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Remediation Plan</h2>\n'
            "<table><thead><tr><th>Priority</th><th>Action</th>"
            "<th>Framework(s)</th><th>Owner</th><th>Target</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_submission_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML submission timeline."""
        timeline = data.get("submission_timeline", [])
        if not timeline:
            return ""
        rows = ""
        for entry in sorted(timeline, key=lambda x: x.get("deadline", "9999")):
            days = entry.get("days_remaining")
            css = "critical" if days and days < 30 else ""
            rows += (
                f'<tr class="{css}"><td>{entry.get("framework", "")}</td>'
                f'<td>{entry.get("submission_type", "-")}</td>'
                f'<td>{entry.get("deadline", "-")}</td>'
                f'<td>{entry.get("status", "-")}</td>'
                f'<td>{days if days is not None else "-"}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>5. Submission Timeline</h2>\n'
            "<table><thead><tr><th>Framework</th><th>Type</th>"
            "<th>Deadline</th><th>Status</th><th>Days Left</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_framework_notes(self, data: Dict[str, Any]) -> str:
        """Render HTML framework notes."""
        notes = data.get("framework_notes", [])
        if not notes:
            return ""
        items = ""
        for note in notes:
            items += f"<li><strong>{note.get('framework', '')}:</strong> {note.get('note', '')}</li>\n"
        return f'<div class="section">\n<h2>6. Framework Notes</h2>\n<ul>{items}</ul>\n</div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-041 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
