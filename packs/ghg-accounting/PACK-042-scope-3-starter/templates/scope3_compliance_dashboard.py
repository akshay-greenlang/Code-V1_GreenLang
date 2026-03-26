# -*- coding: utf-8 -*-
"""
Scope3ComplianceDashboardTemplate - Multi-Framework Compliance for PACK-042.

Generates a multi-framework Scope 3 compliance readiness dashboard
covering per-framework compliance scores (0-100%), requirements checklist
with pass/fail, gap analysis summary, action items with priority and
effort, and cross-framework coverage matrix.

Frameworks covered:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard
    - ESRS E1 Climate Change (E1-6 para 44-46)
    - CDP Climate Change Questionnaire (C6.5)
    - SBTi Corporate Net-Zero Standard (Scope 3)
    - SEC Climate Disclosure Rule
    - SB 253 (California Climate Corporate Data Accountability Act)

Sections:
    1. Compliance Scorecard
    2. Requirements Checklist
    3. Gap Analysis Summary
    4. Action Items
    5. Cross-Framework Coverage Matrix

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, compliance purple theme)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 42.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "42.0.0"

FRAMEWORKS = [
    "GHG Protocol",
    "ESRS E1",
    "CDP",
    "SBTi",
    "SEC",
    "SB 253",
]


def _score_color_css(score: Optional[float]) -> str:
    """Return CSS class for compliance score."""
    if score is None:
        return ""
    if score >= 80:
        return "score-high"
    if score >= 50:
        return "score-medium"
    return "score-low"


def _score_label(score: Optional[float]) -> str:
    """Return text label for compliance score."""
    if score is None:
        return "N/A"
    if score >= 90:
        return "Ready"
    if score >= 70:
        return "Near Ready"
    if score >= 50:
        return "In Progress"
    return "Significant Gaps"


class Scope3ComplianceDashboardTemplate:
    """
    Multi-framework Scope 3 compliance readiness dashboard template.

    Renders compliance dashboards with framework readiness scores,
    requirements checklists with pass/fail, gap analysis summaries,
    action items with priority and effort, and cross-framework coverage
    matrices. Covers GHG Protocol, ESRS E1, CDP, SBTi, SEC, and SB 253.
    All outputs include SHA-256 provenance hashing for audit trails.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope3ComplianceDashboardTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3ComplianceDashboardTemplate."""
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

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render compliance dashboard as Markdown.

        Args:
            data: Validated compliance data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_scorecard(data),
            self._md_requirements_checklist(data),
            self._md_gap_analysis(data),
            self._md_action_items(data),
            self._md_coverage_matrix(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render compliance dashboard as HTML.

        Args:
            data: Validated compliance data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_scorecard(data),
            self._html_requirements_checklist(data),
            self._html_gap_analysis(data),
            self._html_action_items(data),
            self._html_coverage_matrix(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render compliance dashboard as JSON-serializable dict.

        Args:
            data: Validated compliance data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "scope3_compliance_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "scorecard": data.get("scorecard", []),
            "requirements_checklist": data.get("requirements_checklist", []),
            "gap_analysis": data.get("gap_analysis", []),
            "action_items": data.get("action_items", []),
            "coverage_matrix": data.get("coverage_matrix", []),
            "chart_data": {
                "bar_chart": [
                    {
                        "framework": fw.get("framework_name", ""),
                        "score": fw.get("score", 0),
                        "label": _score_label(fw.get("score")),
                    }
                    for fw in data.get("scorecard", [])
                ],
            },
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Compliance Dashboard - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_scorecard(self, data: Dict[str, Any]) -> str:
        """Render Markdown compliance scorecard."""
        scorecard = data.get("scorecard", [])
        if not scorecard:
            return "## 1. Compliance Scorecard\n\nNo compliance data available."
        lines = [
            "## 1. Compliance Scorecard",
            "",
            "| Framework | Score | Status | Key Requirement | Deadline |",
            "|-----------|-------|--------|----------------|----------|",
        ]
        for fw in scorecard:
            name = fw.get("framework_name", "")
            score = fw.get("score")
            score_str = f"{score:.0f}%" if score is not None else "-"
            status = _score_label(score)
            key_req = fw.get("key_requirement", "-")
            deadline = fw.get("deadline", "-")
            lines.append(
                f"| {name} | {score_str} | {status} | {key_req} | {deadline} |"
            )
        return "\n".join(lines)

    def _md_requirements_checklist(self, data: Dict[str, Any]) -> str:
        """Render Markdown requirements checklist."""
        checklist = data.get("requirements_checklist", [])
        if not checklist:
            return "## 2. Requirements Checklist\n\nNo requirements checklist available."
        lines = [
            "## 2. Requirements Checklist",
            "",
        ]
        for fw in checklist:
            fw_name = fw.get("framework_name", "")
            reqs = fw.get("requirements", [])
            lines.append(f"### {fw_name}")
            lines.append("")
            lines.append("| Requirement | Status | Notes |")
            lines.append("|------------|--------|-------|")
            for req in reqs:
                name = req.get("requirement", "")
                status = req.get("status", "-")
                icon = "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "PARTIAL"
                notes = req.get("notes", "-")
                lines.append(f"| {name} | **{icon}** | {notes} |")
            lines.append("")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis summary."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return "## 3. Gap Analysis Summary\n\nNo gaps identified."
        lines = [
            "## 3. Gap Analysis Summary",
            "",
            "| Framework | Gap | Severity | Impact | Remediation |",
            "|-----------|-----|----------|--------|-------------|",
        ]
        for g in gaps:
            fw = g.get("framework_name", "")
            gap = g.get("gap_description", "")
            severity = g.get("severity", "-")
            impact = g.get("impact", "-")
            remediation = g.get("remediation", "-")
            lines.append(
                f"| {fw} | {gap} | {severity} | {impact} | {remediation} |"
            )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown action items."""
        actions = data.get("action_items", [])
        if not actions:
            return "## 4. Action Items\n\nNo action items defined."
        lines = [
            "## 4. Action Items",
            "",
            "| Priority | Action | Framework(s) | Effort | Owner | Deadline |",
            "|----------|--------|-------------|--------|-------|----------|",
        ]
        for a in actions:
            priority = a.get("priority", "-")
            action = a.get("action", "")
            frameworks = a.get("frameworks", "-")
            if isinstance(frameworks, list):
                frameworks = ", ".join(frameworks)
            effort = a.get("effort", "-")
            owner = a.get("owner", "-")
            deadline = a.get("deadline", "-")
            lines.append(
                f"| {priority} | {action} | {frameworks} | {effort} | {owner} | {deadline} |"
            )
        return "\n".join(lines)

    def _md_coverage_matrix(self, data: Dict[str, Any]) -> str:
        """Render Markdown cross-framework coverage matrix."""
        matrix = data.get("coverage_matrix", [])
        if not matrix:
            return "## 5. Cross-Framework Coverage Matrix\n\nNo coverage matrix available."
        fw_names = [fw for fw in FRAMEWORKS]
        header = "| Requirement |"
        sep = "|------------|"
        for fw in fw_names:
            header += f" {fw} |"
            sep += "-----|"
        lines = [
            "## 5. Cross-Framework Coverage Matrix",
            "",
            header,
            sep,
        ]
        for item in matrix:
            req = item.get("requirement", "")
            row = f"| {req} |"
            coverage = item.get("coverage", {})
            for fw in fw_names:
                key = fw.lower().replace(" ", "_")
                val = coverage.get(key, "-")
                row += f" {val} |"
            lines.append(row)
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scope 3 Compliance Dashboard - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#8E44AD;border-bottom:3px solid #8E44AD;padding-bottom:0.5rem;}\n"
            "h2{color:#6C3483;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#8E44AD;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f4ecf7;font-weight:600;color:#6C3483;}\n"
            "tr:nth-child(even){background:#faf5fd;}\n"
            ".score-high{color:#27AE60;font-weight:700;}\n"
            ".score-medium{color:#F39C12;font-weight:700;}\n"
            ".score-low{color:#E74C3C;font-weight:700;}\n"
            ".metric-card{display:inline-block;background:#f4ecf7;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:150px;"
            "border-top:3px solid #8E44AD;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#6C3483;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".pass{color:#27AE60;font-weight:700;}\n"
            ".fail{color:#E74C3C;font-weight:700;}\n"
            ".partial{color:#F39C12;font-weight:700;}\n"
            ".severity-high{color:#E74C3C;font-weight:700;}\n"
            ".severity-medium{color:#F39C12;font-weight:700;}\n"
            ".severity-low{color:#27AE60;font-weight:600;}\n"
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
            f"<h1>Scope 3 Compliance Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_scorecard(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance scorecard with cards."""
        scorecard = data.get("scorecard", [])
        if not scorecard:
            return ""
        cards = ""
        for fw in scorecard:
            name = fw.get("framework_name", "")
            score = fw.get("score")
            score_str = f"{score:.0f}%" if score is not None else "N/A"
            css = _score_color_css(score)
            cards += (
                f'<div class="metric-card">'
                f'<div class="metric-value {css}">{score_str}</div>'
                f'<div class="metric-label">{name}</div></div>\n'
            )
        rows = ""
        for fw in scorecard:
            name = fw.get("framework_name", "")
            score = fw.get("score")
            score_str = f"{score:.0f}%" if score is not None else "-"
            css = _score_color_css(score)
            status = _score_label(score)
            key_req = fw.get("key_requirement", "-")
            deadline = fw.get("deadline", "-")
            rows += (
                f'<tr><td>{name}</td><td class="{css}">{score_str}</td>'
                f"<td>{status}</td><td>{key_req}</td><td>{deadline}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>1. Compliance Scorecard</h2>\n"
            f"<div>{cards}</div>\n"
            "<table><thead><tr><th>Framework</th><th>Score</th><th>Status</th>"
            f"<th>Key Requirement</th><th>Deadline</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_requirements_checklist(self, data: Dict[str, Any]) -> str:
        """Render HTML requirements checklist."""
        checklist = data.get("requirements_checklist", [])
        if not checklist:
            return ""
        sections = ""
        for fw in checklist:
            fw_name = fw.get("framework_name", "")
            reqs = fw.get("requirements", [])
            rows = ""
            for req in reqs:
                name = req.get("requirement", "")
                status = req.get("status", "-")
                css = "pass" if status == "PASS" else "fail" if status == "FAIL" else "partial"
                notes = req.get("notes", "-")
                rows += f'<tr><td>{name}</td><td class="{css}">{status}</td><td>{notes}</td></tr>\n'
            sections += (
                f"<h3>{fw_name}</h3>\n"
                "<table><thead><tr><th>Requirement</th><th>Status</th>"
                f"<th>Notes</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Requirements Checklist</h2>\n"
            f"{sections}</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis summary."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            fw = g.get("framework_name", "")
            gap = g.get("gap_description", "")
            severity = g.get("severity", "-")
            sev_css = f"severity-{severity.lower()}" if severity in ("High", "Medium", "Low") else ""
            impact = g.get("impact", "-")
            remediation = g.get("remediation", "-")
            rows += (
                f'<tr><td>{fw}</td><td>{gap}</td><td class="{sev_css}">{severity}</td>'
                f"<td>{impact}</td><td>{remediation}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Gap Analysis Summary</h2>\n"
            "<table><thead><tr><th>Framework</th><th>Gap</th><th>Severity</th>"
            f"<th>Impact</th><th>Remediation</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items."""
        actions = data.get("action_items", [])
        if not actions:
            return ""
        rows = ""
        for a in actions:
            priority = a.get("priority", "-")
            p_css = f"severity-{priority.lower()}" if priority in ("High", "Medium", "Low") else ""
            action = a.get("action", "")
            frameworks = a.get("frameworks", "-")
            if isinstance(frameworks, list):
                frameworks = ", ".join(frameworks)
            effort = a.get("effort", "-")
            owner = a.get("owner", "-")
            deadline = a.get("deadline", "-")
            rows += (
                f'<tr><td class="{p_css}">{priority}</td><td>{action}</td>'
                f"<td>{frameworks}</td><td>{effort}</td><td>{owner}</td>"
                f"<td>{deadline}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Action Items</h2>\n"
            "<table><thead><tr><th>Priority</th><th>Action</th><th>Framework(s)</th>"
            f"<th>Effort</th><th>Owner</th><th>Deadline</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_coverage_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML cross-framework coverage matrix."""
        matrix = data.get("coverage_matrix", [])
        if not matrix:
            return ""
        fw_headers = "".join(f"<th>{fw}</th>" for fw in FRAMEWORKS)
        rows = ""
        for item in matrix:
            req = item.get("requirement", "")
            row = f"<td>{req}</td>"
            coverage = item.get("coverage", {})
            for fw in FRAMEWORKS:
                key = fw.lower().replace(" ", "_")
                val = coverage.get(key, "-")
                css = "pass" if val == "Yes" else "fail" if val == "No" else ""
                row += f'<td class="{css}">{val}</td>'
            rows += f"<tr>{row}</tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. Cross-Framework Coverage Matrix</h2>\n"
            f"<table><thead><tr><th>Requirement</th>{fw_headers}</tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
