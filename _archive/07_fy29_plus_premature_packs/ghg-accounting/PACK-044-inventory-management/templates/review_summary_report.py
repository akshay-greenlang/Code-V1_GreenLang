# -*- coding: utf-8 -*-
"""
ReviewSummaryReport - Review Decisions and Comments for PACK-044.

Generates a review summary report covering review cycles, reviewer
decisions, review comments, open issues, and sign-off status.

Sections:
    1. Review Cycle Summary
    2. Reviewer Decisions
    3. Review Comments and Issues
    4. Sign-Off Status

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "44.0.0"


class ReviewSummaryReport:
    """
    Review summary report template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ReviewSummaryReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render review summary as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_cycle_summary(data),
            self._md_decisions(data),
            self._md_comments(data),
            self._md_signoff(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render review summary as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_cycle_summary(data),
            self._html_decisions(data),
            self._html_comments(data),
            self._html_signoff(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render review summary as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "review_summary_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "review_cycle": data.get("review_cycle", {}),
            "decisions": data.get("decisions", []),
            "comments": data.get("comments", []),
            "signoffs": data.get("signoffs", []),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            f"# Review Summary Report - {company}\n\n"
            f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_cycle_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown review cycle summary."""
        cycle = data.get("review_cycle", {})
        if not cycle:
            return "## 1. Review Cycle Summary\n\nNo review cycle data."
        return (
            "## 1. Review Cycle Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Review Cycle | {cycle.get('cycle_name', '-')} |\n"
            f"| Start Date | {cycle.get('start_date', '-')} |\n"
            f"| End Date | {cycle.get('end_date', '-')} |\n"
            f"| Status | **{cycle.get('status', '-')}** |\n"
            f"| Reviewers | {cycle.get('reviewer_count', 0)} |\n"
            f"| Items Reviewed | {cycle.get('items_reviewed', 0)} |\n"
            f"| Issues Found | {cycle.get('issues_found', 0)} |\n"
            f"| Issues Resolved | {cycle.get('issues_resolved', 0)} |"
        )

    def _md_decisions(self, data: Dict[str, Any]) -> str:
        """Render Markdown reviewer decisions."""
        decisions = data.get("decisions", [])
        if not decisions:
            return ""
        lines = [
            "## 2. Reviewer Decisions",
            "",
            "| Reviewer | Scope | Decision | Date | Condition |",
            "|---------|-------|----------|------|-----------|",
        ]
        for d in decisions:
            lines.append(
                f"| {d.get('reviewer', '')} | {d.get('scope', '')} | "
                f"**{d.get('decision', '')}** | {d.get('date', '')} | {d.get('condition', '-')} |"
            )
        return "\n".join(lines)

    def _md_comments(self, data: Dict[str, Any]) -> str:
        """Render Markdown review comments."""
        comments = data.get("comments", [])
        if not comments:
            return ""
        lines = [
            "## 3. Review Comments and Issues",
            "",
            "| ID | Reviewer | Category | Severity | Comment | Status |",
            "|----|---------|----------|---------|---------|--------|",
        ]
        for c in comments:
            lines.append(
                f"| {c.get('comment_id', '')} | {c.get('reviewer', '')} | "
                f"{c.get('category', '')} | {c.get('severity', 'info')} | "
                f"{c.get('comment', '')} | **{c.get('status', 'open')}** |"
            )
        return "\n".join(lines)

    def _md_signoff(self, data: Dict[str, Any]) -> str:
        """Render Markdown sign-off status."""
        signoffs = data.get("signoffs", [])
        if not signoffs:
            return ""
        lines = [
            "## 4. Sign-Off Status",
            "",
            "| Role | Name | Status | Date | Comments |",
            "|------|------|--------|------|----------|",
        ]
        for s in signoffs:
            lines.append(
                f"| {s.get('role', '')} | {s.get('name', '')} | "
                f"**{s.get('status', 'pending')}** | {s.get('date', '-')} | {s.get('comments', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Review Summary - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Review Summary &mdash; {company}</h1><hr></div>'

    def _html_cycle_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML cycle summary."""
        cycle = data.get("review_cycle", {})
        if not cycle:
            return ""
        return f'<div><h2>1. Review Cycle</h2><p>Cycle: {cycle.get("cycle_name", "-")} | Status: <strong>{cycle.get("status", "-")}</strong></p></div>'

    def _html_decisions(self, data: Dict[str, Any]) -> str:
        """Render HTML decisions."""
        decisions = data.get("decisions", [])
        if not decisions:
            return ""
        rows = ""
        for d in decisions:
            rows += f"<tr><td>{d.get('reviewer', '')}</td><td>{d.get('scope', '')}</td><td><strong>{d.get('decision', '')}</strong></td><td>{d.get('date', '')}</td></tr>\n"
        return (
            '<div><h2>2. Decisions</h2>\n'
            "<table><thead><tr><th>Reviewer</th><th>Scope</th><th>Decision</th><th>Date</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_comments(self, data: Dict[str, Any]) -> str:
        """Render HTML comments."""
        comments = data.get("comments", [])
        if not comments:
            return ""
        rows = ""
        for c in comments:
            rows += f"<tr><td>{c.get('reviewer', '')}</td><td>{c.get('category', '')}</td><td>{c.get('severity', '')}</td><td>{c.get('comment', '')}</td><td><strong>{c.get('status', 'open')}</strong></td></tr>\n"
        return (
            '<div><h2>3. Comments</h2>\n'
            "<table><thead><tr><th>Reviewer</th><th>Category</th><th>Severity</th><th>Comment</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_signoff(self, data: Dict[str, Any]) -> str:
        """Render HTML sign-off."""
        signoffs = data.get("signoffs", [])
        if not signoffs:
            return ""
        rows = ""
        for s in signoffs:
            rows += f"<tr><td>{s.get('role', '')}</td><td>{s.get('name', '')}</td><td><strong>{s.get('status', 'pending')}</strong></td><td>{s.get('date', '-')}</td></tr>\n"
        return (
            '<div><h2>4. Sign-Off</h2>\n'
            "<table><thead><tr><th>Role</th><th>Name</th><th>Status</th><th>Date</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div style="font-size:0.85rem;color:#666;"><hr>'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>"
            f'<p class="provenance">Provenance Hash: {provenance}</p></div>'
        )
