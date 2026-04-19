# -*- coding: utf-8 -*-
"""
ChangeLogReport - Change Log and Impact Tracking for PACK-044.

Generates a change log report covering inventory changes, their emission
impact, approval status, change categories, and audit trail entries.

Sections:
    1. Change Summary
    2. Change Details
    3. Emission Impact Analysis
    4. Approval Status

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


class ChangeLogReport:
    """
    Change log report template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ChangeLogReport()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ChangeLogReport."""
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
        """Render change log report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_changes(data),
            self._md_impact(data),
            self._md_approvals(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render change log report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_changes(data),
            self._html_impact(data),
            self._html_approvals(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render change log report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "change_log_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "change_summary": data.get("change_summary", {}),
            "changes": data.get("changes", []),
            "emission_impacts": data.get("emission_impacts", []),
            "approvals": data.get("approvals", []),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Change Log Report - {company}\n\n"
            f"**Period:** {period} | **Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown change summary."""
        summary = data.get("change_summary", {})
        if not summary:
            return "## 1. Change Summary\n\nNo changes recorded."
        return (
            "## 1. Change Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Changes | {summary.get('total_changes', 0)} |\n"
            f"| Data Changes | {summary.get('data_changes', 0)} |\n"
            f"| Methodology Changes | {summary.get('methodology_changes', 0)} |\n"
            f"| EF Changes | {summary.get('ef_changes', 0)} |\n"
            f"| Boundary Changes | {summary.get('boundary_changes', 0)} |\n"
            f"| Pending Approval | {summary.get('pending_approval', 0)} |"
        )

    def _md_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown change details."""
        changes = data.get("changes", [])
        if not changes:
            return ""
        lines = [
            "## 2. Change Details",
            "",
            "| ID | Date | Category | Description | Changed By | Impact |",
            "|----|------|----------|------------|-----------|--------|",
        ]
        for c in changes:
            lines.append(
                f"| {c.get('change_id', '')} | {c.get('date', '')} | "
                f"{c.get('category', '')} | {c.get('description', '')} | "
                f"{c.get('changed_by', '')} | {c.get('impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission impact analysis."""
        impacts = data.get("emission_impacts", [])
        if not impacts:
            return ""
        lines = [
            "## 3. Emission Impact Analysis",
            "",
            "| Change ID | Scope | Before (tCO2e) | After (tCO2e) | Delta | % Change |",
            "|----------|-------|---------------|--------------|-------|----------|",
        ]
        for imp in impacts:
            before = imp.get("before_tco2e", 0.0)
            after = imp.get("after_tco2e", 0.0)
            delta = after - before
            pct = (delta / before * 100) if before > 0 else 0.0
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {imp.get('change_id', '')} | {imp.get('scope', '')} | "
                f"{before:,.1f} | {after:,.1f} | {sign}{delta:,.1f} | {sign}{pct:.1f}% |"
            )
        return "\n".join(lines)

    def _md_approvals(self, data: Dict[str, Any]) -> str:
        """Render Markdown approval status."""
        approvals = data.get("approvals", [])
        if not approvals:
            return ""
        lines = [
            "## 4. Approval Status",
            "",
            "| Change ID | Approver | Status | Date | Comments |",
            "|----------|---------|--------|------|----------|",
        ]
        for a in approvals:
            lines.append(
                f"| {a.get('change_id', '')} | {a.get('approver', '')} | "
                f"**{a.get('status', '')}** | {a.get('date', '')} | {a.get('comments', '-')} |"
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
            f'<meta charset="UTF-8"><title>Change Log - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #e9c46a;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Change Log &mdash; {company}</h1><hr></div>'

    def _html_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML summary."""
        summary = data.get("change_summary", {})
        if not summary:
            return ""
        return f'<div><h2>1. Summary</h2><p>Total changes: {summary.get("total_changes", 0)}</p></div>'

    def _html_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML change details."""
        changes = data.get("changes", [])
        if not changes:
            return ""
        rows = ""
        for c in changes:
            rows += f"<tr><td>{c.get('change_id', '')}</td><td>{c.get('date', '')}</td><td>{c.get('category', '')}</td><td>{c.get('description', '')}</td><td>{c.get('changed_by', '')}</td></tr>\n"
        return (
            '<div><h2>2. Changes</h2>\n'
            "<table><thead><tr><th>ID</th><th>Date</th><th>Category</th><th>Description</th><th>By</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML emission impact."""
        impacts = data.get("emission_impacts", [])
        if not impacts:
            return ""
        rows = ""
        for imp in impacts:
            before = imp.get("before_tco2e", 0.0)
            after = imp.get("after_tco2e", 0.0)
            delta = after - before
            rows += f"<tr><td>{imp.get('change_id', '')}</td><td>{imp.get('scope', '')}</td><td>{before:,.1f}</td><td>{after:,.1f}</td><td>{delta:+,.1f}</td></tr>\n"
        return (
            '<div><h2>3. Emission Impact</h2>\n'
            "<table><thead><tr><th>ID</th><th>Scope</th><th>Before</th><th>After</th><th>Delta</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_approvals(self, data: Dict[str, Any]) -> str:
        """Render HTML approvals."""
        approvals = data.get("approvals", [])
        if not approvals:
            return ""
        rows = ""
        for a in approvals:
            rows += f"<tr><td>{a.get('change_id', '')}</td><td>{a.get('approver', '')}</td><td><strong>{a.get('status', '')}</strong></td><td>{a.get('date', '')}</td></tr>\n"
        return (
            '<div><h2>4. Approvals</h2>\n'
            "<table><thead><tr><th>ID</th><th>Approver</th><th>Status</th><th>Date</th></tr></thead>\n"
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
