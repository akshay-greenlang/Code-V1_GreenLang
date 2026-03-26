# -*- coding: utf-8 -*-
"""
AuditTrailReport - Chronological Audit Entries for PACK-045.

Generates an audit trail report covering chronological entries, approval
records, verification status tracking, change history, and data integrity
verification through provenance hashes.

Sections:
    1. Audit Summary
    2. Chronological Audit Entries
    3. Approval Records
    4. Verification Status
    5. Integrity Chain

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


class AuditTrailReport:
    """
    Audit trail report template.

    Renders chronological audit entries for base year management activities
    including data changes, approvals, recalculations, and verifications.
    All outputs include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = AuditTrailReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AuditTrailReport."""
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
        """Render audit trail report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_entries(data),
            self._md_approvals(data),
            self._md_verification(data),
            self._md_integrity_chain(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render audit trail report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_entries(data),
            self._html_approvals(data),
            self._html_verification(data),
            self._html_integrity_chain(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render audit trail report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "audit_trail_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "total_entries": data.get("total_entries", 0),
            "entries": data.get("entries", []),
            "approvals": data.get("approvals", []),
            "verifications": data.get("verifications", []),
            "integrity_chain": data.get("integrity_chain", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        total = data.get("total_entries", 0)
        return (
            f"# Base Year Audit Trail - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Total Entries:** {total} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown audit summary."""
        entries = data.get("entries", [])
        by_type: Dict[str, int] = {}
        for e in entries:
            etype = e.get("event_type", "other")
            by_type[etype] = by_type.get(etype, 0) + 1
        lines = ["## 1. Audit Summary", ""]
        for k, v in sorted(by_type.items()):
            lines.append(f"- **{k}:** {v} entries")
        return "\n".join(lines)

    def _md_entries(self, data: Dict[str, Any]) -> str:
        """Render Markdown chronological audit entries."""
        entries = data.get("entries", [])
        if not entries:
            return "## 2. Audit Entries\n\nNo entries recorded."
        lines = [
            "## 2. Chronological Audit Entries",
            "",
            "| # | Timestamp | Event Type | User | Description | Hash |",
            "|---|----------|-----------|------|------------|------|",
        ]
        for i, e in enumerate(entries, 1):
            ts = e.get("timestamp", "")
            etype = e.get("event_type", "")
            user = e.get("user", "")
            desc = e.get("description", "")
            ehash = e.get("entry_hash", "")[:12]
            lines.append(f"| {i} | {ts} | {etype} | {user} | {desc} | `{ehash}...` |")
        return "\n".join(lines)

    def _md_approvals(self, data: Dict[str, Any]) -> str:
        """Render Markdown approval records."""
        approvals = data.get("approvals", [])
        if not approvals:
            return ""
        lines = [
            "## 3. Approval Records",
            "",
            "| Date | Approver | Role | Action | Scope | Comments |",
            "|------|---------|------|--------|-------|----------|",
        ]
        for a in approvals:
            date = a.get("date", "")
            approver = a.get("approver", "")
            role = a.get("role", "")
            action = a.get("action", "")
            scope = a.get("scope", "")
            comments = a.get("comments", "-")
            lines.append(f"| {date} | {approver} | {role} | **{action}** | {scope} | {comments} |")
        return "\n".join(lines)

    def _md_verification(self, data: Dict[str, Any]) -> str:
        """Render Markdown verification status."""
        verifications = data.get("verifications", [])
        if not verifications:
            return ""
        lines = [
            "## 4. Verification Status",
            "",
            "| Scope | Verifier | Level | Status | Date | Opinion |",
            "|-------|---------|-------|--------|------|---------|",
        ]
        for v in verifications:
            scope = v.get("scope", "")
            verifier = v.get("verifier", "")
            level = v.get("assurance_level", "")
            status = v.get("status", "pending")
            date = v.get("date", "-")
            opinion = v.get("opinion", "-")
            lines.append(f"| {scope} | {verifier} | {level} | **{status.upper()}** | {date} | {opinion} |")
        return "\n".join(lines)

    def _md_integrity_chain(self, data: Dict[str, Any]) -> str:
        """Render Markdown integrity chain."""
        chain = data.get("integrity_chain", [])
        if not chain:
            return ""
        lines = [
            "## 5. Integrity Chain",
            "",
            "| Version | Timestamp | SHA-256 Hash | Previous Hash |",
            "|---------|----------|-------------|---------------|",
        ]
        for c in chain:
            ver = c.get("version", "")
            ts = c.get("timestamp", "")
            current = c.get("hash", "")[:16]
            prev = c.get("previous_hash", "")[:16]
            lines.append(f"| {ver} | {ts} | `{current}...` | `{prev}...` |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Audit Trail - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".verified{color:#2a9d8f;font-weight:700;}\n"
            ".pending{color:#e9c46a;font-weight:700;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        total = data.get("total_entries", 0)
        return (
            '<div class="section">\n'
            f"<h1>Base Year Audit Trail &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f"<strong>Entries:</strong> {total}</p>\n<hr>\n</div>"
        )

    def _html_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML audit summary."""
        entries = data.get("entries", [])
        by_type: Dict[str, int] = {}
        for e in entries:
            etype = e.get("event_type", "other")
            by_type[etype] = by_type.get(etype, 0) + 1
        items = "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in sorted(by_type.items()))
        return f'<div class="section">\n<h2>1. Summary</h2>\n<ul>{items}</ul>\n</div>'

    def _html_entries(self, data: Dict[str, Any]) -> str:
        """Render HTML chronological audit entries table."""
        entries = data.get("entries", [])
        if not entries:
            return ""
        rows = ""
        for i, e in enumerate(entries, 1):
            ts = e.get("timestamp", "")
            etype = e.get("event_type", "")
            user = e.get("user", "")
            desc = e.get("description", "")
            rows += f"<tr><td>{i}</td><td>{ts}</td><td>{etype}</td><td>{user}</td><td>{desc}</td></tr>\n"
        return (
            '<div class="section">\n<h2>2. Audit Entries</h2>\n'
            "<table><thead><tr><th>#</th><th>Timestamp</th><th>Event</th>"
            "<th>User</th><th>Description</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_approvals(self, data: Dict[str, Any]) -> str:
        """Render HTML approval records table."""
        approvals = data.get("approvals", [])
        if not approvals:
            return ""
        rows = ""
        for a in approvals:
            date = a.get("date", "")
            approver = a.get("approver", "")
            role = a.get("role", "")
            action = a.get("action", "")
            rows += f"<tr><td>{date}</td><td>{approver}</td><td>{role}</td><td><strong>{action}</strong></td></tr>\n"
        return (
            '<div class="section">\n<h2>3. Approvals</h2>\n'
            "<table><thead><tr><th>Date</th><th>Approver</th>"
            "<th>Role</th><th>Action</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_verification(self, data: Dict[str, Any]) -> str:
        """Render HTML verification status table."""
        verifications = data.get("verifications", [])
        if not verifications:
            return ""
        rows = ""
        for v in verifications:
            scope = v.get("scope", "")
            verifier = v.get("verifier", "")
            status = v.get("status", "pending")
            css = "verified" if status == "verified" else "pending"
            rows += (
                f'<tr><td>{scope}</td><td>{verifier}</td>'
                f'<td class="{css}"><strong>{status.upper()}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>4. Verification</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Verifier</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_integrity_chain(self, data: Dict[str, Any]) -> str:
        """Render HTML integrity chain table."""
        chain = data.get("integrity_chain", [])
        if not chain:
            return ""
        rows = ""
        for c in chain:
            ver = c.get("version", "")
            ts = c.get("timestamp", "")
            h = c.get("hash", "")[:16]
            rows += f"<tr><td>{ver}</td><td>{ts}</td><td><code>{h}...</code></td></tr>\n"
        return (
            '<div class="section">\n<h2>5. Integrity Chain</h2>\n'
            "<table><thead><tr><th>Version</th><th>Timestamp</th>"
            "<th>Hash</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
