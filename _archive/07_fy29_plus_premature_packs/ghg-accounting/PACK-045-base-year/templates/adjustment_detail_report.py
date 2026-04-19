# -*- coding: utf-8 -*-
"""
AdjustmentDetailReport - Adjustment Lines and Before/After Comparison for PACK-045.

Generates a base year adjustment detail report covering individual adjustment
line items, before/after emission comparisons, approval status tracking,
net impact summaries, and version references.

Sections:
    1. Adjustment Summary
    2. Adjustment Line Items
    3. Before/After Comparison
    4. Approval Status
    5. Net Impact Analysis

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


def _change_indicator(before: float, after: float) -> str:
    """Return change direction indicator."""
    if after > before:
        return "UP"
    if after < before:
        return "DOWN"
    return "UNCHANGED"


def _change_pct(before: float, after: float) -> str:
    """Return percentage change string."""
    if before == 0:
        return "N/A"
    pct = ((after - before) / before) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


class AdjustmentDetailReport:
    """
    Base year adjustment detail report template.

    Renders individual adjustment line items, before/after emission values,
    approval workflow status, and net impact analysis. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = AdjustmentDetailReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AdjustmentDetailReport."""
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
        """Render adjustment detail report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_line_items(data),
            self._md_before_after(data),
            self._md_approval_status(data),
            self._md_net_impact(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render adjustment detail report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_line_items(data),
            self._html_before_after(data),
            self._html_approval_status(data),
            self._html_net_impact(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render adjustment detail report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "adjustment_detail_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "adjustment_version": self._get_val(data, "adjustment_version", ""),
            "total_adjustments": data.get("total_adjustments", 0),
            "net_change_tco2e": data.get("net_change_tco2e", 0),
            "line_items": data.get("line_items", []),
            "before_after": data.get("before_after", {}),
            "approvals": data.get("approvals", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        version = self._get_val(data, "adjustment_version", "")
        return (
            f"# Base Year Adjustment Detail - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Adjustment Version:** {version} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown adjustment summary."""
        total = data.get("total_adjustments", 0)
        net = data.get("net_change_tco2e", 0)
        approved = sum(1 for a in data.get("approvals", []) if a.get("status") == "approved")
        pending = sum(1 for a in data.get("approvals", []) if a.get("status") == "pending")
        return (
            "## 1. Adjustment Summary\n\n"
            f"- **Total Adjustments:** {total}\n"
            f"- **Net Change:** {net:+,.1f} tCO2e\n"
            f"- **Approved:** {approved} | **Pending:** {pending}"
        )

    def _md_line_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown adjustment line items."""
        items = data.get("line_items", [])
        if not items:
            return "## 2. Adjustment Line Items\n\nNo adjustments recorded."
        lines = [
            "## 2. Adjustment Line Items",
            "",
            "| # | Scope | Category | Reason | Before (tCO2e) | After (tCO2e) | Change |",
            "|---|-------|----------|--------|---------------|--------------|--------|",
        ]
        for i, item in enumerate(items, 1):
            scope = item.get("scope", "")
            cat = item.get("category", "")
            reason = item.get("reason", "")
            before = item.get("before_tco2e", 0)
            after = item.get("after_tco2e", 0)
            change = _change_pct(before, after)
            lines.append(
                f"| {i} | {scope} | {cat} | {reason} | "
                f"{before:,.1f} | {after:,.1f} | {change} |"
            )
        return "\n".join(lines)

    def _md_before_after(self, data: Dict[str, Any]) -> str:
        """Render Markdown before/after comparison by scope."""
        ba = data.get("before_after", {})
        scopes = ba.get("scopes", [])
        if not scopes:
            return ""
        lines = [
            "## 3. Before/After Comparison by Scope",
            "",
            "| Scope | Before (tCO2e) | After (tCO2e) | Change (tCO2e) | Change % |",
            "|-------|---------------|--------------|---------------|----------|",
        ]
        for s in scopes:
            name = s.get("scope_name", "")
            before = s.get("before_tco2e", 0)
            after = s.get("after_tco2e", 0)
            delta = after - before
            pct = _change_pct(before, after)
            lines.append(
                f"| {name} | {before:,.1f} | {after:,.1f} | "
                f"{delta:+,.1f} | {pct} |"
            )
        total_before = ba.get("total_before_tco2e", 0)
        total_after = ba.get("total_after_tco2e", 0)
        total_delta = total_after - total_before
        total_pct = _change_pct(total_before, total_after)
        lines.append(
            f"| **Total** | **{total_before:,.1f}** | **{total_after:,.1f}** | "
            f"**{total_delta:+,.1f}** | **{total_pct}** |"
        )
        return "\n".join(lines)

    def _md_approval_status(self, data: Dict[str, Any]) -> str:
        """Render Markdown approval status."""
        approvals = data.get("approvals", [])
        if not approvals:
            return ""
        lines = [
            "## 4. Approval Status",
            "",
            "| Reviewer | Role | Status | Date | Comments |",
            "|---------|------|--------|------|----------|",
        ]
        for a in approvals:
            name = a.get("reviewer", "")
            role = a.get("role", "")
            status = a.get("status", "pending")
            date = a.get("date", "-")
            comments = a.get("comments", "-")
            lines.append(f"| {name} | {role} | **{status.upper()}** | {date} | {comments} |")
        return "\n".join(lines)

    def _md_net_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown net impact analysis."""
        ba = data.get("before_after", {})
        total_before = ba.get("total_before_tco2e", 0)
        total_after = ba.get("total_after_tco2e", 0)
        delta = total_after - total_before
        pct = _change_pct(total_before, total_after)
        direction = _change_indicator(total_before, total_after)
        return (
            "## 5. Net Impact Analysis\n\n"
            f"- **Original Base Year Total:** {total_before:,.1f} tCO2e\n"
            f"- **Adjusted Base Year Total:** {total_after:,.1f} tCO2e\n"
            f"- **Net Change:** {delta:+,.1f} tCO2e ({pct})\n"
            f"- **Direction:** {direction}"
        )

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
            f"<title>Adjustment Detail - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".change-up{color:#e76f51;}\n"
            ".change-down{color:#2a9d8f;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".approved{color:#2a9d8f;font-weight:700;}\n"
            ".pending{color:#e9c46a;font-weight:700;}\n"
            ".rejected{color:#e76f51;font-weight:700;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        version = self._get_val(data, "adjustment_version", "")
        return (
            '<div class="section">\n'
            f"<h1>Base Year Adjustment Detail &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f"<strong>Version:</strong> {version}</p>\n<hr>\n</div>"
        )

    def _html_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML adjustment summary."""
        total = data.get("total_adjustments", 0)
        net = data.get("net_change_tco2e", 0)
        return (
            '<div class="section">\n<h2>1. Summary</h2>\n'
            f"<p><strong>Total Adjustments:</strong> {total} | "
            f"<strong>Net Change:</strong> {net:+,.1f} tCO2e</p>\n</div>"
        )

    def _html_line_items(self, data: Dict[str, Any]) -> str:
        """Render HTML adjustment line items table."""
        items = data.get("line_items", [])
        if not items:
            return ""
        rows = ""
        for i, item in enumerate(items, 1):
            scope = item.get("scope", "")
            cat = item.get("category", "")
            reason = item.get("reason", "")
            before = item.get("before_tco2e", 0)
            after = item.get("after_tco2e", 0)
            change = _change_pct(before, after)
            rows += (
                f"<tr><td>{i}</td><td>{scope}</td><td>{cat}</td><td>{reason}</td>"
                f"<td>{before:,.1f}</td><td>{after:,.1f}</td><td>{change}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Adjustment Line Items</h2>\n'
            "<table><thead><tr><th>#</th><th>Scope</th><th>Category</th>"
            "<th>Reason</th><th>Before</th><th>After</th><th>Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_before_after(self, data: Dict[str, Any]) -> str:
        """Render HTML before/after comparison."""
        ba = data.get("before_after", {})
        scopes = ba.get("scopes", [])
        if not scopes:
            return ""
        rows = ""
        for s in scopes:
            name = s.get("scope_name", "")
            before = s.get("before_tco2e", 0)
            after = s.get("after_tco2e", 0)
            delta = after - before
            css = "change-up" if delta > 0 else "change-down"
            rows += (
                f'<tr><td>{name}</td><td>{before:,.1f}</td><td>{after:,.1f}</td>'
                f'<td class="{css}">{delta:+,.1f}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Before/After Comparison</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Before tCO2e</th>"
            "<th>After tCO2e</th><th>Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_approval_status(self, data: Dict[str, Any]) -> str:
        """Render HTML approval status table."""
        approvals = data.get("approvals", [])
        if not approvals:
            return ""
        rows = ""
        for a in approvals:
            name = a.get("reviewer", "")
            role = a.get("role", "")
            status = a.get("status", "pending")
            date = a.get("date", "-")
            css = status.lower()
            rows += (
                f'<tr><td>{name}</td><td>{role}</td>'
                f'<td class="{css}"><strong>{status.upper()}</strong></td>'
                f"<td>{date}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Approval Status</h2>\n'
            "<table><thead><tr><th>Reviewer</th><th>Role</th>"
            "<th>Status</th><th>Date</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_net_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML net impact analysis."""
        ba = data.get("before_after", {})
        total_before = ba.get("total_before_tco2e", 0)
        total_after = ba.get("total_after_tco2e", 0)
        delta = total_after - total_before
        return (
            '<div class="section">\n<h2>5. Net Impact</h2>\n'
            f"<p><strong>Original:</strong> {total_before:,.1f} tCO2e | "
            f"<strong>Adjusted:</strong> {total_after:,.1f} tCO2e | "
            f"<strong>Net:</strong> {delta:+,.1f} tCO2e</p>\n</div>"
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
