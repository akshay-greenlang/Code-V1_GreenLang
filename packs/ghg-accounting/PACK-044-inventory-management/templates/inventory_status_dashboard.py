# -*- coding: utf-8 -*-
"""
InventoryStatusDashboard - Period Status KPIs and Progress for PACK-044.

Generates an inventory status dashboard covering period KPIs, progress bars
for each inventory phase, milestone tracking, overall completion metrics,
and scope-level readiness summaries.

Sections:
    1. Period Overview KPIs
    2. Phase Progress Bars
    3. Milestone Tracker
    4. Scope Readiness Summary
    5. Action Items

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

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


def _progress_bar_text(pct: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int(pct / 100 * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:.0f}%"


def _status_color(pct: float) -> str:
    """Return CSS class for completion percentage."""
    if pct >= 90:
        return "status-complete"
    if pct >= 60:
        return "status-progress"
    if pct >= 30:
        return "status-partial"
    return "status-early"


class InventoryStatusDashboard:
    """
    Inventory period status dashboard template.

    Renders period status KPIs, phase progress bars, milestone tracking,
    scope readiness summaries, and action items. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = InventoryStatusDashboard()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize InventoryStatusDashboard."""
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
        """Render inventory status dashboard as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_period_kpis(data),
            self._md_phase_progress(data),
            self._md_milestones(data),
            self._md_scope_readiness(data),
            self._md_action_items(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render inventory status dashboard as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_period_kpis(data),
            self._html_phase_progress(data),
            self._html_milestones(data),
            self._html_scope_readiness(data),
            self._html_action_items(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render inventory status dashboard as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "inventory_status_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "overall_completion_pct": data.get("overall_completion_pct", 0.0),
            "period_kpis": data.get("period_kpis", {}),
            "phase_progress": data.get("phase_progress", []),
            "milestones": data.get("milestones", []),
            "scope_readiness": data.get("scope_readiness", []),
            "action_items": data.get("action_items", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        overall = data.get("overall_completion_pct", 0.0)
        return (
            f"# GHG Inventory Status Dashboard - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Overall Completion:** {overall:.0f}% | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_period_kpis(self, data: Dict[str, Any]) -> str:
        """Render Markdown period KPIs."""
        kpis = data.get("period_kpis", {})
        if not kpis:
            return "## 1. Period Overview KPIs\n\nNo KPI data available."
        lines = [
            "## 1. Period Overview KPIs",
            "",
            "| KPI | Value | Target | Status |",
            "|-----|-------|--------|--------|",
        ]
        for key, kpi in kpis.items():
            name = kpi.get("name", key)
            value = kpi.get("value", "N/A")
            target = kpi.get("target", "N/A")
            status = kpi.get("status", "-")
            lines.append(f"| {name} | {value} | {target} | **{status}** |")
        return "\n".join(lines)

    def _md_phase_progress(self, data: Dict[str, Any]) -> str:
        """Render Markdown phase progress bars."""
        phases = data.get("phase_progress", [])
        if not phases:
            return "## 2. Phase Progress\n\nNo phase data available."
        lines = [
            "## 2. Phase Progress",
            "",
            "| Phase | Completion | Progress | Status |",
            "|-------|-----------|----------|--------|",
        ]
        for phase in phases:
            name = phase.get("phase_name", "")
            pct = phase.get("completion_pct", 0.0)
            bar = _progress_bar_text(pct, 15)
            status = phase.get("status", "pending")
            lines.append(f"| {name} | {pct:.0f}% | `{bar}` | {status} |")
        return "\n".join(lines)

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        """Render Markdown milestone tracker."""
        milestones = data.get("milestones", [])
        if not milestones:
            return "## 3. Milestone Tracker\n\nNo milestones defined."
        lines = [
            "## 3. Milestone Tracker",
            "",
            "| Milestone | Due Date | Status | Owner | Notes |",
            "|-----------|----------|--------|-------|-------|",
        ]
        for ms in milestones:
            name = ms.get("name", "")
            due = ms.get("due_date", "-")
            status = ms.get("status", "pending")
            owner = ms.get("owner", "-")
            notes = ms.get("notes", "-")
            lines.append(f"| {name} | {due} | **{status}** | {owner} | {notes} |")
        return "\n".join(lines)

    def _md_scope_readiness(self, data: Dict[str, Any]) -> str:
        """Render Markdown scope readiness summary."""
        scopes = data.get("scope_readiness", [])
        if not scopes:
            return ""
        lines = [
            "## 4. Scope Readiness Summary",
            "",
            "| Scope | Data Complete | Calculated | Reviewed | Ready |",
            "|-------|-------------|------------|----------|-------|",
        ]
        for scope in scopes:
            name = scope.get("scope_name", "")
            data_pct = f"{scope.get('data_complete_pct', 0):.0f}%"
            calc = "Yes" if scope.get("calculated") else "No"
            reviewed = "Yes" if scope.get("reviewed") else "No"
            ready = "Yes" if scope.get("ready") else "No"
            lines.append(f"| {name} | {data_pct} | {calc} | {reviewed} | **{ready}** |")
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown action items."""
        items = data.get("action_items", [])
        if not items:
            return ""
        lines = ["## 5. Action Items", ""]
        for item in items:
            priority = item.get("priority", "medium")
            desc = item.get("description", "")
            owner = item.get("owner", "-")
            due = item.get("due_date", "-")
            lines.append(f"- **[{priority.upper()}]** {desc} (Owner: {owner}, Due: {due})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-044 Inventory Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Inventory Status - {company} ({period})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".status-complete{color:#2a9d8f;font-weight:700;}\n"
            ".status-progress{color:#e9c46a;font-weight:700;}\n"
            ".status-partial{color:#f4a261;font-weight:700;}\n"
            ".status-early{color:#e76f51;font-weight:700;}\n"
            ".progress-bg{background:#e8e8e8;border-radius:4px;height:20px;width:200px;display:inline-block;}\n"
            ".progress-fill{border-radius:4px;height:20px;display:inline-block;background:#2a9d8f;}\n"
            ".kpi-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:160px;}\n"
            ".kpi-value{font-size:1.5rem;font-weight:700;color:#1b263b;}\n"
            ".kpi-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        overall = data.get("overall_completion_pct", 0.0)
        return (
            '<div class="section">\n'
            f"<h1>GHG Inventory Status Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Overall Completion:</strong> {overall:.0f}%</p>\n<hr>\n</div>"
        )

    def _html_period_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML period KPIs as cards."""
        kpis = data.get("period_kpis", {})
        if not kpis:
            return ""
        cards = ""
        for key, kpi in kpis.items():
            name = kpi.get("name", key)
            value = kpi.get("value", "N/A")
            cards += (
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-label">{name}</div></div>\n'
            )
        return f'<div class="section">\n<h2>1. Period Overview KPIs</h2>\n<div>{cards}</div>\n</div>'

    def _html_phase_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML phase progress bars."""
        phases = data.get("phase_progress", [])
        if not phases:
            return ""
        rows = ""
        for phase in phases:
            name = phase.get("phase_name", "")
            pct = phase.get("completion_pct", 0.0)
            css = _status_color(pct)
            bar_width = int(pct * 2)
            status = phase.get("status", "pending")
            rows += (
                f'<tr><td>{name}</td><td class="{css}">{pct:.0f}%</td>'
                f'<td><div class="progress-bg">'
                f'<div class="progress-fill" style="width:{bar_width}px"></div>'
                f'</div></td><td>{status}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>2. Phase Progress</h2>\n'
            "<table><thead><tr><th>Phase</th><th>%</th><th>Progress</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        """Render HTML milestone tracker."""
        milestones = data.get("milestones", [])
        if not milestones:
            return ""
        rows = ""
        for ms in milestones:
            name = ms.get("name", "")
            due = ms.get("due_date", "-")
            status = ms.get("status", "pending")
            owner = ms.get("owner", "-")
            rows += f"<tr><td>{name}</td><td>{due}</td><td><strong>{status}</strong></td><td>{owner}</td></tr>\n"
        return (
            '<div class="section">\n<h2>3. Milestone Tracker</h2>\n'
            "<table><thead><tr><th>Milestone</th><th>Due Date</th>"
            "<th>Status</th><th>Owner</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scope_readiness(self, data: Dict[str, Any]) -> str:
        """Render HTML scope readiness."""
        scopes = data.get("scope_readiness", [])
        if not scopes:
            return ""
        rows = ""
        for scope in scopes:
            name = scope.get("scope_name", "")
            data_pct = f"{scope.get('data_complete_pct', 0):.0f}%"
            calc = "Yes" if scope.get("calculated") else "No"
            reviewed = "Yes" if scope.get("reviewed") else "No"
            ready = "Yes" if scope.get("ready") else "No"
            rows += f"<tr><td>{name}</td><td>{data_pct}</td><td>{calc}</td><td>{reviewed}</td><td><strong>{ready}</strong></td></tr>\n"
        return (
            '<div class="section">\n<h2>4. Scope Readiness</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Data</th><th>Calculated</th>"
            "<th>Reviewed</th><th>Ready</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items."""
        items = data.get("action_items", [])
        if not items:
            return ""
        li = ""
        for item in items:
            priority = item.get("priority", "medium").upper()
            desc = item.get("description", "")
            owner = item.get("owner", "-")
            li += f"<li><strong>[{priority}]</strong> {desc} (Owner: {owner})</li>\n"
        return f'<div class="section">\n<h2>5. Action Items</h2>\n<ul>{li}</ul>\n</div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
