# -*- coding: utf-8 -*-
"""
CorrectiveActionTemplate - ISO 50001 Clause 10.2 NC/CA Register for PACK-034.

Generates nonconformity and corrective action register documents aligned
with ISO 50001:2018 Clause 10.1/10.2. Covers NC/CA register table, detailed
nonconformity records (description, clause reference, severity, root cause,
correction, corrective action, verification, status), statistics summary,
trend analysis, and effectiveness review.

Sections:
    1. NC/CA Register Table
    2. Nonconformity Details
    3. Statistics Summary
    4. Trend Analysis
    5. Effectiveness Review

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CorrectiveActionTemplate:
    """
    ISO 50001 nonconformity and corrective action register template.

    Renders NC/CA register documents aligned with ISO 50001:2018
    Clause 10.1/10.2, covering nonconformity details, root cause
    analysis, corrective actions, verification, and effectiveness
    tracking across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CorrectiveActionTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render NC/CA register as Markdown.

        Args:
            data: NC/CA data including nonconformities list with
                  full details and statistics dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_register_table(data),
            self._md_nc_details(data),
            self._md_statistics(data),
            self._md_trend_analysis(data),
            self._md_effectiveness_review(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render NC/CA register as self-contained HTML.

        Args:
            data: NC/CA data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_statistics(data),
            self._html_register_table(data),
            self._html_nc_details(data),
            self._html_trend_analysis(data),
            self._html_effectiveness_review(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>NC/CA Register - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render NC/CA register as structured JSON.

        Args:
            data: NC/CA data dict.

        Returns:
            Dict with structured NC/CA sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        ncs = data.get("nonconformities", [])
        result: Dict[str, Any] = {
            "template": "corrective_action",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "nonconformities": ncs,
            "statistics": self._compute_statistics(ncs, data.get("statistics", {})),
            "trend_analysis": data.get("trend_analysis", []),
            "effectiveness_review": data.get("effectiveness_review", []),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with register metadata."""
        org = data.get("organization_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Nonconformity & Corrective Action Register\n\n"
            f"**Organization:** {org}  \n"
            f"**Register Date:** {data.get('register_date', '')}  \n"
            f"**ISO 50001:2018 Clause:** 10.1, 10.2  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 CorrectiveActionTemplate v34.0.0\n\n---"
        )

    def _md_register_table(self, data: Dict[str, Any]) -> str:
        """Render NC/CA register overview table."""
        ncs = data.get("nonconformities", [])
        if not ncs:
            return "## 1. NC/CA Register\n\n_No nonconformities registered._"
        lines = [
            "## 1. NC/CA Register\n",
            "| NC ID | Title | Clause | Severity | Source | Date Raised | Due Date | Status |",
            "|-------|-------|--------|----------|--------|-----------|----------|--------|",
        ]
        for nc in ncs:
            status = nc.get("status", "Open")
            lines.append(
                f"| NC-{nc.get('id', 0):03d} | {nc.get('title', '-')} "
                f"| {nc.get('clause_ref', '-')} "
                f"| {nc.get('severity', '-')} "
                f"| {nc.get('source', '-')} "
                f"| {nc.get('date_raised', '-')} "
                f"| {nc.get('due_date', '-')} "
                f"| {status} |"
            )
        return "\n".join(lines)

    def _md_nc_details(self, data: Dict[str, Any]) -> str:
        """Render detailed nonconformity records."""
        ncs = data.get("nonconformities", [])
        if not ncs:
            return "## 2. Nonconformity Details\n\n_No nonconformities to detail._"
        lines = ["## 2. Nonconformity Details\n"]
        for i, nc in enumerate(ncs, 1):
            nc_id = nc.get("id", i)
            lines.extend([
                f"### NC-{nc_id:03d}: {nc.get('title', 'Nonconformity')}",
                "",
                "| Field | Detail |",
                "|-------|--------|",
                f"| **NC ID** | NC-{nc_id:03d} |",
                f"| **Date Raised** | {nc.get('date_raised', '-')} |",
                f"| **Source** | {nc.get('source', '-')} |",
                f"| **Clause Reference** | {nc.get('clause_ref', '-')} |",
                f"| **Severity** | {nc.get('severity', '-')} |",
                f"| **Area/Process** | {nc.get('area', '-')} |",
                f"| **Description** | {nc.get('description', '-')} |",
                f"| **Objective Evidence** | {nc.get('evidence', '-')} |",
                f"| **Root Cause Analysis** | {nc.get('root_cause', 'Pending')} |",
                f"| **Root Cause Method** | {nc.get('root_cause_method', '-')} |",
                f"| **Correction (Immediate)** | {nc.get('correction', 'Pending')} |",
                f"| **Corrective Action** | {nc.get('corrective_action', 'Pending')} |",
                f"| **Responsible Person** | {nc.get('responsible', '-')} |",
                f"| **Due Date** | {nc.get('due_date', '-')} |",
                f"| **Verification Method** | {nc.get('verification_method', '-')} |",
                f"| **Verified By** | {nc.get('verified_by', '-')} |",
                f"| **Verification Date** | {nc.get('verification_date', '-')} |",
                f"| **Effectiveness Confirmed** | {nc.get('effectiveness_confirmed', 'Pending')} |",
                f"| **Status** | {nc.get('status', 'Open')} |",
                "",
            ])
        return "\n".join(lines)

    def _md_statistics(self, data: Dict[str, Any]) -> str:
        """Render statistics summary section."""
        ncs = data.get("nonconformities", [])
        stats = self._compute_statistics(ncs, data.get("statistics", {}))
        return (
            "## 3. Statistics Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Nonconformities | {stats['total']} |\n"
            f"| Open | {stats['open']} |\n"
            f"| Closed | {stats['closed']} |\n"
            f"| Overdue | {stats['overdue']} |\n"
            f"| Major NCs | {stats['major']} |\n"
            f"| Minor NCs | {stats['minor']} |\n"
            f"| Average Closure Time | {self._fmt(stats.get('avg_closure_days', 0), 1)} days |\n"
            f"| Closure Rate | {self._fmt(stats.get('closure_rate_pct', 0))}% |\n"
            f"| Effectiveness Rate | {self._fmt(stats.get('effectiveness_rate_pct', 0))}% |"
        )

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis section."""
        trends = data.get("trend_analysis", [])
        if not trends:
            return "## 4. Trend Analysis\n\n_No trend data available._"
        lines = [
            "## 4. Trend Analysis\n",
            "| Period | NCs Raised | NCs Closed | Open Balance | Major | Minor |",
            "|--------|-----------|-----------|-------------|-------|-------|",
        ]
        for t in trends:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {t.get('raised', 0)} "
                f"| {t.get('closed', 0)} "
                f"| {t.get('open_balance', 0)} "
                f"| {t.get('major', 0)} "
                f"| {t.get('minor', 0)} |"
            )
        lines.extend([
            "\n### Common Root Causes\n",
        ])
        root_causes = data.get("common_root_causes", [])
        if root_causes:
            for rc in root_causes:
                lines.append(
                    f"- **{rc.get('cause', '-')}** ({rc.get('count', 0)} occurrences, "
                    f"{self._fmt(rc.get('percentage', 0))}%)"
                )
        else:
            lines.append("_Root cause analysis pending._")
        return "\n".join(lines)

    def _md_effectiveness_review(self, data: Dict[str, Any]) -> str:
        """Render effectiveness review section."""
        reviews = data.get("effectiveness_review", [])
        if not reviews:
            return "## 5. Effectiveness Review\n\n_No effectiveness reviews completed._"
        lines = [
            "## 5. Effectiveness Review\n",
            "| NC ID | Corrective Action | Review Date | Effective | Recurrence | Notes |",
            "|-------|------------------|------------|-----------|-----------|-------|",
        ]
        for r in reviews:
            effective = "Yes" if r.get("effective", False) else "No"
            recurrence = "Yes" if r.get("recurrence", False) else "No"
            lines.append(
                f"| NC-{r.get('nc_id', 0):03d} "
                f"| {r.get('corrective_action', '-')} "
                f"| {r.get('review_date', '-')} "
                f"| {effective} "
                f"| {recurrence} "
                f"| {r.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>NC/CA Register</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'ISO 50001 Clause 10.1/10.2</p>'
        )

    def _html_statistics(self, data: Dict[str, Any]) -> str:
        """Render HTML statistics summary cards."""
        ncs = data.get("nonconformities", [])
        stats = self._compute_statistics(ncs, data.get("statistics", {}))
        return (
            '<h2>Statistics Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total NCs</span>'
            f'<span class="value">{stats["total"]}</span></div>\n'
            f'  <div class="card card-danger"><span class="label">Open</span>'
            f'<span class="value">{stats["open"]}</span></div>\n'
            f'  <div class="card"><span class="label">Closed</span>'
            f'<span class="value">{stats["closed"]}</span></div>\n'
            f'  <div class="card card-warning"><span class="label">Overdue</span>'
            f'<span class="value">{stats["overdue"]}</span></div>\n'
            f'  <div class="card"><span class="label">Closure Rate</span>'
            f'<span class="value">{self._fmt(stats.get("closure_rate_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_register_table(self, data: Dict[str, Any]) -> str:
        """Render HTML NC/CA register table."""
        ncs = data.get("nonconformities", [])
        rows = ""
        for nc in ncs:
            severity_cls = "severity-major" if nc.get("severity", "").lower() == "major" else "severity-minor"
            status = nc.get("status", "Open")
            status_cls = "status-closed" if status.lower() == "closed" else "status-open"
            rows += (
                f'<tr><td>NC-{nc.get("id", 0):03d}</td>'
                f'<td>{nc.get("title", "-")}</td>'
                f'<td>{nc.get("clause_ref", "-")}</td>'
                f'<td class="{severity_cls}">{nc.get("severity", "-")}</td>'
                f'<td>{nc.get("date_raised", "-")}</td>'
                f'<td>{nc.get("due_date", "-")}</td>'
                f'<td class="{status_cls}">{status}</td></tr>\n'
            )
        return (
            '<h2>NC/CA Register</h2>\n'
            '<table>\n<tr><th>NC ID</th><th>Title</th><th>Clause</th>'
            f'<th>Severity</th><th>Raised</th><th>Due</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_nc_details(self, data: Dict[str, Any]) -> str:
        """Render HTML detailed nonconformity records."""
        ncs = data.get("nonconformities", [])
        parts = ['<h2>Nonconformity Details</h2>\n']
        for nc in ncs:
            nc_id = nc.get("id", 0)
            severity_cls = "nc-major" if nc.get("severity", "").lower() == "major" else "nc-minor"
            parts.append(
                f'<div class="nc-detail {severity_cls}">\n'
                f'<h3>NC-{nc_id:03d}: {nc.get("title", "-")}</h3>\n'
                f'<table class="detail-table">\n'
                f'<tr><td><strong>Clause</strong></td><td>{nc.get("clause_ref", "-")}</td></tr>\n'
                f'<tr><td><strong>Severity</strong></td><td>{nc.get("severity", "-")}</td></tr>\n'
                f'<tr><td><strong>Description</strong></td><td>{nc.get("description", "-")}</td></tr>\n'
                f'<tr><td><strong>Root Cause</strong></td><td>{nc.get("root_cause", "Pending")}</td></tr>\n'
                f'<tr><td><strong>Correction</strong></td><td>{nc.get("correction", "Pending")}</td></tr>\n'
                f'<tr><td><strong>Corrective Action</strong></td><td>{nc.get("corrective_action", "Pending")}</td></tr>\n'
                f'<tr><td><strong>Responsible</strong></td><td>{nc.get("responsible", "-")}</td></tr>\n'
                f'<tr><td><strong>Status</strong></td><td>{nc.get("status", "Open")}</td></tr>\n'
                f'</table>\n</div>\n'
            )
        return "".join(parts)

    def _html_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML trend analysis."""
        trends = data.get("trend_analysis", [])
        rows = ""
        for t in trends:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{t.get("raised", 0)}</td>'
                f'<td>{t.get("closed", 0)}</td>'
                f'<td>{t.get("open_balance", 0)}</td></tr>\n'
            )
        return (
            '<h2>Trend Analysis</h2>\n'
            '<table>\n<tr><th>Period</th><th>Raised</th>'
            f'<th>Closed</th><th>Open Balance</th></tr>\n{rows}</table>'
        )

    def _html_effectiveness_review(self, data: Dict[str, Any]) -> str:
        """Render HTML effectiveness review."""
        reviews = data.get("effectiveness_review", [])
        rows = ""
        for r in reviews:
            effective = r.get("effective", False)
            cls = "status-improved" if effective else "status-declined"
            rows += (
                f'<tr><td>NC-{r.get("nc_id", 0):03d}</td>'
                f'<td>{r.get("corrective_action", "-")}</td>'
                f'<td>{r.get("review_date", "-")}</td>'
                f'<td class="{cls}">{"Effective" if effective else "Not Effective"}</td></tr>\n'
            )
        return (
            '<h2>Effectiveness Review</h2>\n'
            '<table>\n<tr><th>NC ID</th><th>Corrective Action</th>'
            f'<th>Review Date</th><th>Effective</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_statistics(
        self,
        ncs: List[Dict[str, Any]],
        provided_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute NC/CA statistics from nonconformities list."""
        if provided_stats and provided_stats.get("total", 0) > 0:
            return provided_stats
        total = len(ncs)
        open_count = sum(1 for n in ncs if n.get("status", "").lower() in ("open", "in progress"))
        closed = sum(1 for n in ncs if n.get("status", "").lower() == "closed")
        overdue = sum(1 for n in ncs if n.get("status", "").lower() == "overdue")
        major = sum(1 for n in ncs if n.get("severity", "").lower() == "major")
        minor = sum(1 for n in ncs if n.get("severity", "").lower() == "minor")
        closure_rate = (closed / total * 100) if total > 0 else 0
        effective = sum(1 for n in ncs if n.get("effectiveness_confirmed", "").lower() == "yes")
        effectiveness_rate = (effective / closed * 100) if closed > 0 else 0
        return {
            "total": total,
            "open": open_count,
            "closed": closed,
            "overdue": overdue,
            "major": major,
            "minor": minor,
            "closure_rate_pct": round(closure_rate, 1),
            "effectiveness_rate_pct": round(effectiveness_rate, 1),
            "avg_closure_days": provided_stats.get("avg_closure_days", 0),
        }

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "h3{color:#495057;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".detail-table{width:100%;}"
            ".detail-table td:first-child{width:200px;background:#f8f9fa;font-weight:500;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:120px;}"
            ".card-danger{border-left:4px solid #dc3545;}"
            ".card-warning{border-left:4px solid #ffc107;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".severity-major{color:#dc3545;font-weight:700;}"
            ".severity-minor{color:#fd7e14;font-weight:600;}"
            ".status-open{color:#dc3545;font-weight:600;}"
            ".status-closed{color:#198754;font-weight:600;}"
            ".status-improved{color:#198754;font-weight:600;}"
            ".status-declined{color:#dc3545;font-weight:600;}"
            ".nc-detail{border:1px solid #dee2e6;border-radius:8px;padding:15px;margin:10px 0;}"
            ".nc-major{border-left:4px solid #dc3545;}"
            ".nc-minor{border-left:4px solid #fd7e14;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
