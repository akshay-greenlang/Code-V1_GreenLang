# -*- coding: utf-8 -*-
"""
BillAuditReportTemplate - Utility bill audit findings report for PACK-036.

Generates comprehensive bill audit reports covering error detection,
financial impact quantification, historical error rate trends, and
corrective action recommendations. Designed for energy managers and
finance teams who need to validate utility billing accuracy and
recover overcharges.

Sections:
    1. Header & Audit Summary
    2. Audit Scope & Methodology
    3. Error Summary by Category
    4. Bill-by-Bill Findings
    5. Financial Impact Analysis
    6. Historical Error Rates
    7. Corrective Actions
    8. Provenance

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash excluding volatile fields."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {
            k: v for k, v in s.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


# ── Error category constants ────────────────────────────────────────

class ErrorCategory(str, Enum):
    """Bill audit error categories."""
    RATE_MISMATCH = "Rate Mismatch"
    METER_READ = "Meter Read Error"
    DEMAND_CHARGE = "Demand Charge Error"
    TAX_SURCHARGE = "Tax/Surcharge Error"
    DUPLICATE = "Duplicate Charge"
    TARIFF_MISAPPLY = "Tariff Misapplication"
    ESTIMATED_READ = "Estimated Read Override"
    OTHER = "Other"


class BillAuditReportTemplate:
    """
    Utility bill audit findings report template.

    Renders bill audit results including errors found across bills,
    financial impact per error, corrective action recommendations,
    and historical error rate trending across markdown, HTML, JSON,
    and CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    SEVERITY_LABELS: Dict[str, str] = {
        "critical": "CRITICAL",
        "high": "HIGH",
        "medium": "MEDIUM",
        "low": "LOW",
        "info": "INFO",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BillAuditReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render bill audit report as Markdown.

        Args:
            data: Bill audit data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_audit_scope(data),
            self._md_error_summary(data),
            self._md_bill_findings(data),
            self._md_financial_impact(data),
            self._md_historical_error_rates(data),
            self._md_corrective_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render bill audit report as self-contained HTML.

        Args:
            data: Bill audit data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_audit_scope(data),
            self._html_error_summary(data),
            self._html_bill_findings(data),
            self._html_financial_impact(data),
            self._html_historical_error_rates(data),
            self._html_corrective_actions(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Bill Audit Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render bill audit report as structured JSON.

        Args:
            data: Bill audit data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "bill_audit_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "audit_scope": data.get("audit_scope", {}),
            "error_summary": self._json_error_summary(data),
            "bill_findings": data.get("bill_findings", []),
            "financial_impact": self._json_financial_impact(data),
            "historical_error_rates": data.get("historical_error_rates", []),
            "corrective_actions": data.get("corrective_actions", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render bill audit findings as CSV.

        Args:
            data: Bill audit data from engine processing.

        Returns:
            CSV string with one row per bill finding.
        """
        self.generated_at = _utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Bill ID", "Bill Date", "Account", "Utility Type",
            "Error Category", "Severity", "Billed Amount",
            "Correct Amount", "Overcharge", "Description",
        ])
        for finding in data.get("bill_findings", []):
            errors = finding.get("errors", [])
            for err in errors:
                writer.writerow([
                    finding.get("bill_id", ""),
                    finding.get("bill_date", ""),
                    finding.get("account_number", ""),
                    finding.get("utility_type", ""),
                    err.get("category", ""),
                    err.get("severity", ""),
                    self._fmt_raw(err.get("billed_amount", 0)),
                    self._fmt_raw(err.get("correct_amount", 0)),
                    self._fmt_raw(err.get("overcharge", 0)),
                    err.get("description", ""),
                ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with audit metadata."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("audit_summary", {})
        return (
            "# Utility Bill Audit Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Audit Period:** {data.get('audit_period', '-')}  \n"
            f"**Bills Audited:** {summary.get('bills_audited', 0)}  \n"
            f"**Errors Found:** {summary.get('errors_found', 0)}  \n"
            f"**Total Overcharge:** {self._fmt_currency(summary.get('total_overcharge', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 BillAuditReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_audit_scope(self, data: Dict[str, Any]) -> str:
        """Render audit scope and methodology section."""
        scope = data.get("audit_scope", {})
        methods = scope.get("methodology", [])
        lines = [
            "## 1. Audit Scope & Methodology\n",
            f"**Accounts Reviewed:** {scope.get('accounts_reviewed', 0)}  ",
            f"**Utility Types:** {', '.join(scope.get('utility_types', []))}  ",
            f"**Date Range:** {scope.get('date_range', '-')}  ",
            f"**Auditor:** {scope.get('auditor', '-')}  ",
            f"**Confidence Threshold:** {self._fmt(scope.get('confidence_threshold_pct', 95))}%",
        ]
        if methods:
            lines.append("\n### Methodology\n")
            for m in methods:
                lines.append(f"- {m}")
        return "\n".join(lines)

    def _md_error_summary(self, data: Dict[str, Any]) -> str:
        """Render error summary by category section."""
        categories = data.get("error_categories", [])
        if not categories:
            return "## 2. Error Summary by Category\n\n_No errors detected._"
        lines = [
            "## 2. Error Summary by Category\n",
            "| Category | Count | Total Overcharge | Avg Overcharge | Severity |",
            "|----------|-------|-----------------|---------------|----------|",
        ]
        for cat in categories:
            lines.append(
                f"| {cat.get('category', '-')} "
                f"| {cat.get('count', 0)} "
                f"| {self._fmt_currency(cat.get('total_overcharge', 0))} "
                f"| {self._fmt_currency(cat.get('avg_overcharge', 0))} "
                f"| {cat.get('severity', '-')} |"
            )
        return "\n".join(lines)

    def _md_bill_findings(self, data: Dict[str, Any]) -> str:
        """Render bill-by-bill findings section."""
        findings = data.get("bill_findings", [])
        if not findings:
            return "## 3. Bill-by-Bill Findings\n\n_All bills validated successfully._"
        lines = [
            "## 3. Bill-by-Bill Findings\n",
        ]
        for f in findings:
            lines.append(
                f"### Bill: {f.get('bill_id', '-')} "
                f"({f.get('bill_date', '-')})\n"
            )
            lines.append(
                f"**Account:** {f.get('account_number', '-')} | "
                f"**Utility:** {f.get('utility_type', '-')} | "
                f"**Billed Total:** {self._fmt_currency(f.get('billed_total', 0))} | "
                f"**Correct Total:** {self._fmt_currency(f.get('correct_total', 0))}\n"
            )
            errors = f.get("errors", [])
            if errors:
                lines.extend([
                    "| # | Category | Severity | Billed | Correct | Overcharge | Description |",
                    "|---|----------|----------|--------|---------|------------|-------------|",
                ])
                for i, err in enumerate(errors, 1):
                    lines.append(
                        f"| {i} | {err.get('category', '-')} "
                        f"| {err.get('severity', '-')} "
                        f"| {self._fmt_currency(err.get('billed_amount', 0))} "
                        f"| {self._fmt_currency(err.get('correct_amount', 0))} "
                        f"| {self._fmt_currency(err.get('overcharge', 0))} "
                        f"| {err.get('description', '-')} |"
                    )
            lines.append("")
        return "\n".join(lines)

    def _md_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render financial impact analysis section."""
        impact = data.get("financial_impact", {})
        return (
            "## 4. Financial Impact Analysis\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Overcharges Detected | {self._fmt_currency(impact.get('total_overcharge', 0))} |\n"
            f"| Confirmed Recoverable | {self._fmt_currency(impact.get('confirmed_recoverable', 0))} |\n"
            f"| Pending Verification | {self._fmt_currency(impact.get('pending_verification', 0))} |\n"
            f"| Already Recovered | {self._fmt_currency(impact.get('already_recovered', 0))} |\n"
            f"| Annualized Savings Potential | {self._fmt_currency(impact.get('annualized_savings', 0))} |\n"
            f"| Error Rate (by value) | {self._fmt(impact.get('error_rate_value_pct', 0))}% |\n"
            f"| Error Rate (by count) | {self._fmt(impact.get('error_rate_count_pct', 0))}% |"
        )

    def _md_historical_error_rates(self, data: Dict[str, Any]) -> str:
        """Render historical error rates section."""
        rates = data.get("historical_error_rates", [])
        if not rates:
            return "## 5. Historical Error Rates\n\n_No historical data available._"
        lines = [
            "## 5. Historical Error Rates\n",
            "| Period | Bills Audited | Errors Found | Error Rate (%) | Overcharge |",
            "|--------|-------------|-------------|---------------|------------|",
        ]
        for r in rates:
            lines.append(
                f"| {r.get('period', '-')} "
                f"| {r.get('bills_audited', 0)} "
                f"| {r.get('errors_found', 0)} "
                f"| {self._fmt(r.get('error_rate_pct', 0))}% "
                f"| {self._fmt_currency(r.get('overcharge', 0))} |"
            )
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Render corrective actions section."""
        actions = data.get("corrective_actions", [])
        if not actions:
            return "## 6. Corrective Actions\n\n_No corrective actions required._"
        lines = [
            "## 6. Corrective Actions\n",
            "| # | Action | Priority | Owner | Due Date | Status | Est. Recovery |",
            "|---|--------|----------|-------|----------|--------|--------------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('priority', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {a.get('status', '-')} "
                f"| {self._fmt_currency(a.get('estimated_recovery', 0))} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Bill audit results should be verified with utility provider before filing disputes.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with audit summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("audit_summary", {})
        errors = summary.get("errors_found", 0)
        err_cls = "card-red" if errors > 0 else "card-green"
        return (
            f'<h1>Utility Bill Audit Report</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Audit Period: {data.get("audit_period", "-")} | '
            f'Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Bills Audited</span>'
            f'<span class="value">{summary.get("bills_audited", 0)}</span></div>\n'
            f'  <div class="card {err_cls}"><span class="label">Errors Found</span>'
            f'<span class="value">{errors}</span></div>\n'
            f'  <div class="card card-red"><span class="label">Total Overcharge</span>'
            f'<span class="value">{self._fmt_currency(summary.get("total_overcharge", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Recoverable</span>'
            f'<span class="value">{self._fmt_currency(summary.get("recoverable", 0))}</span></div>\n'
            f'</div>'
        )

    def _html_audit_scope(self, data: Dict[str, Any]) -> str:
        """Render HTML audit scope section."""
        scope = data.get("audit_scope", {})
        return (
            '<h2>Audit Scope & Methodology</h2>\n'
            '<div class="info-box">'
            f'<p><strong>Accounts:</strong> {scope.get("accounts_reviewed", 0)} | '
            f'<strong>Utilities:</strong> {", ".join(scope.get("utility_types", []))} | '
            f'<strong>Period:</strong> {scope.get("date_range", "-")} | '
            f'<strong>Auditor:</strong> {scope.get("auditor", "-")}</p>'
            '</div>'
        )

    def _html_error_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML error summary table."""
        categories = data.get("error_categories", [])
        rows = ""
        for cat in categories:
            sev = cat.get("severity", "info").lower()
            cls = "severity-critical" if sev == "critical" else (
                "severity-high" if sev == "high" else ""
            )
            rows += (
                f'<tr class="{cls}"><td>{cat.get("category", "-")}</td>'
                f'<td>{cat.get("count", 0)}</td>'
                f'<td>{self._fmt_currency(cat.get("total_overcharge", 0))}</td>'
                f'<td>{self._fmt_currency(cat.get("avg_overcharge", 0))}</td>'
                f'<td>{cat.get("severity", "-")}</td></tr>\n'
            )
        return (
            '<h2>Error Summary by Category</h2>\n'
            '<table>\n<tr><th>Category</th><th>Count</th>'
            '<th>Total Overcharge</th><th>Avg Overcharge</th>'
            f'<th>Severity</th></tr>\n{rows}</table>'
        )

    def _html_bill_findings(self, data: Dict[str, Any]) -> str:
        """Render HTML bill-by-bill findings."""
        findings = data.get("bill_findings", [])
        parts: List[str] = ['<h2>Bill-by-Bill Findings</h2>']
        for f in findings:
            errors = f.get("errors", [])
            err_rows = ""
            for err in errors:
                err_rows += (
                    f'<tr><td>{err.get("category", "-")}</td>'
                    f'<td>{err.get("severity", "-")}</td>'
                    f'<td>{self._fmt_currency(err.get("billed_amount", 0))}</td>'
                    f'<td>{self._fmt_currency(err.get("correct_amount", 0))}</td>'
                    f'<td>{self._fmt_currency(err.get("overcharge", 0))}</td>'
                    f'<td>{err.get("description", "-")}</td></tr>\n'
                )
            parts.append(
                f'<h3>Bill: {f.get("bill_id", "-")} ({f.get("bill_date", "-")})</h3>\n'
                f'<p>Account: {f.get("account_number", "-")} | '
                f'Utility: {f.get("utility_type", "-")} | '
                f'Billed: {self._fmt_currency(f.get("billed_total", 0))} | '
                f'Correct: {self._fmt_currency(f.get("correct_total", 0))}</p>\n'
                f'<table>\n<tr><th>Category</th><th>Severity</th>'
                f'<th>Billed</th><th>Correct</th><th>Overcharge</th>'
                f'<th>Description</th></tr>\n{err_rows}</table>'
            )
        return "\n".join(parts)

    def _html_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML financial impact section."""
        impact = data.get("financial_impact", {})
        fields = [
            ("Total Overcharges", impact.get("total_overcharge", 0)),
            ("Confirmed Recoverable", impact.get("confirmed_recoverable", 0)),
            ("Pending Verification", impact.get("pending_verification", 0)),
            ("Already Recovered", impact.get("already_recovered", 0)),
            ("Annualized Savings", impact.get("annualized_savings", 0)),
        ]
        rows = "".join(
            f'<tr><td>{label}</td><td>{self._fmt_currency(val)}</td></tr>\n'
            for label, val in fields
        )
        return (
            '<h2>Financial Impact Analysis</h2>\n'
            f'<table>\n<tr><th>Metric</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_historical_error_rates(self, data: Dict[str, Any]) -> str:
        """Render HTML historical error rates section."""
        rates = data.get("historical_error_rates", [])
        rows = ""
        for r in rates:
            rows += (
                f'<tr><td>{r.get("period", "-")}</td>'
                f'<td>{r.get("bills_audited", 0)}</td>'
                f'<td>{r.get("errors_found", 0)}</td>'
                f'<td>{self._fmt(r.get("error_rate_pct", 0))}%</td>'
                f'<td>{self._fmt_currency(r.get("overcharge", 0))}</td></tr>\n'
            )
        return (
            '<h2>Historical Error Rates</h2>\n'
            '<table>\n<tr><th>Period</th><th>Bills Audited</th>'
            '<th>Errors</th><th>Error Rate</th>'
            f'<th>Overcharge</th></tr>\n{rows}</table>'
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML corrective actions section."""
        actions = data.get("corrective_actions", [])
        rows = ""
        for a in actions:
            priority = a.get("priority", "").lower()
            cls = "priority-high" if priority in ("critical", "high") else ""
            rows += (
                f'<tr class="{cls}"><td>{a.get("action", "-")}</td>'
                f'<td>{a.get("priority", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td>'
                f'<td>{a.get("due_date", "-")}</td>'
                f'<td>{a.get("status", "-")}</td>'
                f'<td>{self._fmt_currency(a.get("estimated_recovery", 0))}</td></tr>\n'
            )
        return (
            '<h2>Corrective Actions</h2>\n'
            '<table>\n<tr><th>Action</th><th>Priority</th>'
            '<th>Owner</th><th>Due Date</th><th>Status</th>'
            f'<th>Est. Recovery</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_error_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON error summary data."""
        summary = data.get("audit_summary", {})
        return {
            "bills_audited": summary.get("bills_audited", 0),
            "errors_found": summary.get("errors_found", 0),
            "total_overcharge": summary.get("total_overcharge", 0),
            "recoverable": summary.get("recoverable", 0),
            "error_categories": data.get("error_categories", []),
        }

    def _json_financial_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON financial impact data."""
        impact = data.get("financial_impact", {})
        return {
            "total_overcharge": impact.get("total_overcharge", 0),
            "confirmed_recoverable": impact.get("confirmed_recoverable", 0),
            "pending_verification": impact.get("pending_verification", 0),
            "already_recovered": impact.get("already_recovered", 0),
            "annualized_savings": impact.get("annualized_savings", 0),
            "error_rate_value_pct": impact.get("error_rate_value_pct", 0),
            "error_rate_count_pct": impact.get("error_rate_count_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        categories = data.get("error_categories", [])
        history = data.get("historical_error_rates", [])
        return {
            "error_by_category_pie": {
                "type": "pie",
                "labels": [c.get("category", "") for c in categories],
                "values": [c.get("total_overcharge", 0) for c in categories],
            },
            "error_rate_trend_line": {
                "type": "line",
                "labels": [h.get("period", "") for h in history],
                "series": {
                    "error_rate_pct": [h.get("error_rate_pct", 0) for h in history],
                    "overcharge": [h.get("overcharge", 0) for h in history],
                },
            },
            "severity_bar": {
                "type": "bar",
                "labels": [c.get("category", "") for c in categories],
                "series": {
                    "count": [c.get("count", 0) for c in categories],
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "h3{color:#495057;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:160px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
            ".severity-critical{background:#f8d7da !important;}"
            ".severity-high{background:#fff3cd !important;}"
            ".priority-high{background:#fff3cd !important;}"
        )

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

    def _fmt_raw(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value without comma separators (for CSV).

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string without commas.
        """
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}"
        return str(val)

    def _fmt_currency(self, val: Any, symbol: str = "") -> str:
        """Format a currency value.

        Args:
            val: Numeric value.
            symbol: Currency symbol (default from config or empty).

        Returns:
            Formatted currency string.
        """
        sym = symbol or self.config.get("currency_symbol", "EUR")
        if isinstance(val, (int, float)):
            return f"{sym} {val:,.2f}"
        return f"{sym} {val}"

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
