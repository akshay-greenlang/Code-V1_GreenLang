# -*- coding: utf-8 -*-
"""
VerifierQueryRegister - Verifier Query Register for PACK-048.

Generates a verifier query and findings register with an IR/query/finding
log table (sortable by priority, status, category), SLA compliance
dashboard (on-time vs. overdue), outstanding query count by priority,
finding severity distribution, resolution timeline (average days to
resolve), and escalation history.

Regulatory References:
    - ISAE 3410 para 48-52: Inquiries and requests for information
    - ISO 14064-3 clause 6.3.5: Communication with responsible party
    - AA1000AS v3: Stakeholder engagement in assurance
    - CSRD / ESRS: Assurance engagement communication

Sections:
    1. Query/Finding Log Table
    2. SLA Compliance Dashboard
    3. Outstanding Queries by Priority
    4. Finding Severity Distribution
    5. Resolution Timeline
    6. Escalation History
    7. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


class QueryPriority(str, Enum):
    """Query priority classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryStatus(str, Enum):
    """Query status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    CLOSED = "closed"
    OVERDUE = "overdue"


class QueryType(str, Enum):
    """Query type classification."""
    INFORMATION_REQUEST = "information_request"
    QUERY = "query"
    FINDING = "finding"
    RECOMMENDATION = "recommendation"


class FindingSeverity(str, Enum):
    """Finding severity classification."""
    MATERIAL = "material"
    SIGNIFICANT = "significant"
    MINOR = "minor"
    OBSERVATION = "observation"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class QueryEntry(BaseModel):
    """Single query or finding in the register."""
    query_id: str = Field(..., description="Query identifier (e.g., VQ-001)")
    query_type: QueryType = Field(QueryType.QUERY, description="Query type")
    title: str = Field(..., description="Query title")
    description: str = Field("", description="Detailed description")
    category: str = Field("", description="Category (e.g., Scope 1 Data)")
    priority: QueryPriority = Field(QueryPriority.MEDIUM, description="Priority")
    status: QueryStatus = Field(QueryStatus.OPEN, description="Current status")
    severity: Optional[FindingSeverity] = Field(None, description="Finding severity (if finding)")
    raised_by: str = Field("", description="Raised by (verifier name)")
    raised_date: Optional[str] = Field(None, description="Date raised (ISO)")
    sla_due_date: Optional[str] = Field(None, description="SLA due date (ISO)")
    response_date: Optional[str] = Field(None, description="Date responded (ISO)")
    closed_date: Optional[str] = Field(None, description="Date closed (ISO)")
    days_to_resolve: Optional[int] = Field(None, ge=0, description="Days to resolve")
    assignee: str = Field("", description="Assigned responder")
    response_summary: str = Field("", description="Response summary")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence references")
    notes: str = Field("", description="Additional notes")


class SLAComplianceSummary(BaseModel):
    """SLA compliance summary."""
    total_queries: int = Field(0, ge=0, description="Total queries raised")
    on_time: int = Field(0, ge=0, description="Responded on time")
    overdue: int = Field(0, ge=0, description="Overdue responses")
    pending: int = Field(0, ge=0, description="Pending (not yet due)")
    sla_compliance_pct: float = Field(0.0, ge=0, le=100, description="SLA compliance %")
    average_response_days: float = Field(0.0, ge=0, description="Avg days to respond")


class EscalationEntry(BaseModel):
    """Single escalation event."""
    escalation_id: str = Field(default_factory=lambda: f"ESC-{_new_uuid()[:6]}", description="Escalation ID")
    query_id: str = Field("", description="Related query ID")
    escalated_date: Optional[str] = Field(None, description="Escalation date (ISO)")
    escalated_to: str = Field("", description="Escalated to (person/role)")
    reason: str = Field("", description="Escalation reason")
    resolution: str = Field("", description="Escalation resolution")
    resolved_date: Optional[str] = Field(None, description="Resolution date (ISO)")


class QueryRegisterInput(BaseModel):
    """Complete input model for VerifierQueryRegister."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    verifier_name: str = Field("", description="Verifier / assurance provider name")
    engagement_reference: str = Field("", description="Engagement reference number")
    queries: List[QueryEntry] = Field(
        default_factory=list, description="Query/finding log"
    )
    sla_summary: Optional[SLAComplianceSummary] = Field(
        None, description="SLA compliance summary"
    )
    escalations: List[EscalationEntry] = Field(
        default_factory=list, description="Escalation history"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _priority_label(priority: str) -> str:
    """Return display label for priority."""
    return priority.upper()


def _priority_css(priority: str) -> str:
    """Return CSS class for priority."""
    mapping = {
        "critical": "pri-critical",
        "high": "pri-high",
        "medium": "pri-medium",
        "low": "pri-low",
    }
    return mapping.get(priority, "pri-medium")


def _status_label(status: str) -> str:
    """Return display label for status."""
    return status.replace("_", " ").title()


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class VerifierQueryRegister:
    """
    Verifier query register template for PACK-048.

    Renders a query/finding log with SLA compliance dashboard, outstanding
    counts by priority, severity distribution, resolution timeline, and
    escalation history. All outputs include SHA-256 provenance hashing
    for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = VerifierQueryRegister()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VerifierQueryRegister."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

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

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Any:
        """Render in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render query register as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render query register as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render query register as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(data)

    def to_html(self, data: Dict[str, Any]) -> str:
        """Alias for render_html."""
        return self.render_html(data)

    def to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for render_json."""
        return self.render_json(data)

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_query_log(data),
            self._md_sla_dashboard(data),
            self._md_outstanding_by_priority(data),
            self._md_severity_distribution(data),
            self._md_resolution_timeline(data),
            self._md_escalation_history(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        verifier = self._get_val(data, "verifier_name", "")
        return (
            f"# Verifier Query Register - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Verifier:** {verifier} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_query_log(self, data: Dict[str, Any]) -> str:
        """Render Markdown query/finding log table."""
        queries = data.get("queries", [])
        if not queries:
            return "## 1. Query / Finding Log\n\nNo queries raised."
        lines = [
            "## 1. Query / Finding Log",
            "",
            "| ID | Type | Title | Category | Priority | Status | Raised | SLA Due | Assignee |",
            "|----|------|-------|----------|----------|--------|--------|---------|----------|",
        ]
        for q in queries:
            q_type = q.get("query_type", "query").replace("_", " ").title()
            lines.append(
                f"| {q.get('query_id', '')} | "
                f"{q_type} | "
                f"{q.get('title', '')} | "
                f"{q.get('category', '')} | "
                f"**{_priority_label(q.get('priority', 'medium'))}** | "
                f"{_status_label(q.get('status', 'open'))} | "
                f"{q.get('raised_date', '-')} | "
                f"{q.get('sla_due_date', '-')} | "
                f"{q.get('assignee', '')} |"
            )
        return "\n".join(lines)

    def _md_sla_dashboard(self, data: Dict[str, Any]) -> str:
        """Render Markdown SLA compliance dashboard."""
        sla = data.get("sla_summary")
        if not sla:
            return ""
        lines = [
            "## 2. SLA Compliance Dashboard",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Queries | {sla.get('total_queries', 0)} |",
            f"| On Time | {sla.get('on_time', 0)} |",
            f"| Overdue | {sla.get('overdue', 0)} |",
            f"| Pending | {sla.get('pending', 0)} |",
            f"| SLA Compliance | {sla.get('sla_compliance_pct', 0):.1f}% |",
            f"| Avg Response (days) | {sla.get('average_response_days', 0):.1f} |",
        ]
        return "\n".join(lines)

    def _md_outstanding_by_priority(self, data: Dict[str, Any]) -> str:
        """Render Markdown outstanding queries by priority."""
        queries = data.get("queries", [])
        open_queries = [q for q in queries if q.get("status") in ("open", "in_progress", "overdue")]
        if not open_queries:
            return ""
        priority_counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for q in open_queries:
            pri = q.get("priority", "medium")
            if pri in priority_counts:
                priority_counts[pri] += 1
        lines = [
            "## 3. Outstanding Queries by Priority",
            "",
            "| Priority | Count |",
            "|----------|-------|",
        ]
        for pri, count in priority_counts.items():
            if count > 0:
                lines.append(f"| **{pri.upper()}** | {count} |")
        lines.append(f"| **TOTAL** | {sum(priority_counts.values())} |")
        return "\n".join(lines)

    def _md_severity_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown finding severity distribution."""
        queries = data.get("queries", [])
        findings = [q for q in queries if q.get("query_type") == "finding"]
        if not findings:
            return ""
        dist: Dict[str, int] = {"material": 0, "significant": 0, "minor": 0, "observation": 0}
        for f in findings:
            sev = f.get("severity", "observation")
            if sev in dist:
                dist[sev] += 1
        lines = [
            "## 4. Finding Severity Distribution",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ]
        for sev, count in dist.items():
            lines.append(f"| {sev.title()} | {count} |")
        return "\n".join(lines)

    def _md_resolution_timeline(self, data: Dict[str, Any]) -> str:
        """Render Markdown resolution timeline."""
        queries = data.get("queries", [])
        resolved = [q for q in queries if q.get("days_to_resolve") is not None]
        if not resolved:
            return ""
        days_list = [q.get("days_to_resolve", 0) for q in resolved]
        avg_days = sum(days_list) / len(days_list) if days_list else 0
        max_days = max(days_list) if days_list else 0
        min_days = min(days_list) if days_list else 0
        lines = [
            "## 5. Resolution Timeline",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Resolved Queries | {len(resolved)} |",
            f"| Average Days | {avg_days:.1f} |",
            f"| Min Days | {min_days} |",
            f"| Max Days | {max_days} |",
        ]
        return "\n".join(lines)

    def _md_escalation_history(self, data: Dict[str, Any]) -> str:
        """Render Markdown escalation history."""
        escalations = data.get("escalations", [])
        if not escalations:
            return ""
        lines = [
            "## 6. Escalation History",
            "",
            "| ID | Query | Date | Escalated To | Reason | Resolution |",
            "|----|-------|------|-------------|--------|------------|",
        ]
        for e in escalations:
            lines.append(
                f"| {e.get('escalation_id', '')} | "
                f"{e.get('query_id', '')} | "
                f"{e.get('escalated_date', '-')} | "
                f"{e.get('escalated_to', '')} | "
                f"{e.get('reason', '')} | "
                f"{e.get('resolution', '')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_query_log(data),
            self._html_sla_dashboard(data),
            self._html_outstanding_by_priority(data),
            self._html_severity_distribution(data),
            self._html_resolution_timeline(data),
            self._html_escalation_history(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Verifier Query Register - {company}</title>\n"
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
            ".pri-critical{color:#d62828;font-weight:700;}\n"
            ".pri-high{color:#e76f51;font-weight:700;}\n"
            ".pri-medium{color:#e9c46a;font-weight:700;}\n"
            ".pri-low{color:#2a9d8f;}\n"
            ".status-overdue{color:#e76f51;font-weight:700;}\n"
            ".status-open{color:#e9c46a;}\n"
            ".status-closed{color:#2a9d8f;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        verifier = self._get_val(data, "verifier_name", "")
        return (
            '<div class="section">\n'
            f"<h1>Verifier Query Register &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Verifier:</strong> {verifier} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_query_log(self, data: Dict[str, Any]) -> str:
        """Render HTML query log table."""
        queries = data.get("queries", [])
        if not queries:
            return ""
        rows = ""
        for q in queries:
            pri = q.get("priority", "medium")
            q_type = q.get("query_type", "query").replace("_", " ").title()
            rows += (
                f"<tr><td>{q.get('query_id', '')}</td>"
                f"<td>{q_type}</td>"
                f"<td>{q.get('title', '')}</td>"
                f"<td>{q.get('category', '')}</td>"
                f'<td class="{_priority_css(pri)}">{_priority_label(pri)}</td>'
                f"<td>{_status_label(q.get('status', 'open'))}</td>"
                f"<td>{q.get('raised_date', '-')}</td>"
                f"<td>{q.get('sla_due_date', '-')}</td>"
                f"<td>{q.get('assignee', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. Query / Finding Log</h2>\n'
            "<table><thead><tr><th>ID</th><th>Type</th><th>Title</th>"
            "<th>Category</th><th>Priority</th><th>Status</th>"
            "<th>Raised</th><th>SLA Due</th><th>Assignee</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sla_dashboard(self, data: Dict[str, Any]) -> str:
        """Render HTML SLA compliance dashboard."""
        sla = data.get("sla_summary")
        if not sla:
            return ""
        rows = (
            f"<tr><td>Total Queries</td><td>{sla.get('total_queries', 0)}</td></tr>\n"
            f"<tr><td>On Time</td><td>{sla.get('on_time', 0)}</td></tr>\n"
            f"<tr><td>Overdue</td><td>{sla.get('overdue', 0)}</td></tr>\n"
            f"<tr><td>Pending</td><td>{sla.get('pending', 0)}</td></tr>\n"
            f"<tr><td>SLA Compliance</td><td>{sla.get('sla_compliance_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>Avg Response (days)</td><td>{sla.get('average_response_days', 0):.1f}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>2. SLA Compliance Dashboard</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_outstanding_by_priority(self, data: Dict[str, Any]) -> str:
        """Render HTML outstanding queries by priority."""
        queries = data.get("queries", [])
        open_queries = [q for q in queries if q.get("status") in ("open", "in_progress", "overdue")]
        if not open_queries:
            return ""
        priority_counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for q in open_queries:
            pri = q.get("priority", "medium")
            if pri in priority_counts:
                priority_counts[pri] += 1
        rows = ""
        for pri, count in priority_counts.items():
            if count > 0:
                rows += f'<tr><td class="{_priority_css(pri)}">{pri.upper()}</td><td>{count}</td></tr>\n'
        total = sum(priority_counts.values())
        rows += f"<tr><td><strong>TOTAL</strong></td><td><strong>{total}</strong></td></tr>\n"
        return (
            '<div class="section">\n<h2>3. Outstanding Queries by Priority</h2>\n'
            "<table><thead><tr><th>Priority</th><th>Count</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_severity_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML finding severity distribution."""
        queries = data.get("queries", [])
        findings = [q for q in queries if q.get("query_type") == "finding"]
        if not findings:
            return ""
        dist: Dict[str, int] = {"material": 0, "significant": 0, "minor": 0, "observation": 0}
        for f in findings:
            sev = f.get("severity", "observation")
            if sev in dist:
                dist[sev] += 1
        rows = ""
        for sev, count in dist.items():
            rows += f"<tr><td>{sev.title()}</td><td>{count}</td></tr>\n"
        return (
            '<div class="section">\n<h2>4. Finding Severity Distribution</h2>\n'
            "<table><thead><tr><th>Severity</th><th>Count</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_resolution_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML resolution timeline."""
        queries = data.get("queries", [])
        resolved = [q for q in queries if q.get("days_to_resolve") is not None]
        if not resolved:
            return ""
        days_list = [q.get("days_to_resolve", 0) for q in resolved]
        avg_days = sum(days_list) / len(days_list) if days_list else 0
        max_days = max(days_list) if days_list else 0
        min_days = min(days_list) if days_list else 0
        rows = (
            f"<tr><td>Resolved Queries</td><td>{len(resolved)}</td></tr>\n"
            f"<tr><td>Average Days</td><td>{avg_days:.1f}</td></tr>\n"
            f"<tr><td>Min Days</td><td>{min_days}</td></tr>\n"
            f"<tr><td>Max Days</td><td>{max_days}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>5. Resolution Timeline</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_escalation_history(self, data: Dict[str, Any]) -> str:
        """Render HTML escalation history."""
        escalations = data.get("escalations", [])
        if not escalations:
            return ""
        rows = ""
        for e in escalations:
            rows += (
                f"<tr><td>{e.get('escalation_id', '')}</td>"
                f"<td>{e.get('query_id', '')}</td>"
                f"<td>{e.get('escalated_date', '-')}</td>"
                f"<td>{e.get('escalated_to', '')}</td>"
                f"<td>{e.get('reason', '')}</td>"
                f"<td>{e.get('resolution', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Escalation History</h2>\n'
            "<table><thead><tr><th>ID</th><th>Query</th><th>Date</th>"
            "<th>Escalated To</th><th>Reason</th><th>Resolution</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render query register as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "verifier_query_register",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "verifier_name": self._get_val(data, "verifier_name", ""),
            "queries": data.get("queries", []),
            "sla_summary": data.get("sla_summary"),
            "escalations": data.get("escalations", []),
            "chart_data": {
                "priority_bar": self._build_priority_bar(data),
                "severity_pie": self._build_severity_pie(data),
                "sla_gauge": self._build_sla_gauge(data),
                "resolution_histogram": self._build_resolution_histogram(data),
            },
        }

    def _build_priority_bar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build outstanding priority bar chart data."""
        queries = data.get("queries", [])
        open_q = [q for q in queries if q.get("status") in ("open", "in_progress", "overdue")]
        counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for q in open_q:
            pri = q.get("priority", "medium")
            if pri in counts:
                counts[pri] += 1
        return {"labels": list(counts.keys()), "values": list(counts.values())}

    def _build_severity_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build finding severity pie chart data."""
        queries = data.get("queries", [])
        findings = [q for q in queries if q.get("query_type") == "finding"]
        if not findings:
            return {}
        dist: Dict[str, int] = {"material": 0, "significant": 0, "minor": 0, "observation": 0}
        for f in findings:
            sev = f.get("severity", "observation")
            if sev in dist:
                dist[sev] += 1
        return {"labels": list(dist.keys()), "values": list(dist.values())}

    def _build_sla_gauge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build SLA compliance gauge data."""
        sla = data.get("sla_summary")
        if not sla:
            return {}
        return {
            "value": sla.get("sla_compliance_pct", 0),
            "min": 0,
            "max": 100,
            "target": 95,
        }

    def _build_resolution_histogram(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build resolution days histogram data."""
        queries = data.get("queries", [])
        resolved = [q for q in queries if q.get("days_to_resolve") is not None]
        if not resolved:
            return {}
        return {
            "query_ids": [q.get("query_id", "") for q in resolved],
            "days": [q.get("days_to_resolve", 0) for q in resolved],
        }
