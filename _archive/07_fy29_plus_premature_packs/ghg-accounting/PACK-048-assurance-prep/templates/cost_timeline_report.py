# -*- coding: utf-8 -*-
"""
CostTimelineReport - Cost & Timeline Report for PACK-048.

Generates an engagement cost and timeline report with cost breakdown by
component, assurance level comparison (limited vs. reasonable costs),
timeline Gantt chart data (planning, fieldwork, reporting), resource
allocation by role, multi-year cost projection, and verifier RFP summary.

Regulatory References:
    - ISAE 3410: Engagement acceptance and planning
    - ISO 14064-3: Verification programme planning
    - CSRD: Mandatory assurance cost considerations
    - IAASB Quality Management Standards (ISQM 1)

Sections:
    1. Engagement Cost Breakdown
    2. Assurance Level Comparison
    3. Timeline Gantt Chart Data
    4. Resource Allocation
    5. Multi-Year Cost Projection
    6. Verifier RFP Summary
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class PhaseType(str, Enum):
    """Engagement phase type."""
    PLANNING = "planning"
    FIELDWORK = "fieldwork"
    REPORTING = "reporting"
    FOLLOW_UP = "follow_up"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class CostComponent(BaseModel):
    """Single cost component."""
    component_name: str = Field(..., description="Component name")
    cost_amount: float = Field(0.0, ge=0, description="Cost amount (currency)")
    currency: str = Field("EUR", description="Currency code")
    pct_of_total: float = Field(0.0, ge=0, le=100, description="% of total cost")
    hours: Optional[float] = Field(None, ge=0, description="Hours estimated")
    rate_per_hour: Optional[float] = Field(None, ge=0, description="Rate per hour")
    notes: str = Field("", description="Additional notes")

class AssuranceLevelComparison(BaseModel):
    """Cost comparison between assurance levels."""
    level: str = Field(..., description="Assurance level (limited/reasonable)")
    total_cost: float = Field(0.0, ge=0, description="Total estimated cost")
    total_hours: float = Field(0.0, ge=0, description="Total estimated hours")
    duration_weeks: float = Field(0.0, ge=0, description="Duration in weeks")
    key_differences: List[str] = Field(
        default_factory=list, description="Key differences from other level"
    )
    currency: str = Field("EUR", description="Currency code")

class TimelinePhase(BaseModel):
    """Single phase in the engagement timeline."""
    phase_name: str = Field(..., description="Phase name")
    phase_type: PhaseType = Field(PhaseType.PLANNING, description="Phase type")
    start_date: Optional[str] = Field(None, description="Start date (ISO)")
    end_date: Optional[str] = Field(None, description="End date (ISO)")
    duration_days: int = Field(0, ge=0, description="Duration in days")
    milestones: List[str] = Field(default_factory=list, description="Key milestones")
    deliverables: List[str] = Field(default_factory=list, description="Key deliverables")

class ResourceAllocation(BaseModel):
    """Resource allocation by role."""
    role: str = Field(..., description="Role title (e.g., Partner, Manager)")
    name: str = Field("", description="Person name (if known)")
    hours: float = Field(0.0, ge=0, description="Allocated hours")
    rate_per_hour: Optional[float] = Field(None, ge=0, description="Rate per hour")
    total_cost: float = Field(0.0, ge=0, description="Total cost for this role")
    currency: str = Field("EUR", description="Currency code")
    phase: str = Field("", description="Primary phase")

class YearProjection(BaseModel):
    """Single-year cost projection."""
    year: int = Field(..., description="Year")
    estimated_cost: float = Field(0.0, ge=0, description="Estimated cost")
    assurance_level: str = Field("limited", description="Assurance level")
    scope_coverage: str = Field("", description="Scope coverage")
    notes: str = Field("", description="Projection notes")
    currency: str = Field("EUR", description="Currency code")

class VerifierRFPSummary(BaseModel):
    """Verifier RFP summary data."""
    rfp_status: str = Field("", description="RFP status (draft/issued/evaluated)")
    verifiers_invited: int = Field(0, ge=0, description="Number of verifiers invited")
    proposals_received: int = Field(0, ge=0, description="Proposals received")
    preferred_verifier: str = Field("", description="Preferred verifier (if selected)")
    evaluation_criteria: List[str] = Field(
        default_factory=list, description="Evaluation criteria"
    )
    selection_date: Optional[str] = Field(None, description="Selection date (ISO)")
    engagement_start: Optional[str] = Field(None, description="Engagement start (ISO)")

class CostTimelineInput(BaseModel):
    """Complete input model for CostTimelineReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    total_cost: float = Field(0.0, ge=0, description="Total engagement cost")
    currency: str = Field("EUR", description="Currency code")
    cost_components: List[CostComponent] = Field(
        default_factory=list, description="Cost breakdown"
    )
    level_comparison: List[AssuranceLevelComparison] = Field(
        default_factory=list, description="Assurance level comparison"
    )
    timeline_phases: List[TimelinePhase] = Field(
        default_factory=list, description="Timeline phases"
    )
    resources: List[ResourceAllocation] = Field(
        default_factory=list, description="Resource allocation"
    )
    multi_year_projection: List[YearProjection] = Field(
        default_factory=list, description="Multi-year projection"
    )
    rfp_summary: Optional[VerifierRFPSummary] = Field(
        None, description="Verifier RFP summary"
    )

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class CostTimelineReport:
    """
    Cost and timeline report template for PACK-048.

    Renders engagement cost breakdown, assurance level comparison, timeline
    Gantt data, resource allocation, multi-year projection, and RFP summary.
    All outputs include SHA-256 provenance hashing for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = CostTimelineReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CostTimelineReport."""
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
        """Render cost timeline as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render cost timeline as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render cost timeline as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
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
            self._md_cost_breakdown(data),
            self._md_level_comparison(data),
            self._md_timeline(data),
            self._md_resources(data),
            self._md_projection(data),
            self._md_rfp_summary(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        total = data.get("total_cost", 0)
        currency = data.get("currency", "EUR")
        return (
            f"# Cost & Timeline Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Total Cost:** {total:,.0f} {currency} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_cost_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown cost breakdown."""
        components = data.get("cost_components", [])
        if not components:
            return ""
        currency = data.get("currency", "EUR")
        lines = [
            "## 1. Engagement Cost Breakdown",
            "",
            f"| Component | Cost ({currency}) | % of Total | Hours | Rate/Hr |",
            f"|-----------|-----|------------|-------|---------|",
        ]
        for c in components:
            hours = c.get("hours")
            hours_str = f"{hours:,.0f}" if hours is not None else "-"
            rate = c.get("rate_per_hour")
            rate_str = f"{rate:,.0f}" if rate is not None else "-"
            lines.append(
                f"| {c.get('component_name', '')} | "
                f"{c.get('cost_amount', 0):,.0f} | "
                f"{c.get('pct_of_total', 0):.1f}% | "
                f"{hours_str} | "
                f"{rate_str} |"
            )
        return "\n".join(lines)

    def _md_level_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown assurance level comparison."""
        levels = data.get("level_comparison", [])
        if not levels:
            return ""
        lines = [
            "## 2. Assurance Level Comparison",
            "",
            "| Level | Total Cost | Hours | Duration (weeks) |",
            "|-------|-----------|-------|------------------|",
        ]
        for lv in levels:
            curr = lv.get("currency", "EUR")
            lines.append(
                f"| {lv.get('level', '').title()} | "
                f"{lv.get('total_cost', 0):,.0f} {curr} | "
                f"{lv.get('total_hours', 0):,.0f} | "
                f"{lv.get('duration_weeks', 0):.0f} |"
            )
            diffs = lv.get("key_differences", [])
            if diffs:
                for d in diffs:
                    lines.append(f"  - {d}")
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render Markdown timeline phases."""
        phases = data.get("timeline_phases", [])
        if not phases:
            return ""
        lines = [
            "## 3. Engagement Timeline",
            "",
            "| Phase | Type | Start | End | Duration (days) | Deliverables |",
            "|-------|------|-------|-----|-----------------|-------------|",
        ]
        for p in phases:
            deliverables = "; ".join(p.get("deliverables", [])[:3]) or "-"
            lines.append(
                f"| {p.get('phase_name', '')} | "
                f"{p.get('phase_type', 'planning').title()} | "
                f"{p.get('start_date', '-')} | "
                f"{p.get('end_date', '-')} | "
                f"{p.get('duration_days', 0)} | "
                f"{deliverables} |"
            )
        return "\n".join(lines)

    def _md_resources(self, data: Dict[str, Any]) -> str:
        """Render Markdown resource allocation."""
        resources = data.get("resources", [])
        if not resources:
            return ""
        lines = [
            "## 4. Resource Allocation",
            "",
            "| Role | Name | Hours | Rate/Hr | Total Cost | Phase |",
            "|------|------|-------|---------|------------|-------|",
        ]
        for r in resources:
            rate = r.get("rate_per_hour")
            rate_str = f"{rate:,.0f}" if rate is not None else "-"
            lines.append(
                f"| {r.get('role', '')} | "
                f"{r.get('name', '')} | "
                f"{r.get('hours', 0):,.0f} | "
                f"{rate_str} | "
                f"{r.get('total_cost', 0):,.0f} | "
                f"{r.get('phase', '')} |"
            )
        return "\n".join(lines)

    def _md_projection(self, data: Dict[str, Any]) -> str:
        """Render Markdown multi-year projection."""
        projections = data.get("multi_year_projection", [])
        if not projections:
            return ""
        lines = [
            "## 5. Multi-Year Cost Projection",
            "",
            "| Year | Estimated Cost | Level | Scope | Notes |",
            "|------|---------------|-------|-------|-------|",
        ]
        for p in projections:
            curr = p.get("currency", "EUR")
            lines.append(
                f"| {p.get('year', '')} | "
                f"{p.get('estimated_cost', 0):,.0f} {curr} | "
                f"{p.get('assurance_level', 'limited').title()} | "
                f"{p.get('scope_coverage', '')} | "
                f"{p.get('notes', '')} |"
            )
        return "\n".join(lines)

    def _md_rfp_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown RFP summary."""
        rfp = data.get("rfp_summary")
        if not rfp:
            return ""
        lines = [
            "## 6. Verifier RFP Summary",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Status | {rfp.get('rfp_status', '')} |",
            f"| Verifiers Invited | {rfp.get('verifiers_invited', 0)} |",
            f"| Proposals Received | {rfp.get('proposals_received', 0)} |",
            f"| Preferred Verifier | {rfp.get('preferred_verifier', '-')} |",
        ]
        sel_date = rfp.get("selection_date")
        if sel_date:
            lines.append(f"| Selection Date | {sel_date} |")
        criteria = rfp.get("evaluation_criteria", [])
        if criteria:
            lines.append("")
            lines.append("**Evaluation Criteria:**")
            for c in criteria:
                lines.append(f"- {c}")
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
            self._html_cost_breakdown(data),
            self._html_level_comparison(data),
            self._html_timeline(data),
            self._html_resources(data),
            self._html_projection(data),
            self._html_rfp_summary(data),
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
            f"<title>Cost & Timeline - {company}</title>\n"
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
            ".gantt-bar{background:#264653;height:18px;border-radius:3px;display:inline-block;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        total = data.get("total_cost", 0)
        currency = data.get("currency", "EUR")
        return (
            '<div class="section">\n'
            f"<h1>Cost &amp; Timeline Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Total Cost:</strong> {total:,.0f} {currency} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_cost_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML cost breakdown."""
        components = data.get("cost_components", [])
        if not components:
            return ""
        currency = data.get("currency", "EUR")
        rows = ""
        for c in components:
            hours = c.get("hours")
            hours_str = f"{hours:,.0f}" if hours is not None else "-"
            rate = c.get("rate_per_hour")
            rate_str = f"{rate:,.0f}" if rate is not None else "-"
            rows += (
                f"<tr><td>{c.get('component_name', '')}</td>"
                f"<td>{c.get('cost_amount', 0):,.0f}</td>"
                f"<td>{c.get('pct_of_total', 0):.1f}%</td>"
                f"<td>{hours_str}</td>"
                f"<td>{rate_str}</td></tr>\n"
            )
        return (
            f'<div class="section">\n<h2>1. Engagement Cost Breakdown</h2>\n'
            f"<table><thead><tr><th>Component</th><th>Cost ({currency})</th>"
            "<th>% of Total</th><th>Hours</th><th>Rate/Hr</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_level_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML assurance level comparison."""
        levels = data.get("level_comparison", [])
        if not levels:
            return ""
        rows = ""
        for lv in levels:
            curr = lv.get("currency", "EUR")
            rows += (
                f"<tr><td>{lv.get('level', '').title()}</td>"
                f"<td>{lv.get('total_cost', 0):,.0f} {curr}</td>"
                f"<td>{lv.get('total_hours', 0):,.0f}</td>"
                f"<td>{lv.get('duration_weeks', 0):.0f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Assurance Level Comparison</h2>\n'
            "<table><thead><tr><th>Level</th><th>Total Cost</th>"
            "<th>Hours</th><th>Duration (weeks)</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML timeline."""
        phases = data.get("timeline_phases", [])
        if not phases:
            return ""
        rows = ""
        for p in phases:
            deliverables = "; ".join(p.get("deliverables", [])[:3]) or "-"
            rows += (
                f"<tr><td>{p.get('phase_name', '')}</td>"
                f"<td>{p.get('phase_type', 'planning').title()}</td>"
                f"<td>{p.get('start_date', '-')}</td>"
                f"<td>{p.get('end_date', '-')}</td>"
                f"<td>{p.get('duration_days', 0)}</td>"
                f"<td>{deliverables}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Engagement Timeline</h2>\n'
            "<table><thead><tr><th>Phase</th><th>Type</th><th>Start</th>"
            "<th>End</th><th>Days</th><th>Deliverables</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_resources(self, data: Dict[str, Any]) -> str:
        """Render HTML resource allocation."""
        resources = data.get("resources", [])
        if not resources:
            return ""
        rows = ""
        for r in resources:
            rate = r.get("rate_per_hour")
            rate_str = f"{rate:,.0f}" if rate is not None else "-"
            rows += (
                f"<tr><td>{r.get('role', '')}</td>"
                f"<td>{r.get('name', '')}</td>"
                f"<td>{r.get('hours', 0):,.0f}</td>"
                f"<td>{rate_str}</td>"
                f"<td>{r.get('total_cost', 0):,.0f}</td>"
                f"<td>{r.get('phase', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Resource Allocation</h2>\n'
            "<table><thead><tr><th>Role</th><th>Name</th><th>Hours</th>"
            "<th>Rate/Hr</th><th>Total Cost</th><th>Phase</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_projection(self, data: Dict[str, Any]) -> str:
        """Render HTML multi-year projection."""
        projections = data.get("multi_year_projection", [])
        if not projections:
            return ""
        rows = ""
        for p in projections:
            curr = p.get("currency", "EUR")
            rows += (
                f"<tr><td>{p.get('year', '')}</td>"
                f"<td>{p.get('estimated_cost', 0):,.0f} {curr}</td>"
                f"<td>{p.get('assurance_level', 'limited').title()}</td>"
                f"<td>{p.get('scope_coverage', '')}</td>"
                f"<td>{p.get('notes', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Multi-Year Cost Projection</h2>\n'
            "<table><thead><tr><th>Year</th><th>Cost</th><th>Level</th>"
            "<th>Scope</th><th>Notes</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_rfp_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML RFP summary."""
        rfp = data.get("rfp_summary")
        if not rfp:
            return ""
        rows = (
            f"<tr><td>Status</td><td>{rfp.get('rfp_status', '')}</td></tr>\n"
            f"<tr><td>Verifiers Invited</td><td>{rfp.get('verifiers_invited', 0)}</td></tr>\n"
            f"<tr><td>Proposals Received</td><td>{rfp.get('proposals_received', 0)}</td></tr>\n"
            f"<tr><td>Preferred Verifier</td><td>{rfp.get('preferred_verifier', '-')}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>6. Verifier RFP Summary</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
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
        """Render cost timeline as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "cost_timeline_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "total_cost": data.get("total_cost", 0),
            "currency": data.get("currency", "EUR"),
            "cost_components": data.get("cost_components", []),
            "level_comparison": data.get("level_comparison", []),
            "timeline_phases": data.get("timeline_phases", []),
            "resources": data.get("resources", []),
            "multi_year_projection": data.get("multi_year_projection", []),
            "rfp_summary": data.get("rfp_summary"),
            "chart_data": {
                "cost_pie": self._build_cost_pie(data),
                "gantt": self._build_gantt_data(data),
                "projection_line": self._build_projection_line(data),
            },
        }

    def _build_cost_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cost breakdown pie chart data."""
        components = data.get("cost_components", [])
        if not components:
            return {}
        return {
            "labels": [c.get("component_name", "") for c in components],
            "values": [c.get("cost_amount", 0) for c in components],
        }

    def _build_gantt_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Gantt chart data for timeline."""
        phases = data.get("timeline_phases", [])
        return [
            {
                "phase": p.get("phase_name", ""),
                "type": p.get("phase_type", "planning"),
                "start": p.get("start_date", ""),
                "end": p.get("end_date", ""),
                "duration_days": p.get("duration_days", 0),
            }
            for p in phases
        ]

    def _build_projection_line(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build multi-year projection line chart data."""
        projections = data.get("multi_year_projection", [])
        if not projections:
            return {}
        return {
            "years": [p.get("year", 0) for p in projections],
            "costs": [p.get("estimated_cost", 0) for p in projections],
        }
