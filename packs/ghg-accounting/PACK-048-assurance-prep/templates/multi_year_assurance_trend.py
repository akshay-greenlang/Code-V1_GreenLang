# -*- coding: utf-8 -*-
"""
MultiYearAssuranceTrend - Multi-Year Assurance Trend for PACK-048.

Generates a multi-year assurance trend report with readiness score
evolution over 3-5 years, finding recurrence analysis (recurring vs.
new vs. resolved), control maturity evolution, evidence quality trend,
query volume and resolution time trend, cost trajectory, and assurance
opinion history.

Regulatory References:
    - ISAE 3410: Engagement planning and trend analysis
    - ISO 14064-3: Continuous improvement of verification
    - CSRD: Progressive assurance requirements
    - GHG Protocol: Year-over-year improvement tracking

Sections:
    1. Readiness Score Evolution
    2. Finding Recurrence Analysis
    3. Control Maturity Evolution
    4. Evidence Quality Trend
    5. Query Volume & Resolution Trend
    6. Cost Trajectory
    7. Assurance Opinion History
    8. Provenance Footer

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


class OpinionType(str, Enum):
    """Assurance opinion type."""
    UNQUALIFIED = "unqualified"
    QUALIFIED = "qualified"
    ADVERSE = "adverse"
    DISCLAIMER = "disclaimer"
    NOT_ISSUED = "not_issued"


class FindingCategory(str, Enum):
    """Finding recurrence category."""
    NEW = "new"
    RECURRING = "recurring"
    RESOLVED = "resolved"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class YearlyReadinessScore(BaseModel):
    """Readiness score for a single year."""
    year: int = Field(..., description="Year")
    score: float = Field(0.0, ge=0, le=100, description="Readiness score")
    assurance_level: str = Field("limited", description="Assurance level that year")
    notes: str = Field("", description="Year-specific notes")


class FindingRecurrence(BaseModel):
    """Finding recurrence for a single year."""
    year: int = Field(..., description="Year")
    total_findings: int = Field(0, ge=0, description="Total findings")
    new_findings: int = Field(0, ge=0, description="New findings")
    recurring_findings: int = Field(0, ge=0, description="Recurring findings")
    resolved_findings: int = Field(0, ge=0, description="Resolved from prior year")


class ControlMaturityYear(BaseModel):
    """Control maturity for a single year."""
    year: int = Field(..., description="Year")
    average_maturity_score: float = Field(0.0, ge=0, le=5, description="Average maturity (0-5)")
    controls_at_optimizing: int = Field(0, ge=0, description="Controls at optimizing level")
    controls_at_managed: int = Field(0, ge=0, description="Controls at managed level")
    controls_at_defined: int = Field(0, ge=0, description="Controls at defined level")
    controls_at_repeatable: int = Field(0, ge=0, description="Controls at repeatable level")
    controls_at_initial: int = Field(0, ge=0, description="Controls at initial level")


class EvidenceQualityYear(BaseModel):
    """Evidence quality for a single year."""
    year: int = Field(..., description="Year")
    high_pct: float = Field(0.0, ge=0, le=100, description="% high quality")
    medium_pct: float = Field(0.0, ge=0, le=100, description="% medium quality")
    low_pct: float = Field(0.0, ge=0, le=100, description="% low quality")
    insufficient_pct: float = Field(0.0, ge=0, le=100, description="% insufficient")
    total_evidence_items: int = Field(0, ge=0, description="Total evidence items")


class QueryVolumeYear(BaseModel):
    """Query volume and resolution for a single year."""
    year: int = Field(..., description="Year")
    total_queries: int = Field(0, ge=0, description="Total queries raised")
    avg_resolution_days: float = Field(0.0, ge=0, description="Avg days to resolve")
    overdue_pct: float = Field(0.0, ge=0, le=100, description="% overdue")
    findings_count: int = Field(0, ge=0, description="Number of formal findings")


class CostYear(BaseModel):
    """Cost for a single year."""
    year: int = Field(..., description="Year")
    total_cost: float = Field(0.0, ge=0, description="Total cost")
    currency: str = Field("EUR", description="Currency")
    assurance_level: str = Field("limited", description="Assurance level")
    scope_coverage: str = Field("", description="Scope coverage")


class OpinionYear(BaseModel):
    """Assurance opinion for a single year."""
    year: int = Field(..., description="Year")
    opinion_type: OpinionType = Field(OpinionType.NOT_ISSUED, description="Opinion type")
    verifier: str = Field("", description="Verifier name")
    qualifications: List[str] = Field(default_factory=list, description="Qualification details")
    scope: str = Field("", description="Scope of assurance")


class MultiYearTrendInput(BaseModel):
    """Complete input model for MultiYearAssuranceTrend."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Current reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    readiness_scores: List[YearlyReadinessScore] = Field(
        default_factory=list, description="Readiness scores by year"
    )
    finding_recurrence: List[FindingRecurrence] = Field(
        default_factory=list, description="Finding recurrence by year"
    )
    control_maturity: List[ControlMaturityYear] = Field(
        default_factory=list, description="Control maturity by year"
    )
    evidence_quality: List[EvidenceQualityYear] = Field(
        default_factory=list, description="Evidence quality by year"
    )
    query_volume: List[QueryVolumeYear] = Field(
        default_factory=list, description="Query volume by year"
    )
    cost_trajectory: List[CostYear] = Field(
        default_factory=list, description="Cost trajectory by year"
    )
    opinion_history: List[OpinionYear] = Field(
        default_factory=list, description="Assurance opinion history"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _opinion_label(opinion: str) -> str:
    """Return display label for opinion type."""
    return opinion.replace("_", " ").title()


def _opinion_css(opinion: str) -> str:
    """Return CSS class for opinion type."""
    mapping = {
        "unqualified": "op-unqualified",
        "qualified": "op-qualified",
        "adverse": "op-adverse",
        "disclaimer": "op-disclaimer",
        "not_issued": "op-not-issued",
    }
    return mapping.get(opinion, "op-not-issued")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class MultiYearAssuranceTrend:
    """
    Multi-year assurance trend template for PACK-048.

    Renders multi-year trends for readiness scores, finding recurrence,
    control maturity, evidence quality, query resolution, cost, and
    opinion history. All outputs include SHA-256 provenance hashing
    for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = MultiYearAssuranceTrend()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MultiYearAssuranceTrend."""
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
        """Render multi-year trend as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render multi-year trend as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render multi-year trend as JSON dict."""
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
            self._md_readiness_evolution(data),
            self._md_finding_recurrence(data),
            self._md_control_maturity(data),
            self._md_evidence_quality(data),
            self._md_query_volume(data),
            self._md_cost_trajectory(data),
            self._md_opinion_history(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Multi-Year Assurance Trend - {company}\n\n"
            f"**Current Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_readiness_evolution(self, data: Dict[str, Any]) -> str:
        """Render Markdown readiness score evolution."""
        scores = data.get("readiness_scores", [])
        if not scores:
            return ""
        lines = [
            "## 1. Readiness Score Evolution",
            "",
            "| Year | Score | Level | Notes |",
            "|------|-------|-------|-------|",
        ]
        for s in sorted(scores, key=lambda x: x.get("year", 0)):
            lines.append(
                f"| {s.get('year', '')} | "
                f"{s.get('score', 0):.0f} | "
                f"{s.get('assurance_level', 'limited').title()} | "
                f"{s.get('notes', '')} |"
            )
        return "\n".join(lines)

    def _md_finding_recurrence(self, data: Dict[str, Any]) -> str:
        """Render Markdown finding recurrence analysis."""
        findings = data.get("finding_recurrence", [])
        if not findings:
            return ""
        lines = [
            "## 2. Finding Recurrence Analysis",
            "",
            "| Year | Total | New | Recurring | Resolved |",
            "|------|-------|-----|-----------|----------|",
        ]
        for f in sorted(findings, key=lambda x: x.get("year", 0)):
            lines.append(
                f"| {f.get('year', '')} | "
                f"{f.get('total_findings', 0)} | "
                f"{f.get('new_findings', 0)} | "
                f"{f.get('recurring_findings', 0)} | "
                f"{f.get('resolved_findings', 0)} |"
            )
        return "\n".join(lines)

    def _md_control_maturity(self, data: Dict[str, Any]) -> str:
        """Render Markdown control maturity evolution."""
        maturity = data.get("control_maturity", [])
        if not maturity:
            return ""
        lines = [
            "## 3. Control Maturity Evolution",
            "",
            "| Year | Avg Score | Optimizing | Managed | Defined | Repeatable | Initial |",
            "|------|-----------|-----------|---------|---------|-----------|---------|",
        ]
        for m in sorted(maturity, key=lambda x: x.get("year", 0)):
            lines.append(
                f"| {m.get('year', '')} | "
                f"{m.get('average_maturity_score', 0):.1f} | "
                f"{m.get('controls_at_optimizing', 0)} | "
                f"{m.get('controls_at_managed', 0)} | "
                f"{m.get('controls_at_defined', 0)} | "
                f"{m.get('controls_at_repeatable', 0)} | "
                f"{m.get('controls_at_initial', 0)} |"
            )
        return "\n".join(lines)

    def _md_evidence_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown evidence quality trend."""
        quality = data.get("evidence_quality", [])
        if not quality:
            return ""
        lines = [
            "## 4. Evidence Quality Trend",
            "",
            "| Year | High % | Medium % | Low % | Insufficient % | Total Items |",
            "|------|--------|----------|-------|----------------|-------------|",
        ]
        for q in sorted(quality, key=lambda x: x.get("year", 0)):
            lines.append(
                f"| {q.get('year', '')} | "
                f"{q.get('high_pct', 0):.1f}% | "
                f"{q.get('medium_pct', 0):.1f}% | "
                f"{q.get('low_pct', 0):.1f}% | "
                f"{q.get('insufficient_pct', 0):.1f}% | "
                f"{q.get('total_evidence_items', 0)} |"
            )
        return "\n".join(lines)

    def _md_query_volume(self, data: Dict[str, Any]) -> str:
        """Render Markdown query volume trend."""
        queries = data.get("query_volume", [])
        if not queries:
            return ""
        lines = [
            "## 5. Query Volume & Resolution Trend",
            "",
            "| Year | Queries | Avg Days | Overdue % | Findings |",
            "|------|---------|----------|-----------|----------|",
        ]
        for q in sorted(queries, key=lambda x: x.get("year", 0)):
            lines.append(
                f"| {q.get('year', '')} | "
                f"{q.get('total_queries', 0)} | "
                f"{q.get('avg_resolution_days', 0):.1f} | "
                f"{q.get('overdue_pct', 0):.1f}% | "
                f"{q.get('findings_count', 0)} |"
            )
        return "\n".join(lines)

    def _md_cost_trajectory(self, data: Dict[str, Any]) -> str:
        """Render Markdown cost trajectory."""
        costs = data.get("cost_trajectory", [])
        if not costs:
            return ""
        lines = [
            "## 6. Cost Trajectory",
            "",
            "| Year | Cost | Level | Scope |",
            "|------|------|-------|-------|",
        ]
        for c in sorted(costs, key=lambda x: x.get("year", 0)):
            curr = c.get("currency", "EUR")
            lines.append(
                f"| {c.get('year', '')} | "
                f"{c.get('total_cost', 0):,.0f} {curr} | "
                f"{c.get('assurance_level', 'limited').title()} | "
                f"{c.get('scope_coverage', '')} |"
            )
        return "\n".join(lines)

    def _md_opinion_history(self, data: Dict[str, Any]) -> str:
        """Render Markdown assurance opinion history."""
        opinions = data.get("opinion_history", [])
        if not opinions:
            return ""
        lines = [
            "## 7. Assurance Opinion History",
            "",
            "| Year | Opinion | Verifier | Scope | Qualifications |",
            "|------|---------|----------|-------|----------------|",
        ]
        for o in sorted(opinions, key=lambda x: x.get("year", 0)):
            quals = "; ".join(o.get("qualifications", [])[:3]) or "-"
            lines.append(
                f"| {o.get('year', '')} | "
                f"**{_opinion_label(o.get('opinion_type', 'not_issued'))}** | "
                f"{o.get('verifier', '')} | "
                f"{o.get('scope', '')} | "
                f"{quals} |"
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
            self._html_readiness_evolution(data),
            self._html_finding_recurrence(data),
            self._html_control_maturity(data),
            self._html_evidence_quality(data),
            self._html_query_volume(data),
            self._html_cost_trajectory(data),
            self._html_opinion_history(data),
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
            f"<title>Multi-Year Assurance Trend - {company}</title>\n"
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
            ".op-unqualified{color:#2a9d8f;font-weight:700;}\n"
            ".op-qualified{color:#e9c46a;font-weight:700;}\n"
            ".op-adverse{color:#e76f51;font-weight:700;}\n"
            ".op-disclaimer{color:#d62828;font-weight:700;}\n"
            ".op-not-issued{color:#888;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>Multi-Year Assurance Trend &mdash; {company}</h1>\n"
            f"<p><strong>Current Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_readiness_evolution(self, data: Dict[str, Any]) -> str:
        """Render HTML readiness evolution."""
        scores = data.get("readiness_scores", [])
        if not scores:
            return ""
        rows = ""
        for s in sorted(scores, key=lambda x: x.get("year", 0)):
            rows += (
                f"<tr><td>{s.get('year', '')}</td>"
                f"<td>{s.get('score', 0):.0f}</td>"
                f"<td>{s.get('assurance_level', 'limited').title()}</td>"
                f"<td>{s.get('notes', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. Readiness Score Evolution</h2>\n'
            "<table><thead><tr><th>Year</th><th>Score</th>"
            "<th>Level</th><th>Notes</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_finding_recurrence(self, data: Dict[str, Any]) -> str:
        """Render HTML finding recurrence."""
        findings = data.get("finding_recurrence", [])
        if not findings:
            return ""
        rows = ""
        for f in sorted(findings, key=lambda x: x.get("year", 0)):
            rows += (
                f"<tr><td>{f.get('year', '')}</td>"
                f"<td>{f.get('total_findings', 0)}</td>"
                f"<td>{f.get('new_findings', 0)}</td>"
                f"<td>{f.get('recurring_findings', 0)}</td>"
                f"<td>{f.get('resolved_findings', 0)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Finding Recurrence Analysis</h2>\n'
            "<table><thead><tr><th>Year</th><th>Total</th><th>New</th>"
            "<th>Recurring</th><th>Resolved</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_control_maturity(self, data: Dict[str, Any]) -> str:
        """Render HTML control maturity evolution."""
        maturity = data.get("control_maturity", [])
        if not maturity:
            return ""
        rows = ""
        for m in sorted(maturity, key=lambda x: x.get("year", 0)):
            rows += (
                f"<tr><td>{m.get('year', '')}</td>"
                f"<td>{m.get('average_maturity_score', 0):.1f}</td>"
                f"<td>{m.get('controls_at_optimizing', 0)}</td>"
                f"<td>{m.get('controls_at_managed', 0)}</td>"
                f"<td>{m.get('controls_at_defined', 0)}</td>"
                f"<td>{m.get('controls_at_repeatable', 0)}</td>"
                f"<td>{m.get('controls_at_initial', 0)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Control Maturity Evolution</h2>\n'
            "<table><thead><tr><th>Year</th><th>Avg Score</th><th>Optimizing</th>"
            "<th>Managed</th><th>Defined</th><th>Repeatable</th>"
            "<th>Initial</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_evidence_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML evidence quality trend."""
        quality = data.get("evidence_quality", [])
        if not quality:
            return ""
        rows = ""
        for q in sorted(quality, key=lambda x: x.get("year", 0)):
            rows += (
                f"<tr><td>{q.get('year', '')}</td>"
                f"<td>{q.get('high_pct', 0):.1f}%</td>"
                f"<td>{q.get('medium_pct', 0):.1f}%</td>"
                f"<td>{q.get('low_pct', 0):.1f}%</td>"
                f"<td>{q.get('insufficient_pct', 0):.1f}%</td>"
                f"<td>{q.get('total_evidence_items', 0)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Evidence Quality Trend</h2>\n'
            "<table><thead><tr><th>Year</th><th>High %</th><th>Medium %</th>"
            "<th>Low %</th><th>Insufficient %</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_query_volume(self, data: Dict[str, Any]) -> str:
        """Render HTML query volume trend."""
        queries = data.get("query_volume", [])
        if not queries:
            return ""
        rows = ""
        for q in sorted(queries, key=lambda x: x.get("year", 0)):
            rows += (
                f"<tr><td>{q.get('year', '')}</td>"
                f"<td>{q.get('total_queries', 0)}</td>"
                f"<td>{q.get('avg_resolution_days', 0):.1f}</td>"
                f"<td>{q.get('overdue_pct', 0):.1f}%</td>"
                f"<td>{q.get('findings_count', 0)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Query Volume & Resolution Trend</h2>\n'
            "<table><thead><tr><th>Year</th><th>Queries</th><th>Avg Days</th>"
            "<th>Overdue %</th><th>Findings</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_cost_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML cost trajectory."""
        costs = data.get("cost_trajectory", [])
        if not costs:
            return ""
        rows = ""
        for c in sorted(costs, key=lambda x: x.get("year", 0)):
            curr = c.get("currency", "EUR")
            rows += (
                f"<tr><td>{c.get('year', '')}</td>"
                f"<td>{c.get('total_cost', 0):,.0f} {curr}</td>"
                f"<td>{c.get('assurance_level', 'limited').title()}</td>"
                f"<td>{c.get('scope_coverage', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Cost Trajectory</h2>\n'
            "<table><thead><tr><th>Year</th><th>Cost</th>"
            "<th>Level</th><th>Scope</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_opinion_history(self, data: Dict[str, Any]) -> str:
        """Render HTML opinion history."""
        opinions = data.get("opinion_history", [])
        if not opinions:
            return ""
        rows = ""
        for o in sorted(opinions, key=lambda x: x.get("year", 0)):
            op = o.get("opinion_type", "not_issued")
            quals = "; ".join(o.get("qualifications", [])[:3]) or "-"
            rows += (
                f"<tr><td>{o.get('year', '')}</td>"
                f'<td class="{_opinion_css(op)}">{_opinion_label(op)}</td>'
                f"<td>{o.get('verifier', '')}</td>"
                f"<td>{o.get('scope', '')}</td>"
                f"<td>{quals}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>7. Assurance Opinion History</h2>\n'
            "<table><thead><tr><th>Year</th><th>Opinion</th><th>Verifier</th>"
            "<th>Scope</th><th>Qualifications</th></tr></thead>\n"
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
        """Render multi-year trend as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "multi_year_assurance_trend",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "readiness_scores": data.get("readiness_scores", []),
            "finding_recurrence": data.get("finding_recurrence", []),
            "control_maturity": data.get("control_maturity", []),
            "evidence_quality": data.get("evidence_quality", []),
            "query_volume": data.get("query_volume", []),
            "cost_trajectory": data.get("cost_trajectory", []),
            "opinion_history": data.get("opinion_history", []),
            "chart_data": {
                "readiness_line": self._build_readiness_line(data),
                "finding_stacked_bar": self._build_finding_bar(data),
                "maturity_line": self._build_maturity_line(data),
                "quality_stacked_area": self._build_quality_area(data),
                "cost_line": self._build_cost_line(data),
            },
        }

    def _build_readiness_line(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build readiness score line chart data."""
        scores = data.get("readiness_scores", [])
        if not scores:
            return {}
        sorted_s = sorted(scores, key=lambda x: x.get("year", 0))
        return {
            "years": [s.get("year", 0) for s in sorted_s],
            "scores": [s.get("score", 0) for s in sorted_s],
        }

    def _build_finding_bar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build finding recurrence stacked bar chart data."""
        findings = data.get("finding_recurrence", [])
        if not findings:
            return {}
        sorted_f = sorted(findings, key=lambda x: x.get("year", 0))
        return {
            "years": [f.get("year", 0) for f in sorted_f],
            "new": [f.get("new_findings", 0) for f in sorted_f],
            "recurring": [f.get("recurring_findings", 0) for f in sorted_f],
            "resolved": [f.get("resolved_findings", 0) for f in sorted_f],
        }

    def _build_maturity_line(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build maturity line chart data."""
        maturity = data.get("control_maturity", [])
        if not maturity:
            return {}
        sorted_m = sorted(maturity, key=lambda x: x.get("year", 0))
        return {
            "years": [m.get("year", 0) for m in sorted_m],
            "avg_scores": [m.get("average_maturity_score", 0) for m in sorted_m],
        }

    def _build_quality_area(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build quality stacked area chart data."""
        quality = data.get("evidence_quality", [])
        if not quality:
            return {}
        sorted_q = sorted(quality, key=lambda x: x.get("year", 0))
        return {
            "years": [q.get("year", 0) for q in sorted_q],
            "high": [q.get("high_pct", 0) for q in sorted_q],
            "medium": [q.get("medium_pct", 0) for q in sorted_q],
            "low": [q.get("low_pct", 0) for q in sorted_q],
            "insufficient": [q.get("insufficient_pct", 0) for q in sorted_q],
        }

    def _build_cost_line(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cost line chart data."""
        costs = data.get("cost_trajectory", [])
        if not costs:
            return {}
        sorted_c = sorted(costs, key=lambda x: x.get("year", 0))
        return {
            "years": [c.get("year", 0) for c in sorted_c],
            "costs": [c.get("total_cost", 0) for c in sorted_c],
        }
