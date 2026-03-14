# -*- coding: utf-8 -*-
"""
PACK-001 Phase 3: Executive Summary Template
=============================================

Generates a board-level 2-page CSRD executive summary with compliance
status, key metrics, material topics, regulatory deadlines, risk
heatmap, and prioritized action items.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ComplianceStatus(str, Enum):
    """Traffic-light compliance status."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_APPLICABLE = "N/A"


class ActionPriority(str, Enum):
    """Priority levels for action items."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskLevel(str, Enum):
    """Risk severity levels for the heatmap."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ComplianceStatusEntry(BaseModel):
    """Compliance status for a single ESRS standard."""
    standard_id: str = Field(..., description="ESRS standard identifier (e.g. E1, S1)")
    standard_name: str = Field(..., description="Full standard name")
    status: ComplianceStatus = Field(..., description="Traffic-light status")
    completion_pct: float = Field(..., ge=0.0, le=100.0, description="Completion percentage")
    disclosure_count_met: int = Field(0, ge=0, description="Disclosures met")
    disclosure_count_total: int = Field(0, ge=0, description="Total disclosures required")
    notes: Optional[str] = Field(None, description="Brief status notes")


class KeyMetricsDashboard(BaseModel):
    """Headline KPI metrics for the dashboard section."""
    scope1_total_tco2e: Optional[float] = Field(None, ge=0.0, description="Scope 1 total in tCO2e")
    scope2_total_tco2e: Optional[float] = Field(None, ge=0.0, description="Scope 2 total in tCO2e")
    scope3_total_tco2e: Optional[float] = Field(None, ge=0.0, description="Scope 3 total in tCO2e")
    total_ghg_tco2e: Optional[float] = Field(None, ge=0.0, description="Total GHG in tCO2e")
    yoy_change_pct: Optional[float] = Field(None, description="Year-over-year change %")
    intensity_per_revenue: Optional[float] = Field(None, description="tCO2e per EUR million revenue")
    intensity_per_employee: Optional[float] = Field(None, description="tCO2e per employee")
    revenue_eur: Optional[float] = Field(None, ge=0.0, description="Annual revenue in EUR")
    employee_count: Optional[int] = Field(None, ge=0, description="Full-time employee count")
    reporting_period_start: Optional[date] = Field(None, description="Reporting period start")
    reporting_period_end: Optional[date] = Field(None, description="Reporting period end")


class MaterialTopicSummary(BaseModel):
    """Summary of a material topic for the executive view."""
    rank: int = Field(..., ge=1, description="Materiality ranking")
    topic_name: str = Field(..., description="Topic name")
    esrs_standard: str = Field(..., description="Relevant ESRS standard")
    impact_score: float = Field(..., ge=0.0, le=10.0, description="Impact materiality score")
    financial_score: float = Field(..., ge=0.0, le=10.0, description="Financial materiality score")
    is_material: bool = Field(True, description="Whether the topic is material")


class RegulatoryDeadline(BaseModel):
    """Regulatory deadline entry."""
    regulation: str = Field(..., description="Regulation name")
    deadline_date: date = Field(..., description="Deadline date")
    description: str = Field(..., description="What is due")
    days_remaining: Optional[int] = Field(None, description="Days until deadline")
    status: ComplianceStatus = Field(ComplianceStatus.WARNING, description="Readiness status")


class RiskHeatmapEntry(BaseModel):
    """Single cell in the materiality risk heatmap."""
    topic: str = Field(..., description="Risk topic")
    likelihood: float = Field(..., ge=0.0, le=5.0, description="Likelihood score 0-5")
    impact: float = Field(..., ge=0.0, le=5.0, description="Impact score 0-5")
    risk_level: RiskLevel = Field(..., description="Derived risk level")
    esrs_reference: Optional[str] = Field(None, description="ESRS standard reference")


class ActionItem(BaseModel):
    """Action item or recommendation."""
    action_id: str = Field(..., description="Unique action identifier")
    title: str = Field(..., description="Action title")
    description: str = Field(..., description="Detailed description")
    priority: ActionPriority = Field(..., description="Priority level")
    owner: Optional[str] = Field(None, description="Responsible party")
    due_date: Optional[date] = Field(None, description="Target completion date")
    esrs_reference: Optional[str] = Field(None, description="Related ESRS standard")


class ExecutiveSummaryInput(BaseModel):
    """Complete input data for the executive summary template."""
    company_name: str = Field(..., description="Reporting company name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting fiscal year")
    report_date: date = Field(default_factory=date.today, description="Report generation date")
    compliance_entries: List[ComplianceStatusEntry] = Field(
        default_factory=list, description="Per-standard compliance statuses"
    )
    key_metrics: KeyMetricsDashboard = Field(
        default_factory=KeyMetricsDashboard, description="Headline KPI metrics"
    )
    material_topics: List[MaterialTopicSummary] = Field(
        default_factory=list, description="Top material topics (max 5 for summary)"
    )
    deadlines: List[RegulatoryDeadline] = Field(
        default_factory=list, description="Upcoming regulatory deadlines"
    )
    risk_heatmap: List[RiskHeatmapEntry] = Field(
        default_factory=list, description="Risk heatmap entries"
    )
    action_items: List[ActionItem] = Field(
        default_factory=list, description="Prioritized action items"
    )

    @field_validator("material_topics")
    @classmethod
    def limit_material_topics(cls, v: List[MaterialTopicSummary]) -> List[MaterialTopicSummary]:
        """Board summary shows at most 5 material topics."""
        return sorted(v, key=lambda t: t.rank)[:5]


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format a numeric value with thousands separator, or return 'N/A'."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format a percentage value with sign, or return 'N/A'."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with appropriate scale."""
    return _fmt_number(value, decimals=1, suffix=" tCO2e")


def _status_badge(status: ComplianceStatus) -> str:
    """Return a text badge for compliance status."""
    mapping = {
        ComplianceStatus.PASS: "[PASS]",
        ComplianceStatus.WARNING: "[WARNING]",
        ComplianceStatus.FAIL: "[FAIL]",
        ComplianceStatus.NOT_APPLICABLE: "[N/A]",
    }
    return mapping.get(status, "[UNKNOWN]")


def _status_html_class(status: ComplianceStatus) -> str:
    """Return CSS class name for status."""
    return f"status-{status.value.lower().replace('/', '')}"


def _priority_sort_key(priority: ActionPriority) -> int:
    """Numeric sort key for action priority."""
    return {
        ActionPriority.CRITICAL: 0,
        ActionPriority.HIGH: 1,
        ActionPriority.MEDIUM: 2,
        ActionPriority.LOW: 3,
    }.get(priority, 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ExecutiveSummaryTemplate:
    """Generate board-level 2-page CSRD executive summary.

    Sections:
        1. Compliance Status Overview (PASS/WARNING/FAIL per ESRS standard)
        2. Key Metrics Dashboard (Scope 1/2/3 totals, YoY change, intensity)
        3. Material Topics (top 5 with impact/financial scores)
        4. Regulatory Deadline Tracking
        5. Risk Heatmap (materiality matrix summary)
        6. Action Items & Recommendations

    Example:
        >>> template = ExecutiveSummaryTemplate()
        >>> data = ExecutiveSummaryInput(company_name="Acme", reporting_year=2025, ...)
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> payload = template.render_json(data)
    """

    def __init__(self) -> None:
        """Initialize the template renderer."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: ExecutiveSummaryInput) -> str:
        """Render the executive summary as Markdown.

        Args:
            data: Validated executive summary input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_compliance_overview(data),
            self._md_key_metrics(data),
            self._md_material_topics(data),
            self._md_deadlines(data),
            self._md_risk_heatmap(data),
            self._md_action_items(data),
            self._md_footer(data),
        ]
        return "\n\n".join(sections)

    def render_html(self, data: ExecutiveSummaryInput) -> str:
        """Render the executive summary as HTML.

        Args:
            data: Validated executive summary input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_compliance_overview(data),
            self._html_key_metrics(data),
            self._html_material_topics(data),
            self._html_deadlines(data),
            self._html_risk_heatmap(data),
            self._html_action_items(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        return self._wrap_html(data.company_name, data.reporting_year, body)

    def render_json(self, data: ExecutiveSummaryInput) -> Dict[str, Any]:
        """Render the executive summary as a JSON-serializable dict.

        Args:
            data: Validated executive summary input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance(data)
        overall_score = self._overall_compliance_score(data)

        return {
            "template": "executive_summary",
            "version": "1.0.0",
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance_hash,
            "company_name": data.company_name,
            "reporting_year": data.reporting_year,
            "overall_compliance_score_pct": overall_score,
            "compliance_entries": [e.model_dump(mode="json") for e in data.compliance_entries],
            "key_metrics": data.key_metrics.model_dump(mode="json"),
            "material_topics": [t.model_dump(mode="json") for t in data.material_topics],
            "deadlines": [d.model_dump(mode="json") for d in data.deadlines],
            "risk_heatmap": [r.model_dump(mode="json") for r in data.risk_heatmap],
            "action_items": [a.model_dump(mode="json") for a in data.action_items],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: ExecutiveSummaryInput) -> str:
        """Compute SHA-256 provenance hash for the input data."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # METRICS HELPERS
    # --------------------------------------------------------------------- #

    def _overall_compliance_score(self, data: ExecutiveSummaryInput) -> float:
        """Calculate weighted overall compliance score from individual entries."""
        if not data.compliance_entries:
            return 0.0
        applicable = [e for e in data.compliance_entries if e.status != ComplianceStatus.NOT_APPLICABLE]
        if not applicable:
            return 100.0
        return sum(e.completion_pct for e in applicable) / len(applicable)

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTION RENDERERS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ExecutiveSummaryInput) -> str:
        """Render Markdown header block."""
        score = self._overall_compliance_score(data)
        lines = [
            f"# CSRD Executive Summary - {data.company_name}",
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Overall Compliance:** {score:.1f}%",
            "---",
        ]
        return "\n".join(lines)

    def _md_compliance_overview(self, data: ExecutiveSummaryInput) -> str:
        """Render compliance status table."""
        lines = [
            "## 1. Compliance Status Overview",
            "",
            "| Standard | Name | Status | Completion | Disclosures | Notes |",
            "|----------|------|--------|------------|-------------|-------|",
        ]
        for entry in data.compliance_entries:
            badge = _status_badge(entry.status)
            disc = f"{entry.disclosure_count_met}/{entry.disclosure_count_total}"
            notes = entry.notes or "-"
            lines.append(
                f"| {entry.standard_id} | {entry.standard_name} | {badge} "
                f"| {entry.completion_pct:.0f}% | {disc} | {notes} |"
            )
        if not data.compliance_entries:
            lines.append("| - | No data available | [N/A] | - | - | - |")
        return "\n".join(lines)

    def _md_key_metrics(self, data: ExecutiveSummaryInput) -> str:
        """Render key metrics dashboard."""
        m = data.key_metrics
        period = "N/A"
        if m.reporting_period_start and m.reporting_period_end:
            period = f"{m.reporting_period_start.isoformat()} to {m.reporting_period_end.isoformat()}"
        lines = [
            "## 2. Key Metrics Dashboard",
            "",
            f"**Reporting Period:** {period}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Scope 1 Emissions | {_fmt_tco2e(m.scope1_total_tco2e)} |",
            f"| Scope 2 Emissions | {_fmt_tco2e(m.scope2_total_tco2e)} |",
            f"| Scope 3 Emissions | {_fmt_tco2e(m.scope3_total_tco2e)} |",
            f"| **Total GHG Emissions** | **{_fmt_tco2e(m.total_ghg_tco2e)}** |",
            f"| Year-over-Year Change | {_fmt_pct(m.yoy_change_pct)} |",
            f"| Intensity (per EUR M revenue) | {_fmt_number(m.intensity_per_revenue, 2, ' tCO2e/EUR M')} |",
            f"| Intensity (per employee) | {_fmt_number(m.intensity_per_employee, 2, ' tCO2e/FTE')} |",
        ]
        return "\n".join(lines)

    def _md_material_topics(self, data: ExecutiveSummaryInput) -> str:
        """Render material topics summary table."""
        lines = [
            "## 3. Material Topics (Top 5)",
            "",
            "| Rank | Topic | ESRS | Impact Score | Financial Score | Material |",
            "|------|-------|------|-------------|----------------|----------|",
        ]
        for topic in data.material_topics:
            mat = "Yes" if topic.is_material else "No"
            lines.append(
                f"| {topic.rank} | {topic.topic_name} | {topic.esrs_standard} "
                f"| {topic.impact_score:.1f}/10 | {topic.financial_score:.1f}/10 | {mat} |"
            )
        if not data.material_topics:
            lines.append("| - | No material topics reported | - | - | - | - |")
        return "\n".join(lines)

    def _md_deadlines(self, data: ExecutiveSummaryInput) -> str:
        """Render regulatory deadline tracker."""
        lines = [
            "## 4. Regulatory Deadline Tracking",
            "",
            "| Regulation | Deadline | Description | Days Remaining | Status |",
            "|------------|----------|-------------|----------------|--------|",
        ]
        for dl in sorted(data.deadlines, key=lambda d: d.deadline_date):
            days = str(dl.days_remaining) if dl.days_remaining is not None else "N/A"
            lines.append(
                f"| {dl.regulation} | {dl.deadline_date.isoformat()} | {dl.description} "
                f"| {days} | {_status_badge(dl.status)} |"
            )
        if not data.deadlines:
            lines.append("| - | - | No deadlines tracked | - | - |")
        return "\n".join(lines)

    def _md_risk_heatmap(self, data: ExecutiveSummaryInput) -> str:
        """Render risk heatmap summary."""
        lines = [
            "## 5. Risk Heatmap Summary",
            "",
            "| Topic | Likelihood | Impact | Risk Level | ESRS Ref |",
            "|-------|-----------|--------|------------|----------|",
        ]
        for r in sorted(data.risk_heatmap, key=lambda x: x.likelihood * x.impact, reverse=True):
            ref = r.esrs_reference or "-"
            lines.append(
                f"| {r.topic} | {r.likelihood:.1f}/5 | {r.impact:.1f}/5 "
                f"| {r.risk_level.value} | {ref} |"
            )
        if not data.risk_heatmap:
            lines.append("| - | - | - | No risk data | - |")
        return "\n".join(lines)

    def _md_action_items(self, data: ExecutiveSummaryInput) -> str:
        """Render action items and recommendations."""
        sorted_actions = sorted(data.action_items, key=lambda a: _priority_sort_key(a.priority))
        lines = [
            "## 6. Action Items & Recommendations",
            "",
            "| ID | Priority | Title | Owner | Due Date | ESRS Ref |",
            "|----|----------|-------|-------|----------|----------|",
        ]
        for a in sorted_actions:
            due = a.due_date.isoformat() if a.due_date else "TBD"
            owner = a.owner or "Unassigned"
            ref = a.esrs_reference or "-"
            lines.append(
                f"| {a.action_id} | {a.priority.value} | {a.title} "
                f"| {owner} | {due} | {ref} |"
            )
        if not data.action_items:
            lines.append("| - | - | No outstanding actions | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: ExecutiveSummaryInput) -> str:
        """Render Markdown footer with provenance."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTION RENDERERS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, company: str, year: int, body: str) -> str:
        """Wrap body content in a full HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>CSRD Executive Summary - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#222;}\n"
            "table{border-collapse:collapse;width:100%;margin-bottom:1rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem;text-align:left;}\n"
            "th{background:#f5f5f5;}\n"
            ".status-pass{color:#1a7f37;font-weight:bold;}\n"
            ".status-warning{color:#b08800;font-weight:bold;}\n"
            ".status-fail{color:#cf222e;font-weight:bold;}\n"
            ".status-na{color:#888;}\n"
            ".metric-value{font-size:1.25rem;font-weight:bold;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML header."""
        score = self._overall_compliance_score(data)
        return (
            f'<div class="section">\n'
            f"<h1>CSRD Executive Summary &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()} | "
            f'<strong>Overall Compliance:</strong> {score:.1f}%</p>\n'
            f"<hr>\n</div>"
        )

    def _html_compliance_overview(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML compliance status table."""
        rows = []
        for entry in data.compliance_entries:
            css = _status_html_class(entry.status)
            disc = f"{entry.disclosure_count_met}/{entry.disclosure_count_total}"
            notes = entry.notes or "-"
            rows.append(
                f"<tr><td>{entry.standard_id}</td><td>{entry.standard_name}</td>"
                f'<td class="{css}">{entry.status.value}</td>'
                f"<td>{entry.completion_pct:.0f}%</td><td>{disc}</td>"
                f"<td>{notes}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No data available</td></tr>')
        return (
            '<div class="section">\n'
            "<h2>1. Compliance Status Overview</h2>\n"
            "<table><thead><tr><th>Standard</th><th>Name</th><th>Status</th>"
            "<th>Completion</th><th>Disclosures</th><th>Notes</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_key_metrics(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML key metrics."""
        m = data.key_metrics
        period = "N/A"
        if m.reporting_period_start and m.reporting_period_end:
            period = f"{m.reporting_period_start.isoformat()} to {m.reporting_period_end.isoformat()}"
        metrics = [
            ("Scope 1 Emissions", _fmt_tco2e(m.scope1_total_tco2e)),
            ("Scope 2 Emissions", _fmt_tco2e(m.scope2_total_tco2e)),
            ("Scope 3 Emissions", _fmt_tco2e(m.scope3_total_tco2e)),
            ("Total GHG Emissions", _fmt_tco2e(m.total_ghg_tco2e)),
            ("Year-over-Year Change", _fmt_pct(m.yoy_change_pct)),
            ("Intensity (per EUR M revenue)", _fmt_number(m.intensity_per_revenue, 2, " tCO2e/EUR M")),
            ("Intensity (per employee)", _fmt_number(m.intensity_per_employee, 2, " tCO2e/FTE")),
        ]
        rows = "\n".join(
            f'<tr><td>{label}</td><td class="metric-value">{val}</td></tr>'
            for label, val in metrics
        )
        return (
            '<div class="section">\n'
            "<h2>2. Key Metrics Dashboard</h2>\n"
            f"<p><strong>Reporting Period:</strong> {period}</p>\n"
            f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_material_topics(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML material topics table."""
        rows = []
        for t in data.material_topics:
            mat = "Yes" if t.is_material else "No"
            rows.append(
                f"<tr><td>{t.rank}</td><td>{t.topic_name}</td>"
                f"<td>{t.esrs_standard}</td><td>{t.impact_score:.1f}/10</td>"
                f"<td>{t.financial_score:.1f}/10</td><td>{mat}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No material topics reported</td></tr>')
        return (
            '<div class="section">\n'
            "<h2>3. Material Topics (Top 5)</h2>\n"
            "<table><thead><tr><th>Rank</th><th>Topic</th><th>ESRS</th>"
            "<th>Impact</th><th>Financial</th><th>Material</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_deadlines(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML regulatory deadlines."""
        rows = []
        for dl in sorted(data.deadlines, key=lambda d: d.deadline_date):
            css = _status_html_class(dl.status)
            days = str(dl.days_remaining) if dl.days_remaining is not None else "N/A"
            rows.append(
                f"<tr><td>{dl.regulation}</td><td>{dl.deadline_date.isoformat()}</td>"
                f"<td>{dl.description}</td><td>{days}</td>"
                f'<td class="{css}">{dl.status.value}</td></tr>'
            )
        if not rows:
            rows.append('<tr><td colspan="5">No deadlines tracked</td></tr>')
        return (
            '<div class="section">\n'
            "<h2>4. Regulatory Deadline Tracking</h2>\n"
            "<table><thead><tr><th>Regulation</th><th>Deadline</th>"
            "<th>Description</th><th>Days Remaining</th><th>Status</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_risk_heatmap(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML risk heatmap table."""
        rows = []
        for r in sorted(data.risk_heatmap, key=lambda x: x.likelihood * x.impact, reverse=True):
            ref = r.esrs_reference or "-"
            rows.append(
                f"<tr><td>{r.topic}</td><td>{r.likelihood:.1f}/5</td>"
                f"<td>{r.impact:.1f}/5</td><td>{r.risk_level.value}</td>"
                f"<td>{ref}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="5">No risk data</td></tr>')
        return (
            '<div class="section">\n'
            "<h2>5. Risk Heatmap Summary</h2>\n"
            "<table><thead><tr><th>Topic</th><th>Likelihood</th>"
            "<th>Impact</th><th>Risk Level</th><th>ESRS Ref</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_action_items(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML action items table."""
        sorted_actions = sorted(data.action_items, key=lambda a: _priority_sort_key(a.priority))
        rows = []
        for a in sorted_actions:
            due = a.due_date.isoformat() if a.due_date else "TBD"
            owner = a.owner or "Unassigned"
            ref = a.esrs_reference or "-"
            rows.append(
                f"<tr><td>{a.action_id}</td><td>{a.priority.value}</td>"
                f"<td>{a.title}</td><td>{owner}</td><td>{due}</td>"
                f"<td>{ref}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No outstanding actions</td></tr>')
        return (
            '<div class="section">\n'
            "<h2>6. Action Items &amp; Recommendations</h2>\n"
            "<table><thead><tr><th>ID</th><th>Priority</th><th>Title</th>"
            "<th>Owner</th><th>Due Date</th><th>ESRS Ref</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: ExecutiveSummaryInput) -> str:
        """Render HTML footer with provenance."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n'
            "<hr>\n"
            f"<p>Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n"
            "</div>"
        )
