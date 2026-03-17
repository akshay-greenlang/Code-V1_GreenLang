# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Executive Summary Template
========================================================

Generates a board-level EUDR compliance status overview with headline
readiness score, key metrics, per-commodity traffic lights, risk
exposure items, financial impact assessment, prioritized action items,
regulatory timeline, and quarter-over-quarter comparison.

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

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "executive_summary"
TEMPLATE_VERSION = "1.0.0"


# =============================================================================
# ENUMS
# =============================================================================

class ReadinessGrade(str, Enum):
    """EUDR readiness grade (A-F)."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class TrafficLight(str, Enum):
    """Commodity compliance traffic light."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


class CommodityType(str, Enum):
    """EUDR-regulated commodities."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


class ActionPriority(str, Enum):
    """Action priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class HeadlineScore(BaseModel):
    """Section 1: EUDR readiness headline score."""
    readiness_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall EUDR readiness 0-100"
    )
    grade: ReadinessGrade = Field(..., description="Letter grade A-F")
    summary_text: str = Field(
        "", description="One-line summary for the board"
    )

    @field_validator("grade", mode="before")
    @classmethod
    def derive_grade(cls, v: Any, info: Any) -> Any:
        """Derive grade from score if not explicitly set."""
        return v


class KeyMetric(BaseModel):
    """Single key metric for the board summary."""
    label: str = Field(..., description="Metric label")
    value: str = Field(..., description="Metric value (formatted)")
    trend: Optional[str] = Field(None, description="Trend indicator")
    detail: Optional[str] = Field(None, description="Additional detail")


class KeyMetricsSummary(BaseModel):
    """Section 2: Key metrics overview."""
    dds_submitted: int = Field(0, ge=0, description="DDS submitted")
    dds_total: int = Field(0, ge=0, description="Total DDS required")
    suppliers_compliant: int = Field(0, ge=0, description="Compliant suppliers")
    suppliers_total: int = Field(0, ge=0, description="Total suppliers")
    risk_exposure_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Risk exposure"
    )
    geolocation_coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Geo coverage %"
    )
    additional_metrics: List[KeyMetric] = Field(
        default_factory=list, description="Additional metrics"
    )


class CommodityStatusEntry(BaseModel):
    """Section 3: Per-commodity compliance status."""
    commodity: CommodityType = Field(..., description="Commodity")
    traffic_light: TrafficLight = Field(..., description="Traffic light")
    compliance_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Compliance %"
    )
    dds_submitted: int = Field(0, ge=0, description="DDS submitted")
    dds_required: int = Field(0, ge=0, description="DDS required")
    suppliers: int = Field(0, ge=0, description="Supplier count")
    key_issue: Optional[str] = Field(None, description="Key issue summary")


class RiskExposureItem(BaseModel):
    """Section 4: Top risk item requiring board attention."""
    rank: int = Field(..., ge=1, description="Risk rank")
    title: str = Field(..., description="Risk title")
    description: str = Field(..., description="Risk description")
    impact: str = Field("", description="Business impact")
    likelihood: str = Field("", description="Likelihood assessment")
    mitigation_status: str = Field("", description="Mitigation status")


class FinancialImpact(BaseModel):
    """Section 5: Financial impact assessment."""
    total_penalty_exposure_eur: float = Field(
        0.0, ge=0.0, description="Total estimated penalty exposure (EUR)"
    )
    non_compliant_shipment_value_eur: float = Field(
        0.0, ge=0.0, description="Value of non-compliant shipments (EUR)"
    )
    remediation_cost_eur: float = Field(
        0.0, ge=0.0, description="Estimated remediation cost (EUR)"
    )
    market_access_risk_eur: float = Field(
        0.0, ge=0.0, description="Market access risk (EUR)"
    )
    notes: Optional[str] = Field(None, description="Financial impact notes")


class ActionItem(BaseModel):
    """Section 6: Prioritized action item."""
    rank: int = Field(..., ge=1, description="Priority rank")
    title: str = Field(..., description="Action title")
    description: str = Field(..., description="Action description")
    priority: ActionPriority = Field(..., description="Priority level")
    owner: Optional[str] = Field(None, description="Responsible party")
    deadline: Optional[date] = Field(None, description="Target date")
    status: str = Field("OPEN", description="Action status")


class RegulatoryMilestone(BaseModel):
    """Section 7: Regulatory timeline milestone."""
    milestone_id: str = Field(..., description="Milestone identifier")
    title: str = Field(..., description="Milestone title")
    milestone_date: date = Field(..., description="Milestone date")
    description: str = Field("", description="Milestone description")
    applicability: str = Field("ALL", description="Who it applies to")
    days_remaining: int = Field(0, description="Days until milestone")
    readiness_pct: float = Field(0.0, ge=0.0, le=100.0, description="Readiness")


class QuarterlyComparison(BaseModel):
    """Section 8: Quarter-over-quarter metrics."""
    current_quarter: str = Field(..., description="Current quarter (e.g. Q1 2026)")
    previous_quarter: str = Field(..., description="Previous quarter")
    current_score: float = Field(0.0, ge=0.0, le=100.0, description="Current score")
    previous_score: float = Field(0.0, ge=0.0, le=100.0, description="Previous score")
    score_change: float = Field(0.0, description="Score change (signed)")
    dds_change: int = Field(0, description="DDS submitted change")
    suppliers_change: int = Field(0, description="Supplier compliance change")
    risk_change: float = Field(0.0, description="Risk exposure change")
    highlights: List[str] = Field(
        default_factory=list, description="Key improvements"
    )
    concerns: List[str] = Field(
        default_factory=list, description="Key concerns"
    )


class ExecutiveSummaryInput(BaseModel):
    """Complete input for the Executive Summary."""
    company_name: str = Field(..., description="Reporting entity")
    report_date: date = Field(
        default_factory=date.today, description="Report date"
    )
    headline: HeadlineScore = Field(..., description="Headline score")
    key_metrics: KeyMetricsSummary = Field(
        default_factory=KeyMetricsSummary, description="Key metrics"
    )
    commodity_status: List[CommodityStatusEntry] = Field(
        default_factory=list, description="Per-commodity status"
    )
    risk_exposure: List[RiskExposureItem] = Field(
        default_factory=list, description="Top risk items (max 5)"
    )
    financial_impact: FinancialImpact = Field(
        default_factory=FinancialImpact, description="Financial impact"
    )
    action_items: List[ActionItem] = Field(
        default_factory=list, description="Top actions (max 10)"
    )
    regulatory_timeline: List[RegulatoryMilestone] = Field(
        default_factory=list, description="Regulatory milestones"
    )
    quarterly_comparison: Optional[QuarterlyComparison] = Field(
        None, description="QoQ comparison"
    )

    @field_validator("risk_exposure")
    @classmethod
    def limit_risk_items(cls, v: List[RiskExposureItem]) -> List[RiskExposureItem]:
        """Limit to top 5 risk items."""
        return sorted(v, key=lambda x: x.rank)[:5]

    @field_validator("action_items")
    @classmethod
    def limit_action_items(cls, v: List[ActionItem]) -> List[ActionItem]:
        """Limit to top 10 actions."""
        return sorted(v, key=lambda x: x.rank)[:10]


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _grade_badge(grade: ReadinessGrade) -> str:
    """Text badge for grade."""
    return f"[Grade: {grade.value}]"


def _grade_css(grade: ReadinessGrade) -> str:
    """Inline CSS for grade."""
    mapping = {
        ReadinessGrade.A: "color:#1a7f37;font-weight:bold;font-size:2rem;",
        ReadinessGrade.B: "color:#2da44e;font-weight:bold;font-size:2rem;",
        ReadinessGrade.C: "color:#b08800;font-weight:bold;font-size:2rem;",
        ReadinessGrade.D: "color:#e36209;font-weight:bold;font-size:2rem;",
        ReadinessGrade.F: "color:#cf222e;font-weight:bold;font-size:2rem;",
    }
    return mapping.get(grade, "")


def _traffic_badge(light: TrafficLight) -> str:
    """Text badge for traffic light."""
    return f"[{light.value}]"


def _traffic_css(light: TrafficLight) -> str:
    """Inline CSS for traffic light."""
    mapping = {
        TrafficLight.GREEN: "color:#1a7f37;font-weight:bold;",
        TrafficLight.AMBER: "color:#b08800;font-weight:bold;",
        TrafficLight.RED: "color:#cf222e;font-weight:bold;",
    }
    return mapping.get(light, "")


def _fmt_commodity(commodity: CommodityType) -> str:
    """Human-readable commodity name."""
    mapping = {
        CommodityType.CATTLE: "Cattle",
        CommodityType.COCOA: "Cocoa",
        CommodityType.COFFEE: "Coffee",
        CommodityType.OIL_PALM: "Oil Palm",
        CommodityType.RUBBER: "Rubber",
        CommodityType.SOYA: "Soya",
        CommodityType.WOOD: "Wood",
    }
    return mapping.get(commodity, commodity.value)


def _fmt_eur(value: float) -> str:
    """Format EUR value."""
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:,.1f}M"
    if value >= 1_000:
        return f"EUR {value / 1_000:,.1f}K"
    return f"EUR {value:,.2f}"


def _priority_sort(priority: ActionPriority) -> int:
    """Priority sort key."""
    return {
        ActionPriority.CRITICAL: 0,
        ActionPriority.HIGH: 1,
        ActionPriority.MEDIUM: 2,
        ActionPriority.LOW: 3,
    }.get(priority, 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ExecutiveSummary:
    """Generate board-level EUDR compliance executive summary.

    Sections:
        1. Headline - EUDR readiness score and grade
        2. Key Metrics - DDS, suppliers, risk, coverage
        3. Commodity Status - Per-commodity traffic lights
        4. Risk Exposure - Top 5 board-attention items
        5. Financial Impact - Penalty exposure, remediation costs
        6. Action Items - Top 10 prioritized actions
        7. Regulatory Timeline - Upcoming EUDR milestones
        8. Comparison - Quarter-over-quarter improvement

    Example:
        >>> summary = ExecutiveSummary()
        >>> data = ExecutiveSummaryInput(...)
        >>> md = summary.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the Executive Summary template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: ExecutiveSummaryInput) -> str:
        """Render as Markdown.

        Args:
            data: Validated executive summary input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_headline(data),
            self._md_key_metrics(data),
            self._md_commodity_status(data),
            self._md_risk_exposure(data),
            self._md_financial_impact(data),
            self._md_action_items(data),
            self._md_regulatory_timeline(data),
            self._md_comparison(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: ExecutiveSummaryInput) -> str:
        """Render as HTML.

        Args:
            data: Validated executive summary input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_headline(data),
            self._html_key_metrics(data),
            self._html_commodity_status(data),
            self._html_risk_exposure(data),
            self._html_financial_impact(data),
            self._html_action_items(data),
            self._html_regulatory_timeline(data),
            self._html_comparison(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: ExecutiveSummaryInput) -> Dict[str, Any]:
        """Render as JSON-serializable dictionary.

        Args:
            data: Validated executive summary input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance_hash(data)

        result: Dict[str, Any] = {
            "metadata": {
                "pack_id": PACK_ID,
                "template_name": TEMPLATE_NAME,
                "version": TEMPLATE_VERSION,
                "generated_at": self._render_timestamp.isoformat(),
                "provenance_hash": provenance_hash,
            },
            "company_name": data.company_name,
            "report_date": data.report_date.isoformat(),
            "headline": data.headline.model_dump(mode="json"),
            "key_metrics": data.key_metrics.model_dump(mode="json"),
            "commodity_status": [
                c.model_dump(mode="json") for c in data.commodity_status
            ],
            "risk_exposure": [
                r.model_dump(mode="json") for r in data.risk_exposure
            ],
            "financial_impact": data.financial_impact.model_dump(mode="json"),
            "action_items": [
                a.model_dump(mode="json") for a in data.action_items
            ],
            "regulatory_timeline": [
                m.model_dump(mode="json") for m in data.regulatory_timeline
            ],
        }

        if data.quarterly_comparison:
            result["quarterly_comparison"] = data.quarterly_comparison.model_dump(
                mode="json"
            )

        return result

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance_hash(self, data: ExecutiveSummaryInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ExecutiveSummaryInput) -> str:
        """Report header."""
        return (
            f"# EUDR Executive Summary - {data.company_name}\n"
            f"**Report Date:** {data.report_date.isoformat()}\n\n---"
        )

    def _md_headline(self, data: ExecutiveSummaryInput) -> str:
        """Section 1: Headline."""
        h = data.headline
        summary = h.summary_text or "EUDR compliance readiness assessment."
        return (
            "## 1. EUDR Readiness Score\n\n"
            f"**Score: {h.readiness_score:.0f}/100** "
            f"{_grade_badge(h.grade)}\n\n"
            f"_{summary}_"
        )

    def _md_key_metrics(self, data: ExecutiveSummaryInput) -> str:
        """Section 2: Key Metrics."""
        km = data.key_metrics
        dds_pct = (
            (km.dds_submitted / km.dds_total * 100) if km.dds_total > 0 else 0.0
        )
        supplier_pct = (
            (km.suppliers_compliant / km.suppliers_total * 100)
            if km.suppliers_total > 0
            else 0.0
        )
        lines = [
            "## 2. Key Metrics\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| DDS Submitted | {km.dds_submitted}/{km.dds_total} ({dds_pct:.0f}%) |",
            f"| Suppliers Compliant | {km.suppliers_compliant}/{km.suppliers_total} "
            f"({supplier_pct:.0f}%) |",
            f"| Risk Exposure | {km.risk_exposure_score:.1f}/100 |",
            f"| Geolocation Coverage | {km.geolocation_coverage_pct:.1f}% |",
        ]
        for m in km.additional_metrics:
            trend = f" ({m.trend})" if m.trend else ""
            lines.append(f"| {m.label} | {m.value}{trend} |")
        return "\n".join(lines)

    def _md_commodity_status(self, data: ExecutiveSummaryInput) -> str:
        """Section 3: Commodity Status."""
        lines = [
            "## 3. Commodity Status\n",
            "| Commodity | Status | Compliance | DDS | Suppliers | Key Issue |",
            "|-----------|--------|------------|-----|-----------|-----------|",
        ]
        for c in data.commodity_status:
            issue = c.key_issue or "None"
            lines.append(
                f"| {_fmt_commodity(c.commodity)} | {_traffic_badge(c.traffic_light)} "
                f"| {c.compliance_pct:.0f}% | {c.dds_submitted}/{c.dds_required} "
                f"| {c.suppliers} | {issue} |"
            )
        if not data.commodity_status:
            lines.append("| - | No commodity data | - | - | - | - |")
        return "\n".join(lines)

    def _md_risk_exposure(self, data: ExecutiveSummaryInput) -> str:
        """Section 4: Risk Exposure."""
        lines = [
            "## 4. Risk Exposure (Top 5)\n",
            "| # | Risk | Impact | Likelihood | Mitigation |",
            "|---|------|--------|-----------|------------|",
        ]
        for r in data.risk_exposure:
            lines.append(
                f"| {r.rank} | {r.title} | {r.impact} "
                f"| {r.likelihood} | {r.mitigation_status} |"
            )
        if not data.risk_exposure:
            lines.append("| - | No significant risks identified | - | - | - |")
        return "\n".join(lines)

    def _md_financial_impact(self, data: ExecutiveSummaryInput) -> str:
        """Section 5: Financial Impact."""
        fi = data.financial_impact
        notes = fi.notes or "N/A"
        return (
            "## 5. Financial Impact Assessment\n\n"
            "| Category | Estimated Value |\n"
            "|----------|----------------|\n"
            f"| Penalty Exposure | {_fmt_eur(fi.total_penalty_exposure_eur)} |\n"
            f"| Non-Compliant Shipment Value | "
            f"{_fmt_eur(fi.non_compliant_shipment_value_eur)} |\n"
            f"| Remediation Cost | {_fmt_eur(fi.remediation_cost_eur)} |\n"
            f"| Market Access Risk | {_fmt_eur(fi.market_access_risk_eur)} |\n\n"
            f"**Notes:** {notes}"
        )

    def _md_action_items(self, data: ExecutiveSummaryInput) -> str:
        """Section 6: Action Items."""
        lines = [
            "## 6. Action Items (Top 10)\n",
            "| # | Priority | Title | Owner | Deadline | Status |",
            "|---|----------|-------|-------|----------|--------|",
        ]
        for a in data.action_items:
            owner = a.owner or "TBD"
            deadline = a.deadline.isoformat() if a.deadline else "TBD"
            lines.append(
                f"| {a.rank} | [{a.priority.value}] | {a.title} "
                f"| {owner} | {deadline} | {a.status} |"
            )
        if not data.action_items:
            lines.append("| - | No outstanding actions | - | - | - | - |")
        return "\n".join(lines)

    def _md_regulatory_timeline(self, data: ExecutiveSummaryInput) -> str:
        """Section 7: Regulatory Timeline."""
        lines = [
            "## 7. Regulatory Timeline\n",
            "| Milestone | Date | Description | Days Left | Readiness | Applies To |",
            "|-----------|------|-------------|-----------|-----------|------------|",
        ]
        for m in sorted(
            data.regulatory_timeline, key=lambda x: x.milestone_date
        ):
            lines.append(
                f"| {m.title} | {m.milestone_date.isoformat()} "
                f"| {m.description} | {m.days_remaining} "
                f"| {m.readiness_pct:.0f}% | {m.applicability} |"
            )
        if not data.regulatory_timeline:
            lines.append("| - | No milestones tracked | - | - | - | - |")
        return "\n".join(lines)

    def _md_comparison(self, data: ExecutiveSummaryInput) -> str:
        """Section 8: Quarter-over-Quarter Comparison."""
        qc = data.quarterly_comparison
        if qc is None:
            return "## 8. Quarter-over-Quarter Comparison\n\nNo comparison data available."

        sign = "+" if qc.score_change > 0 else ""
        highlights = (
            "\n".join(f"- {h}" for h in qc.highlights)
            if qc.highlights
            else "- None noted"
        )
        concerns = (
            "\n".join(f"- {c}" for c in qc.concerns)
            if qc.concerns
            else "- None noted"
        )
        dds_sign = "+" if qc.dds_change > 0 else ""
        sup_sign = "+" if qc.suppliers_change > 0 else ""
        risk_sign = "+" if qc.risk_change > 0 else ""

        return (
            "## 8. Quarter-over-Quarter Comparison\n\n"
            f"**{qc.previous_quarter} -> {qc.current_quarter}**\n\n"
            "| Metric | Previous | Current | Change |\n"
            "|--------|----------|---------|--------|\n"
            f"| Readiness Score | {qc.previous_score:.0f} "
            f"| {qc.current_score:.0f} | {sign}{qc.score_change:.1f} |\n"
            f"| DDS Submitted | - | - | {dds_sign}{qc.dds_change} |\n"
            f"| Supplier Compliance | - | - | {sup_sign}{qc.suppliers_change} |\n"
            f"| Risk Exposure | - | - | {risk_sign}{qc.risk_change:.1f} |\n\n"
            f"### Highlights\n\n{highlights}\n\n"
            f"### Concerns\n\n{concerns}"
        )

    def _md_provenance(self, data: ExecutiveSummaryInput) -> str:
        """Provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, data: ExecutiveSummaryInput, body: str) -> str:
        """Wrap body in HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>EUDR Executive Summary - {data.company_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1100px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:0.5rem;}\n"
            "h2{color:#2b6cb0;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".grade-box{display:inline-block;padding:1rem 2rem;border:3px solid #ccc;"
            "border-radius:12px;text-align:center;margin:1rem 0;}\n"
            ".grade-score{font-size:2.5rem;font-weight:bold;}\n"
            ".grade-letter{font-size:3rem;font-weight:bold;}\n"
            ".kpi-card{display:inline-block;text-align:center;padding:1rem 2rem;"
            "border:2px solid #e2e8f0;border-radius:8px;margin:0.5rem;"
            "min-width:150px;}\n"
            ".kpi-value{font-size:1.5rem;font-weight:bold;}\n"
            ".kpi-label{font-size:0.85rem;color:#666;}\n"
            ".traffic-green{color:#1a7f37;font-weight:bold;}\n"
            ".traffic-amber{color:#b08800;font-weight:bold;}\n"
            ".traffic-red{color:#cf222e;font-weight:bold;}\n"
            ".financial-alert{background:#fff8f0;border-left:4px solid #e36209;"
            "padding:1rem;margin:1rem 0;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: ExecutiveSummaryInput) -> str:
        """HTML header."""
        return (
            '<div class="section">\n'
            f"<h1>EUDR Executive Summary &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Report Date:</strong> "
            f"{data.report_date.isoformat()}</p>\n<hr>\n</div>"
        )

    def _html_headline(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 1: Headline."""
        h = data.headline
        css = _grade_css(h.grade)
        summary = h.summary_text or "EUDR compliance readiness assessment."
        return (
            '<div class="section">\n<h2>1. EUDR Readiness Score</h2>\n'
            f'<div class="grade-box">'
            f'<div class="grade-score">{h.readiness_score:.0f}/100</div>'
            f'<div class="grade-letter" style="{css}">{h.grade.value}</div>'
            f"</div>\n<p><em>{summary}</em></p>\n</div>"
        )

    def _html_key_metrics(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 2: Key Metrics."""
        km = data.key_metrics
        dds_pct = (
            (km.dds_submitted / km.dds_total * 100) if km.dds_total > 0 else 0.0
        )
        supplier_pct = (
            (km.suppliers_compliant / km.suppliers_total * 100)
            if km.suppliers_total > 0
            else 0.0
        )
        return (
            '<div class="section">\n<h2>2. Key Metrics</h2>\n<div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f"{km.dds_submitted}/{km.dds_total}</div>"
            f'<div class="kpi-label">DDS Submitted ({dds_pct:.0f}%)</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f"{km.suppliers_compliant}/{km.suppliers_total}</div>"
            f'<div class="kpi-label">Suppliers Compliant '
            f"({supplier_pct:.0f}%)</div></div>\n"
            f'<div class="kpi-card"><div class="kpi-value">'
            f"{km.risk_exposure_score:.0f}</div>"
            f'<div class="kpi-label">Risk Exposure</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f"{km.geolocation_coverage_pct:.0f}%</div>"
            f'<div class="kpi-label">Geo Coverage</div></div>\n'
            "</div>\n</div>"
        )

    def _html_commodity_status(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 3: Commodity Status."""
        rows = ""
        for c in data.commodity_status:
            css = _traffic_css(c.traffic_light)
            issue = c.key_issue or "None"
            rows += (
                f"<tr><td>{_fmt_commodity(c.commodity)}</td>"
                f'<td style="{css}">{c.traffic_light.value}</td>'
                f"<td>{c.compliance_pct:.0f}%</td>"
                f"<td>{c.dds_submitted}/{c.dds_required}</td>"
                f"<td>{c.suppliers}</td><td>{issue}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No commodity data</td></tr>'
        return (
            '<div class="section">\n<h2>3. Commodity Status</h2>\n'
            "<table><thead><tr><th>Commodity</th><th>Status</th>"
            "<th>Compliance</th><th>DDS</th><th>Suppliers</th>"
            f"<th>Key Issue</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_risk_exposure(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 4: Risk Exposure."""
        rows = ""
        for r in data.risk_exposure:
            rows += (
                f"<tr><td>{r.rank}</td><td><strong>{r.title}</strong><br>"
                f"<small>{r.description}</small></td>"
                f"<td>{r.impact}</td><td>{r.likelihood}</td>"
                f"<td>{r.mitigation_status}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No significant risks</td></tr>'
        return (
            '<div class="section">\n<h2>4. Risk Exposure (Top 5)</h2>\n'
            "<table><thead><tr><th>#</th><th>Risk</th><th>Impact</th>"
            "<th>Likelihood</th><th>Mitigation</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_financial_impact(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 5: Financial Impact."""
        fi = data.financial_impact
        notes = fi.notes or "N/A"
        return (
            '<div class="section">\n<h2>5. Financial Impact Assessment</h2>\n'
            f'<div class="financial-alert">'
            f"<strong>Total Estimated Exposure:</strong> "
            f"{_fmt_eur(fi.total_penalty_exposure_eur)}</div>\n"
            "<table><tbody>"
            f"<tr><th>Penalty Exposure</th>"
            f"<td>{_fmt_eur(fi.total_penalty_exposure_eur)}</td></tr>"
            f"<tr><th>Non-Compliant Shipment Value</th>"
            f"<td>{_fmt_eur(fi.non_compliant_shipment_value_eur)}</td></tr>"
            f"<tr><th>Remediation Cost</th>"
            f"<td>{_fmt_eur(fi.remediation_cost_eur)}</td></tr>"
            f"<tr><th>Market Access Risk</th>"
            f"<td>{_fmt_eur(fi.market_access_risk_eur)}</td></tr>"
            f"</tbody></table>\n<p><strong>Notes:</strong> {notes}</p>\n</div>"
        )

    def _html_action_items(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 6: Action Items."""
        rows = ""
        for a in data.action_items:
            owner = a.owner or "TBD"
            deadline = a.deadline.isoformat() if a.deadline else "TBD"
            rows += (
                f"<tr><td>{a.rank}</td><td>{a.priority.value}</td>"
                f"<td>{a.title}</td><td>{owner}</td>"
                f"<td>{deadline}</td><td>{a.status}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No outstanding actions</td></tr>'
        return (
            '<div class="section">\n<h2>6. Action Items (Top 10)</h2>\n'
            "<table><thead><tr><th>#</th><th>Priority</th><th>Title</th>"
            "<th>Owner</th><th>Deadline</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_regulatory_timeline(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 7: Regulatory Timeline."""
        rows = ""
        for m in sorted(
            data.regulatory_timeline, key=lambda x: x.milestone_date
        ):
            rows += (
                f"<tr><td>{m.title}</td><td>{m.milestone_date.isoformat()}</td>"
                f"<td>{m.description}</td><td>{m.days_remaining}</td>"
                f"<td>{m.readiness_pct:.0f}%</td><td>{m.applicability}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No milestones tracked</td></tr>'
        return (
            '<div class="section">\n<h2>7. Regulatory Timeline</h2>\n'
            "<table><thead><tr><th>Milestone</th><th>Date</th>"
            "<th>Description</th><th>Days</th><th>Readiness</th>"
            f"<th>Applies To</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_comparison(self, data: ExecutiveSummaryInput) -> str:
        """HTML Section 8: QoQ Comparison."""
        qc = data.quarterly_comparison
        if qc is None:
            return (
                '<div class="section"><h2>8. Quarter-over-Quarter</h2>'
                "<p>No comparison data.</p></div>"
            )

        sign = "+" if qc.score_change > 0 else ""
        change_css = "color:#1a7f37;" if qc.score_change > 0 else (
            "color:#cf222e;" if qc.score_change < 0 else ""
        )

        highlights_html = "".join(f"<li>{h}</li>" for h in qc.highlights)
        concerns_html = "".join(f"<li>{c}</li>" for c in qc.concerns)

        return (
            '<div class="section">\n<h2>8. Quarter-over-Quarter Comparison</h2>\n'
            f"<p><strong>{qc.previous_quarter} &rarr; {qc.current_quarter}</strong></p>\n"
            "<table><thead><tr><th>Metric</th><th>Previous</th>"
            "<th>Current</th><th>Change</th></tr></thead>\n<tbody>"
            f"<tr><td>Readiness Score</td><td>{qc.previous_score:.0f}</td>"
            f"<td>{qc.current_score:.0f}</td>"
            f'<td style="{change_css}">{sign}{qc.score_change:.1f}</td></tr>'
            "</tbody></table>\n"
            f"<h3>Highlights</h3>\n<ul>{highlights_html or '<li>None noted</li>'}</ul>\n"
            f"<h3>Concerns</h3>\n<ul>{concerns_html or '<li>None noted</li>'}</ul>\n"
            "</div>"
        )

    def _html_provenance(self, data: ExecutiveSummaryInput) -> str:
        """HTML provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            f"<p>Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} "
            f"| {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
