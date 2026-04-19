"""
Executive Summary Template - PACK-008 EU Taxonomy Alignment Pack

This module generates board-level executive summaries with headline metrics,
alignment status, KPI summaries (Turnover/CapEx/OpEx), regulatory compliance status,
year-over-year progress, risk areas, and strategic recommendations for EU Taxonomy
compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from executive_summary import ExecutiveSummaryTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = ExecutiveSummaryTemplate()
    >>> report = template.render(data, format="markdown")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """Configuration for Executive Summary generation."""

    include_yoy_progress: bool = Field(
        default=True,
        description="Include year-over-year progress section"
    )
    include_risk_areas: bool = Field(
        default=True,
        description="Include risk areas section"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include strategic recommendations"
    )
    include_peer_comparison: bool = Field(
        default=False,
        description="Include peer/sector comparison if available"
    )


class HeadlineMetric(BaseModel):
    """A single headline metric for the executive dashboard."""

    metric_name: str = Field(..., description="Metric name")
    current_value: str = Field(..., description="Current period value (formatted)")
    prior_value: str = Field(default="", description="Prior period value (formatted)")
    change_direction: str = Field(
        default="",
        description="Change direction: UP, DOWN, FLAT, or N/A"
    )
    change_pct: float = Field(
        default=0.0,
        description="Percentage change from prior period"
    )
    status: str = Field(
        default="ON_TRACK",
        description="Status: ON_TRACK, AT_RISK, OFF_TRACK"
    )


class KPIHighlight(BaseModel):
    """KPI highlight for executive summary."""

    kpi_name: str = Field(..., description="KPI name (Turnover, CapEx, OpEx)")
    total_eur: float = Field(..., ge=0, description="Total amount")
    aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Aligned percentage")
    eligible_pct: float = Field(default=0.0, ge=0, le=100, description="Eligible percentage")
    prior_aligned_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Prior period aligned percentage"
    )
    change_pp: float = Field(
        default=0.0,
        description="Change in percentage points from prior period"
    )


class RegulatoryStatus(BaseModel):
    """Regulatory compliance status item."""

    requirement: str = Field(..., description="Regulatory requirement name")
    regulation: str = Field(..., description="Regulation reference")
    status: str = Field(
        ...,
        description="Status: COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT, NOT_APPLICABLE"
    )
    due_date: str = Field(default="", description="Compliance due date")
    notes: str = Field(default="", description="Additional notes")


class RiskArea(BaseModel):
    """Identified risk area."""

    risk_name: str = Field(..., description="Risk area name")
    severity: str = Field(
        ...,
        description="Severity: CRITICAL, HIGH, MEDIUM, LOW"
    )
    description: str = Field(..., description="Risk description")
    impact_area: str = Field(
        default="",
        description="Impact area (KPI, Disclosure, Compliance)"
    )
    mitigation_status: str = Field(
        default="PLANNED",
        description="Mitigation status"
    )


class StrategicRecommendation(BaseModel):
    """Strategic recommendation for the board."""

    priority: int = Field(..., ge=1, description="Priority ranking")
    recommendation: str = Field(..., description="Recommendation text")
    expected_impact: str = Field(
        default="",
        description="Expected impact on alignment ratios"
    )
    timeframe: str = Field(
        default="",
        description="Implementation timeframe"
    )
    estimated_investment_eur: float = Field(
        default=0.0, ge=0,
        description="Estimated investment required"
    )


class YoYProgress(BaseModel):
    """Year-over-year progress data point."""

    period: str = Field(..., description="Period label")
    turnover_aligned_pct: float = Field(default=0.0, ge=0, le=100)
    capex_aligned_pct: float = Field(default=0.0, ge=0, le=100)
    opex_aligned_pct: float = Field(default=0.0, ge=0, le=100)
    total_activities: int = Field(default=0, ge=0)
    aligned_activities: int = Field(default=0, ge=0)


class ReportData(BaseModel):
    """Data model for Executive Summary."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    headline_metrics: List[HeadlineMetric] = Field(
        default_factory=list,
        description="Headline metrics for dashboard"
    )
    kpi_highlights: List[KPIHighlight] = Field(
        default_factory=list,
        description="KPI highlights (Turnover, CapEx, OpEx)"
    )
    regulatory_status: List[RegulatoryStatus] = Field(
        default_factory=list,
        description="Regulatory compliance status items"
    )
    risk_areas: List[RiskArea] = Field(
        default_factory=list,
        description="Identified risk areas"
    )
    recommendations: List[StrategicRecommendation] = Field(
        default_factory=list,
        description="Strategic recommendations"
    )
    yoy_progress: List[YoYProgress] = Field(
        default_factory=list,
        description="Year-over-year progress data"
    )
    total_activities: int = Field(default=0, ge=0, description="Total activities")
    eligible_activities: int = Field(default=0, ge=0, description="Eligible activities")
    aligned_activities: int = Field(default=0, ge=0, description="Aligned activities")
    board_message: str = Field(
        default="",
        description="Custom message for the board"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class ExecutiveSummaryTemplate:
    """
    Executive Summary Template for EU Taxonomy Alignment Pack.

    Generates board-level overviews with headline metrics, KPI summaries,
    regulatory compliance status, risk areas, and strategic recommendations.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = ExecutiveSummaryTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Headline Metrics" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Executive Summary Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the executive summary.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered summary content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Executive Summary for {data.organization_name} "
            f"in {format} format"
        )

        if format == "markdown":
            content = self._render_markdown(data)
        elif format == "html":
            content = self._render_html(data)
        elif format == "json":
            content = self._render_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        content_hash = self._calculate_hash(content)
        logger.info(f"Summary generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render summary in Markdown format."""
        sections = []

        # Header
        sections.append(f"# EU Taxonomy Alignment - Executive Summary")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Prepared for:** Board of Directors")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Board Message
        if data.board_message:
            sections.append(f"> {data.board_message}")
            sections.append(f"")

        # Headline Metrics
        if data.headline_metrics:
            sections.append(f"## Headline Metrics")
            sections.append(f"")
            sections.append(
                f"| Metric | Current | Prior | Change | Status |"
            )
            sections.append(
                f"|--------|---------|-------|-------:|--------|"
            )
            for m in data.headline_metrics:
                change_str = ""
                if m.change_direction == "UP":
                    change_str = f"+{m.change_pct:.1f}%"
                elif m.change_direction == "DOWN":
                    change_str = f"{m.change_pct:.1f}%"
                elif m.change_direction == "FLAT":
                    change_str = "0.0%"
                else:
                    change_str = "N/A"
                sections.append(
                    f"| {m.metric_name} | {m.current_value} | "
                    f"{m.prior_value or 'N/A'} | {change_str} | {m.status} |"
                )
            sections.append(f"")

        # Alignment Status
        sections.append(f"## Alignment Status")
        sections.append(f"")
        align_rate = (data.aligned_activities / data.total_activities * 100) if data.total_activities > 0 else 0
        elig_rate = (data.eligible_activities / data.total_activities * 100) if data.total_activities > 0 else 0

        sections.append(f"| Category | Count | Percentage |")
        sections.append(f"|----------|------:|----------:|")
        sections.append(f"| Total Activities | {data.total_activities} | 100.0% |")
        sections.append(
            f"| Taxonomy-Eligible | {data.eligible_activities} | {elig_rate:.1f}% |"
        )
        sections.append(
            f"| Taxonomy-Aligned | {data.aligned_activities} | {align_rate:.1f}% |"
        )
        sections.append(f"")

        # KPI Summary
        if data.kpi_highlights:
            sections.append(f"## KPI Summary (Turnover / CapEx / OpEx)")
            sections.append(f"")
            sections.append(
                f"| KPI | Total (EUR) | Aligned % | Eligible % | "
                f"Prior Aligned % | Change (pp) |"
            )
            sections.append(
                f"|-----|----------:|----------:|----------:|"
                f"---------------:|----------:|"
            )
            for kpi in data.kpi_highlights:
                sections.append(
                    f"| {kpi.kpi_name} | {kpi.total_eur:,.0f} | "
                    f"{kpi.aligned_pct:.1f}% | {kpi.eligible_pct:.1f}% | "
                    f"{kpi.prior_aligned_pct:.1f}% | "
                    f"{kpi.change_pp:+.1f} |"
                )
            sections.append(f"")

        # Regulatory Compliance
        if data.regulatory_status:
            sections.append(f"## Regulatory Compliance")
            sections.append(f"")
            sections.append(
                f"| Requirement | Regulation | Status | Due Date |"
            )
            sections.append(
                f"|-------------|------------|--------|----------|"
            )
            for reg in data.regulatory_status:
                sections.append(
                    f"| {reg.requirement} | {reg.regulation} | "
                    f"{reg.status} | {reg.due_date} |"
                )
            sections.append(f"")

        # YoY Progress
        if self.config.include_yoy_progress and data.yoy_progress:
            sections.append(f"## Year-over-Year Progress")
            sections.append(f"")
            sections.append(
                f"| Period | Turnover Aligned % | CapEx Aligned % | "
                f"OpEx Aligned % | Aligned Activities |"
            )
            sections.append(
                f"|--------|-------------------:|----------------:|"
                f"---------------:|-------------------:|"
            )
            for yoy in data.yoy_progress:
                sections.append(
                    f"| {yoy.period} | {yoy.turnover_aligned_pct:.1f}% | "
                    f"{yoy.capex_aligned_pct:.1f}% | "
                    f"{yoy.opex_aligned_pct:.1f}% | "
                    f"{yoy.aligned_activities}/{yoy.total_activities} |"
                )
            sections.append(f"")

        # Risk Areas
        if self.config.include_risk_areas and data.risk_areas:
            sections.append(f"## Risk Areas")
            sections.append(f"")
            for risk in sorted(
                data.risk_areas,
                key=lambda r: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(r.severity, 4)
            ):
                sections.append(
                    f"- **[{risk.severity}]** {risk.risk_name}: {risk.description}"
                )
                if risk.impact_area:
                    sections.append(f"  - Impact: {risk.impact_area}")
                sections.append(f"  - Mitigation: {risk.mitigation_status}")
            sections.append(f"")

        # Strategic Recommendations
        if self.config.include_recommendations and data.recommendations:
            sections.append(f"## Strategic Recommendations")
            sections.append(f"")
            for rec in sorted(data.recommendations, key=lambda r: r.priority):
                sections.append(f"### {rec.priority}. {rec.recommendation}")
                sections.append(f"")
                if rec.expected_impact:
                    sections.append(f"- **Expected Impact:** {rec.expected_impact}")
                if rec.timeframe:
                    sections.append(f"- **Timeframe:** {rec.timeframe}")
                if rec.estimated_investment_eur > 0:
                    sections.append(
                        f"- **Estimated Investment:** EUR {rec.estimated_investment_eur:,.0f}"
                    )
                sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Executive summary generated on {data.report_date} using "
            f"GreenLang EU Taxonomy Alignment Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render summary in HTML format."""
        align_rate = (data.aligned_activities / data.total_activities * 100) if data.total_activities > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy Executive Summary - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-card {{ display: inline-block; width: 22%; margin: 1%; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #27ae60; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #27ae60; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .on-track {{ color: #27ae60; }}
        .at-risk {{ color: #f39c12; }}
        .off-track {{ color: #e74c3c; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
        blockquote {{ border-left: 4px solid #27ae60; padding-left: 15px; color: #555; font-style: italic; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy Alignment - Executive Summary</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Prepared for:</strong> Board of Directors</p>
        <p><strong>Alignment Rate:</strong> <span class="metric-value" style="font-size: 1.2em;">{align_rate:.1f}%</span></p>
    </div>
"""

        if data.board_message:
            html += f"""    <blockquote>{data.board_message}</blockquote>
"""

        if data.headline_metrics:
            html += """    <h2>Headline Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Current</th><th>Prior</th><th>Change</th><th>Status</th></tr>
"""
            for m in data.headline_metrics:
                status_class = m.status.lower().replace("_", "-")
                html += f"""        <tr>
            <td>{m.metric_name}</td><td>{m.current_value}</td>
            <td>{m.prior_value or 'N/A'}</td><td>{m.change_pct:+.1f}%</td>
            <td class="{status_class}">{m.status}</td>
        </tr>
"""
            html += """    </table>
"""

        if data.kpi_highlights:
            html += """    <h2>KPI Summary</h2>
    <table>
        <tr><th>KPI</th><th>Total (EUR)</th><th>Aligned %</th><th>Eligible %</th>
        <th>Prior Aligned %</th><th>Change (pp)</th></tr>
"""
            for kpi in data.kpi_highlights:
                html += f"""        <tr>
            <td>{kpi.kpi_name}</td><td>{kpi.total_eur:,.0f}</td>
            <td>{kpi.aligned_pct:.1f}%</td><td>{kpi.eligible_pct:.1f}%</td>
            <td>{kpi.prior_aligned_pct:.1f}%</td><td>{kpi.change_pp:+.1f}</td>
        </tr>
"""
            html += """    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Executive summary generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render summary in JSON format."""
        report_dict = {
            "report_type": "executive_summary",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "alignment_status": {
                "total_activities": data.total_activities,
                "eligible_activities": data.eligible_activities,
                "aligned_activities": data.aligned_activities,
                "alignment_rate": (
                    data.aligned_activities / data.total_activities * 100
                ) if data.total_activities > 0 else 0,
            },
            "headline_metrics": [m.dict() for m in data.headline_metrics],
            "kpi_highlights": [kpi.dict() for kpi in data.kpi_highlights],
            "regulatory_status": [reg.dict() for reg in data.regulatory_status],
            "risk_areas": [risk.dict() for risk in data.risk_areas],
            "recommendations": [rec.dict() for rec in data.recommendations],
            "yoy_progress": [yoy.dict() for yoy in data.yoy_progress],
            "board_message": data.board_message,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "ExecutiveSummaryTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
