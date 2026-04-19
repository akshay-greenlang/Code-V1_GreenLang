"""
KPI Dashboard Template - PACK-008 EU Taxonomy Alignment Pack

This module generates Turnover/CapEx/OpEx alignment ratio dashboards with year-over-year
trends, activity-level breakdowns, top aligned activities, and improvement opportunities
for EU Taxonomy compliance reporting.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from kpi_dashboard import KPIDashboardTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = KPIDashboardTemplate()
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
    """Configuration for KPI Dashboard generation."""

    include_yoy_trends: bool = Field(
        default=True,
        description="Include year-over-year trend analysis"
    )
    include_activity_breakdown: bool = Field(
        default=True,
        description="Include activity-level KPI breakdown"
    )
    include_improvement_opportunities: bool = Field(
        default=True,
        description="Include improvement opportunity analysis"
    )
    top_activities_count: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of top activities to highlight"
    )
    currency: str = Field(default="EUR", description="Reporting currency")


class KPISummary(BaseModel):
    """Summary of a single KPI (Turnover, CapEx, or OpEx)."""

    kpi_name: str = Field(..., description="KPI name (Turnover, CapEx, OpEx)")
    total_eur: float = Field(..., ge=0, description="Total amount in EUR")
    aligned_eur: float = Field(default=0.0, ge=0, description="Aligned amount in EUR")
    eligible_eur: float = Field(default=0.0, ge=0, description="Eligible amount in EUR")
    non_eligible_eur: float = Field(default=0.0, ge=0, description="Non-eligible amount in EUR")
    aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Aligned percentage")
    eligible_pct: float = Field(default=0.0, ge=0, le=100, description="Eligible percentage")
    non_eligible_pct: float = Field(default=0.0, ge=0, le=100, description="Non-eligible percentage")


class YoYTrend(BaseModel):
    """Year-over-year trend data point."""

    period: str = Field(..., description="Period label (e.g., FY 2023, FY 2024, FY 2025)")
    turnover_aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Turnover aligned %")
    capex_aligned_pct: float = Field(default=0.0, ge=0, le=100, description="CapEx aligned %")
    opex_aligned_pct: float = Field(default=0.0, ge=0, le=100, description="OpEx aligned %")
    turnover_eligible_pct: float = Field(default=0.0, ge=0, le=100, description="Turnover eligible %")
    capex_eligible_pct: float = Field(default=0.0, ge=0, le=100, description="CapEx eligible %")
    opex_eligible_pct: float = Field(default=0.0, ge=0, le=100, description="OpEx eligible %")


class ActivityKPI(BaseModel):
    """KPI breakdown for a single activity."""

    activity_id: str = Field(..., description="Taxonomy activity identifier")
    activity_name: str = Field(..., description="Activity name")
    nace_code: str = Field(..., description="NACE code")
    turnover_eur: float = Field(default=0.0, ge=0, description="Activity turnover")
    turnover_pct_of_total: float = Field(default=0.0, ge=0, le=100, description="% of total turnover")
    capex_eur: float = Field(default=0.0, ge=0, description="Activity CapEx")
    capex_pct_of_total: float = Field(default=0.0, ge=0, le=100, description="% of total CapEx")
    opex_eur: float = Field(default=0.0, ge=0, description="Activity OpEx")
    opex_pct_of_total: float = Field(default=0.0, ge=0, le=100, description="% of total OpEx")
    is_aligned: bool = Field(default=False, description="Alignment status")
    sc_objective: str = Field(default="", description="SC objective if aligned")


class ImprovementOpportunity(BaseModel):
    """Improvement opportunity for increasing alignment."""

    activity_name: str = Field(..., description="Activity name")
    current_status: str = Field(..., description="Current status (Eligible, Partially Aligned)")
    gap_description: str = Field(..., description="Gap preventing full alignment")
    potential_turnover_impact_eur: float = Field(
        default=0.0, ge=0,
        description="Potential turnover impact if aligned"
    )
    potential_capex_impact_eur: float = Field(
        default=0.0, ge=0,
        description="Potential CapEx impact if aligned"
    )
    estimated_effort: str = Field(
        default="MEDIUM",
        description="Estimated effort (LOW, MEDIUM, HIGH)"
    )
    priority: str = Field(default="MEDIUM", description="Priority (CRITICAL, HIGH, MEDIUM, LOW)")


class ReportData(BaseModel):
    """Data model for KPI Dashboard."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    kpi_summary: List[KPISummary] = Field(
        default_factory=list,
        description="KPI summaries (Turnover, CapEx, OpEx)"
    )
    yoy_trends: List[YoYTrend] = Field(
        default_factory=list,
        description="Year-over-year trend data"
    )
    activity_kpis: List[ActivityKPI] = Field(
        default_factory=list,
        description="Activity-level KPI breakdown"
    )
    improvement_opportunities: List[ImprovementOpportunity] = Field(
        default_factory=list,
        description="Improvement opportunities"
    )
    key_highlights: List[str] = Field(
        default_factory=list,
        description="Key dashboard highlights"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class KPIDashboardTemplate:
    """
    KPI Dashboard Template for EU Taxonomy Alignment Pack.

    Generates Turnover/CapEx/OpEx alignment ratio dashboards with year-over-year
    trends, activity-level breakdowns, and improvement opportunities.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = KPIDashboardTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "KPI Summary" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize KPI Dashboard Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the KPI dashboard.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered dashboard content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering KPI Dashboard for {data.organization_name} in {format} format"
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
        logger.info(f"Dashboard generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render dashboard in Markdown format."""
        sections = []

        # Header
        sections.append(f"# EU Taxonomy KPI Dashboard")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # KPI Summary
        if data.kpi_summary:
            sections.append(f"## KPI Summary")
            sections.append(f"")
            sections.append(
                f"| KPI | Total (EUR) | Aligned (EUR) | Aligned % | "
                f"Eligible (EUR) | Eligible % | Non-Eligible % |"
            )
            sections.append(
                f"|-----|----------:|-------------:|----------:|"
                f"--------------:|----------:|---------------:|"
            )
            for kpi in data.kpi_summary:
                sections.append(
                    f"| {kpi.kpi_name} | {kpi.total_eur:,.0f} | "
                    f"{kpi.aligned_eur:,.0f} | {kpi.aligned_pct:.1f}% | "
                    f"{kpi.eligible_eur:,.0f} | {kpi.eligible_pct:.1f}% | "
                    f"{kpi.non_eligible_pct:.1f}% |"
                )
            sections.append(f"")

            # Individual KPI analyses
            for kpi in data.kpi_summary:
                sections.append(f"### {kpi.kpi_name} Analysis")
                sections.append(f"")
                sections.append(f"- **Total:** EUR {kpi.total_eur:,.0f}")
                sections.append(
                    f"- **Taxonomy-Aligned:** EUR {kpi.aligned_eur:,.0f} "
                    f"({kpi.aligned_pct:.1f}%)"
                )
                sections.append(
                    f"- **Taxonomy-Eligible (not aligned):** EUR {kpi.eligible_eur:,.0f} "
                    f"({kpi.eligible_pct:.1f}%)"
                )
                sections.append(
                    f"- **Non-Eligible:** EUR {kpi.non_eligible_eur:,.0f} "
                    f"({kpi.non_eligible_pct:.1f}%)"
                )
                sections.append(f"")

        # YoY Trends
        if self.config.include_yoy_trends and data.yoy_trends:
            sections.append(f"## Year-over-Year Trends")
            sections.append(f"")
            sections.append(
                f"| Period | Turnover Aligned % | CapEx Aligned % | OpEx Aligned % | "
                f"Turnover Eligible % | CapEx Eligible % | OpEx Eligible % |"
            )
            sections.append(
                f"|--------|-------------------:|----------------:|---------------:|"
                f"-------------------:|----------------:|----------------:|"
            )
            for trend in data.yoy_trends:
                sections.append(
                    f"| {trend.period} | {trend.turnover_aligned_pct:.1f}% | "
                    f"{trend.capex_aligned_pct:.1f}% | {trend.opex_aligned_pct:.1f}% | "
                    f"{trend.turnover_eligible_pct:.1f}% | "
                    f"{trend.capex_eligible_pct:.1f}% | "
                    f"{trend.opex_eligible_pct:.1f}% |"
                )
            sections.append(f"")

            # Trend chart
            if len(data.yoy_trends) >= 2:
                sections.append(f"### Alignment Trend Visualization")
                sections.append(f"")
                sections.append(f"```")
                sections.append(self._create_trend_chart(data.yoy_trends))
                sections.append(f"```")
                sections.append(f"")

        # Top Aligned Activities
        if self.config.include_activity_breakdown and data.activity_kpis:
            aligned_acts = [a for a in data.activity_kpis if a.is_aligned]
            top_by_turnover = sorted(
                aligned_acts, key=lambda a: a.turnover_eur, reverse=True
            )[:self.config.top_activities_count]

            if top_by_turnover:
                sections.append(f"## Top Aligned Activities (by Turnover)")
                sections.append(f"")
                sections.append(
                    f"| Rank | Activity | NACE | SC Objective | "
                    f"Turnover (EUR) | % of Total |"
                )
                sections.append(
                    f"|-----:|----------|------|--------------|"
                    f"--------------:|----------:|"
                )
                for idx, act in enumerate(top_by_turnover, 1):
                    sections.append(
                        f"| {idx} | {act.activity_name[:35]} | {act.nace_code} | "
                        f"{act.sc_objective} | {act.turnover_eur:,.0f} | "
                        f"{act.turnover_pct_of_total:.1f}% |"
                    )
                sections.append(f"")

        # Improvement Opportunities
        if self.config.include_improvement_opportunities and data.improvement_opportunities:
            sections.append(f"## Improvement Opportunities")
            sections.append(f"")
            sections.append(
                f"The following activities represent opportunities to increase "
                f"taxonomy alignment ratios:"
            )
            sections.append(f"")
            sections.append(
                f"| Activity | Status | Gap | Turnover Impact (EUR) | "
                f"Effort | Priority |"
            )
            sections.append(
                f"|----------|--------|-----|-----------------------|"
                f"--------|----------|"
            )
            for opp in sorted(
                data.improvement_opportunities,
                key=lambda o: o.potential_turnover_impact_eur,
                reverse=True
            ):
                sections.append(
                    f"| {opp.activity_name[:30]} | {opp.current_status} | "
                    f"{opp.gap_description[:35]} | "
                    f"{opp.potential_turnover_impact_eur:,.0f} | "
                    f"{opp.estimated_effort} | {opp.priority} |"
                )
            sections.append(f"")

        # Key Highlights
        if data.key_highlights:
            sections.append(f"## Key Highlights")
            sections.append(f"")
            for idx, highlight in enumerate(data.key_highlights, 1):
                sections.append(f"{idx}. {highlight}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Dashboard generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render dashboard in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy KPI Dashboard - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background-color: #27ae60; color: white; text-align: center; }}
        td:first-child {{ text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #27ae60; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .kpi-card {{ display: inline-block; width: 30%; margin: 1%; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #27ae60; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy KPI Dashboard</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
    </div>
"""

        if data.kpi_summary:
            html += """    <h2>KPI Summary</h2>
    <table>
        <tr><th>KPI</th><th>Total (EUR)</th><th>Aligned (EUR)</th><th>Aligned %</th>
        <th>Eligible (EUR)</th><th>Eligible %</th></tr>
"""
            for kpi in data.kpi_summary:
                html += f"""        <tr>
            <td>{kpi.kpi_name}</td><td>{kpi.total_eur:,.0f}</td>
            <td>{kpi.aligned_eur:,.0f}</td><td><span class="metric">{kpi.aligned_pct:.1f}%</span></td>
            <td>{kpi.eligible_eur:,.0f}</td><td>{kpi.eligible_pct:.1f}%</td>
        </tr>
"""
            html += """    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Dashboard generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render dashboard in JSON format."""
        report_dict = {
            "report_type": "kpi_dashboard",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "kpi_summary": [kpi.dict() for kpi in data.kpi_summary],
            "yoy_trends": [trend.dict() for trend in data.yoy_trends],
            "activity_kpis": [act.dict() for act in data.activity_kpis],
            "improvement_opportunities": [opp.dict() for opp in data.improvement_opportunities],
            "key_highlights": data.key_highlights,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "KPIDashboardTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_trend_chart(self, trends: List[YoYTrend]) -> str:
        """Create text-based trend chart for alignment percentages."""
        chart_lines = []
        chart_lines.append("Turnover Alignment % Trend")
        chart_lines.append("")

        max_pct = max(t.turnover_aligned_pct for t in trends) if trends else 1
        scale = 40 / max_pct if max_pct > 0 else 1

        for trend in trends:
            bar_length = int(trend.turnover_aligned_pct * scale)
            bar = "█" * bar_length
            chart_lines.append(
                f"{trend.period:12} |{bar} {trend.turnover_aligned_pct:.1f}%"
            )

        chart_lines.append("")
        return "\n".join(chart_lines)
