"""
Portfolio Dashboard Template - PACK-007 EUDR Professional Pack

This module generates multi-operator portfolio dashboards with operator compliance scores,
aggregated risk summaries, DDS submission tracking, supplier pool statistics, cross-operator
benchmarking, and cost allocation for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from portfolio_dashboard import PortfolioDashboardTemplate, ReportData
    >>> data = ReportData(
    ...     portfolio_name="EU Forest Products Portfolio",
    ...     report_date="2026-03-15",
    ...     total_operators=25
    ... )
    >>> template = PortfolioDashboardTemplate()
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
    """Configuration for Portfolio Dashboard generation."""

    include_cost_allocation: bool = Field(
        default=True,
        description="Include cost allocation breakdown"
    )
    include_supplier_pool: bool = Field(
        default=True,
        description="Include supplier pool statistics"
    )
    include_cross_operator_benchmark: bool = Field(
        default=True,
        description="Include cross-operator benchmarking"
    )
    top_performers_count: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of top performers to highlight"
    )


class OperatorScore(BaseModel):
    """Operator compliance score summary."""

    operator_id: str = Field(..., description="Operator identifier")
    operator_name: str = Field(..., description="Operator name")
    country: str = Field(..., description="Primary country of operation")
    compliance_score: float = Field(..., ge=0, le=100, description="Overall compliance score")
    risk_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    dds_submissions_total: int = Field(..., ge=0, description="Total DDS submissions")
    dds_submissions_approved: int = Field(..., ge=0, description="Approved submissions")
    active_suppliers: int = Field(..., ge=0, description="Number of active suppliers")
    total_volume_tonnes: float = Field(..., ge=0, description="Total volume in tonnes")
    last_assessment_date: str = Field(..., description="Date of last assessment")


class DDSSubmissionSummary(BaseModel):
    """DDS submission tracking summary."""

    period: str = Field(..., description="Reporting period (e.g., 2026-Q1)")
    total_submissions: int = Field(..., ge=0, description="Total submissions")
    approved: int = Field(..., ge=0, description="Approved submissions")
    pending: int = Field(..., ge=0, description="Pending review")
    rejected: int = Field(..., ge=0, description="Rejected submissions")
    avg_processing_days: float = Field(..., ge=0, description="Average processing time in days")


class SupplierPoolStats(BaseModel):
    """Supplier pool statistics."""

    total_suppliers: int = Field(..., ge=0, description="Total unique suppliers")
    active_suppliers: int = Field(..., ge=0, description="Active suppliers")
    certified_suppliers: int = Field(..., ge=0, description="Certified suppliers")
    high_risk_suppliers: int = Field(..., ge=0, description="High-risk suppliers")
    avg_supplier_score: float = Field(..., ge=0, le=100, description="Average supplier score")
    shared_suppliers: int = Field(
        ...,
        ge=0,
        description="Suppliers used by multiple operators"
    )
    top_countries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top supplier countries with counts"
    )


class CostAllocation(BaseModel):
    """Cost allocation summary."""

    operator_id: str = Field(..., description="Operator identifier")
    operator_name: str = Field(..., description="Operator name")
    total_cost_eur: float = Field(..., ge=0, description="Total compliance cost in EUR")
    cost_per_tonne_eur: float = Field(..., ge=0, description="Cost per tonne in EUR")
    cost_per_dds_eur: float = Field(..., ge=0, description="Cost per DDS in EUR")
    cost_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by category"
    )


class RiskAggregation(BaseModel):
    """Aggregated risk summary."""

    risk_category: str = Field(..., description="Risk category")
    high_risk_operators: int = Field(..., ge=0, description="Operators with high risk")
    medium_risk_operators: int = Field(..., ge=0, description="Operators with medium risk")
    low_risk_operators: int = Field(..., ge=0, description="Operators with low risk")
    total_exposure_tonnes: float = Field(..., ge=0, description="Total volume at this risk level")
    mitigation_actions_required: int = Field(
        ...,
        ge=0,
        description="Number of mitigation actions required"
    )


class ReportData(BaseModel):
    """Data model for Portfolio Dashboard."""

    portfolio_name: str = Field(..., description="Portfolio name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., 2026-Q1)")
    total_operators: int = Field(..., ge=0, description="Total operators in portfolio")
    operator_scores: List[OperatorScore] = Field(
        default_factory=list,
        description="Operator compliance scores"
    )
    dds_tracking: List[DDSSubmissionSummary] = Field(
        default_factory=list,
        description="DDS submission tracking"
    )
    supplier_pool: SupplierPoolStats = Field(..., description="Supplier pool statistics")
    cost_allocations: List[CostAllocation] = Field(
        default_factory=list,
        description="Cost allocations by operator"
    )
    risk_aggregation: List[RiskAggregation] = Field(
        default_factory=list,
        description="Aggregated risk summary"
    )
    portfolio_compliance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Portfolio-wide compliance score"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key portfolio insights"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class PortfolioDashboardTemplate:
    """
    Portfolio Dashboard Template for EUDR Professional Pack.

    Generates comprehensive multi-operator portfolio dashboards with compliance scores,
    aggregated risk summaries, DDS tracking, supplier pool statistics, and cost allocation.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = PortfolioDashboardTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Portfolio Overview" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Portfolio Dashboard Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the portfolio dashboard.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Portfolio Dashboard for {data.portfolio_name} in {format} format"
        )

        if format == "markdown":
            content = self._render_markdown(data)
        elif format == "html":
            content = self._render_html(data)
        elif format == "json":
            content = self._render_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Add provenance hash
        content_hash = self._calculate_hash(content)
        logger.info(f"Report generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render report in Markdown format."""
        sections = []

        # Header
        sections.append(f"# Portfolio Dashboard: {data.portfolio_name}")
        sections.append(f"")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Total Operators:** {data.total_operators}")
        sections.append(f"**Portfolio Compliance Score:** {data.portfolio_compliance_score:.1f}/100")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        sections.append(
            f"This portfolio dashboard provides a comprehensive overview of {data.total_operators} "
            f"operators within the **{data.portfolio_name}**. The portfolio achieved an overall "
            f"compliance score of **{data.portfolio_compliance_score:.1f}/100** for the reporting "
            f"period {data.reporting_period}."
        )
        sections.append(f"")

        # Operator Compliance Scores
        if data.operator_scores:
            sections.append(f"## Operator Compliance Scores")
            sections.append(f"")
            sections.append(
                f"Compliance performance across all {len(data.operator_scores)} operators:"
            )
            sections.append(f"")

            sections.append(
                f"| Operator | Country | Score | Risk | DDS (App/Total) | Suppliers | Volume (t) | Last Assessment |"
            )
            sections.append(
                f"|----------|---------|-------|------|-----------------|-----------|------------|-----------------|"
            )

            for op in sorted(data.operator_scores, key=lambda x: x.compliance_score, reverse=True):
                dds_ratio = f"{op.dds_submissions_approved}/{op.dds_submissions_total}"
                sections.append(
                    f"| {op.operator_name} | {op.country} | {op.compliance_score:.1f} | "
                    f"{op.risk_level} | {dds_ratio} | {op.active_suppliers} | "
                    f"{op.total_volume_tonnes:,.0f} | {op.last_assessment_date} |"
                )
            sections.append(f"")

            # Top performers
            top_performers = sorted(
                data.operator_scores,
                key=lambda x: x.compliance_score,
                reverse=True
            )[:self.config.top_performers_count]

            sections.append(f"### Top {self.config.top_performers_count} Performers")
            sections.append(f"")
            for idx, op in enumerate(top_performers, 1):
                sections.append(
                    f"{idx}. **{op.operator_name}** ({op.country}): {op.compliance_score:.1f}/100"
                )
            sections.append(f"")

            # Risk distribution
            risk_dist = self._calculate_risk_distribution(data.operator_scores)
            sections.append(f"### Risk Distribution")
            sections.append(f"")
            sections.append(f"- **High Risk:** {risk_dist['HIGH']} operators")
            sections.append(f"- **Medium Risk:** {risk_dist['MEDIUM']} operators")
            sections.append(f"- **Low Risk:** {risk_dist['LOW']} operators")
            sections.append(f"")

        # Aggregated Risk Summary
        if data.risk_aggregation:
            sections.append(f"## Aggregated Risk Summary")
            sections.append(f"")
            sections.append(f"Portfolio-wide risk aggregation by category:")
            sections.append(f"")

            sections.append(
                f"| Risk Category | High | Medium | Low | Exposure (t) | Actions Required |"
            )
            sections.append(
                f"|---------------|------|--------|-----|--------------|------------------|"
            )

            for risk in data.risk_aggregation:
                sections.append(
                    f"| {risk.risk_category} | {risk.high_risk_operators} | "
                    f"{risk.medium_risk_operators} | {risk.low_risk_operators} | "
                    f"{risk.total_exposure_tonnes:,.0f} | {risk.mitigation_actions_required} |"
                )
            sections.append(f"")

        # DDS Submission Tracking
        if data.dds_tracking:
            sections.append(f"## DDS Submission Tracking")
            sections.append(f"")
            sections.append(f"Due Diligence Statement submission status across portfolio:")
            sections.append(f"")

            sections.append(
                f"| Period | Total | Approved | Pending | Rejected | Avg Processing (days) |"
            )
            sections.append(
                f"|--------|-------|----------|---------|----------|-----------------------|"
            )

            for dds in data.dds_tracking:
                sections.append(
                    f"| {dds.period} | {dds.total_submissions} | {dds.approved} | "
                    f"{dds.pending} | {dds.rejected} | {dds.avg_processing_days:.1f} |"
                )
            sections.append(f"")

            # Overall stats
            total_dds = sum(d.total_submissions for d in data.dds_tracking)
            total_approved = sum(d.approved for d in data.dds_tracking)
            approval_rate = (total_approved / total_dds * 100) if total_dds > 0 else 0

            sections.append(
                f"**Overall Approval Rate:** {approval_rate:.1f}% ({total_approved:,} of {total_dds:,} submissions)"
            )
            sections.append(f"")

        # Supplier Pool Statistics
        if self.config.include_supplier_pool:
            sections.append(f"## Supplier Pool Statistics")
            sections.append(f"")
            pool = data.supplier_pool

            sections.append(f"| Metric | Value |")
            sections.append(f"|--------|-------|")
            sections.append(f"| Total Suppliers | {pool.total_suppliers:,} |")
            sections.append(f"| Active Suppliers | {pool.active_suppliers:,} |")
            sections.append(f"| Certified Suppliers | {pool.certified_suppliers:,} |")
            sections.append(f"| High-Risk Suppliers | {pool.high_risk_suppliers:,} |")
            sections.append(f"| Shared Suppliers | {pool.shared_suppliers:,} |")
            sections.append(f"| Avg Supplier Score | {pool.avg_supplier_score:.1f}/100 |")
            sections.append(f"")

            if pool.top_countries:
                sections.append(f"### Top Supplier Countries")
                sections.append(f"")
                for idx, country_data in enumerate(pool.top_countries[:10], 1):
                    sections.append(
                        f"{idx}. {country_data['country']}: {country_data['count']:,} suppliers "
                        f"({country_data['percentage']:.1f}%)"
                    )
                sections.append(f"")

        # Cost Allocation
        if self.config.include_cost_allocation and data.cost_allocations:
            sections.append(f"## Cost Allocation")
            sections.append(f"")
            sections.append(f"EUDR compliance costs by operator:")
            sections.append(f"")

            sections.append(
                f"| Operator | Total Cost (EUR) | Cost/Tonne (EUR) | Cost/DDS (EUR) |"
            )
            sections.append(
                f"|----------|------------------|------------------|----------------|"
            )

            for cost in sorted(data.cost_allocations, key=lambda x: x.total_cost_eur, reverse=True):
                sections.append(
                    f"| {cost.operator_name} | €{cost.total_cost_eur:,.2f} | "
                    f"€{cost.cost_per_tonne_eur:.2f} | €{cost.cost_per_dds_eur:.2f} |"
                )
            sections.append(f"")

            # Total portfolio cost
            total_cost = sum(c.total_cost_eur for c in data.cost_allocations)
            sections.append(f"**Total Portfolio Cost:** €{total_cost:,.2f}")
            sections.append(f"")

        # Cross-Operator Benchmarking
        if self.config.include_cross_operator_benchmark and data.operator_scores:
            sections.append(f"## Cross-Operator Benchmarking")
            sections.append(f"")

            avg_score = sum(op.compliance_score for op in data.operator_scores) / len(data.operator_scores)
            sections.append(f"**Portfolio Average Score:** {avg_score:.1f}/100")
            sections.append(f"")

            # Performance tiers
            excellent = [op for op in data.operator_scores if op.compliance_score >= 80]
            good = [op for op in data.operator_scores if 60 <= op.compliance_score < 80]
            needs_improvement = [op for op in data.operator_scores if op.compliance_score < 60]

            sections.append(f"### Performance Tiers")
            sections.append(f"")
            sections.append(f"- **Excellent (≥80):** {len(excellent)} operators ({len(excellent) / len(data.operator_scores) * 100:.1f}%)")
            sections.append(f"- **Good (60-79):** {len(good)} operators ({len(good) / len(data.operator_scores) * 100:.1f}%)")
            sections.append(f"- **Needs Improvement (<60):** {len(needs_improvement)} operators ({len(needs_improvement) / len(data.operator_scores) * 100:.1f}%)")
            sections.append(f"")

        # Key Insights
        if data.key_insights:
            sections.append(f"## Key Insights")
            sections.append(f"")
            for idx, insight in enumerate(data.key_insights, 1):
                sections.append(f"{idx}. {insight}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Dashboard generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Portfolio Dashboard - {data.portfolio_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #2980b9; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Portfolio Dashboard: {data.portfolio_name}</h1>

    <div class="summary">
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Total Operators:</strong> {data.total_operators}</p>
        <p><strong>Portfolio Compliance Score:</strong> <span class="metric">{data.portfolio_compliance_score:.1f}/100</span></p>
    </div>

    <h2>Executive Summary</h2>
    <p>
        This portfolio dashboard provides a comprehensive overview of <strong>{data.total_operators}</strong>
        operators within the {data.portfolio_name}. The portfolio achieved an overall compliance score of
        <span class="metric">{data.portfolio_compliance_score:.1f}/100</span> for the reporting period
        {data.reporting_period}.
    </p>
"""

        if data.operator_scores:
            html += f"""
    <h2>Operator Compliance Scores</h2>
    <table>
        <tr>
            <th>Operator</th><th>Country</th><th>Score</th><th>Risk</th>
            <th>DDS (App/Total)</th><th>Suppliers</th><th>Volume (t)</th><th>Last Assessment</th>
        </tr>
"""
            for op in sorted(data.operator_scores, key=lambda x: x.compliance_score, reverse=True):
                dds_ratio = f"{op.dds_submissions_approved}/{op.dds_submissions_total}"
                html += f"""        <tr>
            <td>{op.operator_name}</td>
            <td>{op.country}</td>
            <td>{op.compliance_score:.1f}</td>
            <td>{op.risk_level}</td>
            <td>{dds_ratio}</td>
            <td>{op.active_suppliers}</td>
            <td>{op.total_volume_tonnes:,.0f}</td>
            <td>{op.last_assessment_date}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.dds_tracking:
            html += f"""
    <h2>DDS Submission Tracking</h2>
    <table>
        <tr><th>Period</th><th>Total</th><th>Approved</th><th>Pending</th><th>Rejected</th><th>Avg Processing (days)</th></tr>
"""
            for dds in data.dds_tracking:
                html += f"""        <tr>
            <td>{dds.period}</td>
            <td>{dds.total_submissions}</td>
            <td>{dds.approved}</td>
            <td>{dds.pending}</td>
            <td>{dds.rejected}</td>
            <td>{dds.avg_processing_days:.1f}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if self.config.include_supplier_pool:
            pool = data.supplier_pool
            html += f"""
    <h2>Supplier Pool Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Suppliers</td><td>{pool.total_suppliers:,}</td></tr>
        <tr><td>Active Suppliers</td><td>{pool.active_suppliers:,}</td></tr>
        <tr><td>Certified Suppliers</td><td>{pool.certified_suppliers:,}</td></tr>
        <tr><td>High-Risk Suppliers</td><td>{pool.high_risk_suppliers:,}</td></tr>
        <tr><td>Shared Suppliers</td><td>{pool.shared_suppliers:,}</td></tr>
        <tr><td>Avg Supplier Score</td><td>{pool.avg_supplier_score:.1f}/100</td></tr>
    </table>
"""

        if self.config.include_cost_allocation and data.cost_allocations:
            html += f"""
    <h2>Cost Allocation</h2>
    <table>
        <tr><th>Operator</th><th>Total Cost (EUR)</th><th>Cost/Tonne (EUR)</th><th>Cost/DDS (EUR)</th></tr>
"""
            for cost in sorted(data.cost_allocations, key=lambda x: x.total_cost_eur, reverse=True):
                html += f"""        <tr>
            <td>{cost.operator_name}</td>
            <td>€{cost.total_cost_eur:,.2f}</td>
            <td>€{cost.cost_per_tonne_eur:.2f}</td>
            <td>€{cost.cost_per_dds_eur:.2f}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.key_insights:
            html += f"""
    <h2>Key Insights</h2>
    <ol>
"""
            for insight in data.key_insights:
                html += f"        <li>{insight}</li>\n"
            html += f"""    </ol>
"""

        html += f"""
    <div class="footer">
        <p><em>Dashboard generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "portfolio_dashboard",
            "portfolio_name": data.portfolio_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "total_operators": data.total_operators,
            "portfolio_compliance_score": data.portfolio_compliance_score,
            "operator_scores": [op.dict() for op in data.operator_scores],
            "dds_tracking": [dds.dict() for dds in data.dds_tracking],
            "supplier_pool": data.supplier_pool.dict(),
            "cost_allocations": [cost.dict() for cost in data.cost_allocations],
            "risk_aggregation": [risk.dict() for risk in data.risk_aggregation],
            "key_insights": data.key_insights,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_risk_distribution(self, operators: List[OperatorScore]) -> Dict[str, int]:
        """Calculate risk distribution across operators."""
        dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for op in operators:
            if op.risk_level in dist:
                dist[op.risk_level] += 1
        return dist
