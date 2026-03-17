"""
Annual Compliance Report Template - PACK-007 EUDR Professional Pack

This module generates annual compliance summary reports with compliance trajectory,
risk evolution, supplier trends, DDS statistics, audit findings, regulatory changes applied,
and next-year priorities for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from annual_compliance_report import AnnualComplianceReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="International Timber Ltd",
    ...     report_date="2026-03-15",
    ...     reporting_year=2025
    ... )
    >>> template = AnnualComplianceReportTemplate()
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
    """Configuration for Annual Compliance Report generation."""

    include_quarterly_breakdown: bool = Field(
        default=True,
        description="Include quarterly performance breakdown"
    )
    include_supplier_trends: bool = Field(
        default=True,
        description="Include supplier performance trends"
    )
    include_audit_findings: bool = Field(
        default=True,
        description="Include audit findings summary"
    )
    include_next_year_plan: bool = Field(
        default=True,
        description="Include next year priorities and plan"
    )


class QuarterlyPerformance(BaseModel):
    """Quarterly performance summary."""

    quarter: str = Field(..., description="Quarter (e.g., 2025-Q1)")
    compliance_score: float = Field(..., ge=0, le=100, description="Quarterly compliance score")
    dds_submitted: int = Field(..., ge=0, description="DDS submissions")
    dds_approved: int = Field(..., ge=0, description="DDS approvals")
    risk_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    key_achievement: Optional[str] = Field(None, description="Key achievement this quarter")


class RiskEvolutionPoint(BaseModel):
    """Risk evolution data point."""

    period: str = Field(..., description="Time period")
    overall_risk_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    deforestation_risk: float = Field(..., ge=0, le=100, description="Deforestation risk")
    traceability_risk: float = Field(..., ge=0, le=100, description="Traceability risk")
    documentation_risk: float = Field(..., ge=0, le=100, description="Documentation risk")


class SupplierTrend(BaseModel):
    """Supplier trend analysis."""

    metric: str = Field(..., description="Metric name")
    year_start_value: float = Field(..., description="Value at year start")
    year_end_value: float = Field(..., description="Value at year end")
    change_absolute: float = Field(..., description="Absolute change")
    change_percentage: float = Field(..., description="Percentage change")
    trend_direction: str = Field(..., description="IMPROVING, DECLINING, STABLE")


class DDSStatistics(BaseModel):
    """DDS submission statistics."""

    total_dds_submitted: int = Field(..., ge=0, description="Total DDS submitted")
    total_dds_approved: int = Field(..., ge=0, description="Total DDS approved")
    total_dds_rejected: int = Field(..., ge=0, description="Total DDS rejected")
    avg_processing_time_days: float = Field(..., ge=0, description="Average processing time")
    approval_rate: float = Field(..., ge=0, le=100, description="Approval rate percentage")
    total_volume_tonnes: float = Field(..., ge=0, description="Total volume covered (tonnes)")


class AuditFinding(BaseModel):
    """Audit finding summary."""

    audit_date: str = Field(..., description="Audit date")
    audit_type: str = Field(..., description="INTERNAL, CA_INSPECTION, THIRD_PARTY")
    finding_type: str = Field(..., description="CRITICAL, MAJOR, MINOR, OBSERVATION")
    finding_description: str = Field(..., description="Finding description")
    resolution_status: str = Field(..., description="OPEN, IN_PROGRESS, RESOLVED")
    resolution_date: Optional[str] = Field(None, description="Resolution date if resolved")


class RegulatoryChangeApplied(BaseModel):
    """Regulatory change applied during the year."""

    change_id: str = Field(..., description="Change identifier")
    change_description: str = Field(..., description="Description of change")
    implementation_date: str = Field(..., description="Date implemented")
    impact_on_operations: str = Field(..., description="Impact assessment")


class NextYearPriority(BaseModel):
    """Next year priority."""

    priority_id: str = Field(..., description="Priority identifier")
    priority_area: str = Field(..., description="Area (e.g., Traceability, Risk Assessment)")
    objective: str = Field(..., description="Objective description")
    target_metrics: List[str] = Field(
        default_factory=list,
        description="Target metrics/KPIs"
    )
    estimated_investment: Optional[float] = Field(
        None,
        description="Estimated investment (EUR)"
    )


class ReportData(BaseModel):
    """Data model for Annual Compliance Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_year: int = Field(..., description="Year being reported (e.g., 2025)")
    overall_compliance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall annual compliance score"
    )
    quarterly_performance: List[QuarterlyPerformance] = Field(
        default_factory=list,
        description="Quarterly performance data"
    )
    risk_evolution: List[RiskEvolutionPoint] = Field(
        default_factory=list,
        description="Risk evolution over the year"
    )
    supplier_trends: List[SupplierTrend] = Field(
        default_factory=list,
        description="Supplier performance trends"
    )
    dds_statistics: DDSStatistics = Field(..., description="DDS submission statistics")
    audit_findings: List[AuditFinding] = Field(
        default_factory=list,
        description="Audit findings during the year"
    )
    regulatory_changes_applied: List[RegulatoryChangeApplied] = Field(
        default_factory=list,
        description="Regulatory changes implemented"
    )
    next_year_priorities: List[NextYearPriority] = Field(
        default_factory=list,
        description="Priorities for next year"
    )
    key_achievements: List[str] = Field(
        default_factory=list,
        description="Key achievements during the year"
    )
    challenges_faced: List[str] = Field(
        default_factory=list,
        description="Key challenges encountered"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class AnnualComplianceReportTemplate:
    """
    Annual Compliance Report Template for EUDR Professional Pack.

    Generates comprehensive annual compliance summary reports with performance trajectory,
    risk evolution, supplier trends, DDS statistics, audit findings, and next-year priorities.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = AnnualComplianceReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Annual Compliance Report" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Annual Compliance Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the annual compliance report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Annual Compliance Report for {data.operator_name} in {format} format"
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
        sections.append(f"# Annual Compliance Report {data.reporting_year}")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Reporting Year:** {data.reporting_year}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Overall Compliance Score:** {data.overall_compliance_score:.1f}/100")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        sections.append(
            f"This annual compliance report summarizes **{data.operator_name}** EUDR compliance "
            f"performance for the year {data.reporting_year}. The operator achieved an overall "
            f"compliance score of **{data.overall_compliance_score:.1f}/100**, with "
            f"**{data.dds_statistics.total_dds_submitted}** Due Diligence Statements submitted "
            f"and an approval rate of **{data.dds_statistics.approval_rate:.1f}%**."
        )
        sections.append(f"")

        # Year in Review
        sections.append(f"## Year in Review")
        sections.append(f"")

        # Compliance Trajectory
        if self.config.include_quarterly_breakdown and data.quarterly_performance:
            sections.append(f"### Quarterly Performance")
            sections.append(f"")

            sections.append(
                f"| Quarter | Compliance Score | DDS Submitted | DDS Approved | Risk Level | Key Achievement |"
            )
            sections.append(
                f"|---------|------------------|---------------|--------------|------------|-----------------|"
            )

            for qtr in data.quarterly_performance:
                achievement = qtr.key_achievement or "N/A"
                sections.append(
                    f"| {qtr.quarter} | {qtr.compliance_score:.1f} | {qtr.dds_submitted} | "
                    f"{qtr.dds_approved} | {qtr.risk_level} | {achievement[:30]}... |"
                )
            sections.append(f"")

            # Trajectory analysis
            if len(data.quarterly_performance) >= 2:
                first_qtr = data.quarterly_performance[0]
                last_qtr = data.quarterly_performance[-1]
                score_change = last_qtr.compliance_score - first_qtr.compliance_score
                trend = "improved" if score_change > 0 else "declined" if score_change < 0 else "remained stable"

                sections.append(
                    f"**Annual Trajectory:** Compliance score {trend} by {abs(score_change):.1f} points "
                    f"from {first_qtr.quarter} to {last_qtr.quarter}."
                )
                sections.append(f"")

        # Risk Evolution
        if data.risk_evolution:
            sections.append(f"### Risk Evolution")
            sections.append(f"")
            sections.append(
                f"Risk profile evolution throughout {data.reporting_year}:"
            )
            sections.append(f"")

            sections.append(
                f"| Period | Overall Risk | Deforestation | Traceability | Documentation |"
            )
            sections.append(
                f"|--------|--------------|---------------|--------------|---------------|"
            )

            for point in data.risk_evolution:
                sections.append(
                    f"| {point.period} | {point.overall_risk_score:.1f} | "
                    f"{point.deforestation_risk:.1f} | {point.traceability_risk:.1f} | "
                    f"{point.documentation_risk:.1f} |"
                )
            sections.append(f"")

        # Supplier Trends
        if self.config.include_supplier_trends and data.supplier_trends:
            sections.append(f"### Supplier Performance Trends")
            sections.append(f"")
            sections.append(
                f"Year-over-year supplier performance metrics:"
            )
            sections.append(f"")

            sections.append(
                f"| Metric | Year Start | Year End | Change | % Change | Trend |"
            )
            sections.append(
                f"|--------|------------|----------|--------|----------|-------|"
            )

            for trend in data.supplier_trends:
                change_sign = "+" if trend.change_absolute >= 0 else ""
                sections.append(
                    f"| {trend.metric} | {trend.year_start_value:.1f} | {trend.year_end_value:.1f} | "
                    f"{change_sign}{trend.change_absolute:.1f} | {trend.change_percentage:+.1f}% | "
                    f"{trend.trend_direction} |"
                )
            sections.append(f"")

        # DDS Statistics
        sections.append(f"## Due Diligence Statement Statistics")
        sections.append(f"")
        dds = data.dds_statistics

        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Total DDS Submitted | {dds.total_dds_submitted:,} |")
        sections.append(f"| Total DDS Approved | {dds.total_dds_approved:,} |")
        sections.append(f"| Total DDS Rejected | {dds.total_dds_rejected:,} |")
        sections.append(f"| Approval Rate | {dds.approval_rate:.1f}% |")
        sections.append(f"| Avg Processing Time | {dds.avg_processing_time_days:.1f} days |")
        sections.append(f"| Total Volume Covered | {dds.total_volume_tonnes:,.0f} tonnes |")
        sections.append(f"")

        # Audit Findings
        if self.config.include_audit_findings and data.audit_findings:
            sections.append(f"## Audit Findings Summary")
            sections.append(f"")
            sections.append(
                f"Audit findings from internal, CA, and third-party audits:"
            )
            sections.append(f"")

            sections.append(
                f"| Audit Date | Type | Finding Type | Description | Status | Resolution Date |"
            )
            sections.append(
                f"|------------|------|--------------|-------------|--------|-----------------|"
            )

            for finding in sorted(data.audit_findings, key=lambda x: x.audit_date, reverse=True):
                resolution = finding.resolution_date or "Pending"
                sections.append(
                    f"| {finding.audit_date} | {finding.audit_type} | {finding.finding_type} | "
                    f"{finding.finding_description[:40]}... | {finding.resolution_status} | {resolution} |"
                )
            sections.append(f"")

            # Findings summary
            critical = len([f for f in data.audit_findings if f.finding_type == "CRITICAL"])
            major = len([f for f in data.audit_findings if f.finding_type == "MAJOR"])
            minor = len([f for f in data.audit_findings if f.finding_type == "MINOR"])
            resolved = len([f for f in data.audit_findings if f.resolution_status == "RESOLVED"])

            sections.append(f"### Findings Summary")
            sections.append(f"")
            sections.append(f"- **Critical Findings:** {critical}")
            sections.append(f"- **Major Findings:** {major}")
            sections.append(f"- **Minor Findings:** {minor}")
            sections.append(f"- **Resolved Findings:** {resolved} of {len(data.audit_findings)} ({resolved / len(data.audit_findings) * 100:.1f}%)")
            sections.append(f"")

        # Regulatory Changes Applied
        if data.regulatory_changes_applied:
            sections.append(f"## Regulatory Changes Implemented")
            sections.append(f"")
            sections.append(
                f"Regulatory changes successfully implemented during {data.reporting_year}:"
            )
            sections.append(f"")

            for change in data.regulatory_changes_applied:
                sections.append(f"**{change.change_id}** - Implemented {change.implementation_date}")
                sections.append(f"- **Change:** {change.change_description}")
                sections.append(f"- **Impact:** {change.impact_on_operations}")
                sections.append(f"")

        # Key Achievements
        if data.key_achievements:
            sections.append(f"## Key Achievements")
            sections.append(f"")
            for idx, achievement in enumerate(data.key_achievements, 1):
                sections.append(f"{idx}. {achievement}")
            sections.append(f"")

        # Challenges Faced
        if data.challenges_faced:
            sections.append(f"## Challenges Faced")
            sections.append(f"")
            for idx, challenge in enumerate(data.challenges_faced, 1):
                sections.append(f"{idx}. {challenge}")
            sections.append(f"")

        # Next Year Priorities
        if self.config.include_next_year_plan and data.next_year_priorities:
            sections.append(f"## Priorities for {data.reporting_year + 1}")
            sections.append(f"")
            sections.append(
                f"Strategic priorities and planned initiatives for the coming year:"
            )
            sections.append(f"")

            for priority in data.next_year_priorities:
                sections.append(f"### {priority.priority_id}: {priority.priority_area}")
                sections.append(f"")
                sections.append(f"**Objective:** {priority.objective}")
                sections.append(f"")

                if priority.target_metrics:
                    sections.append(f"**Target Metrics:**")
                    sections.append(f"")
                    for metric in priority.target_metrics:
                        sections.append(f"- {metric}")
                    sections.append(f"")

                if priority.estimated_investment:
                    sections.append(f"**Estimated Investment:** €{priority.estimated_investment:,.0f}")
                    sections.append(f"")

        # Conclusion
        sections.append(f"## Conclusion")
        sections.append(f"")
        sections.append(
            f"The year {data.reporting_year} demonstrated **{data.operator_name}** commitment to "
            f"EUDR compliance with an overall score of {data.overall_compliance_score:.1f}/100. "
        )

        if data.overall_compliance_score >= 80:
            sections.append(
                f"The operator achieved strong compliance performance, positioning well for "
                f"continued regulatory adherence in {data.reporting_year + 1}."
            )
        elif data.overall_compliance_score >= 60:
            sections.append(
                f"The operator achieved satisfactory compliance, with identified areas for "
                f"improvement in {data.reporting_year + 1}."
            )
        else:
            sections.append(
                f"The operator faces significant compliance challenges requiring urgent attention "
                f"and remediation in {data.reporting_year + 1}."
            )
        sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Annual Compliance Report {data.reporting_year} - {data.operator_name}</title>
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
    <h1>Annual Compliance Report {data.reporting_year}</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Reporting Year:</strong> {data.reporting_year}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Overall Compliance Score:</strong> <span class="metric">{data.overall_compliance_score:.1f}/100</span></p>
    </div>

    <h2>Executive Summary</h2>
    <p>
        This annual compliance report summarizes <strong>{data.operator_name}</strong> EUDR compliance
        performance for the year {data.reporting_year}. The operator achieved an overall compliance
        score of <span class="metric">{data.overall_compliance_score:.1f}/100</span>.
    </p>
"""

        if data.quarterly_performance:
            html += f"""
    <h2>Quarterly Performance</h2>
    <table>
        <tr><th>Quarter</th><th>Compliance Score</th><th>DDS Submitted</th><th>DDS Approved</th><th>Risk Level</th></tr>
"""
            for qtr in data.quarterly_performance:
                html += f"""        <tr>
            <td>{qtr.quarter}</td>
            <td>{qtr.compliance_score:.1f}</td>
            <td>{qtr.dds_submitted}</td>
            <td>{qtr.dds_approved}</td>
            <td>{qtr.risk_level}</td>
        </tr>
"""
            html += f"""    </table>
"""

        dds = data.dds_statistics
        html += f"""
    <h2>Due Diligence Statement Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total DDS Submitted</td><td>{dds.total_dds_submitted:,}</td></tr>
        <tr><td>Total DDS Approved</td><td>{dds.total_dds_approved:,}</td></tr>
        <tr><td>Approval Rate</td><td>{dds.approval_rate:.1f}%</td></tr>
        <tr><td>Avg Processing Time</td><td>{dds.avg_processing_time_days:.1f} days</td></tr>
        <tr><td>Total Volume Covered</td><td>{dds.total_volume_tonnes:,.0f} tonnes</td></tr>
    </table>
"""

        if data.key_achievements:
            html += f"""
    <h2>Key Achievements</h2>
    <ol>
"""
            for achievement in data.key_achievements:
                html += f"        <li>{achievement}</li>\n"
            html += f"""    </ol>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "annual_compliance",
            "operator_name": data.operator_name,
            "reporting_year": data.reporting_year,
            "report_date": data.report_date,
            "overall_compliance_score": data.overall_compliance_score,
            "quarterly_performance": [qtr.dict() for qtr in data.quarterly_performance],
            "risk_evolution": [point.dict() for point in data.risk_evolution],
            "supplier_trends": [trend.dict() for trend in data.supplier_trends],
            "dds_statistics": data.dds_statistics.dict(),
            "audit_findings": [finding.dict() for finding in data.audit_findings],
            "regulatory_changes_applied": [change.dict() for change in data.regulatory_changes_applied],
            "next_year_priorities": [priority.dict() for priority in data.next_year_priorities],
            "key_achievements": data.key_achievements,
            "challenges_faced": data.challenges_faced,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
