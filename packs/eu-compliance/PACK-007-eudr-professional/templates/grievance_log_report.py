"""
Grievance Log Report Template - PACK-007 EUDR Professional Pack

This module generates grievance and complaint tracking reports with complaint register,
investigation status, resolution statistics, average resolution time, geographic distribution,
and linked suppliers/plots for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from grievance_log_report import GrievanceLogReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Sustainable Commodities Inc",
    ...     report_date="2026-03-15",
    ...     reporting_period="2025-Q1 to 2026-Q1"
    ... )
    >>> template = GrievanceLogReportTemplate()
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
    """Configuration for Grievance Log Report generation."""

    include_investigation_details: bool = Field(
        default=True,
        description="Include investigation details"
    )
    include_geographic_analysis: bool = Field(
        default=True,
        description="Include geographic distribution analysis"
    )
    include_trend_analysis: bool = Field(
        default=True,
        description="Include trend analysis over time"
    )
    anonymize_complainants: bool = Field(
        default=True,
        description="Anonymize complainant information in reports"
    )


class GrievanceRecord(BaseModel):
    """Grievance/complaint record."""

    grievance_id: str = Field(..., description="Unique grievance identifier")
    submission_date: str = Field(..., description="Date grievance was submitted")
    grievance_type: str = Field(
        ...,
        description="DEFORESTATION, LABOR_RIGHTS, INDIGENOUS_RIGHTS, CORRUPTION, OTHER"
    )
    severity: str = Field(..., description="CRITICAL, HIGH, MEDIUM, LOW")
    complainant_type: str = Field(
        ...,
        description="COMMUNITY, NGO, EMPLOYEE, ANONYMOUS, OTHER"
    )
    description: str = Field(..., description="Brief description of grievance")
    linked_suppliers: List[str] = Field(
        default_factory=list,
        description="Linked supplier IDs"
    )
    linked_plots: List[str] = Field(
        default_factory=list,
        description="Linked plot IDs"
    )
    country: str = Field(..., description="Country where issue occurred")
    region: Optional[str] = Field(None, description="Region/state")
    status: str = Field(
        ...,
        description="RECEIVED, UNDER_INVESTIGATION, RESOLVED, CLOSED, ESCALATED"
    )


class InvestigationStatus(BaseModel):
    """Investigation status details."""

    grievance_id: str = Field(..., description="Grievance identifier")
    investigation_started: Optional[str] = Field(None, description="Investigation start date")
    investigator: Optional[str] = Field(None, description="Lead investigator")
    findings_summary: Optional[str] = Field(None, description="Summary of findings")
    evidence_collected: List[str] = Field(
        default_factory=list,
        description="Types of evidence collected"
    )
    investigation_status: str = Field(
        ...,
        description="NOT_STARTED, IN_PROGRESS, COMPLETED, SUSPENDED"
    )


class ResolutionRecord(BaseModel):
    """Grievance resolution record."""

    grievance_id: str = Field(..., description="Grievance identifier")
    resolution_date: str = Field(..., description="Date resolved")
    resolution_type: str = Field(
        ...,
        description="REMEDIATED, UNFOUNDED, WITHDRAWN, ESCALATED"
    )
    resolution_description: str = Field(..., description="Description of resolution")
    days_to_resolve: int = Field(..., ge=0, description="Days from submission to resolution")
    complainant_satisfied: Optional[bool] = Field(
        None,
        description="Complainant satisfaction (if available)"
    )


class GeographicDistribution(BaseModel):
    """Geographic distribution of grievances."""

    country: str = Field(..., description="Country")
    region: Optional[str] = Field(None, description="Region/state")
    total_grievances: int = Field(..., ge=0, description="Total grievances")
    critical_grievances: int = Field(..., ge=0, description="Critical grievances")
    resolved_grievances: int = Field(..., ge=0, description="Resolved grievances")
    avg_resolution_days: Optional[float] = Field(None, description="Avg days to resolution")


class TrendDataPoint(BaseModel):
    """Trend data point."""

    period: str = Field(..., description="Time period (e.g., 2025-Q1)")
    total_grievances: int = Field(..., ge=0, description="Total grievances")
    resolved: int = Field(..., ge=0, description="Resolved grievances")
    avg_resolution_days: float = Field(..., ge=0, description="Avg resolution time")


class OverallStatistics(BaseModel):
    """Overall grievance statistics."""

    total_grievances: int = Field(..., ge=0, description="Total grievances recorded")
    open_grievances: int = Field(..., ge=0, description="Currently open grievances")
    resolved_grievances: int = Field(..., ge=0, description="Resolved grievances")
    escalated_grievances: int = Field(..., ge=0, description="Escalated grievances")
    avg_resolution_time_days: float = Field(..., ge=0, description="Avg resolution time")
    resolution_rate: float = Field(..., ge=0, le=100, description="Resolution rate %")
    critical_grievances: int = Field(..., ge=0, description="Critical severity grievances")


class ReportData(BaseModel):
    """Data model for Grievance Log Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., 2025-Q1 to 2026-Q1)")
    statistics: OverallStatistics = Field(..., description="Overall statistics")
    grievance_records: List[GrievanceRecord] = Field(
        default_factory=list,
        description="Grievance records"
    )
    investigation_statuses: List[InvestigationStatus] = Field(
        default_factory=list,
        description="Investigation status details"
    )
    resolution_records: List[ResolutionRecord] = Field(
        default_factory=list,
        description="Resolution records"
    )
    geographic_distribution: List[GeographicDistribution] = Field(
        default_factory=list,
        description="Geographic distribution"
    )
    trend_data: List[TrendDataPoint] = Field(
        default_factory=list,
        description="Trend data over time"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from grievance analysis"
    )
    improvement_actions: List[str] = Field(
        default_factory=list,
        description="Actions to improve grievance management"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class GrievanceLogReportTemplate:
    """
    Grievance Log Report Template for EUDR Professional Pack.

    Generates comprehensive grievance and complaint tracking reports with complaint register,
    investigation status, resolution statistics, geographic distribution, and trend analysis.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = GrievanceLogReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Grievance Log Report" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Grievance Log Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the grievance log report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Grievance Log Report for {data.operator_name} in {format} format"
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
        sections.append(f"# Grievance Log Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        stats = data.statistics
        sections.append(
            f"This grievance log report covers {data.reporting_period} for **{data.operator_name}**. "
            f"During this period, **{stats.total_grievances}** grievances were recorded, of which "
            f"**{stats.resolved_grievances}** have been resolved ({stats.resolution_rate:.1f}% resolution rate). "
            f"The average resolution time is **{stats.avg_resolution_time_days:.1f} days**. "
            f"Currently, **{stats.open_grievances}** grievances remain open."
        )
        sections.append(f"")

        if stats.critical_grievances > 0:
            sections.append(
                f"⚠️ **Critical Alert:** {stats.critical_grievances} critical severity grievances "
                f"require immediate attention."
            )
            sections.append(f"")

        # Overall Statistics
        sections.append(f"## Overall Statistics")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Total Grievances | {stats.total_grievances:,} |")
        sections.append(f"| Open Grievances | {stats.open_grievances:,} |")
        sections.append(f"| Resolved Grievances | {stats.resolved_grievances:,} |")
        sections.append(f"| Escalated Grievances | {stats.escalated_grievances:,} |")
        sections.append(f"| Critical Grievances | {stats.critical_grievances:,} |")
        sections.append(f"| Resolution Rate | {stats.resolution_rate:.1f}% |")
        sections.append(f"| Avg Resolution Time | {stats.avg_resolution_time_days:.1f} days |")
        sections.append(f"")

        # Grievance Register
        if data.grievance_records:
            sections.append(f"## Grievance Register")
            sections.append(f"")
            sections.append(
                f"Complete register of all grievances received during the reporting period:"
            )
            sections.append(f"")

            sections.append(
                f"| ID | Date | Type | Severity | Complainant | Country | Status | Linked Suppliers |"
            )
            sections.append(
                f"|----|------|------|----------|-------------|---------|--------|------------------|"
            )

            for record in sorted(data.grievance_records, key=lambda x: x.submission_date, reverse=True):
                complainant = "ANONYMIZED" if self.config.anonymize_complainants else record.complainant_type
                suppliers = ", ".join(record.linked_suppliers[:2]) if record.linked_suppliers else "None"
                if len(record.linked_suppliers) > 2:
                    suppliers += f" +{len(record.linked_suppliers) - 2}"

                sections.append(
                    f"| {record.grievance_id} | {record.submission_date} | {record.grievance_type} | "
                    f"{record.severity} | {complainant} | {record.country} | {record.status} | {suppliers} |"
                )
            sections.append(f"")

            # Critical and high severity grievances
            critical_high = [r for r in data.grievance_records if r.severity in ["CRITICAL", "HIGH"]]
            if critical_high:
                sections.append(f"### Critical & High Severity Grievances")
                sections.append(f"")
                sections.append(
                    f"**{len(critical_high)}** grievances require priority attention:"
                )
                sections.append(f"")
                for record in critical_high[:10]:
                    sections.append(f"**{record.grievance_id}** - {record.grievance_type} ({record.severity})")
                    sections.append(f"- **Submitted:** {record.submission_date}")
                    sections.append(f"- **Location:** {record.country}, {record.region or 'N/A'}")
                    sections.append(f"- **Status:** {record.status}")
                    sections.append(f"- **Description:** {record.description}")
                    sections.append(f"")

        # Investigation Status
        if self.config.include_investigation_details and data.investigation_statuses:
            sections.append(f"## Investigation Status")
            sections.append(f"")
            sections.append(
                f"Current status of ongoing investigations:"
            )
            sections.append(f"")

            sections.append(
                f"| Grievance ID | Started | Investigator | Status | Evidence Collected |"
            )
            sections.append(
                f"|--------------|---------|--------------|--------|-------------------|"
            )

            for inv in data.investigation_statuses:
                started = inv.investigation_started or "Not started"
                investigator = inv.investigator or "Unassigned"
                evidence = ", ".join(inv.evidence_collected[:2]) if inv.evidence_collected else "None"

                sections.append(
                    f"| {inv.grievance_id} | {started} | {investigator} | "
                    f"{inv.investigation_status} | {evidence} |"
                )
            sections.append(f"")

        # Resolution Statistics
        if data.resolution_records:
            sections.append(f"## Resolution Statistics")
            sections.append(f"")
            sections.append(
                f"Details of resolved grievances:"
            )
            sections.append(f"")

            sections.append(
                f"| Grievance ID | Resolution Date | Type | Days to Resolve | Satisfied |"
            )
            sections.append(
                f"|--------------|-----------------|------|-----------------|-----------|"
            )

            for res in sorted(data.resolution_records, key=lambda x: x.resolution_date, reverse=True)[:20]:
                satisfied = "Yes" if res.complainant_satisfied else "No" if res.complainant_satisfied is False else "Unknown"
                sections.append(
                    f"| {res.grievance_id} | {res.resolution_date} | {res.resolution_type} | "
                    f"{res.days_to_resolve} | {satisfied} |"
                )
            sections.append(f"")

            # Resolution metrics
            avg_days = sum(r.days_to_resolve for r in data.resolution_records) / len(data.resolution_records)
            min_days = min(r.days_to_resolve for r in data.resolution_records)
            max_days = max(r.days_to_resolve for r in data.resolution_records)

            sections.append(f"### Resolution Metrics")
            sections.append(f"")
            sections.append(f"- **Average Time to Resolution:** {avg_days:.1f} days")
            sections.append(f"- **Fastest Resolution:** {min_days} days")
            sections.append(f"- **Longest Resolution:** {max_days} days")
            sections.append(f"")

            # Resolution types
            by_type = {}
            for res in data.resolution_records:
                by_type[res.resolution_type] = by_type.get(res.resolution_type, 0) + 1

            sections.append(f"### Resolution Types")
            sections.append(f"")
            for res_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                sections.append(f"- **{res_type}:** {count} ({count / len(data.resolution_records) * 100:.1f}%)")
            sections.append(f"")

        # Geographic Distribution
        if self.config.include_geographic_analysis and data.geographic_distribution:
            sections.append(f"## Geographic Distribution")
            sections.append(f"")
            sections.append(
                f"Grievances by country and region:"
            )
            sections.append(f"")

            sections.append(
                f"| Country | Region | Total | Critical | Resolved | Avg Days |"
            )
            sections.append(
                f"|---------|--------|-------|----------|----------|----------|"
            )

            for geo in sorted(data.geographic_distribution, key=lambda x: x.total_grievances, reverse=True):
                region = geo.region or "All regions"
                avg_days = f"{geo.avg_resolution_days:.1f}" if geo.avg_resolution_days else "N/A"
                sections.append(
                    f"| {geo.country} | {region} | {geo.total_grievances} | "
                    f"{geo.critical_grievances} | {geo.resolved_grievances} | {avg_days} |"
                )
            sections.append(f"")

            # Hotspots
            hotspots = [g for g in data.geographic_distribution if g.total_grievances >= 5 or g.critical_grievances >= 2]
            if hotspots:
                sections.append(f"### Geographic Hotspots")
                sections.append(f"")
                sections.append(
                    f"**{len(hotspots)}** locations identified as hotspots (≥5 grievances or ≥2 critical):"
                )
                sections.append(f"")
                for geo in hotspots:
                    sections.append(
                        f"- **{geo.country}** ({geo.region or 'All regions'}): {geo.total_grievances} total, "
                        f"{geo.critical_grievances} critical"
                    )
                sections.append(f"")

        # Trend Analysis
        if self.config.include_trend_analysis and data.trend_data:
            sections.append(f"## Trend Analysis")
            sections.append(f"")
            sections.append(
                f"Grievance trends over time:"
            )
            sections.append(f"")

            sections.append(
                f"| Period | Total Grievances | Resolved | Avg Resolution Days |"
            )
            sections.append(
                f"|--------|------------------|----------|---------------------|"
            )

            for trend in data.trend_data:
                sections.append(
                    f"| {trend.period} | {trend.total_grievances} | {trend.resolved} | "
                    f"{trend.avg_resolution_days:.1f} |"
                )
            sections.append(f"")

            # Trend direction
            if len(data.trend_data) >= 2:
                latest = data.trend_data[-1]
                previous = data.trend_data[-2]
                change = latest.total_grievances - previous.total_grievances
                direction = "increased" if change > 0 else "decreased" if change < 0 else "remained stable"

                sections.append(
                    f"**Trend Direction:** Grievances {direction} by {abs(change)} "
                    f"from {previous.period} to {latest.period}."
                )
                sections.append(f"")

        # Key Findings
        if data.key_findings:
            sections.append(f"## Key Findings")
            sections.append(f"")
            for idx, finding in enumerate(data.key_findings, 1):
                sections.append(f"{idx}. {finding}")
            sections.append(f"")

        # Improvement Actions
        if data.improvement_actions:
            sections.append(f"## Recommended Improvement Actions")
            sections.append(f"")
            for idx, action in enumerate(data.improvement_actions, 1):
                sections.append(f"{idx}. {action}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )
        sections.append(f"")
        if self.config.anonymize_complainants:
            sections.append(
                f"**Note:** Complainant information has been anonymized to protect privacy."
            )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        stats = data.statistics

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Grievance Log Report - {data.operator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #e74c3c; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #c0392b; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .critical {{ background-color: #ffebee; color: #c0392b; font-weight: bold; padding: 10px; margin: 10px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Grievance Log Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
    </div>
"""

        if stats.critical_grievances > 0:
            html += f"""
    <div class="critical">
        <strong>Critical Alert:</strong> {stats.critical_grievances} critical severity grievances require immediate attention.
    </div>
"""

        html += f"""
    <h2>Overall Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Grievances</td><td>{stats.total_grievances:,}</td></tr>
        <tr><td>Open Grievances</td><td>{stats.open_grievances:,}</td></tr>
        <tr><td>Resolved Grievances</td><td>{stats.resolved_grievances:,}</td></tr>
        <tr><td>Resolution Rate</td><td>{stats.resolution_rate:.1f}%</td></tr>
        <tr><td>Avg Resolution Time</td><td>{stats.avg_resolution_time_days:.1f} days</td></tr>
    </table>
"""

        if data.grievance_records:
            html += f"""
    <h2>Grievance Register</h2>
    <table>
        <tr><th>ID</th><th>Date</th><th>Type</th><th>Severity</th><th>Country</th><th>Status</th></tr>
"""
            for record in sorted(data.grievance_records, key=lambda x: x.submission_date, reverse=True)[:30]:
                row_class = ' class="critical"' if record.severity == "CRITICAL" else ''
                html += f"""        <tr{row_class}>
            <td>{record.grievance_id}</td>
            <td>{record.submission_date}</td>
            <td>{record.grievance_type}</td>
            <td>{record.severity}</td>
            <td>{record.country}</td>
            <td>{record.status}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.geographic_distribution:
            html += f"""
    <h2>Geographic Distribution</h2>
    <table>
        <tr><th>Country</th><th>Region</th><th>Total</th><th>Critical</th><th>Resolved</th></tr>
"""
            for geo in sorted(data.geographic_distribution, key=lambda x: x.total_grievances, reverse=True)[:20]:
                html += f"""        <tr>
            <td>{geo.country}</td>
            <td>{geo.region or 'All regions'}</td>
            <td>{geo.total_grievances}</td>
            <td>{geo.critical_grievances}</td>
            <td>{geo.resolved_grievances}</td>
        </tr>
"""
            html += f"""    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
"""
        if self.config.anonymize_complainants:
            html += f"""        <p><strong>Note:</strong> Complainant information has been anonymized to protect privacy.</p>
"""
        html += f"""    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "grievance_log",
            "operator_name": data.operator_name,
            "reporting_period": data.reporting_period,
            "report_date": data.report_date,
            "statistics": data.statistics.dict(),
            "grievance_records": [record.dict() for record in data.grievance_records],
            "investigation_statuses": [inv.dict() for inv in data.investigation_statuses],
            "resolution_records": [res.dict() for res in data.resolution_records],
            "geographic_distribution": [geo.dict() for geo in data.geographic_distribution],
            "trend_data": [trend.dict() for trend in data.trend_data],
            "key_findings": data.key_findings,
            "improvement_actions": data.improvement_actions,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
                "anonymize_complainants": self.config.anonymize_complainants,
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
