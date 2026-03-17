"""
Gap Analysis Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates gap analysis reports with severity-classified gap inventories,
remediation roadmaps, cost estimations, and priority matrices for EU Taxonomy alignment
assessment.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from gap_analysis_report import GapAnalysisReportTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = GapAnalysisReportTemplate()
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
    """Configuration for Gap Analysis Report generation."""

    include_cost_estimates: bool = Field(
        default=True,
        description="Include cost estimation per gap"
    )
    include_remediation_timeline: bool = Field(
        default=True,
        description="Include remediation timeline"
    )
    include_priority_matrix: bool = Field(
        default=True,
        description="Include priority matrix (impact vs effort)"
    )
    max_gaps_detail: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of gaps to show in detail"
    )
    currency: str = Field(default="EUR", description="Reporting currency")


class GapItem(BaseModel):
    """A single identified gap in taxonomy alignment."""

    gap_id: str = Field(..., description="Unique gap identifier")
    activity_name: str = Field(..., description="Affected activity name")
    activity_id: str = Field(default="", description="Taxonomy activity identifier")
    gap_category: str = Field(
        ...,
        description="Gap category: TSC, DNSH, MS, or DATA"
    )
    objective: str = Field(
        default="",
        description="Environmental objective if applicable (CCM, CCA, etc.)"
    )
    severity: str = Field(
        ...,
        description="Severity: CRITICAL, HIGH, MEDIUM, or LOW"
    )
    description: str = Field(..., description="Gap description")
    current_state: str = Field(default="", description="Current state assessment")
    required_state: str = Field(default="", description="Required state for compliance")
    remediation_action: str = Field(default="", description="Recommended remediation")
    estimated_cost_eur: float = Field(
        default=0.0, ge=0,
        description="Estimated remediation cost in EUR"
    )
    estimated_duration_weeks: int = Field(
        default=0, ge=0,
        description="Estimated remediation duration in weeks"
    )
    responsible_party: str = Field(default="", description="Responsible party or department")
    target_date: str = Field(default="", description="Target completion date")
    impact_on_turnover_eur: float = Field(
        default=0.0, ge=0,
        description="Turnover at risk due to this gap"
    )


class RemediationPhase(BaseModel):
    """A phase in the remediation roadmap."""

    phase_name: str = Field(..., description="Phase name")
    phase_number: int = Field(..., ge=1, description="Phase sequence number")
    start_date: str = Field(default="", description="Phase start date")
    end_date: str = Field(default="", description="Phase end date")
    gap_ids: List[str] = Field(
        default_factory=list,
        description="Gap IDs addressed in this phase"
    )
    total_cost_eur: float = Field(default=0.0, ge=0, description="Phase cost")
    description: str = Field(default="", description="Phase description")


class ReportData(BaseModel):
    """Data model for Gap Analysis Report."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    total_activities_assessed: int = Field(
        default=0, ge=0,
        description="Total activities assessed"
    )
    gaps: List[GapItem] = Field(
        default_factory=list,
        description="Identified gaps"
    )
    remediation_phases: List[RemediationPhase] = Field(
        default_factory=list,
        description="Remediation roadmap phases"
    )
    total_estimated_cost_eur: float = Field(
        default=0.0, ge=0,
        description="Total estimated remediation cost"
    )
    total_turnover_at_risk_eur: float = Field(
        default=0.0, ge=0,
        description="Total turnover at risk from identified gaps"
    )
    key_recommendations: List[str] = Field(
        default_factory=list,
        description="Key recommendations"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class GapAnalysisReportTemplate:
    """
    Gap Analysis Report Template for EU Taxonomy Alignment Pack.

    Generates gap inventory with severity classification, remediation roadmaps,
    cost estimations, and priority matrices for taxonomy alignment gaps.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = GapAnalysisReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Gap Summary" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Gap Analysis Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the gap analysis report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Gap Analysis Report for {data.organization_name} "
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
        logger.info(f"Report generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render report in Markdown format."""
        sections = []

        # Header
        sections.append(f"# EU Taxonomy Gap Analysis Report")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Activities Assessed:** {data.total_activities_assessed}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Gap Summary
        severity_counts = self._count_by_severity(data.gaps)
        category_counts = self._count_by_category(data.gaps)

        sections.append(f"## Gap Summary")
        sections.append(f"")
        sections.append(f"**Total Gaps Identified:** {len(data.gaps)}")
        sections.append(
            f"**Total Estimated Remediation Cost:** EUR {data.total_estimated_cost_eur:,.0f}"
        )
        sections.append(
            f"**Total Turnover at Risk:** EUR {data.total_turnover_at_risk_eur:,.0f}"
        )
        sections.append(f"")

        sections.append(f"### By Severity")
        sections.append(f"")
        sections.append(f"| Severity | Count | % of Total |")
        sections.append(f"|----------|------:|----------:|")
        total_gaps = len(data.gaps) if data.gaps else 1
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = severity_counts.get(severity, 0)
            sections.append(
                f"| {severity} | {count} | {count / total_gaps * 100:.1f}% |"
            )
        sections.append(f"")

        sections.append(f"### By Category")
        sections.append(f"")
        sections.append(f"| Category | Count | % of Total |")
        sections.append(f"|----------|------:|----------:|")
        for cat in ["TSC", "DNSH", "MS", "DATA"]:
            count = category_counts.get(cat, 0)
            sections.append(
                f"| {cat} | {count} | {count / total_gaps * 100:.1f}% |"
            )
        sections.append(f"")

        # Critical Gaps
        critical_gaps = [g for g in data.gaps if g.severity == "CRITICAL"]
        if critical_gaps:
            sections.append(f"## Critical Gaps")
            sections.append(f"")
            sections.append(
                f"The following **{len(critical_gaps)}** gaps require immediate attention:"
            )
            sections.append(f"")
            for gap in critical_gaps:
                sections.append(f"### {gap.gap_id}: {gap.activity_name}")
                sections.append(f"")
                sections.append(f"- **Category:** {gap.gap_category}")
                if gap.objective:
                    sections.append(f"- **Objective:** {gap.objective}")
                sections.append(f"- **Description:** {gap.description}")
                sections.append(f"- **Current State:** {gap.current_state}")
                sections.append(f"- **Required State:** {gap.required_state}")
                sections.append(f"- **Remediation:** {gap.remediation_action}")
                sections.append(
                    f"- **Estimated Cost:** EUR {gap.estimated_cost_eur:,.0f}"
                )
                sections.append(
                    f"- **Turnover at Risk:** EUR {gap.impact_on_turnover_eur:,.0f}"
                )
                sections.append(f"")

        # TSC Gaps
        tsc_gaps = [g for g in data.gaps if g.gap_category == "TSC"]
        if tsc_gaps:
            sections.append(f"## TSC Gaps ({len(tsc_gaps)})")
            sections.append(f"")
            sections.append(
                f"| Gap ID | Activity | Objective | Severity | Description | Cost (EUR) |"
            )
            sections.append(
                f"|--------|----------|-----------|----------|-------------|----------:|"
            )
            for gap in tsc_gaps[:self.config.max_gaps_detail]:
                sections.append(
                    f"| {gap.gap_id} | {gap.activity_name[:25]} | "
                    f"{gap.objective} | {gap.severity} | "
                    f"{gap.description[:40]} | {gap.estimated_cost_eur:,.0f} |"
                )
            sections.append(f"")

        # DNSH Gaps
        dnsh_gaps = [g for g in data.gaps if g.gap_category == "DNSH"]
        if dnsh_gaps:
            sections.append(f"## DNSH Gaps ({len(dnsh_gaps)})")
            sections.append(f"")
            sections.append(
                f"| Gap ID | Activity | Objective | Severity | Description | Cost (EUR) |"
            )
            sections.append(
                f"|--------|----------|-----------|----------|-------------|----------:|"
            )
            for gap in dnsh_gaps[:self.config.max_gaps_detail]:
                sections.append(
                    f"| {gap.gap_id} | {gap.activity_name[:25]} | "
                    f"{gap.objective} | {gap.severity} | "
                    f"{gap.description[:40]} | {gap.estimated_cost_eur:,.0f} |"
                )
            sections.append(f"")

        # MS Gaps
        ms_gaps = [g for g in data.gaps if g.gap_category == "MS"]
        if ms_gaps:
            sections.append(f"## Minimum Safeguards Gaps ({len(ms_gaps)})")
            sections.append(f"")
            sections.append(
                f"| Gap ID | Activity | Severity | Description | Remediation | Cost (EUR) |"
            )
            sections.append(
                f"|--------|----------|----------|-------------|-------------|----------:|"
            )
            for gap in ms_gaps[:self.config.max_gaps_detail]:
                sections.append(
                    f"| {gap.gap_id} | {gap.activity_name[:25]} | "
                    f"{gap.severity} | {gap.description[:35]} | "
                    f"{gap.remediation_action[:35]} | {gap.estimated_cost_eur:,.0f} |"
                )
            sections.append(f"")

        # Remediation Timeline
        if self.config.include_remediation_timeline and data.remediation_phases:
            sections.append(f"## Remediation Timeline")
            sections.append(f"")
            sections.append(
                f"| Phase | Name | Start | End | Gaps | Cost (EUR) |"
            )
            sections.append(
                f"|------:|------|-------|-----|-----:|----------:|"
            )
            for phase in data.remediation_phases:
                sections.append(
                    f"| {phase.phase_number} | {phase.phase_name} | "
                    f"{phase.start_date} | {phase.end_date} | "
                    f"{len(phase.gap_ids)} | {phase.total_cost_eur:,.0f} |"
                )
            sections.append(f"")

        # Cost Estimates
        if self.config.include_cost_estimates and data.gaps:
            sections.append(f"## Cost Estimates")
            sections.append(f"")
            sections.append(
                f"**Total Estimated Remediation Cost:** EUR {data.total_estimated_cost_eur:,.0f}"
            )
            sections.append(f"")

            sections.append(f"### By Category")
            sections.append(f"")
            sections.append(f"| Category | Cost (EUR) | % of Total |")
            sections.append(f"|----------|----------:|----------:|")
            for cat in ["TSC", "DNSH", "MS", "DATA"]:
                cat_cost = sum(
                    g.estimated_cost_eur for g in data.gaps if g.gap_category == cat
                )
                cat_pct = (cat_cost / data.total_estimated_cost_eur * 100) if data.total_estimated_cost_eur > 0 else 0
                sections.append(
                    f"| {cat} | {cat_cost:,.0f} | {cat_pct:.1f}% |"
                )
            sections.append(f"")

            sections.append(f"### By Severity")
            sections.append(f"")
            sections.append(f"| Severity | Cost (EUR) | % of Total |")
            sections.append(f"|----------|----------:|----------:|")
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                sev_cost = sum(
                    g.estimated_cost_eur for g in data.gaps if g.severity == sev
                )
                sev_pct = (sev_cost / data.total_estimated_cost_eur * 100) if data.total_estimated_cost_eur > 0 else 0
                sections.append(
                    f"| {sev} | {sev_cost:,.0f} | {sev_pct:.1f}% |"
                )
            sections.append(f"")

        # Priority Matrix
        if self.config.include_priority_matrix and data.gaps:
            sections.append(f"## Priority Matrix")
            sections.append(f"")
            sections.append(
                f"Gaps prioritized by impact (turnover at risk) vs. effort (cost):"
            )
            sections.append(f"")
            sections.append(f"### Quick Wins (High Impact, Low Cost)")
            sections.append(f"")
            quick_wins = [
                g for g in data.gaps
                if g.impact_on_turnover_eur > 0
                and g.estimated_cost_eur < (data.total_estimated_cost_eur / len(data.gaps) if data.gaps else 1)
            ]
            if quick_wins:
                for gap in sorted(quick_wins, key=lambda g: g.impact_on_turnover_eur, reverse=True)[:5]:
                    sections.append(
                        f"- **{gap.gap_id}** ({gap.activity_name}): "
                        f"EUR {gap.impact_on_turnover_eur:,.0f} impact, "
                        f"EUR {gap.estimated_cost_eur:,.0f} cost"
                    )
            else:
                sections.append(f"No quick wins identified.")
            sections.append(f"")

        # Key Recommendations
        if data.key_recommendations:
            sections.append(f"## Key Recommendations")
            sections.append(f"")
            for idx, rec in enumerate(data.key_recommendations, 1):
                sections.append(f"{idx}. {rec}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        severity_counts = self._count_by_severity(data.gaps)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy Gap Analysis - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #e74c3c; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .critical {{ color: #c0392b; font-weight: bold; }}
        .high {{ color: #e67e22; font-weight: bold; }}
        .medium {{ color: #f1c40f; font-weight: bold; }}
        .low {{ color: #27ae60; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy Gap Analysis Report</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Total Gaps:</strong> {len(data.gaps)}</p>
        <p><strong>Critical:</strong> <span class="critical">{severity_counts.get('CRITICAL', 0)}</span> |
           <strong>High:</strong> <span class="high">{severity_counts.get('HIGH', 0)}</span> |
           <strong>Medium:</strong> <span class="medium">{severity_counts.get('MEDIUM', 0)}</span> |
           <strong>Low:</strong> <span class="low">{severity_counts.get('LOW', 0)}</span></p>
        <p><strong>Total Remediation Cost:</strong> EUR {data.total_estimated_cost_eur:,.0f}</p>
    </div>
"""

        if data.gaps:
            html += """    <h2>Gap Inventory</h2>
    <table>
        <tr><th>Gap ID</th><th>Activity</th><th>Category</th><th>Severity</th>
        <th>Description</th><th>Cost (EUR)</th></tr>
"""
            for gap in data.gaps[:self.config.max_gaps_detail]:
                html += f"""        <tr>
            <td>{gap.gap_id}</td><td>{gap.activity_name}</td>
            <td>{gap.gap_category}</td>
            <td class="{gap.severity.lower()}">{gap.severity}</td>
            <td>{gap.description[:60]}</td>
            <td>{gap.estimated_cost_eur:,.0f}</td>
        </tr>
"""
            html += """    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "gap_analysis",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_gaps": len(data.gaps),
                "by_severity": self._count_by_severity(data.gaps),
                "by_category": self._count_by_category(data.gaps),
                "total_estimated_cost_eur": data.total_estimated_cost_eur,
                "total_turnover_at_risk_eur": data.total_turnover_at_risk_eur,
            },
            "gaps": [gap.dict() for gap in data.gaps],
            "remediation_phases": [phase.dict() for phase in data.remediation_phases],
            "key_recommendations": data.key_recommendations,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "GapAnalysisReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _count_by_severity(self, gaps: List[GapItem]) -> Dict[str, int]:
        """Count gaps by severity level."""
        counts: Dict[str, int] = {}
        for gap in gaps:
            counts[gap.severity] = counts.get(gap.severity, 0) + 1
        return counts

    def _count_by_category(self, gaps: List[GapItem]) -> Dict[str, int]:
        """Count gaps by category."""
        counts: Dict[str, int] = {}
        for gap in gaps:
            counts[gap.gap_category] = counts.get(gap.gap_category, 0) + 1
        return counts
