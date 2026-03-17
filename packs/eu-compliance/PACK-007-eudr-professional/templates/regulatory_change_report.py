"""
Regulatory Change Report Template - PACK-007 EUDR Professional Pack

This module generates regulatory amendment impact reports with change timelines, gap analysis,
migration checklists, affected processes, implementation status, and cross-regulation impacts
for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from regulatory_change_report import RegulatoryChangeReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="European Importers AG",
    ...     report_date="2026-03-15",
    ...     regulation_name="EUDR Regulation (EU) 2023/1115"
    ... )
    >>> template = RegulatoryChangeReportTemplate()
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
    """Configuration for Regulatory Change Report generation."""

    include_gap_analysis: bool = Field(
        default=True,
        description="Include detailed gap analysis"
    )
    include_migration_checklist: bool = Field(
        default=True,
        description="Include migration action checklist"
    )
    include_cross_regulation_impacts: bool = Field(
        default=True,
        description="Include cross-regulation impact analysis"
    )


class RegulatoryChange(BaseModel):
    """Regulatory change item."""

    change_id: str = Field(..., description="Change identifier")
    change_type: str = Field(
        ...,
        description="NEW_REQUIREMENT, MODIFIED_REQUIREMENT, REMOVED_REQUIREMENT, CLARIFICATION"
    )
    article_reference: str = Field(..., description="Article/section reference")
    change_description: str = Field(..., description="Description of the change")
    effective_date: str = Field(..., description="Date change becomes effective")
    impact_severity: str = Field(..., description="HIGH, MEDIUM, LOW")
    compliance_required_by: str = Field(..., description="Deadline for compliance")


class GapAnalysisItem(BaseModel):
    """Gap analysis item for regulatory change."""

    gap_id: str = Field(..., description="Gap identifier")
    change_reference: str = Field(..., description="Related change ID")
    current_state: str = Field(..., description="Current compliance state")
    required_state: str = Field(..., description="Required future state")
    gap_description: str = Field(..., description="Description of the gap")
    gap_severity: str = Field(..., description="CRITICAL, MAJOR, MINOR")
    affected_processes: List[str] = Field(
        default_factory=list,
        description="Business processes affected"
    )


class MigrationChecklistItem(BaseModel):
    """Migration action checklist item."""

    action_id: str = Field(..., description="Action identifier")
    action_description: str = Field(..., description="Action required")
    related_change: str = Field(..., description="Related change ID")
    responsible_party: str = Field(..., description="Responsible person/team")
    target_completion_date: str = Field(..., description="Target completion date")
    status: str = Field(..., description="NOT_STARTED, IN_PROGRESS, COMPLETED, BLOCKED")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Dependent action IDs"
    )
    estimated_effort_hours: Optional[float] = Field(None, description="Estimated effort")


class AffectedProcess(BaseModel):
    """Business process affected by regulatory change."""

    process_id: str = Field(..., description="Process identifier")
    process_name: str = Field(..., description="Process name")
    current_version: str = Field(..., description="Current process version")
    changes_required: List[str] = Field(..., description="Changes required")
    impact_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    update_priority: str = Field(..., description="URGENT, HIGH, MEDIUM, LOW")


class ImplementationStatus(BaseModel):
    """Implementation status summary."""

    area: str = Field(..., description="Implementation area")
    total_actions: int = Field(..., ge=0, description="Total actions required")
    completed_actions: int = Field(..., ge=0, description="Actions completed")
    in_progress_actions: int = Field(..., ge=0, description="Actions in progress")
    blocked_actions: int = Field(..., ge=0, description="Actions blocked")
    completion_percentage: float = Field(..., ge=0, le=100, description="Completion %")


class CrossRegulationImpact(BaseModel):
    """Cross-regulation impact analysis."""

    regulation_name: str = Field(..., description="Related regulation name")
    relationship: str = Field(
        ...,
        description="COMPLEMENTARY, CONFLICTING, SUPERSEDED, ALIGNED"
    )
    impact_description: str = Field(..., description="Description of impact")
    action_required: Optional[str] = Field(None, description="Action required if any")
    priority: str = Field(..., description="HIGH, MEDIUM, LOW")


class ReportData(BaseModel):
    """Data model for Regulatory Change Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    regulation_name: str = Field(..., description="Name of regulation (e.g., EUDR)")
    amendment_name: Optional[str] = Field(None, description="Amendment name/number if applicable")
    changes: List[RegulatoryChange] = Field(
        default_factory=list,
        description="Regulatory changes"
    )
    gap_analysis: List[GapAnalysisItem] = Field(
        default_factory=list,
        description="Gap analysis items"
    )
    migration_checklist: List[MigrationChecklistItem] = Field(
        default_factory=list,
        description="Migration checklist"
    )
    affected_processes: List[AffectedProcess] = Field(
        default_factory=list,
        description="Affected business processes"
    )
    implementation_status: List[ImplementationStatus] = Field(
        default_factory=list,
        description="Implementation status by area"
    )
    cross_regulation_impacts: List[CrossRegulationImpact] = Field(
        default_factory=list,
        description="Cross-regulation impacts"
    )
    overall_readiness: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall readiness percentage"
    )
    critical_deadline: Optional[str] = Field(
        None,
        description="Most critical upcoming deadline"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class RegulatoryChangeReportTemplate:
    """
    Regulatory Change Report Template for EUDR Professional Pack.

    Generates comprehensive regulatory amendment impact reports with change timelines,
    gap analysis, migration checklists, affected processes, implementation status,
    and cross-regulation impacts.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = RegulatoryChangeReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Regulatory Change Impact" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Regulatory Change Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the regulatory change report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Regulatory Change Report for {data.operator_name} in {format} format"
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
        sections.append(f"# Regulatory Change Impact Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Regulation:** {data.regulation_name}")
        if data.amendment_name:
            sections.append(f"**Amendment:** {data.amendment_name}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Overall Readiness:** {data.overall_readiness:.1f}%")
        if data.critical_deadline:
            sections.append(f"**Critical Deadline:** {data.critical_deadline}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        sections.append(
            f"This report analyzes the impact of recent changes to **{data.regulation_name}** "
            f"on **{data.operator_name}** operations. A total of **{len(data.changes)}** "
            f"regulatory changes have been identified, resulting in **{len(data.gap_analysis)}** "
            f"compliance gaps and **{len(data.migration_checklist)}** required actions. "
            f"Overall readiness is **{data.overall_readiness:.1f}%**."
        )
        sections.append(f"")

        if data.critical_deadline:
            sections.append(
                f"⚠️ **Critical Deadline:** {data.critical_deadline} - Immediate action required."
            )
            sections.append(f"")

        # Change Timeline
        if data.changes:
            sections.append(f"## Regulatory Change Timeline")
            sections.append(f"")
            sections.append(
                f"Chronological overview of regulatory changes:"
            )
            sections.append(f"")

            sections.append(
                f"| Change ID | Type | Article | Description | Effective Date | Impact | Deadline |"
            )
            sections.append(
                f"|-----------|------|---------|-------------|----------------|--------|----------|"
            )

            for change in sorted(data.changes, key=lambda x: x.effective_date):
                sections.append(
                    f"| {change.change_id} | {change.change_type} | {change.article_reference} | "
                    f"{change.change_description[:40]}... | {change.effective_date} | "
                    f"{change.impact_severity} | {change.compliance_required_by} |"
                )
            sections.append(f"")

            # High impact changes
            high_impact = [c for c in data.changes if c.impact_severity == "HIGH"]
            if high_impact:
                sections.append(f"### High Impact Changes")
                sections.append(f"")
                sections.append(
                    f"**{len(high_impact)}** changes classified as high impact:"
                )
                sections.append(f"")
                for change in high_impact:
                    sections.append(f"**{change.change_id}** - {change.article_reference}")
                    sections.append(f"- **Change:** {change.change_description}")
                    sections.append(f"- **Effective:** {change.effective_date}")
                    sections.append(f"- **Deadline:** {change.compliance_required_by}")
                    sections.append(f"")

        # Gap Analysis
        if self.config.include_gap_analysis and data.gap_analysis:
            sections.append(f"## Gap Analysis")
            sections.append(f"")
            sections.append(
                f"Identified gaps between current state and required future state:"
            )
            sections.append(f"")

            sections.append(
                f"| Gap ID | Change Ref | Current State | Required State | Severity | Affected Processes |"
            )
            sections.append(
                f"|--------|------------|---------------|----------------|----------|--------------------|"
            )

            for gap in sorted(data.gap_analysis, key=lambda x: (x.gap_severity == "CRITICAL", x.gap_severity == "MAJOR"), reverse=True):
                processes = ", ".join(gap.affected_processes[:2]) if gap.affected_processes else "N/A"
                sections.append(
                    f"| {gap.gap_id} | {gap.change_reference} | {gap.current_state[:20]}... | "
                    f"{gap.required_state[:20]}... | {gap.gap_severity} | {processes} |"
                )
            sections.append(f"")

            # Gap summary by severity
            critical_gaps = len([g for g in data.gap_analysis if g.gap_severity == "CRITICAL"])
            major_gaps = len([g for g in data.gap_analysis if g.gap_severity == "MAJOR"])
            minor_gaps = len([g for g in data.gap_analysis if g.gap_severity == "MINOR"])

            sections.append(f"### Gap Summary")
            sections.append(f"")
            sections.append(f"- **Critical Gaps:** {critical_gaps}")
            sections.append(f"- **Major Gaps:** {major_gaps}")
            sections.append(f"- **Minor Gaps:** {minor_gaps}")
            sections.append(f"- **Total Gaps:** {len(data.gap_analysis)}")
            sections.append(f"")

        # Migration Checklist
        if self.config.include_migration_checklist and data.migration_checklist:
            sections.append(f"## Migration Action Checklist")
            sections.append(f"")
            sections.append(
                f"Required actions to achieve compliance with regulatory changes:"
            )
            sections.append(f"")

            sections.append(
                f"| Action ID | Action Description | Related Change | Responsible | Target Date | Status | Effort (hrs) |"
            )
            sections.append(
                f"|-----------|-------------------|----------------|-------------|-------------|--------|--------------|"
            )

            for action in data.migration_checklist:
                effort = f"{action.estimated_effort_hours:.0f}" if action.estimated_effort_hours else "N/A"
                sections.append(
                    f"| {action.action_id} | {action.action_description[:30]}... | "
                    f"{action.related_change} | {action.responsible_party} | "
                    f"{action.target_completion_date} | {action.status} | {effort} |"
                )
            sections.append(f"")

            # Status summary
            not_started = len([a for a in data.migration_checklist if a.status == "NOT_STARTED"])
            in_progress = len([a for a in data.migration_checklist if a.status == "IN_PROGRESS"])
            completed = len([a for a in data.migration_checklist if a.status == "COMPLETED"])
            blocked = len([a for a in data.migration_checklist if a.status == "BLOCKED"])

            sections.append(f"### Action Status Summary")
            sections.append(f"")
            sections.append(f"- **Not Started:** {not_started}")
            sections.append(f"- **In Progress:** {in_progress}")
            sections.append(f"- **Completed:** {completed}")
            sections.append(f"- **Blocked:** {blocked}")
            sections.append(f"")

            if blocked > 0:
                sections.append(f"⚠️ **{blocked} actions are blocked** - immediate attention required")
                sections.append(f"")

        # Affected Processes
        if data.affected_processes:
            sections.append(f"## Affected Business Processes")
            sections.append(f"")
            sections.append(
                f"Business processes requiring updates due to regulatory changes:"
            )
            sections.append(f"")

            sections.append(
                f"| Process ID | Process Name | Current Ver | Changes Required | Impact | Priority |"
            )
            sections.append(
                f"|------------|--------------|-------------|------------------|--------|----------|"
            )

            for process in sorted(data.affected_processes, key=lambda x: (x.update_priority == "URGENT", x.update_priority == "HIGH"), reverse=True):
                changes_count = len(process.changes_required)
                sections.append(
                    f"| {process.process_id} | {process.process_name} | {process.current_version} | "
                    f"{changes_count} changes | {process.impact_level} | {process.update_priority} |"
                )
            sections.append(f"")

        # Implementation Status
        if data.implementation_status:
            sections.append(f"## Implementation Status")
            sections.append(f"")
            sections.append(
                f"Progress by implementation area:"
            )
            sections.append(f"")

            sections.append(
                f"| Area | Total | Completed | In Progress | Blocked | Completion % |"
            )
            sections.append(
                f"|------|-------|-----------|-------------|---------|--------------|"
            )

            for status in sorted(data.implementation_status, key=lambda x: x.completion_percentage, reverse=True):
                sections.append(
                    f"| {status.area} | {status.total_actions} | {status.completed_actions} | "
                    f"{status.in_progress_actions} | {status.blocked_actions} | "
                    f"{status.completion_percentage:.1f}% |"
                )
            sections.append(f"")

            # Progress visualization
            sections.append(f"### Progress Visualization")
            sections.append(f"")
            sections.append(f"```")
            sections.append(self._create_progress_chart(data.implementation_status))
            sections.append(f"```")
            sections.append(f"")

        # Cross-Regulation Impacts
        if self.config.include_cross_regulation_impacts and data.cross_regulation_impacts:
            sections.append(f"## Cross-Regulation Impact Analysis")
            sections.append(f"")
            sections.append(
                f"Impact of EUDR changes on compliance with other regulations:"
            )
            sections.append(f"")

            sections.append(
                f"| Regulation | Relationship | Impact Description | Action Required | Priority |"
            )
            sections.append(
                f"|------------|--------------|-------------------|-----------------|----------|"
            )

            for impact in data.cross_regulation_impacts:
                action = impact.action_required or "None"
                sections.append(
                    f"| {impact.regulation_name} | {impact.relationship} | "
                    f"{impact.impact_description[:40]}... | {action[:30]}... | {impact.priority} |"
                )
            sections.append(f"")

            # Highlight conflicts
            conflicts = [i for i in data.cross_regulation_impacts if i.relationship == "CONFLICTING"]
            if conflicts:
                sections.append(f"### ⚠️ Regulatory Conflicts Identified")
                sections.append(f"")
                for conflict in conflicts:
                    sections.append(f"**{conflict.regulation_name}**")
                    sections.append(f"- {conflict.impact_description}")
                    sections.append(f"- Action: {conflict.action_required}")
                    sections.append(f"")

        # Recommendations
        sections.append(f"## Recommendations")
        sections.append(f"")

        if data.overall_readiness < 70:
            sections.append(
                f"1. **Accelerate Implementation:** Overall readiness is {data.overall_readiness:.1f}%. "
                f"Immediate action required to meet compliance deadlines."
            )

        if data.gap_analysis:
            critical_gaps = [g for g in data.gap_analysis if g.gap_severity == "CRITICAL"]
            if critical_gaps:
                sections.append(
                    f"2. **Address Critical Gaps:** {len(critical_gaps)} critical gaps require "
                    f"immediate remediation."
                )

        if data.migration_checklist:
            blocked = [a for a in data.migration_checklist if a.status == "BLOCKED"]
            if blocked:
                sections.append(
                    f"3. **Unblock Actions:** {len(blocked)} actions are blocked - resolve "
                    f"dependencies urgently."
                )

        sections.append(
            f"4. **Monitor Regulatory Updates:** Subscribe to official regulatory update channels "
            f"to track future amendments."
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
    <title>Regulatory Change Impact Report - {data.operator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e67e22; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #e67e22; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #d35400; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .warning {{ background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0; }}
        pre {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Regulatory Change Impact Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Regulation:</strong> {data.regulation_name}</p>
"""

        if data.amendment_name:
            html += f"        <p><strong>Amendment:</strong> {data.amendment_name}</p>\n"

        html += f"""        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Overall Readiness:</strong> <span class="metric">{data.overall_readiness:.1f}%</span></p>
"""

        if data.critical_deadline:
            html += f"""        <div class="warning">
            <strong>Critical Deadline:</strong> {data.critical_deadline}
        </div>
"""

        html += f"""    </div>

    <h2>Executive Summary</h2>
    <p>
        This report analyzes the impact of recent changes to <strong>{data.regulation_name}</strong>
        on {data.operator_name} operations. A total of <strong>{len(data.changes)}</strong>
        regulatory changes have been identified, resulting in {len(data.gap_analysis)} compliance gaps
        and {len(data.migration_checklist)} required actions.
    </p>
"""

        if data.changes:
            html += f"""
    <h2>Regulatory Change Timeline</h2>
    <table>
        <tr><th>Change ID</th><th>Type</th><th>Article</th><th>Description</th><th>Effective Date</th><th>Impact</th></tr>
"""
            for change in sorted(data.changes, key=lambda x: x.effective_date)[:30]:
                html += f"""        <tr>
            <td>{change.change_id}</td>
            <td>{change.change_type}</td>
            <td>{change.article_reference}</td>
            <td>{change.change_description[:60]}...</td>
            <td>{change.effective_date}</td>
            <td>{change.impact_severity}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.implementation_status:
            html += f"""
    <h2>Implementation Status</h2>
    <table>
        <tr><th>Area</th><th>Total</th><th>Completed</th><th>In Progress</th><th>Blocked</th><th>Completion %</th></tr>
"""
            for status in data.implementation_status:
                html += f"""        <tr>
            <td>{status.area}</td>
            <td>{status.total_actions}</td>
            <td>{status.completed_actions}</td>
            <td>{status.in_progress_actions}</td>
            <td>{status.blocked_actions}</td>
            <td>{status.completion_percentage:.1f}%</td>
        </tr>
"""
            html += f"""    </table>
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
            "report_type": "regulatory_change_impact",
            "operator_name": data.operator_name,
            "regulation_name": data.regulation_name,
            "amendment_name": data.amendment_name,
            "report_date": data.report_date,
            "overall_readiness": data.overall_readiness,
            "critical_deadline": data.critical_deadline,
            "changes": [change.dict() for change in data.changes],
            "gap_analysis": [gap.dict() for gap in data.gap_analysis],
            "migration_checklist": [action.dict() for action in data.migration_checklist],
            "affected_processes": [process.dict() for process in data.affected_processes],
            "implementation_status": [status.dict() for status in data.implementation_status],
            "cross_regulation_impacts": [impact.dict() for impact in data.cross_regulation_impacts],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_progress_chart(self, statuses: List[ImplementationStatus]) -> str:
        """Create text-based progress chart."""
        chart_lines = []
        chart_lines.append("Implementation Progress by Area")
        chart_lines.append("")

        if not statuses:
            return "No implementation data available"

        max_pct = 100
        scale = 50 / max_pct

        for status in sorted(statuses, key=lambda x: x.completion_percentage, reverse=True):
            bar_length = int(status.completion_percentage * scale)
            bar = "█" * bar_length
            chart_lines.append(
                f"{status.area[:25]:25} |{bar} {status.completion_percentage:.1f}%"
            )

        return "\n".join(chart_lines)
