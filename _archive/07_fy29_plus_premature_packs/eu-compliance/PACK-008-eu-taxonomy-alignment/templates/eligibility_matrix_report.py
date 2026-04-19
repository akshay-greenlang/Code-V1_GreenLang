"""
Eligibility Matrix Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates activity-level eligibility results per environmental objective,
NACE sector breakdowns, eligible vs. non-eligible ratios, and objective-level matrices
for EU Taxonomy eligibility screening.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from eligibility_matrix_report import EligibilityMatrixReportTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025",
    ...     total_activities=42
    ... )
    >>> template = EligibilityMatrixReportTemplate()
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
    """Configuration for Eligibility Matrix Report generation."""

    include_nace_breakdown: bool = Field(
        default=True,
        description="Include NACE sector breakdown section"
    )
    include_objective_matrix: bool = Field(
        default=True,
        description="Include 6-objective eligibility matrix"
    )
    include_ratios: bool = Field(
        default=True,
        description="Include eligible vs non-eligible ratio analysis"
    )
    max_activities_detail: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of activities to show in detail"
    )


class ObjectiveEligibility(BaseModel):
    """Eligibility status for a single environmental objective."""

    ccm: bool = Field(default=False, description="Climate Change Mitigation eligibility")
    cca: bool = Field(default=False, description="Climate Change Adaptation eligibility")
    wtr: bool = Field(default=False, description="Water and Marine Resources eligibility")
    ce: bool = Field(default=False, description="Circular Economy eligibility")
    ppc: bool = Field(default=False, description="Pollution Prevention and Control eligibility")
    bio: bool = Field(default=False, description="Biodiversity and Ecosystems eligibility")


class ActivityEligibility(BaseModel):
    """Eligibility assessment for a single economic activity."""

    activity_id: str = Field(..., description="Taxonomy activity identifier")
    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field(..., description="NACE sector code")
    nace_description: str = Field(default="", description="NACE sector description")
    is_eligible: bool = Field(..., description="Overall eligibility status")
    objectives: ObjectiveEligibility = Field(
        default_factory=ObjectiveEligibility,
        description="Per-objective eligibility"
    )
    turnover_eur: float = Field(default=0.0, ge=0, description="Turnover in EUR")
    capex_eur: float = Field(default=0.0, ge=0, description="CapEx in EUR")
    opex_eur: float = Field(default=0.0, ge=0, description="OpEx in EUR")
    delegated_act: str = Field(
        default="Climate DA",
        description="Applicable Delegated Act"
    )


class NACESectorSummary(BaseModel):
    """Summary of eligibility by NACE sector."""

    nace_section: str = Field(..., description="NACE section letter (e.g., C, D, F)")
    nace_description: str = Field(..., description="NACE section name")
    total_activities: int = Field(..., ge=0, description="Total activities in sector")
    eligible_activities: int = Field(..., ge=0, description="Eligible activities in sector")
    eligible_turnover_eur: float = Field(default=0.0, ge=0, description="Eligible turnover")
    total_turnover_eur: float = Field(default=0.0, ge=0, description="Total turnover")
    eligibility_rate: float = Field(
        default=0.0, ge=0, le=100,
        description="Eligibility rate as percentage"
    )


class ReportData(BaseModel):
    """Data model for Eligibility Matrix Report."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    total_activities: int = Field(..., ge=0, description="Total economic activities assessed")
    activities: List[ActivityEligibility] = Field(
        default_factory=list,
        description="Activity-level eligibility results"
    )
    nace_sectors: List[NACESectorSummary] = Field(
        default_factory=list,
        description="NACE sector breakdown"
    )
    total_turnover_eur: float = Field(default=0.0, ge=0, description="Total turnover")
    total_capex_eur: float = Field(default=0.0, ge=0, description="Total CapEx")
    total_opex_eur: float = Field(default=0.0, ge=0, description="Total OpEx")
    notes: List[str] = Field(
        default_factory=list,
        description="Additional notes or caveats"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class EligibilityMatrixReportTemplate:
    """
    Eligibility Matrix Report Template for EU Taxonomy Alignment Pack.

    Generates activity-level eligibility results per environmental objective with
    NACE sector breakdowns, eligible vs. non-eligible ratios, and a 6-column
    objective matrix.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = EligibilityMatrixReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Eligibility Matrix" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Eligibility Matrix Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the eligibility matrix report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Eligibility Matrix Report for {data.organization_name} "
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
        sections.append(f"# EU Taxonomy Eligibility Matrix Report")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Total Activities Assessed:** {data.total_activities}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Summary
        eligible = [a for a in data.activities if a.is_eligible]
        non_eligible = [a for a in data.activities if not a.is_eligible]
        elig_rate = (len(eligible) / len(data.activities) * 100) if data.activities else 0

        sections.append(f"## Summary")
        sections.append(f"")
        sections.append(
            f"Of **{data.total_activities}** economic activities assessed, "
            f"**{len(eligible)}** ({elig_rate:.1f}%) are taxonomy-eligible and "
            f"**{len(non_eligible)}** ({100 - elig_rate:.1f}%) are non-eligible."
        )
        sections.append(f"")

        eligible_turnover = sum(a.turnover_eur for a in eligible)
        eligible_capex = sum(a.capex_eur for a in eligible)
        eligible_opex = sum(a.opex_eur for a in eligible)

        sections.append(f"| Metric | Eligible | Non-Eligible | Total | Eligible % |")
        sections.append(f"|--------|----------|--------------|-------|------------|")

        t_rate = (eligible_turnover / data.total_turnover_eur * 100) if data.total_turnover_eur > 0 else 0
        c_rate = (eligible_capex / data.total_capex_eur * 100) if data.total_capex_eur > 0 else 0
        o_rate = (eligible_opex / data.total_opex_eur * 100) if data.total_opex_eur > 0 else 0

        sections.append(
            f"| Turnover | EUR {eligible_turnover:,.0f} | "
            f"EUR {data.total_turnover_eur - eligible_turnover:,.0f} | "
            f"EUR {data.total_turnover_eur:,.0f} | {t_rate:.1f}% |"
        )
        sections.append(
            f"| CapEx | EUR {eligible_capex:,.0f} | "
            f"EUR {data.total_capex_eur - eligible_capex:,.0f} | "
            f"EUR {data.total_capex_eur:,.0f} | {c_rate:.1f}% |"
        )
        sections.append(
            f"| OpEx | EUR {eligible_opex:,.0f} | "
            f"EUR {data.total_opex_eur - eligible_opex:,.0f} | "
            f"EUR {data.total_opex_eur:,.0f} | {o_rate:.1f}% |"
        )
        sections.append(f"")

        # Activity List
        if data.activities:
            sections.append(f"## Activity List")
            sections.append(f"")
            display_activities = data.activities[:self.config.max_activities_detail]
            sections.append(
                f"| Activity ID | Activity Name | NACE | Eligible | "
                f"Turnover (EUR) | Delegated Act |"
            )
            sections.append(
                f"|-------------|---------------|------|----------|"
                f"----------------|---------------|"
            )
            for act in display_activities:
                elig_flag = "Yes" if act.is_eligible else "No"
                sections.append(
                    f"| {act.activity_id} | {act.activity_name} | {act.nace_code} | "
                    f"{elig_flag} | {act.turnover_eur:,.0f} | {act.delegated_act} |"
                )
            if len(data.activities) > self.config.max_activities_detail:
                sections.append(
                    f"| ... | *{len(data.activities) - self.config.max_activities_detail} "
                    f"more activities* | | | | |"
                )
            sections.append(f"")

        # Objective Matrix (6 columns)
        if self.config.include_objective_matrix and data.activities:
            sections.append(f"## Objective Eligibility Matrix")
            sections.append(f"")
            sections.append(
                f"Eligibility status per environmental objective for each activity:"
            )
            sections.append(f"")
            sections.append(
                f"| Activity | CCM | CCA | WTR | CE | PPC | BIO |"
            )
            sections.append(
                f"|----------|-----|-----|-----|----|-----|-----|"
            )
            for act in display_activities:
                obj = act.objectives
                sections.append(
                    f"| {act.activity_name[:40]} | "
                    f"{'Y' if obj.ccm else '-'} | "
                    f"{'Y' if obj.cca else '-'} | "
                    f"{'Y' if obj.wtr else '-'} | "
                    f"{'Y' if obj.ce else '-'} | "
                    f"{'Y' if obj.ppc else '-'} | "
                    f"{'Y' if obj.bio else '-'} |"
                )
            sections.append(f"")

            # Objective totals
            sections.append(f"### Objective Totals")
            sections.append(f"")
            obj_counts = self._count_objective_eligibility(data.activities)
            total = len(data.activities) if data.activities else 1
            sections.append(f"| Objective | Eligible Activities | Percentage |")
            sections.append(f"|-----------|--------------------:|------------|")
            for obj_name, count in obj_counts.items():
                sections.append(
                    f"| {obj_name} | {count} | {count / total * 100:.1f}% |"
                )
            sections.append(f"")

        # NACE Sector Breakdown
        if self.config.include_nace_breakdown and data.nace_sectors:
            sections.append(f"## NACE Sector Breakdown")
            sections.append(f"")
            sections.append(
                f"| NACE Section | Description | Total | Eligible | "
                f"Rate | Eligible Turnover (EUR) |"
            )
            sections.append(
                f"|--------------|-------------|------:|---------:|"
                f"------:|------------------------:|"
            )
            for sector in sorted(data.nace_sectors, key=lambda s: s.eligible_activities, reverse=True):
                sections.append(
                    f"| {sector.nace_section} | {sector.nace_description} | "
                    f"{sector.total_activities} | {sector.eligible_activities} | "
                    f"{sector.eligibility_rate:.1f}% | {sector.eligible_turnover_eur:,.0f} |"
                )
            sections.append(f"")

        # Eligible vs Non-Eligible Ratios
        if self.config.include_ratios and data.activities:
            sections.append(f"## Eligible vs Non-Eligible Ratios")
            sections.append(f"")
            sections.append(f"| KPI | Eligible Ratio | Non-Eligible Ratio |")
            sections.append(f"|-----|---------------:|-------------------:|")
            sections.append(f"| Activity Count | {elig_rate:.1f}% | {100 - elig_rate:.1f}% |")
            sections.append(f"| Turnover | {t_rate:.1f}% | {100 - t_rate:.1f}% |")
            sections.append(f"| CapEx | {c_rate:.1f}% | {100 - c_rate:.1f}% |")
            sections.append(f"| OpEx | {o_rate:.1f}% | {100 - o_rate:.1f}% |")
            sections.append(f"")

        # Notes
        if data.notes:
            sections.append(f"## Notes")
            sections.append(f"")
            for idx, note in enumerate(data.notes, 1):
                sections.append(f"{idx}. {note}")
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
        eligible = [a for a in data.activities if a.is_eligible]
        elig_rate = (len(eligible) / len(data.activities) * 100) if data.activities else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy Eligibility Matrix - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #27ae60; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #27ae60; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .eligible {{ color: #27ae60; font-weight: bold; }}
        .non-eligible {{ color: #e74c3c; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy Eligibility Matrix Report</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Total Activities:</strong> {data.total_activities}</p>
        <p><strong>Eligibility Rate:</strong> <span class="metric">{elig_rate:.1f}%</span></p>
    </div>

    <h2>Summary</h2>
    <p>
        Of <strong>{data.total_activities}</strong> economic activities assessed,
        <span class="eligible">{len(eligible)}</span> ({elig_rate:.1f}%) are taxonomy-eligible.
    </p>
"""

        if self.config.include_objective_matrix and data.activities:
            html += """    <h2>Objective Eligibility Matrix</h2>
    <table>
        <tr><th>Activity</th><th>CCM</th><th>CCA</th><th>WTR</th><th>CE</th><th>PPC</th><th>BIO</th></tr>
"""
            for act in data.activities[:self.config.max_activities_detail]:
                obj = act.objectives
                html += f"""        <tr>
            <td>{act.activity_name}</td>
            <td>{'Y' if obj.ccm else '-'}</td>
            <td>{'Y' if obj.cca else '-'}</td>
            <td>{'Y' if obj.wtr else '-'}</td>
            <td>{'Y' if obj.ce else '-'}</td>
            <td>{'Y' if obj.ppc else '-'}</td>
            <td>{'Y' if obj.bio else '-'}</td>
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
        eligible = [a for a in data.activities if a.is_eligible]
        non_eligible = [a for a in data.activities if not a.is_eligible]

        report_dict = {
            "report_type": "eligibility_matrix",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_activities": data.total_activities,
                "eligible_activities": len(eligible),
                "non_eligible_activities": len(non_eligible),
                "eligibility_rate": (
                    len(eligible) / len(data.activities) * 100
                ) if data.activities else 0,
                "total_turnover_eur": data.total_turnover_eur,
                "total_capex_eur": data.total_capex_eur,
                "total_opex_eur": data.total_opex_eur,
            },
            "activities": [act.dict() for act in data.activities],
            "nace_sectors": [sector.dict() for sector in data.nace_sectors],
            "objective_totals": self._count_objective_eligibility(data.activities),
            "notes": data.notes,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "EligibilityMatrixReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _count_objective_eligibility(
        self, activities: List[ActivityEligibility]
    ) -> Dict[str, int]:
        """Count the number of eligible activities per environmental objective."""
        counts = {
            "Climate Change Mitigation (CCM)": 0,
            "Climate Change Adaptation (CCA)": 0,
            "Water and Marine Resources (WTR)": 0,
            "Circular Economy (CE)": 0,
            "Pollution Prevention and Control (PPC)": 0,
            "Biodiversity and Ecosystems (BIO)": 0,
        }
        for act in activities:
            if act.objectives.ccm:
                counts["Climate Change Mitigation (CCM)"] += 1
            if act.objectives.cca:
                counts["Climate Change Adaptation (CCA)"] += 1
            if act.objectives.wtr:
                counts["Water and Marine Resources (WTR)"] += 1
            if act.objectives.ce:
                counts["Circular Economy (CE)"] += 1
            if act.objectives.ppc:
                counts["Pollution Prevention and Control (PPC)"] += 1
            if act.objectives.bio:
                counts["Biodiversity and Ecosystems (BIO)"] += 1
        return counts
