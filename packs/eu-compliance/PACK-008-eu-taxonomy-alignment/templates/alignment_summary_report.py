"""
Alignment Summary Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates portfolio-level alignment results including Substantial Contribution
pass rates, DNSH matrix summaries, Minimum Safeguards verification status, and aligned
vs. eligible ratio analysis across the six environmental objectives.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from alignment_summary_report import AlignmentSummaryReportTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025",
    ...     total_activities=42
    ... )
    >>> template = AlignmentSummaryReportTemplate()
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
    """Configuration for Alignment Summary Report generation."""

    include_sc_details: bool = Field(
        default=True,
        description="Include Substantial Contribution pass rates by objective"
    )
    include_dnsh_matrix: bool = Field(
        default=True,
        description="Include DNSH matrix summary"
    )
    include_ms_status: bool = Field(
        default=True,
        description="Include Minimum Safeguards verification status"
    )
    include_activity_detail: bool = Field(
        default=True,
        description="Include activity-level alignment detail"
    )
    max_activities_detail: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of activities in detail section"
    )


class SCPassRate(BaseModel):
    """Substantial Contribution pass rate for an environmental objective."""

    objective: str = Field(..., description="Environmental objective name")
    objective_code: str = Field(..., description="Objective code (CCM, CCA, etc.)")
    activities_assessed: int = Field(..., ge=0, description="Activities assessed for this objective")
    activities_passed: int = Field(..., ge=0, description="Activities that passed SC")
    pass_rate: float = Field(..., ge=0, le=100, description="Pass rate percentage")


class DNSHObjectiveResult(BaseModel):
    """DNSH assessment result for a single objective."""

    objective: str = Field(..., description="Environmental objective name")
    objective_code: str = Field(..., description="Objective code")
    activities_assessed: int = Field(..., ge=0, description="Activities assessed")
    activities_passed: int = Field(..., ge=0, description="Activities that passed DNSH")
    pass_rate: float = Field(..., ge=0, le=100, description="DNSH pass rate")
    common_failures: List[str] = Field(
        default_factory=list,
        description="Most common failure reasons"
    )


class MSVerificationResult(BaseModel):
    """Minimum Safeguards verification result per topic."""

    topic: str = Field(..., description="MS topic (Human Rights, Anti-Corruption, etc.)")
    status: str = Field(..., description="PASS, FAIL, or PARTIAL")
    activities_verified: int = Field(..., ge=0, description="Activities verified")
    activities_compliant: int = Field(..., ge=0, description="Compliant activities")
    findings: List[str] = Field(
        default_factory=list,
        description="Key findings"
    )


class ActivityAlignment(BaseModel):
    """Alignment assessment result for a single activity."""

    activity_id: str = Field(..., description="Taxonomy activity identifier")
    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field(..., description="NACE sector code")
    is_eligible: bool = Field(..., description="Eligibility status")
    is_aligned: bool = Field(..., description="Overall alignment status")
    sc_pass: bool = Field(default=False, description="Substantial Contribution passed")
    sc_objective: str = Field(default="", description="SC objective (e.g., CCM)")
    dnsh_pass: bool = Field(default=False, description="DNSH passed (all objectives)")
    ms_pass: bool = Field(default=False, description="Minimum Safeguards passed")
    turnover_eur: float = Field(default=0.0, ge=0, description="Activity turnover in EUR")
    capex_eur: float = Field(default=0.0, ge=0, description="Activity CapEx in EUR")
    opex_eur: float = Field(default=0.0, ge=0, description="Activity OpEx in EUR")


class ReportData(BaseModel):
    """Data model for Alignment Summary Report."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    total_activities: int = Field(..., ge=0, description="Total economic activities")
    eligible_activities: int = Field(default=0, ge=0, description="Eligible activities")
    aligned_activities: int = Field(default=0, ge=0, description="Aligned activities")
    sc_pass_rates: List[SCPassRate] = Field(
        default_factory=list,
        description="SC pass rates by objective"
    )
    dnsh_results: List[DNSHObjectiveResult] = Field(
        default_factory=list,
        description="DNSH results by objective"
    )
    ms_results: List[MSVerificationResult] = Field(
        default_factory=list,
        description="Minimum Safeguards results by topic"
    )
    activities: List[ActivityAlignment] = Field(
        default_factory=list,
        description="Activity-level alignment results"
    )
    total_turnover_eur: float = Field(default=0.0, ge=0, description="Total turnover")
    total_capex_eur: float = Field(default=0.0, ge=0, description="Total CapEx")
    total_opex_eur: float = Field(default=0.0, ge=0, description="Total OpEx")
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key alignment findings"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class AlignmentSummaryReportTemplate:
    """
    Alignment Summary Report Template for EU Taxonomy Alignment Pack.

    Generates portfolio-level alignment results including SC/DNSH/MS pass rates,
    aligned vs. eligible ratios, and activity-level alignment detail.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = AlignmentSummaryReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Alignment Overview" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Alignment Summary Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the alignment summary report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Alignment Summary Report for {data.organization_name} "
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
        sections.append(f"# EU Taxonomy Alignment Summary Report")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        elig_rate = (data.eligible_activities / data.total_activities * 100) if data.total_activities > 0 else 0
        align_rate = (data.aligned_activities / data.total_activities * 100) if data.total_activities > 0 else 0
        align_of_elig = (data.aligned_activities / data.eligible_activities * 100) if data.eligible_activities > 0 else 0

        sections.append(
            f"Of **{data.total_activities}** total economic activities, "
            f"**{data.eligible_activities}** ({elig_rate:.1f}%) are taxonomy-eligible and "
            f"**{data.aligned_activities}** ({align_rate:.1f}%) are fully taxonomy-aligned. "
            f"The alignment-to-eligibility conversion rate is **{align_of_elig:.1f}%**."
        )
        sections.append(f"")

        # Alignment Overview
        sections.append(f"## Alignment Overview")
        sections.append(f"")
        aligned = [a for a in data.activities if a.is_aligned]
        aligned_turnover = sum(a.turnover_eur for a in aligned)
        aligned_capex = sum(a.capex_eur for a in aligned)
        aligned_opex = sum(a.opex_eur for a in aligned)

        sections.append(f"| Metric | Value | Aligned | Aligned % |")
        sections.append(f"|--------|------:|--------:|----------:|")
        sections.append(
            f"| Activities | {data.total_activities} | "
            f"{data.aligned_activities} | {align_rate:.1f}% |"
        )

        t_pct = (aligned_turnover / data.total_turnover_eur * 100) if data.total_turnover_eur > 0 else 0
        c_pct = (aligned_capex / data.total_capex_eur * 100) if data.total_capex_eur > 0 else 0
        o_pct = (aligned_opex / data.total_opex_eur * 100) if data.total_opex_eur > 0 else 0

        sections.append(
            f"| Turnover (EUR) | {data.total_turnover_eur:,.0f} | "
            f"{aligned_turnover:,.0f} | {t_pct:.1f}% |"
        )
        sections.append(
            f"| CapEx (EUR) | {data.total_capex_eur:,.0f} | "
            f"{aligned_capex:,.0f} | {c_pct:.1f}% |"
        )
        sections.append(
            f"| OpEx (EUR) | {data.total_opex_eur:,.0f} | "
            f"{aligned_opex:,.0f} | {o_pct:.1f}% |"
        )
        sections.append(f"")

        # SC Pass Rates by Objective
        if self.config.include_sc_details and data.sc_pass_rates:
            sections.append(f"## Substantial Contribution Pass Rates")
            sections.append(f"")
            sections.append(
                f"| Objective | Code | Assessed | Passed | Pass Rate |"
            )
            sections.append(
                f"|-----------|------|--------:|---------:|----------:|"
            )
            for sc in data.sc_pass_rates:
                sections.append(
                    f"| {sc.objective} | {sc.objective_code} | "
                    f"{sc.activities_assessed} | {sc.activities_passed} | "
                    f"{sc.pass_rate:.1f}% |"
                )
            sections.append(f"")

        # DNSH Matrix Summary
        if self.config.include_dnsh_matrix and data.dnsh_results:
            sections.append(f"## DNSH Matrix Summary")
            sections.append(f"")
            sections.append(
                f"| Objective | Code | Assessed | Passed | Pass Rate | Common Failures |"
            )
            sections.append(
                f"|-----------|------|--------:|---------:|----------:|-----------------|"
            )
            for dnsh in data.dnsh_results:
                failures = "; ".join(dnsh.common_failures[:2]) if dnsh.common_failures else "None"
                sections.append(
                    f"| {dnsh.objective} | {dnsh.objective_code} | "
                    f"{dnsh.activities_assessed} | {dnsh.activities_passed} | "
                    f"{dnsh.pass_rate:.1f}% | {failures} |"
                )
            sections.append(f"")

        # MS Verification Status
        if self.config.include_ms_status and data.ms_results:
            sections.append(f"## Minimum Safeguards Verification Status")
            sections.append(f"")
            sections.append(f"| Topic | Status | Verified | Compliant | Key Findings |")
            sections.append(f"|-------|--------|--------:|---------:|--------------|")
            for ms in data.ms_results:
                findings_str = "; ".join(ms.findings[:2]) if ms.findings else "None"
                sections.append(
                    f"| {ms.topic} | {ms.status} | "
                    f"{ms.activities_verified} | {ms.activities_compliant} | "
                    f"{findings_str} |"
                )
            sections.append(f"")

        # Activity-Level Detail
        if self.config.include_activity_detail and data.activities:
            sections.append(f"## Activity-Level Detail")
            sections.append(f"")
            display_activities = data.activities[:self.config.max_activities_detail]
            sections.append(
                f"| Activity | NACE | Eligible | SC | DNSH | MS | Aligned | "
                f"Turnover (EUR) |"
            )
            sections.append(
                f"|----------|------|----------|----|----- |----|---------|"
                f"----------------|"
            )
            for act in display_activities:
                sections.append(
                    f"| {act.activity_name[:35]} | {act.nace_code} | "
                    f"{'Y' if act.is_eligible else 'N'} | "
                    f"{'Y' if act.sc_pass else 'N'} | "
                    f"{'Y' if act.dnsh_pass else 'N'} | "
                    f"{'Y' if act.ms_pass else 'N'} | "
                    f"{'Y' if act.is_aligned else 'N'} | "
                    f"{act.turnover_eur:,.0f} |"
                )
            if len(data.activities) > self.config.max_activities_detail:
                sections.append(
                    f"| ... | *{len(data.activities) - self.config.max_activities_detail} "
                    f"more* | | | | | | |"
                )
            sections.append(f"")

        # Key Findings
        if data.key_findings:
            sections.append(f"## Key Findings")
            sections.append(f"")
            for idx, finding in enumerate(data.key_findings, 1):
                sections.append(f"{idx}. {finding}")
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
        align_rate = (data.aligned_activities / data.total_activities * 100) if data.total_activities > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy Alignment Summary - {data.organization_name}</title>
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
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy Alignment Summary Report</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Alignment Rate:</strong> <span class="metric">{align_rate:.1f}%</span></p>
    </div>

    <h2>Alignment Overview</h2>
    <p>
        Of <strong>{data.total_activities}</strong> economic activities,
        <strong>{data.aligned_activities}</strong> ({align_rate:.1f}%) are fully taxonomy-aligned.
    </p>
"""

        if data.sc_pass_rates:
            html += """    <h2>Substantial Contribution Pass Rates</h2>
    <table>
        <tr><th>Objective</th><th>Code</th><th>Assessed</th><th>Passed</th><th>Pass Rate</th></tr>
"""
            for sc in data.sc_pass_rates:
                html += f"""        <tr>
            <td>{sc.objective}</td><td>{sc.objective_code}</td>
            <td>{sc.activities_assessed}</td><td>{sc.activities_passed}</td>
            <td>{sc.pass_rate:.1f}%</td>
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
            "report_type": "alignment_summary",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_activities": data.total_activities,
                "eligible_activities": data.eligible_activities,
                "aligned_activities": data.aligned_activities,
                "alignment_rate": (
                    data.aligned_activities / data.total_activities * 100
                ) if data.total_activities > 0 else 0,
                "alignment_to_eligibility_rate": (
                    data.aligned_activities / data.eligible_activities * 100
                ) if data.eligible_activities > 0 else 0,
            },
            "sc_pass_rates": [sc.dict() for sc in data.sc_pass_rates],
            "dnsh_results": [dnsh.dict() for dnsh in data.dnsh_results],
            "ms_results": [ms.dict() for ms in data.ms_results],
            "activities": [act.dict() for act in data.activities],
            "key_findings": data.key_findings,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "AlignmentSummaryReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
