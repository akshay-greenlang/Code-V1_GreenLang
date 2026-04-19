"""
DNSH Assessment Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates Do No Significant Harm (DNSH) assessment reports with a 6-objective
matrix visualization, climate risk assessment results (for CCA DNSH), water/circular
economy/pollution/biodiversity assessment results, and evidence links per objective.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from dnsh_assessment_report import DNSHAssessmentReportTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = DNSHAssessmentReportTemplate()
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
    """Configuration for DNSH Assessment Report generation."""

    include_climate_risk: bool = Field(
        default=True,
        description="Include climate risk assessment section (CCA DNSH)"
    )
    include_water_assessment: bool = Field(
        default=True,
        description="Include water assessment section"
    )
    include_circular_economy: bool = Field(
        default=True,
        description="Include circular economy assessment"
    )
    include_pollution: bool = Field(
        default=True,
        description="Include pollution assessment"
    )
    include_biodiversity: bool = Field(
        default=True,
        description="Include biodiversity assessment"
    )
    include_evidence_links: bool = Field(
        default=True,
        description="Include evidence links per objective"
    )
    max_activities_detail: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum activities to show in detail"
    )


class ObjectiveAssessment(BaseModel):
    """DNSH assessment result for a single objective within an activity."""

    objective_code: str = Field(..., description="Objective code (CCM, CCA, WTR, CE, PPC, BIO)")
    objective_name: str = Field(..., description="Full objective name")
    is_applicable: bool = Field(default=True, description="Whether DNSH is applicable")
    status: str = Field(
        ...,
        description="Status: PASS, FAIL, NOT_APPLICABLE, NOT_ASSESSED"
    )
    criteria_description: str = Field(
        default="",
        description="DNSH criteria description"
    )
    assessment_notes: str = Field(
        default="",
        description="Assessment notes or findings"
    )
    evidence_documents: List[str] = Field(
        default_factory=list,
        description="Evidence document references"
    )
    failure_reason: str = Field(
        default="",
        description="Reason for failure if FAIL"
    )


class ActivityDNSHResult(BaseModel):
    """DNSH assessment result for a single activity."""

    activity_id: str = Field(..., description="Taxonomy activity identifier")
    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field(..., description="NACE sector code")
    sc_objective: str = Field(
        ...,
        description="Substantial Contribution objective"
    )
    overall_dnsh_status: str = Field(
        ...,
        description="Overall DNSH status: PASS, FAIL, PARTIAL"
    )
    objective_results: List[ObjectiveAssessment] = Field(
        default_factory=list,
        description="Per-objective DNSH results"
    )
    objectives_passed: int = Field(default=0, ge=0, description="Objectives passed")
    objectives_failed: int = Field(default=0, ge=0, description="Objectives failed")
    objectives_na: int = Field(default=0, ge=0, description="Objectives not applicable")
    turnover_eur: float = Field(default=0.0, ge=0, description="Activity turnover")


class ClimateRiskResult(BaseModel):
    """Climate risk assessment result for CCA DNSH."""

    activity_name: str = Field(..., description="Activity name")
    physical_risk_type: str = Field(
        default="",
        description="Type of physical climate risk (acute, chronic)"
    )
    hazards_identified: List[str] = Field(
        default_factory=list,
        description="Climate hazards identified"
    )
    vulnerability_level: str = Field(
        default="LOW",
        description="Vulnerability level: LOW, MEDIUM, HIGH"
    )
    adaptation_measures: List[str] = Field(
        default_factory=list,
        description="Adaptation measures in place"
    )
    risk_assessment_status: str = Field(
        default="COMPLETED",
        description="Status: COMPLETED, IN_PROGRESS, NOT_STARTED"
    )
    climate_scenario: str = Field(
        default="RCP 8.5",
        description="Climate scenario used"
    )


class ObjectiveSummary(BaseModel):
    """Summary statistics for a single environmental objective."""

    objective_code: str = Field(..., description="Objective code")
    objective_name: str = Field(..., description="Objective full name")
    total_assessed: int = Field(default=0, ge=0, description="Total activities assessed")
    passed: int = Field(default=0, ge=0, description="Activities passed")
    failed: int = Field(default=0, ge=0, description="Activities failed")
    not_applicable: int = Field(default=0, ge=0, description="Not applicable")
    pass_rate: float = Field(default=0.0, ge=0, le=100, description="Pass rate")
    common_failures: List[str] = Field(
        default_factory=list,
        description="Common failure reasons"
    )


class ReportData(BaseModel):
    """Data model for DNSH Assessment Report."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    activity_results: List[ActivityDNSHResult] = Field(
        default_factory=list,
        description="Activity-level DNSH results"
    )
    objective_summaries: List[ObjectiveSummary] = Field(
        default_factory=list,
        description="Per-objective summary statistics"
    )
    climate_risk_results: List[ClimateRiskResult] = Field(
        default_factory=list,
        description="Climate risk assessment results"
    )
    total_activities: int = Field(default=0, ge=0, description="Total activities assessed")
    overall_pass_count: int = Field(default=0, ge=0, description="Activities passing all DNSH")
    overall_fail_count: int = Field(default=0, ge=0, description="Activities failing one or more DNSH")
    notes: List[str] = Field(default_factory=list, description="Report notes")

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class DNSHAssessmentReportTemplate:
    """
    DNSH Assessment Report Template for EU Taxonomy Alignment Pack.

    Generates 6-objective DNSH matrix visualizations, climate risk assessments,
    water/CE/pollution/biodiversity results, and evidence links.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = DNSHAssessmentReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "DNSH Summary" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize DNSH Assessment Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the DNSH assessment report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering DNSH Assessment Report for {data.organization_name} "
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
        sections.append(f"# EU Taxonomy DNSH Assessment Report")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # DNSH Summary
        pass_rate = (data.overall_pass_count / data.total_activities * 100) if data.total_activities > 0 else 0

        sections.append(f"## DNSH Summary")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|------:|")
        sections.append(f"| Total Activities Assessed | {data.total_activities} |")
        sections.append(f"| Pass All DNSH Objectives | {data.overall_pass_count} |")
        sections.append(f"| Fail One or More DNSH | {data.overall_fail_count} |")
        sections.append(f"| Overall DNSH Pass Rate | {pass_rate:.1f}% |")
        sections.append(f"")

        # Matrix View (6x6)
        if data.activity_results:
            sections.append(f"## DNSH Matrix View")
            sections.append(f"")
            sections.append(
                f"DNSH assessment results per activity and environmental objective "
                f"(SC = Substantial Contribution objective, not assessed for DNSH):"
            )
            sections.append(f"")
            sections.append(
                f"| Activity | SC Obj | CCM | CCA | WTR | CE | PPC | BIO | Overall |"
            )
            sections.append(
                f"|----------|--------|-----|-----|-----|----|-----|-----|---------|"
            )
            for act in data.activity_results[:self.config.max_activities_detail]:
                obj_map = {o.objective_code: o.status for o in act.objective_results}
                sections.append(
                    f"| {act.activity_name[:30]} | {act.sc_objective} | "
                    f"{self._status_symbol(obj_map.get('CCM', 'N/A'))} | "
                    f"{self._status_symbol(obj_map.get('CCA', 'N/A'))} | "
                    f"{self._status_symbol(obj_map.get('WTR', 'N/A'))} | "
                    f"{self._status_symbol(obj_map.get('CE', 'N/A'))} | "
                    f"{self._status_symbol(obj_map.get('PPC', 'N/A'))} | "
                    f"{self._status_symbol(obj_map.get('BIO', 'N/A'))} | "
                    f"{act.overall_dnsh_status} |"
                )
            sections.append(f"")
            sections.append(
                f"*Legend: P = Pass, F = Fail, SC = Substantial Contribution objective (N/A for DNSH), - = Not Applicable*"
            )
            sections.append(f"")

        # Per-Objective Results
        if data.objective_summaries:
            sections.append(f"## Per-Objective Results")
            sections.append(f"")
            sections.append(
                f"| Objective | Code | Assessed | Passed | Failed | N/A | Pass Rate |"
            )
            sections.append(
                f"|-----------|------|--------:|-------:|-------:|----:|----------:|"
            )
            for obj in data.objective_summaries:
                sections.append(
                    f"| {obj.objective_name} | {obj.objective_code} | "
                    f"{obj.total_assessed} | {obj.passed} | "
                    f"{obj.failed} | {obj.not_applicable} | "
                    f"{obj.pass_rate:.1f}% |"
                )
            sections.append(f"")

            # Common failures per objective
            for obj in data.objective_summaries:
                if obj.common_failures:
                    sections.append(f"### {obj.objective_name} ({obj.objective_code}) - Common Failures")
                    sections.append(f"")
                    for idx, failure in enumerate(obj.common_failures, 1):
                        sections.append(f"{idx}. {failure}")
                    sections.append(f"")

        # Climate Risk Assessment
        if self.config.include_climate_risk and data.climate_risk_results:
            sections.append(f"## Climate Risk Assessment (CCA DNSH)")
            sections.append(f"")
            sections.append(
                f"Climate risk and vulnerability assessment results per "
                f"Appendix A of the Climate Delegated Act:"
            )
            sections.append(f"")
            sections.append(
                f"| Activity | Risk Type | Vulnerability | Scenario | "
                f"Status | Hazards |"
            )
            sections.append(
                f"|----------|-----------|---------------|----------|"
                f"--------|---------|"
            )
            for cr in data.climate_risk_results:
                hazards_str = ", ".join(cr.hazards_identified[:3])
                sections.append(
                    f"| {cr.activity_name[:25]} | {cr.physical_risk_type} | "
                    f"{cr.vulnerability_level} | {cr.climate_scenario} | "
                    f"{cr.risk_assessment_status} | {hazards_str} |"
                )
            sections.append(f"")

            # Adaptation measures
            for cr in data.climate_risk_results:
                if cr.adaptation_measures:
                    sections.append(f"### Adaptation Measures: {cr.activity_name}")
                    sections.append(f"")
                    for measure in cr.adaptation_measures:
                        sections.append(f"- {measure}")
                    sections.append(f"")

        # Water Assessment
        if self.config.include_water_assessment:
            wtr_results = [
                act for act in data.activity_results
                if any(o.objective_code == "WTR" and o.is_applicable for o in act.objective_results)
            ]
            if wtr_results:
                sections.append(f"## Water and Marine Resources Assessment (WTR)")
                sections.append(f"")
                sections.append(
                    f"| Activity | WTR DNSH Status | Key Criteria | Notes |"
                )
                sections.append(
                    f"|----------|----------------|--------------|-------|"
                )
                for act in wtr_results:
                    wtr_obj = next(
                        (o for o in act.objective_results if o.objective_code == "WTR"), None
                    )
                    if wtr_obj:
                        sections.append(
                            f"| {act.activity_name[:30]} | {wtr_obj.status} | "
                            f"{wtr_obj.criteria_description[:35]} | "
                            f"{wtr_obj.assessment_notes[:30]} |"
                        )
                sections.append(f"")

        # Circular Economy Assessment
        if self.config.include_circular_economy:
            ce_results = [
                act for act in data.activity_results
                if any(o.objective_code == "CE" and o.is_applicable for o in act.objective_results)
            ]
            if ce_results:
                sections.append(f"## Circular Economy Assessment (CE)")
                sections.append(f"")
                sections.append(f"| Activity | CE DNSH Status | Notes |")
                sections.append(f"|----------|---------------|-------|")
                for act in ce_results:
                    ce_obj = next(
                        (o for o in act.objective_results if o.objective_code == "CE"), None
                    )
                    if ce_obj:
                        sections.append(
                            f"| {act.activity_name[:35]} | {ce_obj.status} | "
                            f"{ce_obj.assessment_notes[:40]} |"
                        )
                sections.append(f"")

        # Pollution Assessment
        if self.config.include_pollution:
            ppc_results = [
                act for act in data.activity_results
                if any(o.objective_code == "PPC" and o.is_applicable for o in act.objective_results)
            ]
            if ppc_results:
                sections.append(f"## Pollution Prevention and Control Assessment (PPC)")
                sections.append(f"")
                sections.append(f"| Activity | PPC DNSH Status | Notes |")
                sections.append(f"|----------|----------------|-------|")
                for act in ppc_results:
                    ppc_obj = next(
                        (o for o in act.objective_results if o.objective_code == "PPC"), None
                    )
                    if ppc_obj:
                        sections.append(
                            f"| {act.activity_name[:35]} | {ppc_obj.status} | "
                            f"{ppc_obj.assessment_notes[:40]} |"
                        )
                sections.append(f"")

        # Biodiversity Assessment
        if self.config.include_biodiversity:
            bio_results = [
                act for act in data.activity_results
                if any(o.objective_code == "BIO" and o.is_applicable for o in act.objective_results)
            ]
            if bio_results:
                sections.append(f"## Biodiversity and Ecosystems Assessment (BIO)")
                sections.append(f"")
                sections.append(f"| Activity | BIO DNSH Status | Notes |")
                sections.append(f"|----------|----------------|-------|")
                for act in bio_results:
                    bio_obj = next(
                        (o for o in act.objective_results if o.objective_code == "BIO"), None
                    )
                    if bio_obj:
                        sections.append(
                            f"| {act.activity_name[:35]} | {bio_obj.status} | "
                            f"{bio_obj.assessment_notes[:40]} |"
                        )
                sections.append(f"")

        # Evidence Links
        if self.config.include_evidence_links and data.activity_results:
            all_evidence: List[str] = []
            for act in data.activity_results:
                for obj in act.objective_results:
                    all_evidence.extend(obj.evidence_documents)
            unique_evidence = sorted(set(all_evidence))

            if unique_evidence:
                sections.append(f"## Evidence Links")
                sections.append(f"")
                for idx, doc in enumerate(unique_evidence, 1):
                    sections.append(f"{idx}. {doc}")
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
        pass_rate = (data.overall_pass_count / data.total_activities * 100) if data.total_activities > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy DNSH Assessment - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #27ae60; color: white; }}
        td:first-child {{ text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .na {{ color: #95a5a6; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy DNSH Assessment Report</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>DNSH Pass Rate:</strong> <span class="pass">{pass_rate:.1f}%</span></p>
    </div>
"""

        if data.activity_results:
            html += """    <h2>DNSH Matrix View</h2>
    <table>
        <tr><th>Activity</th><th>SC Obj</th><th>CCM</th><th>CCA</th><th>WTR</th>
        <th>CE</th><th>PPC</th><th>BIO</th><th>Overall</th></tr>
"""
            for act in data.activity_results[:self.config.max_activities_detail]:
                obj_map = {o.objective_code: o.status for o in act.objective_results}
                html += f"""        <tr>
            <td>{act.activity_name}</td><td>{act.sc_objective}</td>
            <td class="{self._html_class(obj_map.get('CCM', 'N/A'))}">{self._status_symbol(obj_map.get('CCM', 'N/A'))}</td>
            <td class="{self._html_class(obj_map.get('CCA', 'N/A'))}">{self._status_symbol(obj_map.get('CCA', 'N/A'))}</td>
            <td class="{self._html_class(obj_map.get('WTR', 'N/A'))}">{self._status_symbol(obj_map.get('WTR', 'N/A'))}</td>
            <td class="{self._html_class(obj_map.get('CE', 'N/A'))}">{self._status_symbol(obj_map.get('CE', 'N/A'))}</td>
            <td class="{self._html_class(obj_map.get('PPC', 'N/A'))}">{self._status_symbol(obj_map.get('PPC', 'N/A'))}</td>
            <td class="{self._html_class(obj_map.get('BIO', 'N/A'))}">{self._status_symbol(obj_map.get('BIO', 'N/A'))}</td>
            <td class="{self._html_class(act.overall_dnsh_status)}">{act.overall_dnsh_status}</td>
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
            "report_type": "dnsh_assessment",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_activities": data.total_activities,
                "overall_pass_count": data.overall_pass_count,
                "overall_fail_count": data.overall_fail_count,
                "pass_rate": (
                    data.overall_pass_count / data.total_activities * 100
                ) if data.total_activities > 0 else 0,
            },
            "objective_summaries": [obj.dict() for obj in data.objective_summaries],
            "activity_results": [act.dict() for act in data.activity_results],
            "climate_risk_results": [cr.dict() for cr in data.climate_risk_results],
            "notes": data.notes,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "DNSHAssessmentReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _status_symbol(self, status: str) -> str:
        """Convert status to display symbol."""
        status_map = {
            "PASS": "P",
            "FAIL": "F",
            "NOT_APPLICABLE": "-",
            "NOT_ASSESSED": "?",
            "N/A": "SC",
        }
        return status_map.get(status, status)

    def _html_class(self, status: str) -> str:
        """Get HTML CSS class for a status value."""
        class_map = {
            "PASS": "pass",
            "FAIL": "fail",
            "NOT_APPLICABLE": "na",
            "NOT_ASSESSED": "na",
            "N/A": "na",
        }
        return class_map.get(status, "")
