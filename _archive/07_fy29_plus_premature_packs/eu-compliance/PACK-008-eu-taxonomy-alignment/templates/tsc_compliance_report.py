"""
TSC Compliance Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates per-activity Technical Screening Criteria (TSC) results including
evidence status for each criterion, non-compliance details with remediation suggestions,
and criteria checklists per environmental objective.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from tsc_compliance_report import TSCComplianceReportTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = TSCComplianceReportTemplate()
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
    """Configuration for TSC Compliance Report generation."""

    include_evidence_inventory: bool = Field(
        default=True,
        description="Include evidence inventory section"
    )
    include_remediation_actions: bool = Field(
        default=True,
        description="Include remediation actions for non-compliant criteria"
    )
    include_criteria_checklist: bool = Field(
        default=True,
        description="Include criteria checklist per activity"
    )
    max_activities_detail: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum activities to show in detail"
    )


class CriterionResult(BaseModel):
    """Result for a single technical screening criterion."""

    criterion_id: str = Field(..., description="Criterion identifier")
    criterion_description: str = Field(..., description="Criterion description")
    criterion_type: str = Field(
        default="quantitative",
        description="Type: quantitative or qualitative"
    )
    threshold: str = Field(default="", description="Required threshold or standard")
    actual_value: str = Field(default="", description="Actual measured value")
    status: str = Field(
        ...,
        description="Status: PASS, FAIL, PARTIAL, or NOT_ASSESSED"
    )
    evidence_status: str = Field(
        default="MISSING",
        description="Evidence status: COMPLETE, PARTIAL, MISSING"
    )
    evidence_documents: List[str] = Field(
        default_factory=list,
        description="List of evidence document references"
    )
    non_compliance_reason: str = Field(
        default="",
        description="Reason for non-compliance if FAIL"
    )
    remediation_suggestion: str = Field(
        default="",
        description="Suggested remediation action"
    )


class ActivityTSCResult(BaseModel):
    """TSC evaluation result for a single activity."""

    activity_id: str = Field(..., description="Taxonomy activity identifier")
    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field(..., description="NACE sector code")
    sc_objective: str = Field(
        ...,
        description="SC objective assessed (CCM, CCA, etc.)"
    )
    delegated_act: str = Field(
        default="Climate DA",
        description="Applicable Delegated Act"
    )
    da_article: str = Field(
        default="",
        description="Delegated Act article reference"
    )
    overall_status: str = Field(
        ...,
        description="Overall TSC status: COMPLIANT, NON_COMPLIANT, PARTIAL"
    )
    criteria: List[CriterionResult] = Field(
        default_factory=list,
        description="Individual criterion results"
    )
    total_criteria: int = Field(default=0, ge=0, description="Total criteria count")
    passed_criteria: int = Field(default=0, ge=0, description="Passed criteria count")
    failed_criteria: int = Field(default=0, ge=0, description="Failed criteria count")
    turnover_eur: float = Field(default=0.0, ge=0, description="Activity turnover")


class EvidenceItem(BaseModel):
    """Evidence document in the inventory."""

    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document name or title")
    document_type: str = Field(
        ...,
        description="Type: certification, report, measurement, audit, policy"
    )
    activities_covered: List[str] = Field(
        default_factory=list,
        description="Activity IDs this document covers"
    )
    criteria_covered: List[str] = Field(
        default_factory=list,
        description="Criterion IDs this document supports"
    )
    status: str = Field(
        default="VALID",
        description="Status: VALID, EXPIRED, PENDING, MISSING"
    )
    expiry_date: str = Field(default="", description="Document expiry date")
    source: str = Field(default="", description="Document source")


class ReportData(BaseModel):
    """Data model for TSC Compliance Report."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    activity_results: List[ActivityTSCResult] = Field(
        default_factory=list,
        description="Activity-level TSC results"
    )
    evidence_inventory: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence document inventory"
    )
    total_activities: int = Field(default=0, ge=0, description="Total activities assessed")
    compliant_activities: int = Field(default=0, ge=0, description="Compliant activities")
    non_compliant_activities: int = Field(default=0, ge=0, description="Non-compliant activities")
    notes: List[str] = Field(default_factory=list, description="Report notes")

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class TSCComplianceReportTemplate:
    """
    TSC Compliance Report Template for EU Taxonomy Alignment Pack.

    Generates per-activity technical screening criteria results with evidence status,
    non-compliance details, and remediation suggestions.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = TSCComplianceReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "TSC Overview" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize TSC Compliance Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the TSC compliance report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering TSC Compliance Report for {data.organization_name} "
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
        sections.append(f"# EU Taxonomy TSC Compliance Report")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # TSC Overview
        compliance_rate = (data.compliant_activities / data.total_activities * 100) if data.total_activities > 0 else 0

        sections.append(f"## TSC Overview")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|------:|")
        sections.append(f"| Total Activities Assessed | {data.total_activities} |")
        sections.append(f"| Compliant Activities | {data.compliant_activities} |")
        sections.append(f"| Non-Compliant Activities | {data.non_compliant_activities} |")
        sections.append(f"| Compliance Rate | {compliance_rate:.1f}% |")
        sections.append(f"")

        # Activity-Level Results
        if data.activity_results:
            sections.append(f"## Activity-Level Results")
            sections.append(f"")
            sections.append(
                f"| Activity | NACE | Objective | DA Article | Status | "
                f"Passed | Failed | Total | Turnover (EUR) |"
            )
            sections.append(
                f"|----------|------|-----------|------------|--------|"
                f"------:|-------:|------:|--------------:|"
            )
            for act in data.activity_results[:self.config.max_activities_detail]:
                sections.append(
                    f"| {act.activity_name[:30]} | {act.nace_code} | "
                    f"{act.sc_objective} | {act.da_article} | "
                    f"{act.overall_status} | "
                    f"{act.passed_criteria} | {act.failed_criteria} | "
                    f"{act.total_criteria} | {act.turnover_eur:,.0f} |"
                )
            sections.append(f"")

        # Criteria Checklist per activity
        if self.config.include_criteria_checklist and data.activity_results:
            sections.append(f"## Criteria Checklist")
            sections.append(f"")
            for act in data.activity_results[:self.config.max_activities_detail]:
                if not act.criteria:
                    continue
                sections.append(f"### {act.activity_name} ({act.sc_objective})")
                sections.append(f"")
                sections.append(
                    f"| Criterion | Type | Threshold | Actual | Status | Evidence |"
                )
                sections.append(
                    f"|-----------|------|-----------|--------|--------|----------|"
                )
                for crit in act.criteria:
                    sections.append(
                        f"| {crit.criterion_description[:40]} | "
                        f"{crit.criterion_type[:5]} | "
                        f"{crit.threshold[:15]} | "
                        f"{crit.actual_value[:15]} | "
                        f"{crit.status} | {crit.evidence_status} |"
                    )
                sections.append(f"")

        # Evidence Inventory
        if self.config.include_evidence_inventory and data.evidence_inventory:
            sections.append(f"## Evidence Inventory")
            sections.append(f"")
            sections.append(
                f"| Document ID | Name | Type | Status | "
                f"Activities | Expiry |"
            )
            sections.append(
                f"|-------------|------|------|--------|"
                f"-----------|--------|"
            )
            for ev in data.evidence_inventory:
                act_count = len(ev.activities_covered)
                sections.append(
                    f"| {ev.document_id} | {ev.document_name[:30]} | "
                    f"{ev.document_type} | {ev.status} | "
                    f"{act_count} | {ev.expiry_date} |"
                )
            sections.append(f"")

        # Non-Compliance Summary
        non_compliant = [
            act for act in data.activity_results
            if act.overall_status in ("NON_COMPLIANT", "PARTIAL")
        ]
        if non_compliant:
            sections.append(f"## Non-Compliance Summary")
            sections.append(f"")
            for act in non_compliant:
                failed = [c for c in act.criteria if c.status == "FAIL"]
                if not failed:
                    continue
                sections.append(f"### {act.activity_name}")
                sections.append(f"")
                for crit in failed:
                    sections.append(f"- **{crit.criterion_id}:** {crit.criterion_description}")
                    if crit.non_compliance_reason:
                        sections.append(f"  - Reason: {crit.non_compliance_reason}")
                    if crit.remediation_suggestion:
                        sections.append(f"  - Remediation: {crit.remediation_suggestion}")
                sections.append(f"")

        # Remediation Actions
        if self.config.include_remediation_actions and non_compliant:
            sections.append(f"## Remediation Actions")
            sections.append(f"")
            sections.append(
                f"| Activity | Criterion | Remediation Action |"
            )
            sections.append(
                f"|----------|-----------|-------------------|"
            )
            for act in non_compliant:
                for crit in act.criteria:
                    if crit.status == "FAIL" and crit.remediation_suggestion:
                        sections.append(
                            f"| {act.activity_name[:25]} | "
                            f"{crit.criterion_id} | "
                            f"{crit.remediation_suggestion[:50]} |"
                        )
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
        compliance_rate = (data.compliant_activities / data.total_activities * 100) if data.total_activities > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy TSC Compliance - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #27ae60; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .partial {{ color: #f39c12; font-weight: bold; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy TSC Compliance Report</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Compliance Rate:</strong> <span class="pass">{compliance_rate:.1f}%</span></p>
    </div>
"""

        if data.activity_results:
            html += """    <h2>Activity-Level Results</h2>
    <table>
        <tr><th>Activity</th><th>NACE</th><th>Objective</th><th>Status</th>
        <th>Passed</th><th>Failed</th><th>Total</th></tr>
"""
            for act in data.activity_results[:self.config.max_activities_detail]:
                status_class = act.overall_status.lower().replace("non_compliant", "fail")
                html += f"""        <tr>
            <td>{act.activity_name}</td><td>{act.nace_code}</td>
            <td>{act.sc_objective}</td>
            <td class="{status_class}">{act.overall_status}</td>
            <td>{act.passed_criteria}</td><td>{act.failed_criteria}</td>
            <td>{act.total_criteria}</td>
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
            "report_type": "tsc_compliance",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_activities": data.total_activities,
                "compliant_activities": data.compliant_activities,
                "non_compliant_activities": data.non_compliant_activities,
                "compliance_rate": (
                    data.compliant_activities / data.total_activities * 100
                ) if data.total_activities > 0 else 0,
            },
            "activity_results": [act.dict() for act in data.activity_results],
            "evidence_inventory": [ev.dict() for ev in data.evidence_inventory],
            "notes": data.notes,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "TSCComplianceReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
