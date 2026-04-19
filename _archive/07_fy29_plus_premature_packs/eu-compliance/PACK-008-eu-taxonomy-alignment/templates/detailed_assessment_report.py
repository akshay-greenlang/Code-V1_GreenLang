"""
Detailed Assessment Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates full audit trail reports with activity-level detail for every
assessment result (SC, DNSH, MS, KPI), evidence inventories, provenance hashes,
assumptions and methodology documentation, and appendices for EU Taxonomy compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for every assessment record to ensure full traceability.

Example:
    >>> from detailed_assessment_report import DetailedAssessmentReportTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = DetailedAssessmentReportTemplate()
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
    """Configuration for Detailed Assessment Report generation."""

    include_evidence_inventory: bool = Field(
        default=True,
        description="Include full evidence inventory"
    )
    include_provenance_trail: bool = Field(
        default=True,
        description="Include provenance hash trail"
    )
    include_assumptions: bool = Field(
        default=True,
        description="Include assumptions and methodology"
    )
    include_appendices: bool = Field(
        default=True,
        description="Include appendices"
    )
    max_activities_per_section: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Maximum activities per section"
    )
    generate_per_activity_hash: bool = Field(
        default=True,
        description="Generate SHA-256 hash per activity assessment"
    )


class SCDetail(BaseModel):
    """Substantial Contribution assessment detail."""

    objective: str = Field(..., description="SC objective (CCM, CCA, etc.)")
    criteria_count: int = Field(default=0, ge=0, description="Number of criteria evaluated")
    criteria_passed: int = Field(default=0, ge=0, description="Criteria passed")
    status: str = Field(..., description="PASS, FAIL, NOT_ASSESSED")
    key_criteria: List[str] = Field(
        default_factory=list,
        description="Key criteria evaluated"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Evidence document references"
    )
    notes: str = Field(default="", description="Assessment notes")


class DNSHDetail(BaseModel):
    """DNSH assessment detail per objective."""

    objective_code: str = Field(..., description="Objective code")
    objective_name: str = Field(..., description="Full objective name")
    status: str = Field(..., description="PASS, FAIL, NOT_APPLICABLE")
    criteria_description: str = Field(default="", description="Criteria applied")
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Evidence references"
    )
    notes: str = Field(default="", description="Assessment notes")


class MSDetail(BaseModel):
    """Minimum Safeguards assessment detail."""

    topic: str = Field(
        ...,
        description="MS topic (Human Rights, Anti-Corruption, Taxation, Fair Competition)"
    )
    status: str = Field(..., description="PASS, FAIL, PARTIAL")
    procedures_in_place: bool = Field(default=False, description="Procedures verified")
    outcome_assessment: str = Field(default="", description="Outcome assessment summary")
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Evidence references"
    )
    notes: str = Field(default="", description="Assessment notes")


class KPIDetail(BaseModel):
    """KPI calculation detail for an activity."""

    turnover_eur: float = Field(default=0.0, ge=0, description="Activity turnover")
    turnover_pct: float = Field(default=0.0, ge=0, le=100, description="% of total turnover")
    capex_eur: float = Field(default=0.0, ge=0, description="Activity CapEx")
    capex_pct: float = Field(default=0.0, ge=0, le=100, description="% of total CapEx")
    opex_eur: float = Field(default=0.0, ge=0, description="Activity OpEx")
    opex_pct: float = Field(default=0.0, ge=0, le=100, description="% of total OpEx")
    allocation_method: str = Field(
        default="direct",
        description="Allocation method (direct, pro-rata, estimated)"
    )
    data_source: str = Field(default="", description="Financial data source")


class ActivityAssessment(BaseModel):
    """Complete assessment record for a single economic activity."""

    activity_id: str = Field(..., description="Taxonomy activity identifier")
    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field(..., description="NACE code")
    nace_description: str = Field(default="", description="NACE description")
    delegated_act: str = Field(default="", description="Applicable Delegated Act")
    da_article: str = Field(default="", description="DA article reference")
    is_eligible: bool = Field(..., description="Eligibility status")
    is_aligned: bool = Field(default=False, description="Overall alignment status")
    sc_detail: Optional[SCDetail] = Field(None, description="SC assessment detail")
    dnsh_details: List[DNSHDetail] = Field(
        default_factory=list,
        description="DNSH assessment details per objective"
    )
    ms_details: List[MSDetail] = Field(
        default_factory=list,
        description="MS assessment details per topic"
    )
    kpi_detail: Optional[KPIDetail] = Field(None, description="KPI calculation detail")
    assessment_date: str = Field(default="", description="Date of assessment")
    assessor: str = Field(default="", description="Assessor name or system")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EvidenceDocument(BaseModel):
    """Evidence document in the inventory."""

    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document name")
    document_type: str = Field(
        ...,
        description="Type: certification, audit_report, measurement, policy, procedure"
    )
    issuer: str = Field(default="", description="Document issuer")
    issue_date: str = Field(default="", description="Issue date")
    expiry_date: str = Field(default="", description="Expiry date")
    status: str = Field(default="VALID", description="VALID, EXPIRED, PENDING")
    activities_covered: List[str] = Field(
        default_factory=list,
        description="Activity IDs covered"
    )
    hash_value: str = Field(default="", description="Document hash for integrity")


class ProvenanceRecord(BaseModel):
    """Provenance trail record."""

    record_id: str = Field(..., description="Record identifier")
    activity_id: str = Field(..., description="Activity identifier")
    operation: str = Field(
        ...,
        description="Operation type (eligibility_check, sc_assessment, etc.)"
    )
    timestamp: str = Field(..., description="Operation timestamp (ISO format)")
    input_hash: str = Field(..., description="SHA-256 hash of input data")
    output_hash: str = Field(..., description="SHA-256 hash of output data")
    agent: str = Field(default="", description="Agent/engine that performed the operation")
    version: str = Field(default="1.0.0", description="Agent/engine version")


class Assumption(BaseModel):
    """Documented assumption."""

    assumption_id: str = Field(..., description="Assumption identifier")
    category: str = Field(
        ...,
        description="Category: methodology, data, regulatory, financial"
    )
    description: str = Field(..., description="Assumption description")
    justification: str = Field(default="", description="Justification")
    impact: str = Field(
        default="LOW",
        description="Impact if assumption is wrong: LOW, MEDIUM, HIGH"
    )
    activities_affected: List[str] = Field(
        default_factory=list,
        description="Activity IDs affected"
    )


class ReportData(BaseModel):
    """Data model for Detailed Assessment Report."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    assessments: List[ActivityAssessment] = Field(
        default_factory=list,
        description="Complete activity assessments"
    )
    evidence_inventory: List[EvidenceDocument] = Field(
        default_factory=list,
        description="Evidence document inventory"
    )
    provenance_records: List[ProvenanceRecord] = Field(
        default_factory=list,
        description="Provenance trail records"
    )
    assumptions: List[Assumption] = Field(
        default_factory=list,
        description="Documented assumptions"
    )
    methodology_notes: List[str] = Field(
        default_factory=list,
        description="Methodology notes"
    )
    appendix_items: List[str] = Field(
        default_factory=list,
        description="Appendix references"
    )
    total_activities: int = Field(default=0, ge=0, description="Total activities")
    total_aligned: int = Field(default=0, ge=0, description="Total aligned")
    total_eligible: int = Field(default=0, ge=0, description="Total eligible")

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class DetailedAssessmentReportTemplate:
    """
    Detailed Assessment Report Template for EU Taxonomy Alignment Pack.

    Generates full audit trail reports with activity-level detail, evidence inventories,
    SHA-256 provenance hashes, assumptions, methodology, and appendices.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = DetailedAssessmentReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Assessment Overview" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Detailed Assessment Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the detailed assessment report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Detailed Assessment Report for {data.organization_name} "
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
        sections.append(f"# EU Taxonomy Detailed Assessment Report")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Report Hash:** {self._calculate_hash(data.organization_name + data.report_date)}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Assessment Overview
        align_rate = (data.total_aligned / data.total_activities * 100) if data.total_activities > 0 else 0

        sections.append(f"## Assessment Overview")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|------:|")
        sections.append(f"| Total Activities Assessed | {data.total_activities} |")
        sections.append(f"| Taxonomy-Eligible | {data.total_eligible} |")
        sections.append(f"| Taxonomy-Aligned | {data.total_aligned} |")
        sections.append(f"| Alignment Rate | {align_rate:.1f}% |")
        sections.append(f"| Evidence Documents | {len(data.evidence_inventory)} |")
        sections.append(f"| Provenance Records | {len(data.provenance_records)} |")
        sections.append(f"| Documented Assumptions | {len(data.assumptions)} |")
        sections.append(f"")

        # Activity Catalog
        if data.assessments:
            sections.append(f"## Activity Catalog")
            sections.append(f"")
            sections.append(
                f"| # | Activity ID | Activity Name | NACE | Eligible | "
                f"Aligned | SC | DNSH | MS |"
            )
            sections.append(
                f"|--:|-------------|---------------|------|----------|"
                f"---------|----|----- |----|"
            )
            for idx, act in enumerate(
                data.assessments[:self.config.max_activities_per_section], 1
            ):
                sc_status = act.sc_detail.status if act.sc_detail else "N/A"
                dnsh_pass = all(
                    d.status in ("PASS", "NOT_APPLICABLE") for d in act.dnsh_details
                ) if act.dnsh_details else False
                ms_pass = all(
                    m.status == "PASS" for m in act.ms_details
                ) if act.ms_details else False
                sections.append(
                    f"| {idx} | {act.activity_id} | {act.activity_name[:30]} | "
                    f"{act.nace_code} | "
                    f"{'Y' if act.is_eligible else 'N'} | "
                    f"{'Y' if act.is_aligned else 'N'} | "
                    f"{sc_status[:4]} | "
                    f"{'Y' if dnsh_pass else 'N'} | "
                    f"{'Y' if ms_pass else 'N'} |"
                )
            sections.append(f"")

        # Per-Activity Assessment Detail
        if data.assessments:
            sections.append(f"## Per-Activity Assessment Detail")
            sections.append(f"")
            for act in data.assessments[:self.config.max_activities_per_section]:
                act_hash = self._generate_activity_hash(act)

                sections.append(f"### {act.activity_id}: {act.activity_name}")
                sections.append(f"")
                sections.append(f"**NACE:** {act.nace_code} ({act.nace_description})")
                sections.append(f"**Delegated Act:** {act.delegated_act} ({act.da_article})")
                sections.append(f"**Eligible:** {'Yes' if act.is_eligible else 'No'}")
                sections.append(f"**Aligned:** {'Yes' if act.is_aligned else 'No'}")
                sections.append(f"**Assessment Date:** {act.assessment_date}")
                sections.append(f"**Assessor:** {act.assessor}")
                sections.append(f"**Provenance Hash:** `{act_hash}`")
                sections.append(f"")

                # SC Detail
                if act.sc_detail:
                    sc = act.sc_detail
                    sections.append(f"#### Substantial Contribution ({sc.objective})")
                    sections.append(f"")
                    sections.append(f"- **Status:** {sc.status}")
                    sections.append(
                        f"- **Criteria:** {sc.criteria_passed}/{sc.criteria_count} passed"
                    )
                    if sc.key_criteria:
                        sections.append(f"- **Key Criteria:**")
                        for crit in sc.key_criteria:
                            sections.append(f"  - {crit}")
                    if sc.evidence_refs:
                        sections.append(
                            f"- **Evidence:** {', '.join(sc.evidence_refs)}"
                        )
                    if sc.notes:
                        sections.append(f"- **Notes:** {sc.notes}")
                    sections.append(f"")

                # DNSH Details
                if act.dnsh_details:
                    sections.append(f"#### DNSH Assessment")
                    sections.append(f"")
                    sections.append(f"| Objective | Status | Evidence |")
                    sections.append(f"|-----------|--------|----------|")
                    for dnsh in act.dnsh_details:
                        ev_str = ", ".join(dnsh.evidence_refs) if dnsh.evidence_refs else "None"
                        sections.append(
                            f"| {dnsh.objective_name} ({dnsh.objective_code}) | "
                            f"{dnsh.status} | {ev_str} |"
                        )
                    sections.append(f"")

                # MS Details
                if act.ms_details:
                    sections.append(f"#### Minimum Safeguards")
                    sections.append(f"")
                    sections.append(f"| Topic | Status | Procedures | Evidence |")
                    sections.append(f"|-------|--------|------------|----------|")
                    for ms in act.ms_details:
                        ev_str = ", ".join(ms.evidence_refs) if ms.evidence_refs else "None"
                        sections.append(
                            f"| {ms.topic} | {ms.status} | "
                            f"{'Yes' if ms.procedures_in_place else 'No'} | {ev_str} |"
                        )
                    sections.append(f"")

                # KPI Detail
                if act.kpi_detail:
                    kpi = act.kpi_detail
                    sections.append(f"#### Financial KPIs")
                    sections.append(f"")
                    sections.append(
                        f"- **Turnover:** EUR {kpi.turnover_eur:,.0f} "
                        f"({kpi.turnover_pct:.1f}% of total)"
                    )
                    sections.append(
                        f"- **CapEx:** EUR {kpi.capex_eur:,.0f} "
                        f"({kpi.capex_pct:.1f}% of total)"
                    )
                    sections.append(
                        f"- **OpEx:** EUR {kpi.opex_eur:,.0f} "
                        f"({kpi.opex_pct:.1f}% of total)"
                    )
                    sections.append(
                        f"- **Allocation Method:** {kpi.allocation_method}"
                    )
                    sections.append(f"- **Data Source:** {kpi.data_source}")
                    sections.append(f"")

                sections.append(f"---")
                sections.append(f"")

        # Evidence Inventory
        if self.config.include_evidence_inventory and data.evidence_inventory:
            sections.append(f"## Evidence Inventory")
            sections.append(f"")
            sections.append(
                f"| Doc ID | Name | Type | Issuer | Status | "
                f"Expiry | Activities | Hash |"
            )
            sections.append(
                f"|--------|------|------|--------|--------|"
                f"--------|-----------|------|"
            )
            for ev in data.evidence_inventory:
                sections.append(
                    f"| {ev.document_id} | {ev.document_name[:25]} | "
                    f"{ev.document_type} | {ev.issuer[:15]} | "
                    f"{ev.status} | {ev.expiry_date} | "
                    f"{len(ev.activities_covered)} | "
                    f"`{ev.hash_value[:12]}...` |"
                )
            sections.append(f"")

        # Provenance Trail
        if self.config.include_provenance_trail and data.provenance_records:
            sections.append(f"## Provenance Trail")
            sections.append(f"")
            sections.append(
                f"Complete audit trail with SHA-256 hashes for data integrity verification:"
            )
            sections.append(f"")
            sections.append(
                f"| Record ID | Activity | Operation | Timestamp | "
                f"Input Hash | Output Hash | Agent |"
            )
            sections.append(
                f"|-----------|----------|-----------|-----------|"
                f"-----------|-------------|-------|"
            )
            for rec in data.provenance_records:
                sections.append(
                    f"| {rec.record_id} | {rec.activity_id} | "
                    f"{rec.operation} | {rec.timestamp} | "
                    f"`{rec.input_hash[:12]}...` | "
                    f"`{rec.output_hash[:12]}...` | "
                    f"{rec.agent} |"
                )
            sections.append(f"")

        # Assumptions & Methodology
        if self.config.include_assumptions:
            if data.assumptions:
                sections.append(f"## Assumptions")
                sections.append(f"")
                sections.append(
                    f"| ID | Category | Description | Impact | Activities Affected |"
                )
                sections.append(
                    f"|----|----------|-------------|--------|--------------------:|"
                )
                for assumption in data.assumptions:
                    sections.append(
                        f"| {assumption.assumption_id} | {assumption.category} | "
                        f"{assumption.description[:45]} | {assumption.impact} | "
                        f"{len(assumption.activities_affected)} |"
                    )
                sections.append(f"")

            if data.methodology_notes:
                sections.append(f"## Methodology")
                sections.append(f"")
                for idx, note in enumerate(data.methodology_notes, 1):
                    sections.append(f"{idx}. {note}")
                sections.append(f"")

        # Appendices
        if self.config.include_appendices and data.appendix_items:
            sections.append(f"## Appendices")
            sections.append(f"")
            for idx, item in enumerate(data.appendix_items, 1):
                sections.append(f"**Appendix {chr(64 + idx)}:** {item}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        report_hash = self._calculate_hash("\n".join(sections))
        sections.append(f"**Report Integrity Hash:** `{report_hash}`")
        sections.append(f"")
        sections.append(
            f"*Detailed assessment report generated on {data.report_date} using "
            f"GreenLang EU Taxonomy Alignment Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        align_rate = (data.total_aligned / data.total_activities * 100) if data.total_activities > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EU Taxonomy Detailed Assessment - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
        h4 {{ color: #7f8c8d; margin-top: 15px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .hash {{ font-family: monospace; font-size: 0.85em; color: #7f8c8d; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .activity-card {{ border: 1px solid #bdc3c7; border-radius: 5px; padding: 15px; margin: 15px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EU Taxonomy Detailed Assessment Report</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Alignment Rate:</strong> <span class="pass">{align_rate:.1f}%</span></p>
        <p><strong>Activities:</strong> {data.total_aligned}/{data.total_activities} aligned</p>
    </div>

    <h2>Assessment Overview</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Activities</td><td>{data.total_activities}</td></tr>
        <tr><td>Eligible</td><td>{data.total_eligible}</td></tr>
        <tr><td>Aligned</td><td>{data.total_aligned}</td></tr>
        <tr><td>Evidence Documents</td><td>{len(data.evidence_inventory)}</td></tr>
        <tr><td>Provenance Records</td><td>{len(data.provenance_records)}</td></tr>
    </table>
"""

        if data.assessments:
            html += """    <h2>Activity Catalog</h2>
    <table>
        <tr><th>#</th><th>Activity ID</th><th>Activity</th><th>NACE</th>
        <th>Eligible</th><th>Aligned</th><th>SC</th><th>DNSH</th><th>MS</th></tr>
"""
            for idx, act in enumerate(
                data.assessments[:self.config.max_activities_per_section], 1
            ):
                sc_status = act.sc_detail.status if act.sc_detail else "N/A"
                dnsh_pass = all(
                    d.status in ("PASS", "NOT_APPLICABLE") for d in act.dnsh_details
                ) if act.dnsh_details else False
                ms_pass = all(
                    m.status == "PASS" for m in act.ms_details
                ) if act.ms_details else False
                html += f"""        <tr>
            <td>{idx}</td><td>{act.activity_id}</td>
            <td>{act.activity_name}</td><td>{act.nace_code}</td>
            <td class="{'pass' if act.is_eligible else 'fail'}">{'Y' if act.is_eligible else 'N'}</td>
            <td class="{'pass' if act.is_aligned else 'fail'}">{'Y' if act.is_aligned else 'N'}</td>
            <td>{sc_status}</td>
            <td class="{'pass' if dnsh_pass else 'fail'}">{'Y' if dnsh_pass else 'N'}</td>
            <td class="{'pass' if ms_pass else 'fail'}">{'Y' if ms_pass else 'N'}</td>
        </tr>
"""
            html += """    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Detailed assessment report generated on {data.report_date} using
        GreenLang EU Taxonomy Alignment Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        # Generate per-activity hashes
        assessments_with_hashes = []
        for act in data.assessments:
            act_dict = act.dict()
            act_dict["computed_provenance_hash"] = self._generate_activity_hash(act)
            assessments_with_hashes.append(act_dict)

        report_dict = {
            "report_type": "detailed_assessment",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_activities": data.total_activities,
                "total_eligible": data.total_eligible,
                "total_aligned": data.total_aligned,
                "alignment_rate": (
                    data.total_aligned / data.total_activities * 100
                ) if data.total_activities > 0 else 0,
                "evidence_documents": len(data.evidence_inventory),
                "provenance_records": len(data.provenance_records),
                "assumptions_count": len(data.assumptions),
            },
            "assessments": assessments_with_hashes,
            "evidence_inventory": [ev.dict() for ev in data.evidence_inventory],
            "provenance_trail": [rec.dict() for rec in data.provenance_records],
            "assumptions": [a.dict() for a in data.assumptions],
            "methodology_notes": data.methodology_notes,
            "appendices": data.appendix_items,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "DetailedAssessmentReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
                "report_hash": self._calculate_hash(
                    json.dumps(assessments_with_hashes, default=str)
                ),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_activity_hash(self, activity: ActivityAssessment) -> str:
        """Generate SHA-256 provenance hash for a single activity assessment."""
        hash_input = (
            f"{activity.activity_id}|"
            f"{activity.activity_name}|"
            f"{activity.nace_code}|"
            f"{activity.is_eligible}|"
            f"{activity.is_aligned}|"
            f"{activity.sc_detail.status if activity.sc_detail else 'N/A'}|"
            f"{len(activity.dnsh_details)}|"
            f"{len(activity.ms_details)}|"
            f"{activity.assessment_date}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()
