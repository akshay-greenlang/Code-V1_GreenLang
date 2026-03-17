"""
Audit Readiness Report Template - PACK-007 EUDR Professional Pack

This module generates audit readiness reports for Competent Authority inspections, including
evidence inventory, compliance checklists per EUDR article, gap summaries, remediation status,
retention verification, and mock audit results.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from audit_readiness_report import AuditReadinessReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Forest Imports Ltd",
    ...     report_date="2026-03-15",
    ...     audit_readiness_score=85.5
    ... )
    >>> template = AuditReadinessReportTemplate()
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
    """Configuration for Audit Readiness Report generation."""

    include_evidence_inventory: bool = Field(
        default=True,
        description="Include detailed evidence inventory"
    )
    include_article_checklists: bool = Field(
        default=True,
        description="Include compliance checklists per EUDR article"
    )
    include_mock_audit: bool = Field(
        default=True,
        description="Include mock audit results"
    )
    include_remediation_plan: bool = Field(
        default=True,
        description="Include remediation action plan"
    )


class EvidenceItem(BaseModel):
    """Evidence inventory item."""

    evidence_id: str = Field(..., description="Evidence identifier")
    evidence_type: str = Field(..., description="Type of evidence (e.g., DDS, GPS, Certificate)")
    eudr_article: str = Field(..., description="Related EUDR article")
    description: str = Field(..., description="Description of evidence")
    retention_period_years: int = Field(..., ge=0, description="Required retention period")
    current_status: str = Field(..., description="AVAILABLE, MISSING, INCOMPLETE")
    last_updated: str = Field(..., description="Last update date")
    storage_location: str = Field(..., description="Storage location/system")


class ComplianceChecklistItem(BaseModel):
    """Compliance checklist item per EUDR article."""

    article: str = Field(..., description="EUDR article reference (e.g., Art. 9)")
    requirement: str = Field(..., description="Specific requirement")
    compliance_status: str = Field(..., description="COMPLIANT, PARTIAL, NON_COMPLIANT")
    evidence_references: List[str] = Field(
        default_factory=list,
        description="References to supporting evidence"
    )
    gaps_identified: Optional[str] = Field(None, description="Description of gaps if any")
    remediation_action: Optional[str] = Field(None, description="Remediation action if needed")


class GapSummary(BaseModel):
    """Gap summary by category."""

    category: str = Field(..., description="Gap category")
    critical_gaps: int = Field(..., ge=0, description="Number of critical gaps")
    major_gaps: int = Field(..., ge=0, description="Number of major gaps")
    minor_gaps: int = Field(..., ge=0, description="Number of minor gaps")
    total_gaps: int = Field(..., ge=0, description="Total gaps")
    priority: str = Field(..., description="HIGH, MEDIUM, LOW")


class RemediationAction(BaseModel):
    """Remediation action item."""

    action_id: str = Field(..., description="Action identifier")
    gap_description: str = Field(..., description="Gap being addressed")
    action_required: str = Field(..., description="Required remediation action")
    responsible_party: str = Field(..., description="Responsible person/team")
    target_completion_date: str = Field(..., description="Target completion date")
    status: str = Field(..., description="NOT_STARTED, IN_PROGRESS, COMPLETED")
    priority: str = Field(..., description="HIGH, MEDIUM, LOW")


class MockAuditResult(BaseModel):
    """Mock audit finding."""

    finding_id: str = Field(..., description="Finding identifier")
    audit_area: str = Field(..., description="Area audited")
    finding_type: str = Field(..., description="CRITICAL, MAJOR, MINOR, OBSERVATION")
    description: str = Field(..., description="Finding description")
    eudr_article: str = Field(..., description="Related EUDR article")
    recommendation: str = Field(..., description="Auditor recommendation")


class RetentionVerification(BaseModel):
    """Document retention verification."""

    document_category: str = Field(..., description="Document category")
    total_documents: int = Field(..., ge=0, description="Total documents in category")
    compliant_retention: int = Field(..., ge=0, description="Documents with compliant retention")
    non_compliant: int = Field(..., ge=0, description="Documents with retention issues")
    compliance_rate: float = Field(..., ge=0, le=100, description="Compliance rate percentage")


class ReportData(BaseModel):
    """Data model for Audit Readiness Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    audit_readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall audit readiness score"
    )
    expected_audit_date: Optional[str] = Field(
        None,
        description="Expected audit date if known"
    )
    evidence_inventory: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence inventory"
    )
    compliance_checklist: List[ComplianceChecklistItem] = Field(
        default_factory=list,
        description="Compliance checklist items"
    )
    gap_summary: List[GapSummary] = Field(
        default_factory=list,
        description="Gap summary by category"
    )
    remediation_actions: List[RemediationAction] = Field(
        default_factory=list,
        description="Remediation action plan"
    )
    retention_verification: List[RetentionVerification] = Field(
        default_factory=list,
        description="Document retention verification"
    )
    mock_audit_results: List[MockAuditResult] = Field(
        default_factory=list,
        description="Mock audit findings"
    )
    readiness_assessment: str = Field(
        ...,
        description="Overall readiness assessment (READY, PARTIALLY_READY, NOT_READY)"
    )
    key_strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths identified"
    )
    critical_actions: List[str] = Field(
        default_factory=list,
        description="Critical actions before audit"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class AuditReadinessReportTemplate:
    """
    Audit Readiness Report Template for EUDR Professional Pack.

    Generates comprehensive audit readiness reports for Competent Authority inspections,
    including evidence inventory, compliance checklists, gap analysis, remediation plans,
    and mock audit results.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = AuditReadinessReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Audit Readiness Assessment" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Audit Readiness Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the audit readiness report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Audit Readiness Report for {data.operator_name} in {format} format"
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
        sections.append(f"# Audit Readiness Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Audit Readiness Score:** {data.audit_readiness_score:.1f}/100")
        sections.append(f"**Readiness Status:** {data.readiness_assessment}")
        if data.expected_audit_date:
            sections.append(f"**Expected Audit Date:** {data.expected_audit_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        sections.append(
            f"This audit readiness report assesses **{data.operator_name}** preparedness for "
            f"Competent Authority inspection under EUDR. The operator achieved an audit readiness "
            f"score of **{data.audit_readiness_score:.1f}/100**, with an overall status of "
            f"**{data.readiness_assessment}**."
        )
        sections.append(f"")

        # Readiness Assessment
        sections.append(f"## Readiness Assessment")
        sections.append(f"")

        readiness_color = {
            "READY": "✓ GREEN",
            "PARTIALLY_READY": "⚠ YELLOW",
            "NOT_READY": "✗ RED"
        }
        sections.append(
            f"**Status:** {readiness_color.get(data.readiness_assessment, data.readiness_assessment)}"
        )
        sections.append(f"")

        # Key Strengths
        if data.key_strengths:
            sections.append(f"### Key Strengths")
            sections.append(f"")
            for strength in data.key_strengths:
                sections.append(f"- {strength}")
            sections.append(f"")

        # Critical Actions
        if data.critical_actions:
            sections.append(f"### Critical Actions Required Before Audit")
            sections.append(f"")
            for idx, action in enumerate(data.critical_actions, 1):
                sections.append(f"{idx}. {action}")
            sections.append(f"")

        # Evidence Inventory
        if self.config.include_evidence_inventory and data.evidence_inventory:
            sections.append(f"## Evidence Inventory")
            sections.append(f"")
            sections.append(
                f"Comprehensive inventory of evidence documents required for EUDR compliance:"
            )
            sections.append(f"")

            sections.append(
                f"| Evidence ID | Type | EUDR Article | Status | Retention (yrs) | Last Updated | Storage |"
            )
            sections.append(
                f"|-------------|------|--------------|--------|-----------------|--------------|---------|"
            )

            for evidence in data.evidence_inventory:
                sections.append(
                    f"| {evidence.evidence_id} | {evidence.evidence_type} | {evidence.eudr_article} | "
                    f"{evidence.current_status} | {evidence.retention_period_years} | "
                    f"{evidence.last_updated} | {evidence.storage_location} |"
                )
            sections.append(f"")

            # Evidence status summary
            status_counts = {}
            for evidence in data.evidence_inventory:
                status = evidence.current_status
                status_counts[status] = status_counts.get(status, 0) + 1

            sections.append(f"### Evidence Status Summary")
            sections.append(f"")
            for status, count in sorted(status_counts.items()):
                sections.append(f"- **{status}:** {count} items")
            sections.append(f"")

        # Compliance Checklist
        if self.config.include_article_checklists and data.compliance_checklist:
            sections.append(f"## Compliance Checklist by EUDR Article")
            sections.append(f"")
            sections.append(
                f"Detailed compliance status for each EUDR requirement:"
            )
            sections.append(f"")

            # Group by article
            by_article = {}
            for item in data.compliance_checklist:
                if item.article not in by_article:
                    by_article[item.article] = []
                by_article[item.article].append(item)

            for article in sorted(by_article.keys()):
                sections.append(f"### {article}")
                sections.append(f"")

                sections.append(
                    f"| Requirement | Status | Evidence | Gaps/Actions |"
                )
                sections.append(
                    f"|-------------|--------|----------|--------------|"
                )

                for item in by_article[article]:
                    evidence = ", ".join(item.evidence_references[:2]) if item.evidence_references else "None"
                    gaps = item.gaps_identified or item.remediation_action or "N/A"
                    sections.append(
                        f"| {item.requirement} | {item.compliance_status} | {evidence} | {gaps} |"
                    )
                sections.append(f"")

        # Gap Summary
        if data.gap_summary:
            sections.append(f"## Gap Summary")
            sections.append(f"")
            sections.append(f"Identified gaps by category and severity:")
            sections.append(f"")

            sections.append(
                f"| Category | Critical | Major | Minor | Total | Priority |"
            )
            sections.append(
                f"|----------|----------|-------|-------|-------|----------|"
            )

            for gap in sorted(data.gap_summary, key=lambda x: x.critical_gaps, reverse=True):
                sections.append(
                    f"| {gap.category} | {gap.critical_gaps} | {gap.major_gaps} | "
                    f"{gap.minor_gaps} | {gap.total_gaps} | {gap.priority} |"
                )
            sections.append(f"")

            # Total gaps
            total_critical = sum(g.critical_gaps for g in data.gap_summary)
            total_major = sum(g.major_gaps for g in data.gap_summary)
            total_minor = sum(g.minor_gaps for g in data.gap_summary)

            sections.append(
                f"**Total Gaps:** {total_critical} critical, {total_major} major, {total_minor} minor"
            )
            sections.append(f"")

        # Remediation Action Plan
        if self.config.include_remediation_plan and data.remediation_actions:
            sections.append(f"## Remediation Action Plan")
            sections.append(f"")
            sections.append(
                f"Action plan to address identified gaps before audit:"
            )
            sections.append(f"")

            sections.append(
                f"| Action ID | Gap | Action Required | Responsible | Target Date | Status | Priority |"
            )
            sections.append(
                f"|-----------|-----|-----------------|-------------|-------------|--------|----------|"
            )

            for action in sorted(
                data.remediation_actions,
                key=lambda x: (x.priority == "HIGH", x.priority == "MEDIUM"),
                reverse=True
            ):
                sections.append(
                    f"| {action.action_id} | {action.gap_description[:30]}... | "
                    f"{action.action_required[:30]}... | {action.responsible_party} | "
                    f"{action.target_completion_date} | {action.status} | {action.priority} |"
                )
            sections.append(f"")

            # Status summary
            status_counts = {}
            for action in data.remediation_actions:
                status = action.status
                status_counts[status] = status_counts.get(status, 0) + 1

            sections.append(f"### Remediation Status")
            sections.append(f"")
            for status, count in sorted(status_counts.items()):
                sections.append(f"- **{status}:** {count} actions")
            sections.append(f"")

        # Document Retention Verification
        if data.retention_verification:
            sections.append(f"## Document Retention Verification")
            sections.append(f"")
            sections.append(
                f"Verification of document retention compliance (EUDR requires 5-year retention):"
            )
            sections.append(f"")

            sections.append(
                f"| Document Category | Total | Compliant | Non-Compliant | Compliance Rate |"
            )
            sections.append(
                f"|-------------------|-------|-----------|---------------|-----------------|"
            )

            for retention in data.retention_verification:
                sections.append(
                    f"| {retention.document_category} | {retention.total_documents} | "
                    f"{retention.compliant_retention} | {retention.non_compliant} | "
                    f"{retention.compliance_rate:.1f}% |"
                )
            sections.append(f"")

        # Mock Audit Results
        if self.config.include_mock_audit and data.mock_audit_results:
            sections.append(f"## Mock Audit Results")
            sections.append(f"")
            sections.append(
                f"Findings from mock audit conducted in preparation for CA inspection:"
            )
            sections.append(f"")

            # Group by severity
            critical = [f for f in data.mock_audit_results if f.finding_type == "CRITICAL"]
            major = [f for f in data.mock_audit_results if f.finding_type == "MAJOR"]
            minor = [f for f in data.mock_audit_results if f.finding_type == "MINOR"]

            if critical:
                sections.append(f"### Critical Findings")
                sections.append(f"")
                for finding in critical:
                    sections.append(f"**{finding.finding_id}** - {finding.audit_area} ({finding.eudr_article})")
                    sections.append(f"- **Finding:** {finding.description}")
                    sections.append(f"- **Recommendation:** {finding.recommendation}")
                    sections.append(f"")

            if major:
                sections.append(f"### Major Findings")
                sections.append(f"")
                sections.append(
                    f"| Finding ID | Area | Article | Description | Recommendation |"
                )
                sections.append(
                    f"|------------|------|---------|-------------|----------------|"
                )
                for finding in major:
                    sections.append(
                        f"| {finding.finding_id} | {finding.audit_area} | {finding.eudr_article} | "
                        f"{finding.description[:40]}... | {finding.recommendation[:40]}... |"
                    )
                sections.append(f"")

            # Summary
            sections.append(f"### Mock Audit Summary")
            sections.append(f"")
            sections.append(f"- **Critical Findings:** {len(critical)}")
            sections.append(f"- **Major Findings:** {len(major)}")
            sections.append(f"- **Minor Findings:** {len(minor)}")
            sections.append(f"- **Observations:** {len([f for f in data.mock_audit_results if f.finding_type == 'OBSERVATION'])}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )
        sections.append(f"")
        sections.append(f"**Note:** This report is for internal preparation only and does not constitute official CA audit results.")

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Audit Readiness Report - {data.operator_name}</title>
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
        .critical {{ background-color: #ffebee; border-left: 4px solid #c0392b; padding: 10px; margin: 10px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Audit Readiness Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Audit Readiness Score:</strong> <span class="metric">{data.audit_readiness_score:.1f}/100</span></p>
        <p><strong>Readiness Status:</strong> {data.readiness_assessment}</p>
"""

        if data.expected_audit_date:
            html += f"        <p><strong>Expected Audit Date:</strong> {data.expected_audit_date}</p>\n"

        html += f"""    </div>

    <h2>Executive Summary</h2>
    <p>
        This audit readiness report assesses <strong>{data.operator_name}</strong> preparedness
        for Competent Authority inspection under EUDR. The operator achieved an audit readiness
        score of <span class="metric">{data.audit_readiness_score:.1f}/100</span>, with an
        overall status of <strong>{data.readiness_assessment}</strong>.
    </p>
"""

        if data.critical_actions:
            html += f"""
    <h2>Critical Actions Required Before Audit</h2>
    <ol>
"""
            for action in data.critical_actions:
                html += f"        <li class=\"critical\">{action}</li>\n"
            html += f"""    </ol>
"""

        if self.config.include_evidence_inventory and data.evidence_inventory:
            html += f"""
    <h2>Evidence Inventory</h2>
    <table>
        <tr><th>Evidence ID</th><th>Type</th><th>EUDR Article</th><th>Status</th><th>Retention (yrs)</th><th>Storage</th></tr>
"""
            for evidence in data.evidence_inventory[:50]:
                html += f"""        <tr>
            <td>{evidence.evidence_id}</td>
            <td>{evidence.evidence_type}</td>
            <td>{evidence.eudr_article}</td>
            <td>{evidence.current_status}</td>
            <td>{evidence.retention_period_years}</td>
            <td>{evidence.storage_location}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.gap_summary:
            html += f"""
    <h2>Gap Summary</h2>
    <table>
        <tr><th>Category</th><th>Critical</th><th>Major</th><th>Minor</th><th>Total</th><th>Priority</th></tr>
"""
            for gap in data.gap_summary:
                html += f"""        <tr>
            <td>{gap.category}</td>
            <td>{gap.critical_gaps}</td>
            <td>{gap.major_gaps}</td>
            <td>{gap.minor_gaps}</td>
            <td>{gap.total_gaps}</td>
            <td>{gap.priority}</td>
        </tr>
"""
            html += f"""    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
        <p><strong>Note:</strong> This report is for internal preparation only.</p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "audit_readiness",
            "operator_name": data.operator_name,
            "report_date": data.report_date,
            "audit_readiness_score": data.audit_readiness_score,
            "readiness_assessment": data.readiness_assessment,
            "expected_audit_date": data.expected_audit_date,
            "evidence_inventory": [item.dict() for item in data.evidence_inventory],
            "compliance_checklist": [item.dict() for item in data.compliance_checklist],
            "gap_summary": [gap.dict() for gap in data.gap_summary],
            "remediation_actions": [action.dict() for action in data.remediation_actions],
            "retention_verification": [ret.dict() for ret in data.retention_verification],
            "mock_audit_results": [finding.dict() for finding in data.mock_audit_results],
            "key_strengths": data.key_strengths,
            "critical_actions": data.critical_actions,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
