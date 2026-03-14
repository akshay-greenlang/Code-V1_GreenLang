# -*- coding: utf-8 -*-
"""
PACK-001 Phase 3: Auditor Package Template
============================================

Generates a comprehensive external auditor evidence package containing
calculation audit trails, data lineage documentation, source data
references, compliance checklists, methodology documentation, data
quality assessments, control environment descriptions, and management
assertions.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ChecklistResult(str, Enum):
    """Result of a compliance checklist item."""
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_TESTED = "NOT_TESTED"
    NOT_APPLICABLE = "N/A"


class DataSourceCategory(str, Enum):
    """Category of data source for provenance tracking."""
    ERP_SYSTEM = "ERP_SYSTEM"
    FILE_UPLOAD = "FILE_UPLOAD"
    DATABASE_QUERY = "DATABASE_QUERY"
    API_CALL = "API_CALL"
    MANUAL_ENTRY = "MANUAL_ENTRY"
    IOT_SENSOR = "IOT_SENSOR"
    THIRD_PARTY = "THIRD_PARTY"
    CALCULATED = "CALCULATED"


class LineageStepType(str, Enum):
    """Type of step in data lineage chain."""
    INGESTION = "INGESTION"
    VALIDATION = "VALIDATION"
    TRANSFORMATION = "TRANSFORMATION"
    CALCULATION = "CALCULATION"
    AGGREGATION = "AGGREGATION"
    REVIEW = "REVIEW"
    EXPORT = "EXPORT"


class ControlEffectiveness(str, Enum):
    """Effectiveness rating for control environment."""
    EFFECTIVE = "EFFECTIVE"
    PARTIALLY_EFFECTIVE = "PARTIALLY_EFFECTIVE"
    INEFFECTIVE = "INEFFECTIVE"
    NOT_EVALUATED = "NOT_EVALUATED"


class QualityDimension(str, Enum):
    """Data quality assessment dimensions."""
    COMPLETENESS = "COMPLETENESS"
    ACCURACY = "ACCURACY"
    TIMELINESS = "TIMELINESS"
    CONSISTENCY = "CONSISTENCY"
    VALIDITY = "VALIDITY"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CalculationAuditEntry(BaseModel):
    """Single calculation in the audit trail."""
    calculation_id: str = Field(..., description="Unique calculation identifier")
    description: str = Field(..., description="What was calculated")
    formula: str = Field(..., description="Formula applied (e.g. 'activity_data * EF * GWP')")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Named input values")
    output_value: float = Field(..., description="Calculation result")
    output_unit: str = Field("tCO2e", description="Result unit")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Calculation timestamp")
    agent_id: Optional[str] = Field(None, description="GreenLang agent that performed calculation")
    provenance_hash: str = Field("", description="SHA-256 hash of inputs+formula+output")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            raw = json.dumps({
                "formula": self.formula,
                "inputs": self.inputs,
                "output": self.output_value,
            }, sort_keys=True, default=str)
            self.provenance_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DataLineageRecord(BaseModel):
    """A step in the data lineage chain from source to output."""
    step_number: int = Field(..., ge=1, description="Step sequence number")
    step_type: LineageStepType = Field(..., description="Type of lineage step")
    description: str = Field(..., description="What happened at this step")
    input_ref: str = Field("", description="Reference to input data or prior step")
    output_ref: str = Field("", description="Reference to output data")
    agent_id: Optional[str] = Field(None, description="Agent responsible")
    timestamp: Optional[datetime] = Field(None, description="Step timestamp")
    hash_before: Optional[str] = Field(None, description="Data hash before transformation")
    hash_after: Optional[str] = Field(None, description="Data hash after transformation")


class SourceDataReference(BaseModel):
    """Reference to an original source data artifact."""
    reference_id: str = Field(..., description="Unique reference identifier")
    source_category: DataSourceCategory = Field(..., description="Source category")
    source_name: str = Field(..., description="Source system or file name")
    description: str = Field("", description="What data was obtained")
    file_path: Optional[str] = Field(None, description="File path if file-based")
    query_text: Optional[str] = Field(None, description="Database query if applicable")
    api_endpoint: Optional[str] = Field(None, description="API endpoint if applicable")
    record_count: Optional[int] = Field(None, ge=0, description="Number of records obtained")
    date_retrieved: Optional[date] = Field(None, description="When data was retrieved")
    hash_value: Optional[str] = Field(None, description="SHA-256 hash of retrieved data")


class ComplianceChecklistItem(BaseModel):
    """Single item in the compliance checklist."""
    rule_id: str = Field(..., description="Rule identifier (e.g. ESRS-E1-001)")
    rule_description: str = Field(..., description="What the rule requires")
    standard_reference: str = Field("", description="ESRS / GHG Protocol reference")
    result: ChecklistResult = Field(..., description="Pass/Fail result")
    evidence_references: List[str] = Field(default_factory=list, description="Evidence refs")
    findings: Optional[str] = Field(None, description="Auditor findings or notes")
    remediation: Optional[str] = Field(None, description="Required remediation if failed")


class DataQualityAssessment(BaseModel):
    """Assessment of data quality across dimensions."""
    dimension: QualityDimension = Field(..., description="Quality dimension")
    score_pct: float = Field(..., ge=0.0, le=100.0, description="Score 0-100%")
    metric_description: str = Field("", description="How score was determined")
    issues_found: int = Field(0, ge=0, description="Number of issues identified")
    issue_details: List[str] = Field(default_factory=list, description="Issue descriptions")
    recommendation: Optional[str] = Field(None, description="Improvement recommendation")


class ControlEnvironmentEntry(BaseModel):
    """Control environment description entry."""
    control_id: str = Field(..., description="Control identifier")
    control_name: str = Field(..., description="Control name")
    description: str = Field("", description="Control description")
    control_type: str = Field("preventive", description="Preventive, detective, corrective")
    frequency: str = Field("", description="How often control operates")
    effectiveness: ControlEffectiveness = Field(
        ControlEffectiveness.NOT_EVALUATED, description="Effectiveness rating"
    )
    evidence: Optional[str] = Field(None, description="Evidence of operation")


class ManagementAssertion(BaseModel):
    """Management assertion for the auditor."""
    assertion_id: str = Field(..., description="Assertion identifier")
    assertion_text: str = Field(..., description="The assertion statement")
    category: str = Field("", description="Category (completeness, accuracy, etc.)")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence")
    signatory: Optional[str] = Field(None, description="Management signatory")
    sign_date: Optional[date] = Field(None, description="Assertion date")


class AuditorPackageInput(BaseModel):
    """Full input for the auditor evidence package."""
    company_name: str = Field(..., description="Reporting entity")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Fiscal year")
    report_date: date = Field(default_factory=date.today, description="Package generation date")
    audit_period_start: Optional[date] = Field(None, description="Audit period start")
    audit_period_end: Optional[date] = Field(None, description="Audit period end")
    calculation_audit_trail: List[CalculationAuditEntry] = Field(
        default_factory=list, description="Calculation audit entries"
    )
    data_lineage: List[DataLineageRecord] = Field(
        default_factory=list, description="Data lineage records"
    )
    source_references: List[SourceDataReference] = Field(
        default_factory=list, description="Source data references"
    )
    compliance_checklist: List[ComplianceChecklistItem] = Field(
        default_factory=list, description="Compliance checklist items"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology documentation notes"
    )
    quality_assessments: List[DataQualityAssessment] = Field(
        default_factory=list, description="Data quality assessments"
    )
    control_environment: List[ControlEnvironmentEntry] = Field(
        default_factory=list, description="Control environment entries"
    )
    management_assertions: List[ManagementAssertion] = Field(
        default_factory=list, description="Management assertions"
    )

    @property
    def checklist_pass_count(self) -> int:
        """Number of checklist items that passed."""
        return sum(1 for c in self.compliance_checklist if c.result == ChecklistResult.PASS)

    @property
    def checklist_fail_count(self) -> int:
        """Number of checklist items that failed."""
        return sum(1 for c in self.compliance_checklist if c.result == ChecklistResult.FAIL)

    @property
    def checklist_total_tested(self) -> int:
        """Total checklist items tested."""
        return sum(
            1 for c in self.compliance_checklist
            if c.result not in (ChecklistResult.NOT_TESTED, ChecklistResult.NOT_APPLICABLE)
        )

    @property
    def overall_quality_score(self) -> float:
        """Average data quality score across all dimensions."""
        if not self.quality_assessments:
            return 0.0
        return sum(q.score_pct for q in self.quality_assessments) / len(self.quality_assessments)


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _result_badge(result: ChecklistResult) -> str:
    """Text badge for checklist result."""
    return f"[{result.value}]"


def _result_css(result: ChecklistResult) -> str:
    """CSS class for checklist result."""
    return f"result-{result.value.lower().replace('/', '')}"


def _effectiveness_badge(eff: ControlEffectiveness) -> str:
    """Text badge for control effectiveness."""
    return f"[{eff.value.replace('_', ' ')}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class AuditorPackageTemplate:
    """Generate comprehensive external auditor evidence package.

    Package contents:
        1. Calculation Audit Trail (every formula, input, output, timestamp)
        2. Data Lineage Documentation (source -> transformation -> output)
        3. Source Data References (file paths, database queries, API calls)
        4. Compliance Checklist (235 rules, pass/fail, evidence)
        5. Methodology Documentation (GHG Protocol, ESRS guidance)
        6. Data Quality Assessment (completeness, accuracy, timeliness)
        7. Control Environment Description
        8. Management Assertions & Representations

    Example:
        >>> template = AuditorPackageTemplate()
        >>> data = AuditorPackageInput(company_name="Acme", reporting_year=2025, ...)
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the auditor package template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: AuditorPackageInput) -> str:
        """Render the auditor package as Markdown.

        Args:
            data: Validated auditor package input.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_cover(data),
            self._md_table_of_contents(),
            self._md_calculation_audit(data),
            self._md_data_lineage(data),
            self._md_source_references(data),
            self._md_compliance_checklist(data),
            self._md_methodology(data),
            self._md_quality_assessment(data),
            self._md_control_environment(data),
            self._md_management_assertions(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: AuditorPackageInput) -> str:
        """Render the package as HTML.

        Args:
            data: Validated input.

        Returns:
            HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_cover(data),
            self._html_calculation_audit(data),
            self._html_compliance_checklist(data),
            self._html_quality_assessment(data),
            self._html_control_environment(data),
            self._html_management_assertions(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.company_name, data.reporting_year, body)

    def render_json(self, data: AuditorPackageInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated input.

        Returns:
            Dictionary for serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)

        return {
            "template": "auditor_package",
            "version": "1.0.0",
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "company_name": data.company_name,
            "reporting_year": data.reporting_year,
            "audit_period": {
                "start": data.audit_period_start.isoformat() if data.audit_period_start else None,
                "end": data.audit_period_end.isoformat() if data.audit_period_end else None,
            },
            "summary": {
                "calculation_entries": len(data.calculation_audit_trail),
                "lineage_steps": len(data.data_lineage),
                "source_references": len(data.source_references),
                "checklist_items": len(data.compliance_checklist),
                "checklist_pass": data.checklist_pass_count,
                "checklist_fail": data.checklist_fail_count,
                "checklist_tested": data.checklist_total_tested,
                "overall_quality_score_pct": data.overall_quality_score,
                "controls_count": len(data.control_environment),
                "assertions_count": len(data.management_assertions),
            },
            "calculation_audit_trail": [
                e.model_dump(mode="json") for e in data.calculation_audit_trail
            ],
            "data_lineage": [r.model_dump(mode="json") for r in data.data_lineage],
            "source_references": [r.model_dump(mode="json") for r in data.source_references],
            "compliance_checklist": [
                c.model_dump(mode="json") for c in data.compliance_checklist
            ],
            "methodology_notes": data.methodology_notes,
            "quality_assessments": [
                q.model_dump(mode="json") for q in data.quality_assessments
            ],
            "control_environment": [
                c.model_dump(mode="json") for c in data.control_environment
            ],
            "management_assertions": [
                a.model_dump(mode="json") for a in data.management_assertions
            ],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: AuditorPackageInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_cover(self, data: AuditorPackageInput) -> str:
        """Cover page."""
        period = "N/A"
        if data.audit_period_start and data.audit_period_end:
            period = f"{data.audit_period_start.isoformat()} to {data.audit_period_end.isoformat()}"
        return (
            f"# Auditor Evidence Package - {data.company_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Package Date:** {data.report_date.isoformat()} | "
            f"**Audit Period:** {period}\n\n"
            f"**Checklist Results:** {data.checklist_pass_count} passed / "
            f"{data.checklist_fail_count} failed / "
            f"{data.checklist_total_tested} tested | "
            f"**Data Quality:** {data.overall_quality_score:.1f}%\n\n---"
        )

    def _md_table_of_contents(self) -> str:
        """Table of contents."""
        return (
            "## Table of Contents\n\n"
            "1. Calculation Audit Trail\n"
            "2. Data Lineage Documentation\n"
            "3. Source Data References\n"
            "4. Compliance Checklist\n"
            "5. Methodology Documentation\n"
            "6. Data Quality Assessment\n"
            "7. Control Environment\n"
            "8. Management Assertions"
        )

    def _md_calculation_audit(self, data: AuditorPackageInput) -> str:
        """Calculation audit trail section."""
        lines = [
            "## 1. Calculation Audit Trail",
            "",
            f"**Total Calculations:** {len(data.calculation_audit_trail)}",
            "",
        ]
        if not data.calculation_audit_trail:
            lines.append("No calculation audit entries recorded.")
            return "\n".join(lines)
        lines.extend([
            "| ID | Description | Formula | Output | Unit | Agent | Hash (first 12) |",
            "|----|-------------|---------|--------|------|-------|------------------|",
        ])
        for e in data.calculation_audit_trail:
            agent = e.agent_id or "-"
            short_hash = e.provenance_hash[:12] if e.provenance_hash else "-"
            lines.append(
                f"| {e.calculation_id} | {e.description} | `{e.formula}` "
                f"| {e.output_value:,.4f} | {e.output_unit} | {agent} | `{short_hash}` |"
            )
        return "\n".join(lines)

    def _md_data_lineage(self, data: AuditorPackageInput) -> str:
        """Data lineage section."""
        lines = [
            "## 2. Data Lineage Documentation",
            "",
            f"**Total Steps:** {len(data.data_lineage)}",
            "",
        ]
        if not data.data_lineage:
            lines.append("No data lineage records.")
            return "\n".join(lines)
        lines.extend([
            "| Step | Type | Description | Input Ref | Output Ref | Agent |",
            "|------|------|-------------|-----------|-----------|-------|",
        ])
        for r in sorted(data.data_lineage, key=lambda x: x.step_number):
            agent = r.agent_id or "-"
            lines.append(
                f"| {r.step_number} | {r.step_type.value} | {r.description} "
                f"| {r.input_ref or '-'} | {r.output_ref or '-'} | {agent} |"
            )
        return "\n".join(lines)

    def _md_source_references(self, data: AuditorPackageInput) -> str:
        """Source data references section."""
        lines = [
            "## 3. Source Data References",
            "",
            f"**Total Sources:** {len(data.source_references)}",
            "",
        ]
        if not data.source_references:
            lines.append("No source references recorded.")
            return "\n".join(lines)
        lines.extend([
            "| Ref ID | Category | Source | Description | Records | Retrieved | Hash (first 12) |",
            "|--------|----------|--------|-------------|---------|-----------|------------------|",
        ])
        for r in data.source_references:
            records = str(r.record_count) if r.record_count is not None else "-"
            retrieved = r.date_retrieved.isoformat() if r.date_retrieved else "-"
            short_hash = r.hash_value[:12] if r.hash_value else "-"
            lines.append(
                f"| {r.reference_id} | {r.source_category.value} | {r.source_name} "
                f"| {r.description} | {records} | {retrieved} | `{short_hash}` |"
            )
        return "\n".join(lines)

    def _md_compliance_checklist(self, data: AuditorPackageInput) -> str:
        """Compliance checklist section."""
        total = len(data.compliance_checklist)
        lines = [
            "## 4. Compliance Checklist",
            "",
            f"**Total Rules:** {total} | "
            f"**Passed:** {data.checklist_pass_count} | "
            f"**Failed:** {data.checklist_fail_count} | "
            f"**Tested:** {data.checklist_total_tested}",
            "",
        ]
        if not data.compliance_checklist:
            lines.append("No compliance checklist items.")
            return "\n".join(lines)
        lines.extend([
            "| Rule ID | Description | Standard Ref | Result | Evidence | Findings |",
            "|---------|-------------|-------------|--------|----------|----------|",
        ])
        for c in data.compliance_checklist:
            evidence = ", ".join(c.evidence_references) if c.evidence_references else "-"
            findings = c.findings or "-"
            lines.append(
                f"| {c.rule_id} | {c.rule_description} | {c.standard_reference} "
                f"| {_result_badge(c.result)} | {evidence} | {findings} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: AuditorPackageInput) -> str:
        """Methodology documentation section."""
        lines = ["## 5. Methodology Documentation", ""]
        if not data.methodology_notes:
            lines.append("No methodology notes recorded.")
            return "\n".join(lines)
        for i, note in enumerate(data.methodology_notes, 1):
            lines.append(f"{i}. {note}")
        return "\n".join(lines)

    def _md_quality_assessment(self, data: AuditorPackageInput) -> str:
        """Data quality assessment section."""
        lines = [
            "## 6. Data Quality Assessment",
            "",
            f"**Overall Quality Score:** {data.overall_quality_score:.1f}%",
            "",
        ]
        if not data.quality_assessments:
            lines.append("No quality assessments performed.")
            return "\n".join(lines)
        lines.extend([
            "| Dimension | Score | Issues | Description | Recommendation |",
            "|-----------|-------|--------|-------------|----------------|",
        ])
        for q in data.quality_assessments:
            rec = q.recommendation or "-"
            lines.append(
                f"| {q.dimension.value} | {q.score_pct:.1f}% | {q.issues_found} "
                f"| {q.metric_description} | {rec} |"
            )
        return "\n".join(lines)

    def _md_control_environment(self, data: AuditorPackageInput) -> str:
        """Control environment section."""
        lines = [
            "## 7. Control Environment",
            "",
            f"**Total Controls:** {len(data.control_environment)}",
            "",
        ]
        if not data.control_environment:
            lines.append("No control environment entries.")
            return "\n".join(lines)
        lines.extend([
            "| ID | Name | Type | Frequency | Effectiveness | Evidence |",
            "|----|------|------|-----------|---------------|----------|",
        ])
        for c in data.control_environment:
            evidence = c.evidence or "-"
            lines.append(
                f"| {c.control_id} | {c.control_name} | {c.control_type} "
                f"| {c.frequency} | {_effectiveness_badge(c.effectiveness)} | {evidence} |"
            )
        return "\n".join(lines)

    def _md_management_assertions(self, data: AuditorPackageInput) -> str:
        """Management assertions section."""
        lines = ["## 8. Management Assertions & Representations", ""]
        if not data.management_assertions:
            lines.append("No management assertions recorded.")
            return "\n".join(lines)
        for a in data.management_assertions:
            signatory = a.signatory or "Not specified"
            sign_date = a.sign_date.isoformat() if a.sign_date else "Not dated"
            evidence = ", ".join(a.supporting_evidence) if a.supporting_evidence else "-"
            lines.extend([
                f"### {a.assertion_id} - {a.category}",
                "",
                f"> {a.assertion_text}",
                "",
                f"**Signatory:** {signatory} | **Date:** {sign_date}",
                f"**Supporting Evidence:** {evidence}",
                "",
            ])
        return "\n".join(lines)

    def _md_footer(self, data: AuditorPackageInput) -> str:
        """Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}*\n"
            f"*Package Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, company: str, year: int, body: str) -> str:
        """HTML wrapper."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Auditor Evidence Package - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#222;max-width:1100px;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f5f5f5;}\n"
            ".result-pass{color:#1a7f37;font-weight:bold;}\n"
            ".result-fail{color:#cf222e;font-weight:bold;}\n"
            ".result-partial{color:#b08800;font-weight:bold;}\n"
            ".result-not_tested{color:#888;}\n"
            ".result-na{color:#888;}\n"
            "code{background:#f0f0f0;padding:0.1rem 0.3rem;border-radius:2px;font-size:0.85rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_cover(self, data: AuditorPackageInput) -> str:
        """HTML cover."""
        period = "N/A"
        if data.audit_period_start and data.audit_period_end:
            period = f"{data.audit_period_start.isoformat()} to {data.audit_period_end.isoformat()}"
        return (
            '<div class="section">\n'
            f"<h1>Auditor Evidence Package &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Package Date:</strong> {data.report_date.isoformat()} | "
            f"<strong>Audit Period:</strong> {period}</p>\n"
            f"<p><strong>Checklist:</strong> {data.checklist_pass_count} passed / "
            f"{data.checklist_fail_count} failed / "
            f"{data.checklist_total_tested} tested | "
            f"<strong>Data Quality:</strong> {data.overall_quality_score:.1f}%</p>\n"
            "<hr>\n</div>"
        )

    def _html_calculation_audit(self, data: AuditorPackageInput) -> str:
        """HTML calculation audit trail."""
        rows = []
        for e in data.calculation_audit_trail:
            agent = e.agent_id or "-"
            short_hash = e.provenance_hash[:12] if e.provenance_hash else "-"
            rows.append(
                f"<tr><td>{e.calculation_id}</td><td>{e.description}</td>"
                f"<td><code>{e.formula}</code></td>"
                f"<td>{e.output_value:,.4f}</td><td>{e.output_unit}</td>"
                f"<td>{agent}</td><td><code>{short_hash}</code></td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="7">No calculation audit entries</td></tr>')
        return (
            '<div class="section">\n<h2>1. Calculation Audit Trail</h2>\n'
            f"<p><strong>Total:</strong> {len(data.calculation_audit_trail)}</p>\n"
            "<table><thead><tr><th>ID</th><th>Description</th><th>Formula</th>"
            "<th>Output</th><th>Unit</th><th>Agent</th><th>Hash</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_compliance_checklist(self, data: AuditorPackageInput) -> str:
        """HTML compliance checklist."""
        rows = []
        for c in data.compliance_checklist:
            css = _result_css(c.result)
            evidence = ", ".join(c.evidence_references) if c.evidence_references else "-"
            findings = c.findings or "-"
            rows.append(
                f"<tr><td>{c.rule_id}</td><td>{c.rule_description}</td>"
                f"<td>{c.standard_reference}</td>"
                f'<td class="{css}">{c.result.value}</td>'
                f"<td>{evidence}</td><td>{findings}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No checklist items</td></tr>')
        return (
            '<div class="section">\n<h2>4. Compliance Checklist</h2>\n'
            f"<p><strong>Total:</strong> {len(data.compliance_checklist)} | "
            f"<strong>Passed:</strong> {data.checklist_pass_count} | "
            f"<strong>Failed:</strong> {data.checklist_fail_count}</p>\n"
            "<table><thead><tr><th>Rule</th><th>Description</th><th>Standard</th>"
            "<th>Result</th><th>Evidence</th><th>Findings</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_quality_assessment(self, data: AuditorPackageInput) -> str:
        """HTML data quality assessment."""
        rows = []
        for q in data.quality_assessments:
            rec = q.recommendation or "-"
            rows.append(
                f"<tr><td>{q.dimension.value}</td><td>{q.score_pct:.1f}%</td>"
                f"<td>{q.issues_found}</td><td>{q.metric_description}</td>"
                f"<td>{rec}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="5">No quality assessments</td></tr>')
        return (
            '<div class="section">\n<h2>6. Data Quality Assessment</h2>\n'
            f"<p><strong>Overall Score:</strong> {data.overall_quality_score:.1f}%</p>\n"
            "<table><thead><tr><th>Dimension</th><th>Score</th><th>Issues</th>"
            "<th>Description</th><th>Recommendation</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_control_environment(self, data: AuditorPackageInput) -> str:
        """HTML control environment."""
        rows = []
        for c in data.control_environment:
            evidence = c.evidence or "-"
            rows.append(
                f"<tr><td>{c.control_id}</td><td>{c.control_name}</td>"
                f"<td>{c.control_type}</td><td>{c.frequency}</td>"
                f"<td>{c.effectiveness.value}</td><td>{evidence}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No controls documented</td></tr>')
        return (
            '<div class="section">\n<h2>7. Control Environment</h2>\n'
            "<table><thead><tr><th>ID</th><th>Name</th><th>Type</th>"
            "<th>Frequency</th><th>Effectiveness</th><th>Evidence</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_management_assertions(self, data: AuditorPackageInput) -> str:
        """HTML management assertions."""
        if not data.management_assertions:
            return (
                '<div class="section"><h2>8. Management Assertions</h2>'
                "<p>No assertions recorded.</p></div>"
            )
        parts = ['<div class="section">\n<h2>8. Management Assertions</h2>']
        for a in data.management_assertions:
            signatory = a.signatory or "Not specified"
            sign_date = a.sign_date.isoformat() if a.sign_date else "Not dated"
            parts.append(
                f"<h3>{a.assertion_id} - {a.category}</h3>\n"
                f"<blockquote>{a.assertion_text}</blockquote>\n"
                f"<p><strong>Signatory:</strong> {signatory} | "
                f"<strong>Date:</strong> {sign_date}</p>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _html_footer(self, data: AuditorPackageInput) -> str:
        """HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}</p>\n"
            f"<p>Package Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
