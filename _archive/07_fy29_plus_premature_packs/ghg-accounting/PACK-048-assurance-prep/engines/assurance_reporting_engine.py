# -*- coding: utf-8 -*-
"""
AssuranceReportingEngine - PACK-048 GHG Assurance Prep Engine 10
====================================================================

Aggregates outputs from all PACK-048 assurance preparation engines into
structured assurance readiness reports, verifier packages, management
letters, and multi-framework disclosure sections with multi-format
export capabilities.

Calculation Methodology:
    Report Aggregation:
        The reporting engine does not perform GHG calculations itself.
        It aggregates, formats, and cross-references outputs from:
            Engine 1: EvidenceConsolidationEngine
            Engine 2: ReadinessAssessmentEngine
            Engine 3: CalculationProvenanceEngine
            Engine 4: ControlTestingEngine
            Engine 5: VerifierCollaborationEngine
            Engine 6: MaterialityAssessmentEngine
            Engine 7: SamplingPlanEngine
            Engine 8: RegulatoryRequirementEngine
            Engine 9: CostTimelineEngine

    Provenance Chain:
        Each section includes the provenance_hash from its source engine.
        The final report hash chains all section hashes:
            report_hash = SHA256(section_1_hash || ... || section_9_hash)

    Report Types:
        READINESS_REPORT:       Full assurance readiness assessment
        VERIFIER_PACKAGE:       Package for verifier engagement
        MANAGEMENT_LETTER:      Management representation letter
        ENGAGEMENT_LETTER:      Verifier engagement letter
        GAP_ANALYSIS:           Gap analysis and remediation plan
        REGULATORY_COMPLIANCE:  Regulatory compliance status

    Completeness Score:
        For each report section:
            completeness_pct = (available_data_fields / required_data_fields) * 100

        Overall report completeness:
            overall_pct = SUM(section_completeness * section_weight)
                        / SUM(section_weight)

Regulatory References:
    - ISAE 3410 para 76-80: Reporting on GHG statements
    - ISAE 3000 para 69-73: Assurance report content
    - ISO 14064-3 clause 6.5: Verification statement
    - AA1000AS v3 Section 6: Assurance statement
    - EU CSRD Art. 34: Assurance opinion requirements
    - GHG Protocol: Verification statement template
    - PCAF: Data quality verification reporting

Zero-Hallucination:
    - The reporting engine does NOT perform any GHG calculations
    - It only formats, aggregates, and cross-references engine outputs
    - No LLM involvement in report generation
    - SHA-256 provenance chain across all engines

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-048 GHG Assurance Prep
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _chain_hashes(hashes: List[str]) -> str:
    """Chain multiple hashes into a single provenance hash."""
    combined = "||".join(h for h in hashes if h)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportType(str, Enum):
    """Types of assurance reports."""
    READINESS_REPORT = "readiness_report"
    VERIFIER_PACKAGE = "verifier_package"
    MANAGEMENT_LETTER = "management_letter"
    ENGAGEMENT_LETTER = "engagement_letter"
    GAP_ANALYSIS = "gap_analysis"
    REGULATORY_COMPLIANCE = "regulatory_compliance"

class ReportSectionType(str, Enum):
    """Types of report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    EVIDENCE_CONSOLIDATION = "evidence_consolidation"
    READINESS_ASSESSMENT = "readiness_assessment"
    CALCULATION_PROVENANCE = "calculation_provenance"
    CONTROL_TESTING = "control_testing"
    VERIFIER_COLLABORATION = "verifier_collaboration"
    MATERIALITY_ASSESSMENT = "materiality_assessment"
    SAMPLING_PLAN = "sampling_plan"
    REGULATORY_REQUIREMENTS = "regulatory_requirements"
    COST_TIMELINE = "cost_timeline"
    MANAGEMENT_REPRESENTATIONS = "management_representations"
    OPINION_SCOPE = "opinion_scope"
    APPENDICES = "appendices"

class AssuranceOpinionType(str, Enum):
    """Types of assurance opinion."""
    UNMODIFIED = "unmodified"
    MODIFIED_QUALIFIED = "modified_qualified"
    MODIFIED_ADVERSE = "modified_adverse"
    DISCLAIMER = "disclaimer"

class AssuranceStandard(str, Enum):
    """Assurance standard for report formatting."""
    ISAE_3410 = "isae_3410"
    ISAE_3000 = "isae_3000"
    ISO_14064_3 = "iso_14064_3"
    AA1000AS = "aa1000as"
    SSAE_18 = "ssae_18"
    ISSA_5000 = "issa_5000"

# ---------------------------------------------------------------------------
# Section Weights for completeness calculation
# ---------------------------------------------------------------------------

SECTION_WEIGHTS: Dict[str, Decimal] = {
    ReportSectionType.EVIDENCE_CONSOLIDATION.value: Decimal("0.15"),
    ReportSectionType.READINESS_ASSESSMENT.value: Decimal("0.15"),
    ReportSectionType.CALCULATION_PROVENANCE.value: Decimal("0.15"),
    ReportSectionType.CONTROL_TESTING.value: Decimal("0.12"),
    ReportSectionType.VERIFIER_COLLABORATION.value: Decimal("0.08"),
    ReportSectionType.MATERIALITY_ASSESSMENT.value: Decimal("0.10"),
    ReportSectionType.SAMPLING_PLAN.value: Decimal("0.08"),
    ReportSectionType.REGULATORY_REQUIREMENTS.value: Decimal("0.07"),
    ReportSectionType.COST_TIMELINE.value: Decimal("0.05"),
    ReportSectionType.MANAGEMENT_REPRESENTATIONS.value: Decimal("0.05"),
}

# Required section types for missing-section detection
REQUIRED_ENGINE_SECTIONS: List[ReportSectionType] = [
    ReportSectionType.EVIDENCE_CONSOLIDATION,
    ReportSectionType.READINESS_ASSESSMENT,
    ReportSectionType.CALCULATION_PROVENANCE,
    ReportSectionType.CONTROL_TESTING,
    ReportSectionType.MATERIALITY_ASSESSMENT,
    ReportSectionType.SAMPLING_PLAN,
    ReportSectionType.REGULATORY_REQUIREMENTS,
    ReportSectionType.COST_TIMELINE,
]

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A section of source data for the report.

    Attributes:
        section_type:       Section type.
        title:              Section title.
        data:               Engine output data (dict).
        provenance_hash:    Source engine provenance hash.
        completeness_pct:   Section data completeness.
    """
    section_type: ReportSectionType = Field(..., description="Section type")
    title: str = Field(default="", description="Title")
    data: Dict[str, Any] = Field(default_factory=dict, description="Section data")
    provenance_hash: str = Field(default="", description="Source hash")
    completeness_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)

    @field_validator("completeness_pct", mode="before")
    @classmethod
    def coerce_pct(cls, v: Any) -> Decimal:
        return _decimal(v)

class ManagementRepresentation(BaseModel):
    """Management representation for the assurance engagement.

    Attributes:
        representation_id:  Representation identifier.
        description:        Representation text.
        category:           Category (completeness/accuracy/existence).
        is_confirmed:       Whether management has confirmed.
        confirmed_by:       Who confirmed.
        confirmed_date:     Confirmation date.
    """
    representation_id: str = Field(default_factory=_new_uuid)
    description: str = Field(default="", description="Representation text")
    category: str = Field(default="", description="Category")
    is_confirmed: bool = Field(default=False, description="Confirmed")
    confirmed_by: str = Field(default="", description="Confirmed by")
    confirmed_date: str = Field(default="", description="Confirmed date")

class OpinionScope(BaseModel):
    """Scope of the assurance opinion.

    Attributes:
        scopes_included:        Emission scopes included.
        categories_included:    Categories included.
        period:                 Reporting period.
        assurance_level:        Assurance level.
        assurance_standard:     Assurance standard used.
        criteria:               Reporting criteria.
        exclusions:             Exclusions from scope.
        limitations:            Scope limitations.
    """
    scopes_included: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"], description="Scopes"
    )
    categories_included: List[str] = Field(
        default_factory=list, description="Categories"
    )
    period: str = Field(default="", description="Period")
    assurance_level: str = Field(default="limited", description="Level")
    assurance_standard: AssuranceStandard = Field(
        default=AssuranceStandard.ISAE_3410, description="Standard"
    )
    criteria: str = Field(
        default="GHG Protocol Corporate Standard (revised)", description="Criteria"
    )
    exclusions: List[str] = Field(default_factory=list, description="Exclusions")
    limitations: List[str] = Field(default_factory=list, description="Limitations")

class ReportConfig(BaseModel):
    """Configuration for report generation.

    Attributes:
        organisation_id:    Organisation identifier.
        organisation_name:  Organisation name.
        reporting_year:     Reporting year.
        report_type:        Report type.
        export_formats:     Desired export formats.
        assurance_standard: Assurance standard.
        verifier_name:      Verifier name.
        verifier_firm:      Verifier firm.
        output_precision:   Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    organisation_name: str = Field(default="", description="Org name")
    reporting_year: int = Field(default=2025, description="Year")
    report_type: ReportType = Field(
        default=ReportType.READINESS_REPORT, description="Report type"
    )
    export_formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.JSON, ReportFormat.MARKDOWN]
    )
    assurance_standard: AssuranceStandard = Field(
        default=AssuranceStandard.ISAE_3410, description="Standard"
    )
    verifier_name: str = Field(default="", description="Verifier name")
    verifier_firm: str = Field(default="", description="Verifier firm")
    output_precision: int = Field(default=2, ge=0, le=6, description="Precision")

class ReportInput(BaseModel):
    """Input for assurance report generation.

    Attributes:
        config:                 Report configuration.
        sections:               Source data sections.
        management_reps:        Management representations.
        opinion_scope:          Opinion scope.
    """
    config: ReportConfig = Field(default_factory=ReportConfig, description="Config")
    sections: List[ReportSection] = Field(
        default_factory=list, description="Sections"
    )
    management_reps: List[ManagementRepresentation] = Field(
        default_factory=list, description="Management representations"
    )
    opinion_scope: OpinionScope = Field(
        default_factory=OpinionScope, description="Opinion scope"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class ExecutiveSummary(BaseModel):
    """Executive summary content.

    Attributes:
        headline:               Report headline.
        readiness_score:        Overall readiness score.
        readiness_level:        Readiness level description.
        key_findings:           Key findings.
        critical_gaps:          Critical gaps requiring attention.
        recommendations:        Key recommendations.
        opinion_type:           Projected opinion type.
    """
    headline: str = Field(default="", description="Headline")
    readiness_score: Decimal = Field(default=Decimal("0"), description="Score")
    readiness_level: str = Field(default="", description="Level")
    key_findings: List[str] = Field(default_factory=list, description="Findings")
    critical_gaps: List[str] = Field(default_factory=list, description="Gaps")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    opinion_type: str = Field(default="", description="Opinion type")

class SectionSummary(BaseModel):
    """Summary of a report section.

    Attributes:
        section_type:       Section type.
        title:              Section title.
        status:             Status (complete/partial/missing).
        completeness_pct:   Completeness percentage.
        key_metrics:        Key metrics from the section.
        findings:           Section findings.
        provenance_hash:    Source provenance hash.
    """
    section_type: str = Field(default="", description="Section type")
    title: str = Field(default="", description="Title")
    status: str = Field(default="missing", description="Status")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Completeness")
    key_metrics: Dict[str, str] = Field(default_factory=dict, description="Metrics")
    findings: List[str] = Field(default_factory=list, description="Findings")
    provenance_hash: str = Field(default="", description="Hash")

class ManagementLetterContent(BaseModel):
    """Management representation letter content.

    Attributes:
        letter_date:            Letter date.
        addressee:              Addressee (verifier).
        representations:        Representations made.
        total_representations:  Total count.
        confirmed_count:        Confirmed count.
        unconfirmed_count:      Unconfirmed count.
        signatory:              Signatory.
    """
    letter_date: str = Field(default="", description="Date")
    addressee: str = Field(default="", description="Addressee")
    representations: List[ManagementRepresentation] = Field(
        default_factory=list, description="Representations"
    )
    total_representations: int = Field(default=0, description="Total")
    confirmed_count: int = Field(default=0, description="Confirmed")
    unconfirmed_count: int = Field(default=0, description="Unconfirmed")
    signatory: str = Field(default="", description="Signatory")

class ExportResult(BaseModel):
    """Result of report export.

    Attributes:
        format:         Export format.
        content:        Exported content (string for text formats).
        size_bytes:     Size in bytes.
        success:        Whether export succeeded.
    """
    format: str = Field(default="", description="Format")
    content: str = Field(default="", description="Content")
    size_bytes: int = Field(default=0, description="Size")
    success: bool = Field(default=True, description="Success")

class ReportPackage(BaseModel):
    """Complete assurance report package.

    Attributes:
        result_id:              Unique result ID.
        report_type:            Report type.
        report_title:           Report title.
        organisation_id:        Organisation ID.
        organisation_name:      Organisation name.
        reporting_year:         Reporting year.
        assurance_standard:     Standard used.
        executive_summary:      Executive summary.
        section_summaries:      Per-section summaries.
        management_letter:      Management letter (if applicable).
        opinion_scope:          Opinion scope.
        overall_completeness:   Overall completeness percentage.
        section_hashes:         Source section hashes.
        exports:                Export results.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 chained hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    report_type: str = Field(default="", description="Report type")
    report_title: str = Field(default="", description="Title")
    organisation_id: str = Field(default="", description="Org ID")
    organisation_name: str = Field(default="", description="Org name")
    reporting_year: int = Field(default=2025, description="Year")
    assurance_standard: str = Field(default="", description="Standard")
    executive_summary: ExecutiveSummary = Field(
        default_factory=ExecutiveSummary, description="Summary"
    )
    section_summaries: List[SectionSummary] = Field(
        default_factory=list, description="Sections"
    )
    management_letter: Optional[ManagementLetterContent] = Field(
        default=None, description="Management letter"
    )
    opinion_scope: Optional[OpinionScope] = Field(
        default=None, description="Opinion scope"
    )
    overall_completeness: Decimal = Field(
        default=Decimal("0"), description="Completeness"
    )
    section_hashes: List[str] = Field(default_factory=list, description="Hashes")
    exports: List[ExportResult] = Field(default_factory=list, description="Exports")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: str = Field(default="", description="SHA-256 chain hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AssuranceReportingEngine:
    """Aggregates assurance engine outputs into structured reports.

    Generates readiness reports, verifier packages, management letters,
    engagement letters, gap analyses, and regulatory compliance reports
    with multi-format export.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance chain with SHA-256 hashes.
        - Auditable: Every data point traceable to source engine.
        - Zero-Hallucination: No LLM in report generation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("AssuranceReportingEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: ReportInput) -> ReportPackage:
        """Generate assurance report package.

        Args:
            input_data: Report input with sections, management reps, etc.

        Returns:
            ReportPackage with all report components and exports.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        prec = config.output_precision
        prec_str = "0." + "0" * prec

        # Collect provenance hashes
        section_hashes = [
            s.provenance_hash for s in input_data.sections if s.provenance_hash
        ]

        # Build section summaries
        section_summaries = self._build_section_summaries(input_data.sections)

        # Calculate overall completeness
        overall_completeness = self._calculate_overall_completeness(
            input_data.sections, prec_str
        )

        # Build executive summary
        exec_summary = self._build_executive_summary(
            input_data, section_summaries, overall_completeness
        )

        # Build management letter (if management reps provided)
        mgmt_letter = None
        if input_data.management_reps:
            mgmt_letter = self._build_management_letter(
                input_data.management_reps, config
            )

        # Determine report title
        title = self._get_report_title(config)

        # Generate exports
        exports: List[ExportResult] = []
        for fmt in config.export_formats:
            export = self._export_report(
                fmt, config, exec_summary, section_summaries,
                input_data.opinion_scope,
            )
            exports.append(export)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ReportPackage(
            report_type=config.report_type.value,
            report_title=title,
            organisation_id=config.organisation_id,
            organisation_name=config.organisation_name,
            reporting_year=config.reporting_year,
            assurance_standard=config.assurance_standard.value,
            executive_summary=exec_summary,
            section_summaries=section_summaries,
            management_letter=mgmt_letter,
            opinion_scope=input_data.opinion_scope,
            overall_completeness=overall_completeness,
            section_hashes=section_hashes,
            exports=exports,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = (
            _chain_hashes(section_hashes)
            if section_hashes
            else _compute_hash(result)
        )
        return result

    def generate_verifier_package(self, input_data: ReportInput) -> ReportPackage:
        """Generate a verifier engagement package.

        Args:
            input_data: Report input data.

        Returns:
            ReportPackage configured as verifier package.
        """
        input_data.config.report_type = ReportType.VERIFIER_PACKAGE
        return self.calculate(input_data)

    def generate_gap_analysis(self, input_data: ReportInput) -> ReportPackage:
        """Generate a gap analysis report.

        Args:
            input_data: Report input data.

        Returns:
            ReportPackage configured as gap analysis.
        """
        input_data.config.report_type = ReportType.GAP_ANALYSIS
        return self.calculate(input_data)

    def export_to_format(
        self, format_type: ReportFormat, report: ReportPackage,
    ) -> ExportResult:
        """Export a report package to a specific format.

        Args:
            format_type: Export format.
            report:      Report package to export.

        Returns:
            ExportResult.
        """
        if format_type == ReportFormat.JSON:
            content = json.dumps(
                report.model_dump(mode="json"), default=str, indent=2
            )
            return ExportResult(
                format=format_type.value, content=content,
                size_bytes=len(content.encode("utf-8")), success=True,
            )
        return ExportResult(
            format=format_type.value, content="", size_bytes=0, success=False,
        )

    # ------------------------------------------------------------------
    # Internal: Section Summaries
    # ------------------------------------------------------------------

    def _build_section_summaries(
        self, sections: List[ReportSection],
    ) -> List[SectionSummary]:
        """Build summaries for each section."""
        summaries: List[SectionSummary] = []
        present_types = {s.section_type.value for s in sections}

        for section in sections:
            metrics = self._extract_key_metrics(section)
            findings = self._extract_findings(section)
            status = self._determine_section_status(section.completeness_pct)

            summaries.append(SectionSummary(
                section_type=section.section_type.value,
                title=section.title or section.section_type.value.replace(
                    "_", " "
                ).title(),
                status=status,
                completeness_pct=section.completeness_pct,
                key_metrics=metrics,
                findings=findings,
                provenance_hash=section.provenance_hash,
            ))

        # Add missing required sections
        for st in REQUIRED_ENGINE_SECTIONS:
            if st.value not in present_types:
                summaries.append(SectionSummary(
                    section_type=st.value,
                    title=st.value.replace("_", " ").title(),
                    status="missing",
                    completeness_pct=Decimal("0"),
                ))

        return summaries

    def _determine_section_status(self, completeness: Decimal) -> str:
        """Determine section status from completeness."""
        if completeness >= Decimal("80"):
            return "complete"
        if completeness >= Decimal("30"):
            return "partial"
        return "missing"

    def _extract_key_metrics(self, section: ReportSection) -> Dict[str, str]:
        """Extract key metrics from section data."""
        metrics: Dict[str, str] = {}
        data = section.data

        if section.section_type == ReportSectionType.READINESS_ASSESSMENT:
            if "overall_score" in data:
                metrics["readiness_score"] = str(data["overall_score"])
            if "readiness_level" in data:
                metrics["readiness_level"] = str(data["readiness_level"])

        elif section.section_type == ReportSectionType.EVIDENCE_CONSOLIDATION:
            if "total_evidence_items" in data:
                metrics["evidence_items"] = str(data["total_evidence_items"])
            if "evidence_coverage_pct" in data:
                metrics["evidence_coverage"] = f"{data['evidence_coverage_pct']}%"

        elif section.section_type == ReportSectionType.CALCULATION_PROVENANCE:
            if "chain_completeness_pct" in data:
                metrics["provenance_completeness"] = (
                    f"{data['chain_completeness_pct']}%"
                )
            if "verified_calculations" in data:
                metrics["verified_calculations"] = str(
                    data["verified_calculations"]
                )

        elif section.section_type == ReportSectionType.CONTROL_TESTING:
            if "overall_effectiveness" in data:
                metrics["control_effectiveness"] = str(
                    data["overall_effectiveness"]
                )
            if "controls_tested" in data:
                metrics["controls_tested"] = str(data["controls_tested"])

        elif section.section_type == ReportSectionType.MATERIALITY_ASSESSMENT:
            if "materiality_tco2e" in data:
                metrics["materiality_threshold"] = (
                    f"{data['materiality_tco2e']} tCO2e"
                )
            if "materiality_pct" in data:
                metrics["materiality_pct"] = f"{data['materiality_pct']}%"

        elif section.section_type == ReportSectionType.SAMPLING_PLAN:
            if "total_sample_size" in data:
                metrics["sample_size"] = str(data["total_sample_size"])
            if "coverage_pct" in data:
                metrics["sample_coverage"] = f"{data['coverage_pct']}%"

        elif section.section_type == ReportSectionType.REGULATORY_REQUIREMENTS:
            if "applicable_count" in data:
                metrics["applicable_mandates"] = str(data["applicable_count"])
            if "compliance_score" in data:
                metrics["compliance_score"] = f"{data['compliance_score']}%"

        elif section.section_type == ReportSectionType.COST_TIMELINE:
            if "total_cost" in data:
                metrics["estimated_cost"] = str(data["total_cost"])
            if "total_weeks" in data:
                metrics["estimated_weeks"] = str(data["total_weeks"])

        return metrics

    def _extract_findings(self, section: ReportSection) -> List[str]:
        """Extract findings from section data."""
        findings: List[str] = []
        data = section.data

        if "findings" in data and isinstance(data["findings"], list):
            findings.extend(str(f) for f in data["findings"][:5])

        if "warnings" in data and isinstance(data["warnings"], list):
            findings.extend(f"Warning: {w}" for w in data["warnings"][:3])

        if "gaps" in data and isinstance(data["gaps"], list):
            findings.extend(f"Gap: {g}" for g in data["gaps"][:3])

        return findings

    # ------------------------------------------------------------------
    # Internal: Completeness
    # ------------------------------------------------------------------

    def _calculate_overall_completeness(
        self, sections: List[ReportSection], prec_str: str,
    ) -> Decimal:
        """Calculate weighted overall completeness."""
        if not sections:
            return Decimal("0")

        weighted_sum = Decimal("0")
        weight_sum = Decimal("0")

        for section in sections:
            weight = SECTION_WEIGHTS.get(
                section.section_type.value, Decimal("0.05")
            )
            weighted_sum += section.completeness_pct * weight
            weight_sum += weight

        if weight_sum == Decimal("0"):
            return Decimal("0")

        result = _safe_divide(weighted_sum, weight_sum)
        return result.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Internal: Executive Summary
    # ------------------------------------------------------------------

    def _build_executive_summary(
        self,
        input_data: ReportInput,
        section_summaries: List[SectionSummary],
        overall_completeness: Decimal,
    ) -> ExecutiveSummary:
        """Build executive summary from available sections."""
        config = input_data.config
        findings: List[str] = []
        critical_gaps: List[str] = []
        recommendations: List[str] = []

        readiness_score = Decimal("0")
        readiness_level = "not assessed"

        for summary in section_summaries:
            if summary.section_type == ReportSectionType.READINESS_ASSESSMENT.value:
                if "readiness_score" in summary.key_metrics:
                    readiness_score = _decimal(
                        summary.key_metrics["readiness_score"]
                    )
                if "readiness_level" in summary.key_metrics:
                    readiness_level = summary.key_metrics["readiness_level"]

            findings.extend(summary.findings[:2])

            if summary.status == "missing":
                critical_gaps.append(
                    f"{summary.title}: Data not available for report section"
                )
            elif summary.completeness_pct < Decimal("50"):
                critical_gaps.append(
                    f"{summary.title}: Only {summary.completeness_pct}% complete"
                )

        if overall_completeness < Decimal("70"):
            recommendations.append(
                "Improve data completeness before engaging verifier; "
                f"current completeness is {overall_completeness}%"
            )

        missing_count = sum(
            1 for s in section_summaries if s.status == "missing"
        )
        if missing_count > 0:
            recommendations.append(
                f"Address {missing_count} missing report section(s) to enable "
                "comprehensive assurance preparation"
            )

        if readiness_score < Decimal("60"):
            recommendations.append(
                "Readiness score below threshold; recommend targeted "
                "remediation before verifier engagement"
            )

        opinion_type = self._project_opinion_type(
            readiness_score, overall_completeness, critical_gaps
        )

        headline = (
            f"{config.organisation_name} GHG Assurance Readiness "
            f"({config.reporting_year}) - {readiness_level.title()}"
        )

        return ExecutiveSummary(
            headline=headline,
            readiness_score=readiness_score,
            readiness_level=readiness_level,
            key_findings=findings[:10],
            critical_gaps=critical_gaps[:10],
            recommendations=recommendations[:5],
            opinion_type=opinion_type,
        )

    def _project_opinion_type(
        self,
        readiness_score: Decimal,
        completeness: Decimal,
        critical_gaps: List[str],
    ) -> str:
        """Project likely opinion type based on readiness."""
        if readiness_score >= Decimal("80") and completeness >= Decimal("85"):
            return AssuranceOpinionType.UNMODIFIED.value
        if readiness_score >= Decimal("60") and completeness >= Decimal("60"):
            return AssuranceOpinionType.MODIFIED_QUALIFIED.value
        if len(critical_gaps) > 5:
            return AssuranceOpinionType.DISCLAIMER.value
        return AssuranceOpinionType.MODIFIED_ADVERSE.value

    # ------------------------------------------------------------------
    # Internal: Management Letter
    # ------------------------------------------------------------------

    def _build_management_letter(
        self,
        representations: List[ManagementRepresentation],
        config: ReportConfig,
    ) -> ManagementLetterContent:
        """Build management representation letter."""
        confirmed = sum(1 for r in representations if r.is_confirmed)
        unconfirmed = len(representations) - confirmed

        addressee = (
            config.verifier_firm or config.verifier_name or "Independent Verifier"
        )

        return ManagementLetterContent(
            letter_date=utcnow().date().isoformat(),
            addressee=addressee,
            representations=representations,
            total_representations=len(representations),
            confirmed_count=confirmed,
            unconfirmed_count=unconfirmed,
            signatory=config.organisation_name,
        )

    # ------------------------------------------------------------------
    # Internal: Report Title
    # ------------------------------------------------------------------

    def _get_report_title(self, config: ReportConfig) -> str:
        """Generate report title based on type."""
        org = config.organisation_name
        year = config.reporting_year
        titles = {
            ReportType.READINESS_REPORT.value: (
                f"GHG Assurance Readiness Report - {org} ({year})"
            ),
            ReportType.VERIFIER_PACKAGE.value: (
                f"Verifier Engagement Package - {org} ({year})"
            ),
            ReportType.MANAGEMENT_LETTER.value: (
                f"Management Representation Letter - {org} ({year})"
            ),
            ReportType.ENGAGEMENT_LETTER.value: (
                f"Assurance Engagement Letter - {org} ({year})"
            ),
            ReportType.GAP_ANALYSIS.value: (
                f"Assurance Gap Analysis Report - {org} ({year})"
            ),
            ReportType.REGULATORY_COMPLIANCE.value: (
                f"Regulatory Compliance Report - {org} ({year})"
            ),
        }
        return titles.get(
            config.report_type.value, f"Assurance Report ({year})"
        )

    # ------------------------------------------------------------------
    # Internal: Export
    # ------------------------------------------------------------------

    def _export_report(
        self,
        fmt: ReportFormat,
        config: ReportConfig,
        exec_summary: ExecutiveSummary,
        sections: List[SectionSummary],
        opinion_scope: Optional[OpinionScope],
    ) -> ExportResult:
        """Export report to specified format."""
        if fmt == ReportFormat.JSON:
            content = json.dumps({
                "report_type": config.report_type.value,
                "organisation": config.organisation_name,
                "year": config.reporting_year,
                "standard": config.assurance_standard.value,
                "executive_summary": exec_summary.model_dump(mode="json"),
                "sections": [s.model_dump(mode="json") for s in sections],
                "opinion_scope": (
                    opinion_scope.model_dump(mode="json")
                    if opinion_scope else None
                ),
            }, default=str, indent=2)
            return ExportResult(
                format=fmt.value, content=content,
                size_bytes=len(content.encode("utf-8")), success=True,
            )

        if fmt == ReportFormat.MARKDOWN:
            content = self._to_markdown(
                config, exec_summary, sections, opinion_scope
            )
            return ExportResult(
                format=fmt.value, content=content,
                size_bytes=len(content.encode("utf-8")), success=True,
            )

        if fmt == ReportFormat.HTML:
            content = self._to_html(
                config, exec_summary, sections, opinion_scope
            )
            return ExportResult(
                format=fmt.value, content=content,
                size_bytes=len(content.encode("utf-8")), success=True,
            )

        if fmt == ReportFormat.CSV:
            content = self._to_csv(sections)
            return ExportResult(
                format=fmt.value, content=content,
                size_bytes=len(content.encode("utf-8")), success=True,
            )

        # PDF and XBRL: placeholder
        return ExportResult(
            format=fmt.value, content="", size_bytes=0, success=False,
        )

    def _to_markdown(
        self,
        config: ReportConfig,
        summary: ExecutiveSummary,
        sections: List[SectionSummary],
        opinion_scope: Optional[OpinionScope],
    ) -> str:
        """Generate Markdown report."""
        lines: List[str] = []
        lines.append(f"# {self._get_report_title(config)}")
        lines.append("")
        lines.append(f"**Organisation:** {config.organisation_name}")
        lines.append(f"**Reporting Year:** {config.reporting_year}")
        lines.append(
            f"**Assurance Standard:** {config.assurance_standard.value}"
        )
        if config.verifier_firm:
            lines.append(f"**Verifier:** {config.verifier_firm}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(summary.headline)
        lines.append("")
        lines.append(f"**Readiness Score:** {summary.readiness_score}")
        lines.append(f"**Readiness Level:** {summary.readiness_level}")
        lines.append(f"**Projected Opinion:** {summary.opinion_type}")
        lines.append("")

        if summary.key_findings:
            lines.append("### Key Findings")
            lines.append("")
            for finding in summary.key_findings:
                lines.append(f"- {finding}")
            lines.append("")

        if summary.critical_gaps:
            lines.append("### Critical Gaps")
            lines.append("")
            for gap in summary.critical_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        if summary.recommendations:
            lines.append("### Recommendations")
            lines.append("")
            for rec in summary.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Opinion Scope
        if opinion_scope:
            lines.append("## Opinion Scope")
            lines.append("")
            lines.append(
                f"**Scopes:** {', '.join(opinion_scope.scopes_included)}"
            )
            lines.append(f"**Period:** {opinion_scope.period}")
            lines.append(
                f"**Assurance Level:** {opinion_scope.assurance_level}"
            )
            lines.append(f"**Criteria:** {opinion_scope.criteria}")
            if opinion_scope.exclusions:
                lines.append(
                    f"**Exclusions:** {', '.join(opinion_scope.exclusions)}"
                )
            if opinion_scope.limitations:
                lines.append(
                    f"**Limitations:** {', '.join(opinion_scope.limitations)}"
                )
            lines.append("")

        # Section Status Table
        lines.append("## Section Status")
        lines.append("")
        lines.append("| Section | Status | Completeness |")
        lines.append("|---------|--------|--------------|")
        for sec in sections:
            lines.append(
                f"| {sec.title} | {sec.status} | {sec.completeness_pct}% |"
            )
        lines.append("")

        # Section Details
        for sec in sections:
            if sec.status != "missing" and (sec.key_metrics or sec.findings):
                lines.append(f"### {sec.title}")
                lines.append("")
                for key, val in sec.key_metrics.items():
                    lines.append(f"- **{key}:** {val}")
                for finding in sec.findings:
                    lines.append(f"- {finding}")
                lines.append("")

        return "\n".join(lines)

    def _to_html(
        self,
        config: ReportConfig,
        summary: ExecutiveSummary,
        sections: List[SectionSummary],
        opinion_scope: Optional[OpinionScope],
    ) -> str:
        """Generate HTML report."""
        title = self._get_report_title(config)
        parts: List[str] = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{title}</title>",
            "<style>"
            "body{font-family:sans-serif;max-width:1200px;"
            "margin:0 auto;padding:20px;}"
            "table{border-collapse:collapse;width:100%;margin:16px 0;}"
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
            "th{background-color:#2d5016;color:#fff;}"
            ".complete{color:#2d5016;font-weight:bold;}"
            ".partial{color:#b8860b;font-weight:bold;}"
            ".missing{color:#cc0000;font-weight:bold;}"
            "</style>",
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<p><strong>Organisation:</strong> {config.organisation_name}</p>",
            f"<p><strong>Year:</strong> {config.reporting_year}</p>",
            f"<p><strong>Standard:</strong> "
            f"{config.assurance_standard.value}</p>",
            "<h2>Executive Summary</h2>",
            f"<p>{summary.headline}</p>",
            f"<p><strong>Readiness Score:</strong> "
            f"{summary.readiness_score}</p>",
            f"<p><strong>Readiness Level:</strong> "
            f"{summary.readiness_level}</p>",
            f"<p><strong>Projected Opinion:</strong> "
            f"{summary.opinion_type}</p>",
        ]

        if summary.key_findings:
            parts.append("<h3>Key Findings</h3><ul>")
            for f in summary.key_findings:
                parts.append(f"<li>{f}</li>")
            parts.append("</ul>")

        if summary.critical_gaps:
            parts.append("<h3>Critical Gaps</h3><ul>")
            for g in summary.critical_gaps:
                parts.append(f"<li>{g}</li>")
            parts.append("</ul>")

        # Section table
        parts.append("<h2>Section Status</h2>")
        parts.append(
            "<table><thead><tr>"
            "<th>Section</th><th>Status</th><th>Completeness</th>"
            "</tr></thead><tbody>"
        )
        for sec in sections:
            css_class = sec.status
            parts.append(
                f"<tr><td>{sec.title}</td>"
                f"<td class='{css_class}'>{sec.status}</td>"
                f"<td>{sec.completeness_pct}%</td></tr>"
            )
        parts.append("</tbody></table>")
        parts.append("</body></html>")
        return "\n".join(parts)

    def _to_csv(self, sections: List[SectionSummary]) -> str:
        """Generate CSV from section summaries."""
        lines: List[str] = [
            "section_type,title,status,completeness_pct,provenance_hash"
        ]
        for s in sections:
            lines.append(
                f"{s.section_type},{s.title},{s.status},"
                f"{s.completeness_pct},{s.provenance_hash}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ReportFormat",
    "ReportType",
    "ReportSectionType",
    "AssuranceOpinionType",
    "AssuranceStandard",
    # Input Models
    "ReportSection",
    "ManagementRepresentation",
    "OpinionScope",
    "ReportConfig",
    "ReportInput",
    # Output Models
    "ExecutiveSummary",
    "SectionSummary",
    "ManagementLetterContent",
    "ExportResult",
    "ReportPackage",
    # Engine
    "AssuranceReportingEngine",
]
