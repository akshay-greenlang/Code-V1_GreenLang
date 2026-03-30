# -*- coding: utf-8 -*-
"""
DMAReportEngine - PACK-015 Double Materiality Engine 8
========================================================

Assembles the complete Double Materiality Assessment (DMA) report with
methodology documentation, stakeholder engagement summary, materiality
matrix, IRO register, ESRS disclosure mapping, gap analysis, and
executive summary.

The report structure follows ESRS 2 requirements for documenting the
materiality assessment:
    - ESRS 2 IRO-1: Process description and methodology
    - ESRS 2 IRO-2: List of applicable disclosure requirements
    - ESRS 2 SBM-3: Material impacts, risks, opportunities

Report Sections:
    1. Executive Summary
    2. Methodology
    3. Stakeholder Engagement
    4. Impact Assessment (impact materiality scoring)
    5. Financial Assessment (financial materiality scoring)
    6. Materiality Matrix (double materiality visualization)
    7. IRO Register (impacts, risks, opportunities)
    8. ESRS Mapping (material topics to disclosure requirements)
    9. Gap Analysis (data availability and remediation plan)
    10. Appendices (data tables, audit trail)

Zero-Hallucination:
    - Report assembly uses deterministic template rendering
    - Executive summary is template-based (no LLM generation)
    - All statistics are computed from engine results
    - SHA-256 provenance hash on every report
    - No LLM involvement in any calculation or assembly path
    - Year-over-year comparison uses deterministic delta

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-015 Double Materiality
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

engine_version: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("generated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectionType(str, Enum):
    """Types of sections in the DMA report.

    Corresponds to the logical structure of a compliant DMA report.
    """
    EXECUTIVE_SUMMARY = "executive_summary"
    METHODOLOGY = "methodology"
    STAKEHOLDER = "stakeholder"
    IMPACT_ASSESSMENT = "impact_assessment"
    FINANCIAL_ASSESSMENT = "financial_assessment"
    MATRIX = "matrix"
    IRO_REGISTER = "iro_register"
    ESRS_MAPPING = "esrs_mapping"
    GAP_ANALYSIS = "gap_analysis"
    APPENDIX = "appendix"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_SECTIONS_ORDER: List[SectionType] = [
    SectionType.EXECUTIVE_SUMMARY,
    SectionType.METHODOLOGY,
    SectionType.STAKEHOLDER,
    SectionType.IMPACT_ASSESSMENT,
    SectionType.FINANCIAL_ASSESSMENT,
    SectionType.MATRIX,
    SectionType.IRO_REGISTER,
    SectionType.ESRS_MAPPING,
    SectionType.GAP_ANALYSIS,
    SectionType.APPENDIX,
]
"""Standard ordering of DMA report sections."""

SECTION_TEMPLATES: Dict[str, str] = {
    "executive_summary": (
        "This Double Materiality Assessment (DMA) has been conducted for {company_name} "
        "for the reporting period {reporting_period} in accordance with European "
        "Sustainability Reporting Standards (ESRS), specifically ESRS 1 paragraphs "
        "21-33 and ESRS 2 disclosure requirements IRO-1, IRO-2, and SBM-3.\n\n"
        "The assessment identified {material_count} material sustainability topics "
        "out of {total_count} topics evaluated. Of these, {double_material_count} "
        "topics are material from both impact and financial perspectives (double "
        "material), {impact_only_count} are material from the impact perspective "
        "only, and {financial_only_count} are material from the financial "
        "perspective only.\n\n"
        "The materiality assessment involved engagement with {stakeholder_count} "
        "stakeholder groups and applied a {scoring_methodology} scoring methodology "
        "with impact threshold of {impact_threshold} and financial threshold of "
        "{financial_threshold} on a 1-5 scale.\n\n"
        "Based on the material topics identified, {total_disclosures} ESRS "
        "disclosure requirements are applicable, covering {total_data_points} "
        "individual data points. The gap analysis identified {gap_count} "
        "disclosures requiring additional data collection, with an estimated "
        "{total_effort_hours} hours of effort to achieve full compliance."
    ),
    "methodology": (
        "The double materiality assessment was conducted using the following methodology:\n\n"
        "Scoring Approach: {scoring_approach}\n"
        "Impact Threshold: {impact_threshold} (on 1-5 scale)\n"
        "Financial Threshold: {financial_threshold} (on 1-5 scale)\n"
        "Assessment Date: {assessment_date}\n"
        "Assessor: {assessor_name}\n"
        "Reviewer: {reviewer_name}\n"
        "Review Date: {review_date}\n\n"
        "Stakeholder Engagement Methods:\n{stakeholder_methods}\n\n"
        "Data Sources:\n{data_sources}"
    ),
    "stakeholder": (
        "Stakeholder engagement was conducted as part of the materiality assessment "
        "to incorporate the perspectives of affected and interested parties as "
        "required by ESRS 2 SBM-2.\n\n"
        "Stakeholder Groups Engaged: {stakeholder_count}\n"
        "Total Stakeholders Consulted: {total_consulted}\n\n"
        "{stakeholder_details}"
    ),
    "matrix": (
        "The double materiality matrix positions each sustainability topic "
        "according to its impact materiality score (y-axis) and financial "
        "materiality score (x-axis).\n\n"
        "Total Topics Assessed: {total_count}\n"
        "Double Material (top-right quadrant): {double_material_count}\n"
        "Impact Only (top-left quadrant): {impact_only_count}\n"
        "Financial Only (bottom-right quadrant): {financial_only_count}\n"
        "Not Material (bottom-left quadrant): {not_material_count}"
    ),
    "gap_analysis": (
        "Gap analysis compared required ESRS disclosures against currently "
        "available data.\n\n"
        "Total Disclosures Required: {total_disclosures}\n"
        "Fully Covered: {fully_covered}\n"
        "Partially Covered: {partially_covered}\n"
        "Not Covered: {not_covered}\n\n"
        "Estimated Effort to Close Gaps: {total_effort_hours} hours"
    ),
}
"""Template strings for each report section (no LLM generation)."""

# ESRS 2 disclosure requirements that the DMA must address directly.
ESRS_2_DISCLOSURE_REQUIREMENTS: List[str] = [
    "IRO-1",
    "IRO-2",
    "SBM-3",
]
"""ESRS 2 disclosures directly served by the DMA report."""

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DMAMethodology(BaseModel):
    """Methodology documentation for the double materiality assessment.

    Captures the how of the assessment for ESRS 2 IRO-1 compliance.

    Attributes:
        scoring_approach: Description of the scoring methodology used.
        stakeholder_methods: Methods used for stakeholder engagement.
        data_sources: Sources of data used in the assessment.
        assessment_date: Date the assessment was conducted.
        assessor_name: Name of the person/team conducting the assessment.
        review_date: Date of internal review.
        reviewer_name: Name of the reviewer.
    """
    scoring_approach: str = Field(default="", description="Scoring methodology description")
    stakeholder_methods: List[str] = Field(default_factory=list, description="Stakeholder engagement methods")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    assessment_date: str = Field(default="", description="Assessment date (YYYY-MM-DD)")
    assessor_name: str = Field(default="", description="Assessor name")
    review_date: str = Field(default="", description="Review date (YYYY-MM-DD)")
    reviewer_name: str = Field(default="", description="Reviewer name")

class DMASection(BaseModel):
    """A single section of the DMA report.

    Attributes:
        section_id: Unique section identifier.
        title: Section title.
        content_type: Type of section (from SectionType enum).
        data: Structured data for the section (tables, metrics).
        narrative: Text narrative for the section.
        order: Display order (lower = earlier).
    """
    section_id: str = Field(default_factory=_new_uuid, description="Section identifier")
    title: str = Field(default="", description="Section title")
    content_type: SectionType = Field(default=SectionType.APPENDIX, description="Section type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data")
    narrative: str = Field(default="", description="Text narrative")
    order: int = Field(default=99, ge=0, description="Display order")

class StakeholderSummary(BaseModel):
    """Summary of stakeholder engagement for the DMA.

    Attributes:
        stakeholder_groups: List of stakeholder group names.
        total_consulted: Total number of stakeholders consulted.
        engagement_methods: Methods used (surveys, interviews, workshops).
        key_concerns: Top concerns raised by stakeholders.
        period: Period over which engagement was conducted.
    """
    stakeholder_groups: List[str] = Field(default_factory=list, description="Stakeholder groups")
    total_consulted: int = Field(default=0, ge=0, description="Total stakeholders consulted")
    engagement_methods: List[str] = Field(default_factory=list, description="Engagement methods")
    key_concerns: List[str] = Field(default_factory=list, description="Key concerns raised")
    period: str = Field(default="", description="Engagement period")

class IROEntry(BaseModel):
    """A single Impact, Risk, or Opportunity in the IRO register.

    Attributes:
        iro_id: Unique identifier.
        iro_type: Type (impact, risk, or opportunity).
        esrs_topic: Related ESRS topic.
        matter_name: Related sustainability matter.
        description: Description of the IRO.
        time_horizon: Short, medium, or long term.
        severity: Severity assessment (for impacts).
        likelihood: Likelihood assessment (for risks/opportunities).
        is_material: Whether this IRO is material.
    """
    iro_id: str = Field(default_factory=_new_uuid, description="IRO identifier")
    iro_type: str = Field(default="impact", description="Type: impact, risk, or opportunity")
    esrs_topic: str = Field(default="", description="ESRS topic code")
    matter_name: str = Field(default="", description="Sustainability matter name")
    description: str = Field(default="", description="Description of the IRO")
    time_horizon: str = Field(default="medium_term", description="Time horizon")
    severity: Optional[Decimal] = Field(default=None, description="Severity (0-5)")
    likelihood: Optional[Decimal] = Field(default=None, description="Likelihood (0-5)")
    is_material: bool = Field(default=False, description="Material classification")

class DMAReport(BaseModel):
    """Complete Double Materiality Assessment report.

    Assemblage of all DMA components into a single auditable document.

    Attributes:
        report_id: Unique report identifier.
        company_name: Name of the reporting undertaking.
        reporting_period: Reporting period (e.g. "FY2025").
        methodology: Methodology documentation.
        material_topics: List of material topic matter_ids.
        non_material_topics: List of non-material topic matter_ids.
        matrix_data: Serialized matrix data.
        stakeholder_summary: Stakeholder engagement summary.
        iro_register: List of IRO entries.
        esrs_mapping: Serialized ESRS mapping data.
        gap_analysis: Serialized gap analysis data.
        sections: Ordered list of report sections.
        executive_summary: Generated executive summary text.
        provenance_hash: SHA-256 hash for audit trail.
        generated_at: Report generation timestamp.
        processing_time_ms: Assembly time in milliseconds.
    """
    report_id: str = Field(default_factory=_new_uuid, description="Report identifier")
    company_name: str = Field(default="", description="Company name")
    reporting_period: str = Field(default="", description="Reporting period")
    methodology: DMAMethodology = Field(default_factory=DMAMethodology, description="Methodology")
    material_topics: List[str] = Field(default_factory=list, description="Material topic IDs")
    non_material_topics: List[str] = Field(default_factory=list, description="Non-material topic IDs")
    matrix_data: Dict[str, Any] = Field(default_factory=dict, description="Matrix data")
    stakeholder_summary: StakeholderSummary = Field(
        default_factory=StakeholderSummary, description="Stakeholder summary"
    )
    iro_register: List[IROEntry] = Field(default_factory=list, description="IRO register")
    esrs_mapping: Dict[str, Any] = Field(default_factory=dict, description="ESRS mapping data")
    gap_analysis: Dict[str, Any] = Field(default_factory=dict, description="Gap analysis data")
    sections: List[DMASection] = Field(default_factory=list, description="Report sections")
    executive_summary: str = Field(default="", description="Executive summary text")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    generated_at: Optional[datetime] = Field(default=None, description="Generation timestamp")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time in ms")

class ReportScoreChange(BaseModel):
    """Score change for a matter between two report periods.

    Attributes:
        matter_id: Matter identifier.
        matter_name: Matter name.
        previous_material: Was material in previous period.
        current_material: Is material in current period.
        previous_score: Previous combined score.
        current_score: Current combined score.
        score_delta: Change in score.
    """
    matter_id: str = Field(..., description="Matter identifier")
    matter_name: str = Field(default="", description="Matter name")
    previous_material: bool = Field(default=False)
    current_material: bool = Field(default=False)
    previous_score: Decimal = Field(default=Decimal("0"))
    current_score: Decimal = Field(default=Decimal("0"))
    score_delta: Decimal = Field(default=Decimal("0"))

    @field_validator("previous_score", "current_score", "score_delta", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class DMADelta(BaseModel):
    """Comparison between two DMA reports (current vs. previous period).

    Used for year-over-year change documentation.

    Attributes:
        previous_report_id: Identifier of the previous report.
        current_report_id: Identifier of the current report.
        new_material: Topics that became material.
        no_longer_material: Topics no longer material.
        score_changes: Score changes for common topics.
        methodology_changes: Changes in methodology between periods.
        provenance_hash: SHA-256 hash for audit trail.
    """
    previous_report_id: str = Field(default="", description="Previous report ID")
    current_report_id: str = Field(default="", description="Current report ID")
    new_material: List[str] = Field(default_factory=list, description="Newly material topics")
    no_longer_material: List[str] = Field(default_factory=list, description="No longer material topics")
    score_changes: List[ReportScoreChange] = Field(default_factory=list, description="Score changes")
    methodology_changes: List[str] = Field(default_factory=list, description="Methodology changes")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class ReportAssemblyInput(BaseModel):
    """Input bundle for assembling a DMA report.

    Groups all the data sources needed to assemble the complete report.

    Attributes:
        company_name: Name of the reporting undertaking.
        reporting_period: Reporting period label.
        methodology: Assessment methodology details.
        impact_results: Serialized impact assessment data.
        financial_results: Serialized financial assessment data.
        matrix_data: Serialized materiality matrix data.
        stakeholder_summary: Stakeholder engagement summary.
        iro_entries: List of IRO register entries.
        esrs_mapping: Serialized ESRS mapping result.
        gap_analysis: Serialized gap analysis data.
        material_topic_ids: List of material topic matter_ids.
        non_material_topic_ids: List of non-material topic matter_ids.
    """
    company_name: str = Field(..., min_length=1, description="Company name")
    reporting_period: str = Field(..., min_length=1, description="Reporting period")
    methodology: DMAMethodology = Field(default_factory=DMAMethodology)
    impact_results: Dict[str, Any] = Field(default_factory=dict)
    financial_results: Dict[str, Any] = Field(default_factory=dict)
    matrix_data: Dict[str, Any] = Field(default_factory=dict)
    stakeholder_summary: StakeholderSummary = Field(default_factory=StakeholderSummary)
    iro_entries: List[IROEntry] = Field(default_factory=list)
    esrs_mapping: Dict[str, Any] = Field(default_factory=dict)
    gap_analysis: Dict[str, Any] = Field(default_factory=dict)
    material_topic_ids: List[str] = Field(default_factory=list)
    non_material_topic_ids: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DMAReportEngine:
    """Assembles complete DMA reports with methodology documentation.

    Zero-Hallucination Guarantees:
        - Report assembly uses deterministic template rendering
        - Executive summary is template-based (no LLM generation)
        - All statistics are computed from engine results
        - SHA-256 provenance hash on every report
        - No LLM involvement in any path
        - Year-over-year comparison uses deterministic delta

    Usage::

        engine = DMAReportEngine()
        report = engine.assemble_report(input_data)
    """

    def __init__(self) -> None:
        """Initialize DMAReportEngine."""
        self._section_templates = SECTION_TEMPLATES
        self._section_order = REPORT_SECTIONS_ORDER
        logger.info("DMAReportEngine initialized")

    # ------------------------------------------------------------------
    # Core: Assemble Report
    # ------------------------------------------------------------------

    def assemble_report(
        self,
        assembly_input: ReportAssemblyInput,
    ) -> DMAReport:
        """Assemble a complete DMA report from all assessment components.

        DETERMINISTIC: Same inputs produce the same report.

        Args:
            assembly_input: All data needed for report assembly.

        Returns:
            DMAReport with all sections populated.
        """
        t0 = time.perf_counter()

        # Build sections in standard order
        sections: List[DMASection] = []

        # Section 1: Executive Summary
        exec_summary = self.generate_executive_summary(assembly_input)
        sections.append(DMASection(
            title="Executive Summary",
            content_type=SectionType.EXECUTIVE_SUMMARY,
            narrative=exec_summary,
            order=1,
        ))

        # Section 2: Methodology
        meth_section = self.generate_methodology_section(assembly_input.methodology)
        meth_section.order = 2
        sections.append(meth_section)

        # Section 3: Stakeholder Engagement
        stakeholder_section = self._generate_stakeholder_section(
            assembly_input.stakeholder_summary
        )
        stakeholder_section.order = 3
        sections.append(stakeholder_section)

        # Section 4: Impact Assessment
        sections.append(DMASection(
            title="Impact Materiality Assessment",
            content_type=SectionType.IMPACT_ASSESSMENT,
            data=assembly_input.impact_results,
            narrative="Impact materiality scores for all assessed sustainability topics.",
            order=4,
        ))

        # Section 5: Financial Assessment
        sections.append(DMASection(
            title="Financial Materiality Assessment",
            content_type=SectionType.FINANCIAL_ASSESSMENT,
            data=assembly_input.financial_results,
            narrative="Financial materiality scores for all assessed sustainability topics.",
            order=5,
        ))

        # Section 6: Materiality Matrix
        matrix_narrative = self._render_matrix_narrative(assembly_input.matrix_data)
        sections.append(DMASection(
            title="Double Materiality Matrix",
            content_type=SectionType.MATRIX,
            data=assembly_input.matrix_data,
            narrative=matrix_narrative,
            order=6,
        ))

        # Section 7: IRO Register
        iro_data = {
            "entries": [
                iro.model_dump(mode="json") for iro in assembly_input.iro_entries
            ],
            "total_entries": len(assembly_input.iro_entries),
            "material_entries": sum(
                1 for iro in assembly_input.iro_entries if iro.is_material
            ),
        }
        sections.append(DMASection(
            title="IRO Register (Impacts, Risks, Opportunities)",
            content_type=SectionType.IRO_REGISTER,
            data=iro_data,
            narrative=(
                f"The IRO register contains {len(assembly_input.iro_entries)} entries, "
                f"of which {iro_data['material_entries']} are classified as material."
            ),
            order=7,
        ))

        # Section 8: ESRS Mapping
        sections.append(DMASection(
            title="ESRS Disclosure Mapping",
            content_type=SectionType.ESRS_MAPPING,
            data=assembly_input.esrs_mapping,
            narrative="Mapping of material topics to ESRS disclosure requirements.",
            order=8,
        ))

        # Section 9: Gap Analysis
        gap_narrative = self._render_gap_narrative(assembly_input.gap_analysis)
        sections.append(DMASection(
            title="Gap Analysis and Remediation Plan",
            content_type=SectionType.GAP_ANALYSIS,
            data=assembly_input.gap_analysis,
            narrative=gap_narrative,
            order=9,
        ))

        # Section 10: Appendix
        sections.append(DMASection(
            title="Appendices",
            content_type=SectionType.APPENDIX,
            data={
                "esrs_2_disclosures_addressed": ESRS_2_DISCLOSURE_REQUIREMENTS,
                "engine_version": engine_version,
                "methodology_summary": assembly_input.methodology.model_dump(mode="json"),
            },
            narrative="Supporting documentation and data tables.",
            order=10,
        ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        report = DMAReport(
            company_name=assembly_input.company_name,
            reporting_period=assembly_input.reporting_period,
            methodology=assembly_input.methodology,
            material_topics=assembly_input.material_topic_ids,
            non_material_topics=assembly_input.non_material_topic_ids,
            matrix_data=assembly_input.matrix_data,
            stakeholder_summary=assembly_input.stakeholder_summary,
            iro_register=assembly_input.iro_entries,
            esrs_mapping=assembly_input.esrs_mapping,
            gap_analysis=assembly_input.gap_analysis,
            sections=sections,
            executive_summary=exec_summary,
            generated_at=utcnow(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "DMA report assembled: %s, %d sections, %d material topics, hash=%s",
            assembly_input.company_name,
            len(sections),
            len(assembly_input.material_topic_ids),
            report.provenance_hash[:16],
        )
        return report

    # ------------------------------------------------------------------
    # Executive Summary Generation
    # ------------------------------------------------------------------

    def generate_executive_summary(
        self, assembly_input: ReportAssemblyInput
    ) -> str:
        """Generate executive summary using deterministic templates.

        NO LLM -- uses string formatting with fixed template.

        Args:
            assembly_input: All report data.

        Returns:
            Executive summary text string.
        """
        template = self._section_templates["executive_summary"]

        matrix = assembly_input.matrix_data
        esrs = assembly_input.esrs_mapping
        gap = assembly_input.gap_analysis

        total_count = matrix.get("total_matters", 0)
        material_count = matrix.get("material_count", 0)
        double_material_count = matrix.get("double_material_count", 0)
        impact_only_count = matrix.get("impact_only_count", 0)
        financial_only_count = matrix.get("financial_only_count", 0)

        stakeholder_count = len(assembly_input.stakeholder_summary.stakeholder_groups)
        scoring_methodology = assembly_input.methodology.scoring_approach or "arithmetic mean"
        impact_threshold = matrix.get("impact_threshold", "3.0")
        financial_threshold = matrix.get("financial_threshold", "3.0")

        total_disclosures = esrs.get("total_disclosures", 0)
        total_data_points = esrs.get("total_data_points", 0)

        gap_count = gap.get("not_covered", 0) + gap.get("partially_covered", 0)
        total_effort_hours = gap.get("total_estimated_effort_hours", 0)

        summary = template.format(
            company_name=assembly_input.company_name,
            reporting_period=assembly_input.reporting_period,
            material_count=material_count,
            total_count=total_count,
            double_material_count=double_material_count,
            impact_only_count=impact_only_count,
            financial_only_count=financial_only_count,
            stakeholder_count=stakeholder_count,
            scoring_methodology=scoring_methodology,
            impact_threshold=impact_threshold,
            financial_threshold=financial_threshold,
            total_disclosures=total_disclosures,
            total_data_points=total_data_points,
            gap_count=gap_count,
            total_effort_hours=total_effort_hours,
        )
        return summary

    # ------------------------------------------------------------------
    # Methodology Section
    # ------------------------------------------------------------------

    def generate_methodology_section(
        self, methodology: DMAMethodology
    ) -> DMASection:
        """Generate the methodology documentation section.

        DETERMINISTIC: Template-based rendering.

        Args:
            methodology: Assessment methodology details.

        Returns:
            DMASection with methodology narrative and data.
        """
        template = self._section_templates["methodology"]

        methods_text = "\n".join(
            f"  - {m}" for m in methodology.stakeholder_methods
        ) if methodology.stakeholder_methods else "  - None documented"

        sources_text = "\n".join(
            f"  - {s}" for s in methodology.data_sources
        ) if methodology.data_sources else "  - None documented"

        narrative = template.format(
            scoring_approach=methodology.scoring_approach or "Not specified",
            impact_threshold="Per profile configuration",
            financial_threshold="Per profile configuration",
            assessment_date=methodology.assessment_date or "Not specified",
            assessor_name=methodology.assessor_name or "Not specified",
            reviewer_name=methodology.reviewer_name or "Not specified",
            review_date=methodology.review_date or "Not specified",
            stakeholder_methods=methods_text,
            data_sources=sources_text,
        )

        return DMASection(
            title="Assessment Methodology",
            content_type=SectionType.METHODOLOGY,
            data=methodology.model_dump(mode="json"),
            narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Report Comparison
    # ------------------------------------------------------------------

    def compare_reports(
        self,
        current: DMAReport,
        previous: DMAReport,
    ) -> DMADelta:
        """Compare two DMA reports to identify year-over-year changes.

        DETERMINISTIC: Same reports produce the same delta.

        Args:
            current: Current period's DMA report.
            previous: Previous period's DMA report.

        Returns:
            DMADelta documenting all changes.
        """
        current_material = set(current.material_topics)
        previous_material = set(previous.material_topics)

        new_material = sorted(current_material - previous_material)
        no_longer = sorted(previous_material - current_material)

        # Methodology changes
        meth_changes: List[str] = []
        curr_meth = current.methodology
        prev_meth = previous.methodology
        if curr_meth.scoring_approach != prev_meth.scoring_approach:
            meth_changes.append(
                f"Scoring approach changed from '{prev_meth.scoring_approach}' "
                f"to '{curr_meth.scoring_approach}'"
            )
        if set(curr_meth.stakeholder_methods) != set(prev_meth.stakeholder_methods):
            meth_changes.append("Stakeholder engagement methods changed")
        if set(curr_meth.data_sources) != set(prev_meth.data_sources):
            meth_changes.append("Data sources changed")

        delta = DMADelta(
            previous_report_id=previous.report_id,
            current_report_id=current.report_id,
            new_material=new_material,
            no_longer_material=no_longer,
            score_changes=[],
            methodology_changes=meth_changes,
        )
        delta.provenance_hash = _compute_hash(delta)

        logger.info(
            "Report comparison: %d new material, %d no longer material, "
            "%d methodology changes",
            len(new_material), len(no_longer), len(meth_changes),
        )
        return delta

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_completeness(self, report: DMAReport) -> List[str]:
        """Validate that a DMA report contains all required sections.

        Checks for the presence and minimum content of each required
        section per ESRS 2 requirements.

        Args:
            report: The DMA report to validate.

        Returns:
            List of missing/incomplete section descriptions.
                Empty list means the report is complete.
        """
        issues: List[str] = []

        # Check all required section types are present
        present_types = {s.content_type for s in report.sections}
        for required_type in self._section_order:
            if required_type not in present_types:
                issues.append(f"Missing section: {required_type.value}")

        # Check executive summary is non-empty
        if not report.executive_summary.strip():
            issues.append("Executive summary is empty")

        # Check methodology is documented
        if not report.methodology.scoring_approach:
            issues.append("Methodology scoring approach not documented")
        if not report.methodology.assessment_date:
            issues.append("Assessment date not documented")

        # Check material topics are listed
        if not report.material_topics and not report.non_material_topics:
            issues.append("No topics listed (neither material nor non-material)")

        # Check matrix data is present
        if not report.matrix_data:
            issues.append("Matrix data is empty")

        # Check IRO register has entries
        if not report.iro_register:
            issues.append("IRO register is empty")

        # Check ESRS mapping is present
        if not report.esrs_mapping:
            issues.append("ESRS mapping data is empty")

        # Check provenance hash
        if not report.provenance_hash:
            issues.append("Provenance hash is missing")

        return issues

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_report(
        self, report: DMAReport, fmt: ReportFormat = ReportFormat.JSON
    ) -> bytes:
        """Export the DMA report to the specified format.

        Currently supports JSON export.  PDF, HTML, and XBRL are
        placeholder implementations that serialize to JSON with a
        format marker.

        Args:
            report: The DMA report to export.
            fmt: Output format.

        Returns:
            Serialized report as bytes.
        """
        report_data = report.model_dump(mode="json")

        if fmt == ReportFormat.JSON:
            return json.dumps(
                report_data, indent=2, sort_keys=True, default=str
            ).encode("utf-8")

        elif fmt == ReportFormat.HTML:
            # Structured JSON wrapped in minimal HTML
            html_content = (
                "<!DOCTYPE html>\n<html>\n<head>\n"
                f"<title>DMA Report - {report.company_name}</title>\n"
                "</head>\n<body>\n"
                f"<h1>Double Materiality Assessment: {report.company_name}</h1>\n"
                f"<h2>Reporting Period: {report.reporting_period}</h2>\n"
                f"<p>{report.executive_summary}</p>\n"
            )
            for section in sorted(report.sections, key=lambda s: s.order):
                html_content += f"<h3>{section.title}</h3>\n"
                html_content += f"<p>{section.narrative}</p>\n"
            html_content += (
                f"<footer>Provenance Hash: {report.provenance_hash}</footer>\n"
                "</body>\n</html>"
            )
            return html_content.encode("utf-8")

        elif fmt == ReportFormat.XBRL:
            # Structured JSON with XBRL format marker
            xbrl_wrapper = {
                "format": "xbrl_inline",
                "version": "1.0",
                "report_data": report_data,
            }
            return json.dumps(
                xbrl_wrapper, indent=2, sort_keys=True, default=str
            ).encode("utf-8")

        elif fmt == ReportFormat.PDF:
            # PDF generation requires external library (reportlab, weasyprint).
            # Return JSON with PDF format marker for downstream processing.
            pdf_wrapper = {
                "format": "pdf_pending",
                "version": "1.0",
                "report_data": report_data,
                "note": "PDF rendering requires downstream processor",
            }
            return json.dumps(
                pdf_wrapper, indent=2, sort_keys=True, default=str
            ).encode("utf-8")

        # Fallback to JSON
        return json.dumps(
            report_data, indent=2, sort_keys=True, default=str
        ).encode("utf-8")

    # ------------------------------------------------------------------
    # Internal: Narrative Rendering
    # ------------------------------------------------------------------

    def _generate_stakeholder_section(
        self, summary: StakeholderSummary
    ) -> DMASection:
        """Generate stakeholder engagement section."""
        template = self._section_templates["stakeholder"]

        details_parts: List[str] = []
        for group in summary.stakeholder_groups:
            details_parts.append(f"  - {group}")
        if summary.key_concerns:
            details_parts.append("\nKey Concerns Raised:")
            for concern in summary.key_concerns:
                details_parts.append(f"  - {concern}")

        details_text = "\n".join(details_parts) if details_parts else "No details available."

        narrative = template.format(
            stakeholder_count=len(summary.stakeholder_groups),
            total_consulted=summary.total_consulted,
            stakeholder_details=details_text,
        )

        return DMASection(
            title="Stakeholder Engagement",
            content_type=SectionType.STAKEHOLDER,
            data=summary.model_dump(mode="json"),
            narrative=narrative,
        )

    def _render_matrix_narrative(self, matrix_data: Dict[str, Any]) -> str:
        """Render matrix section narrative from data."""
        template = self._section_templates["matrix"]
        return template.format(
            total_count=matrix_data.get("total_matters", 0),
            double_material_count=matrix_data.get("double_material_count", 0),
            impact_only_count=matrix_data.get("impact_only_count", 0),
            financial_only_count=matrix_data.get("financial_only_count", 0),
            not_material_count=matrix_data.get("not_material_count", 0),
        )

    def _render_gap_narrative(self, gap_data: Dict[str, Any]) -> str:
        """Render gap analysis section narrative from data."""
        template = self._section_templates["gap_analysis"]
        return template.format(
            total_disclosures=gap_data.get("total_disclosures", 0),
            fully_covered=gap_data.get("fully_covered", 0),
            partially_covered=gap_data.get("partially_covered", 0),
            not_covered=gap_data.get("not_covered", 0),
            total_effort_hours=gap_data.get("total_estimated_effort_hours", 0),
        )

    # ------------------------------------------------------------------
    # Utility: Section Access
    # ------------------------------------------------------------------

    def get_section_by_type(
        self, report: DMAReport, section_type: SectionType
    ) -> Optional[DMASection]:
        """Retrieve a specific section from the report by type.

        Args:
            report: The DMA report.
            section_type: The section type to find.

        Returns:
            DMASection if found, None otherwise.
        """
        for section in report.sections:
            if section.content_type == section_type:
                return section
        return None

    def get_sections_ordered(self, report: DMAReport) -> List[DMASection]:
        """Return all sections in display order.

        Args:
            report: The DMA report.

        Returns:
            Sections sorted by order field.
        """
        return sorted(report.sections, key=lambda s: s.order)

    def get_report_statistics(self, report: DMAReport) -> Dict[str, Any]:
        """Compute summary statistics for the report.

        Args:
            report: The DMA report.

        Returns:
            Dictionary with counts and metrics.
        """
        return {
            "report_id": report.report_id,
            "company_name": report.company_name,
            "reporting_period": report.reporting_period,
            "total_sections": len(report.sections),
            "material_topic_count": len(report.material_topics),
            "non_material_topic_count": len(report.non_material_topics),
            "iro_count": len(report.iro_register),
            "material_iro_count": sum(
                1 for iro in report.iro_register if iro.is_material
            ),
            "esrs_2_disclosures_addressed": ESRS_2_DISCLOSURE_REQUIREMENTS,
            "provenance_hash": report.provenance_hash,
            "generated_at": str(report.generated_at) if report.generated_at else None,
            "processing_time_ms": report.processing_time_ms,
        }
