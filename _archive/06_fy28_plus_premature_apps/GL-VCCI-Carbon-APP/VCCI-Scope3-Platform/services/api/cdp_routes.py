# -*- coding: utf-8 -*-
"""
CDP Climate Change Questionnaire API Routes - GL-VCCI Scope 3 Platform v1.1.0

FastAPI router providing endpoints for the enhanced CDP Climate Change
questionnaire auto-population engine. Supports full questionnaire CRUD,
auto-population, validation, gap analysis, score prediction, year-over-year
comparison, and multi-format export.

Endpoints:
    GET  /api/v1/cdp/questionnaire/{year}              - Get full questionnaire
    PUT  /api/v1/cdp/questionnaire/{year}/section/{section} - Update section answers
    POST /api/v1/cdp/questionnaire/{year}/auto-populate - Run auto-population
    GET  /api/v1/cdp/questionnaire/{year}/progress      - Completion tracking
    POST /api/v1/cdp/questionnaire/{year}/validate      - Full validation
    GET  /api/v1/cdp/questionnaire/{year}/gaps           - Data gap analysis
    GET  /api/v1/cdp/questionnaire/{year}/score-prediction - Score prediction
    GET  /api/v1/cdp/questionnaire/compare/{year1}/{year2} - Year comparison
    POST /api/v1/cdp/questionnaire/{year}/export         - Export (excel/pdf/json)
    GET  /api/v1/cdp/sections                            - CDP section metadata

Version: 1.1.0
Date: 2026-03-01
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..agents.reporting.standards.cdp_enhanced import (
    CDPEnhancedGenerator,
    CDPQuestionnaireResponse,
    CDPScorePrediction,
    CDPValidation,
    DataGap,
    YearComparison,
)
from ..agents.reporting.standards.cdp_questionnaire_schema import (
    CDP_QUESTIONNAIRE_SCHEMA,
    get_section_ids,
    get_total_question_count,
    get_auto_populatable_count,
    get_required_question_count,
)

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(prefix="/api/v1/cdp", tags=["cdp"])

# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

# Questionnaire drafts keyed by year
_questionnaire_store: Dict[int, CDPQuestionnaireResponse] = {}

# Input data store keyed by year (for auto-population)
_input_data_store: Dict[int, Dict[str, Any]] = {}

# Generator instance
_generator = CDPEnhancedGenerator()


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class SectionUpdateRequest(BaseModel):
    """Request to update answers for a specific section."""
    answers: Dict[str, Any] = Field(..., description="Question ID -> answer mapping to merge into the section")


class AutoPopulateRequest(BaseModel):
    """Request to run auto-population engine."""
    company_info: Dict[str, Any] = Field(..., description="Company information")
    emissions_data: Dict[str, Any] = Field(..., description="Emissions data (Scope 1/2/3)")
    energy_data: Optional[Dict[str, Any]] = Field(None, description="Energy consumption data")
    targets_data: Optional[Dict[str, Any]] = Field(None, description="Emission reduction targets")
    risks_data: Optional[Dict[str, Any]] = Field(None, description="Climate risks and opportunities")
    governance_data: Optional[Dict[str, Any]] = Field(None, description="Governance and board oversight data")
    engagement_data: Optional[Dict[str, Any]] = Field(None, description="Value chain engagement data")


class ExportRequest(BaseModel):
    """Request to export questionnaire."""
    format: str = Field(default="json", description="Export format: json, excel, or pdf")


class ProgressResponse(BaseModel):
    """Questionnaire completion progress."""
    year: int = Field(..., description="Reporting year")
    overall_completion_pct: float = Field(default=0.0, description="Overall completion percentage")
    auto_population_rate: float = Field(default=0.0, description="Auto-population rate")
    total_questions: int = Field(default=0, description="Total questions")
    total_answered: int = Field(default=0, description="Total answered questions")
    section_progress: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-section progress"
    )
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class SectionMetadata(BaseModel):
    """Metadata for a CDP section."""
    id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    description: str = Field(default="", description="Section description")
    question_count: int = Field(default=0, description="Number of questions")
    weight: float = Field(default=1.0, description="Scoring weight")
    critical: bool = Field(default=False, description="Whether section is critical")


class SectionsResponse(BaseModel):
    """Response listing all CDP sections."""
    version: str = Field(..., description="Questionnaire version")
    total_sections: int = Field(default=0, description="Total number of sections")
    total_questions: int = Field(default=0, description="Total questions across all sections")
    auto_populatable_questions: int = Field(default=0, description="Auto-populatable questions")
    required_questions: int = Field(default=0, description="Required questions")
    sections: List[SectionMetadata] = Field(default_factory=list, description="Section metadata")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional detail")


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/questionnaire/{year}",
    response_model=CDPQuestionnaireResponse,
    summary="Get full CDP questionnaire for a reporting year",
    responses={404: {"model": ErrorResponse, "description": "Questionnaire not found"}},
)
async def get_questionnaire(year: int) -> CDPQuestionnaireResponse:
    """
    Retrieve the full CDP questionnaire for the specified reporting year.

    Returns the complete questionnaire structure with all current answers,
    section completion percentages, and auto-population metadata. If no
    questionnaire exists for the year, returns 404.

    Args:
        year: The reporting year (e.g., 2025).

    Returns:
        CDPQuestionnaireResponse with all sections and answers.
    """
    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}. Run auto-populate first.",
        )

    logger.info("Retrieved CDP questionnaire for year %d", year)
    return questionnaire


@router.put(
    "/questionnaire/{year}/section/{section}",
    response_model=CDPQuestionnaireResponse,
    summary="Update answers for a specific section",
    responses={
        404: {"model": ErrorResponse, "description": "Questionnaire or section not found"},
        400: {"model": ErrorResponse, "description": "Invalid section ID"},
    },
)
async def update_section(
    year: int,
    section: str,
    body: SectionUpdateRequest,
) -> CDPQuestionnaireResponse:
    """
    Update or merge answers for a specific section of the questionnaire.

    Merges the provided answers into the existing section answers. Does not
    remove existing answers unless explicitly set to null. Recalculates
    section and overall completion percentages after the update.

    Args:
        year: The reporting year.
        section: The section ID (e.g., C0, C1, ..., C12).
        body: Request body with answers to merge.

    Returns:
        Updated CDPQuestionnaireResponse.
    """
    section_upper = section.upper()

    # Validate section ID
    valid_sections = get_section_ids()
    if section_upper not in valid_sections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid section ID '{section}'. Valid sections: {', '.join(valid_sections)}.",
        )

    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}. Run auto-populate first.",
        )

    section_resp = questionnaire.sections.get(section_upper)
    if section_resp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section '{section_upper}' not found in questionnaire for year {year}.",
        )

    # Merge answers
    for q_id, answer in body.answers.items():
        if answer is None:
            section_resp.answers.pop(q_id, None)
        else:
            section_resp.answers[q_id] = answer

    # Recalculate section completion
    section_schema = CDP_QUESTIONNAIRE_SCHEMA["sections"].get(section_upper, {})
    total_q = len(section_schema.get("questions", []))
    answered_q = len(section_resp.answers)
    section_resp.auto_populated_count = answered_q
    section_resp.completion_pct = round((answered_q / total_q * 100.0) if total_q > 0 else 0.0, 1)
    section_resp.data_gaps = [
        q["id"] for q in section_schema.get("questions", [])
        if q["id"] not in section_resp.answers
    ]

    # Recalculate overall
    total_all = sum(s.total_questions for s in questionnaire.sections.values())
    answered_all = sum(s.auto_populated_count for s in questionnaire.sections.values())
    questionnaire.total_answered = answered_all
    questionnaire.overall_completion_pct = round(
        (answered_all / total_all * 100.0) if total_all > 0 else 0.0, 1
    )

    logger.info("Updated section %s for year %d: %d answers, %.1f%% complete",
                section_upper, year, answered_q, section_resp.completion_pct)

    return questionnaire


@router.post(
    "/questionnaire/{year}/auto-populate",
    response_model=CDPQuestionnaireResponse,
    summary="Run auto-population engine",
    status_code=status.HTTP_201_CREATED,
)
async def auto_populate(year: int, body: AutoPopulateRequest) -> CDPQuestionnaireResponse:
    """
    Run the enhanced CDP auto-population engine for the specified year.

    Generates a full CDP questionnaire response by mapping the provided
    data sources (emissions, energy, targets, risks, governance, engagement)
    to the 200+ CDP questions. Achieves 95%+ auto-population rate when all
    data sources are provided.

    If a questionnaire already exists for the year, it is replaced.

    Args:
        year: The reporting year.
        body: Input data for auto-population.

    Returns:
        CDPQuestionnaireResponse with auto-populated sections.
    """
    logger.info("Running CDP auto-population for year %d", year)

    try:
        questionnaire = _generator.generate_full_questionnaire(
            company_info=body.company_info,
            emissions_data=body.emissions_data,
            energy_data=body.energy_data,
            targets_data=body.targets_data,
            risks_data=body.risks_data,
            governance_data=body.governance_data,
            engagement_data=body.engagement_data,
            year=year,
        )

        # Store the questionnaire
        _questionnaire_store[year] = questionnaire

        # Store input data for potential re-population
        _input_data_store[year] = {
            "company_info": body.company_info,
            "emissions_data": body.emissions_data,
            "energy_data": body.energy_data,
            "targets_data": body.targets_data,
            "risks_data": body.risks_data,
            "governance_data": body.governance_data,
            "engagement_data": body.engagement_data,
        }

        logger.info(
            "Auto-population complete for year %d: %.1f%% completion, %.1f%% auto-pop rate",
            year, questionnaire.overall_completion_pct, questionnaire.auto_population_rate,
        )

        return questionnaire

    except Exception as e:
        logger.error("Auto-population failed for year %d: %s", year, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Auto-population failed: {str(e)}",
        )


@router.get(
    "/questionnaire/{year}/progress",
    response_model=ProgressResponse,
    summary="Get questionnaire completion progress",
    responses={404: {"model": ErrorResponse, "description": "Questionnaire not found"}},
)
async def get_progress(year: int) -> ProgressResponse:
    """
    Get detailed completion progress for the CDP questionnaire.

    Returns overall and per-section completion percentages, answered
    question counts, and data gap counts.

    Args:
        year: The reporting year.

    Returns:
        ProgressResponse with completion tracking data.
    """
    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}.",
        )

    section_progress: Dict[str, Dict[str, Any]] = {}
    for section_id, section_resp in questionnaire.sections.items():
        section_progress[section_id] = {
            "title": section_resp.title,
            "completion_pct": section_resp.completion_pct,
            "answered": section_resp.auto_populated_count,
            "total": section_resp.total_questions,
            "gaps": len(section_resp.data_gaps),
        }

    return ProgressResponse(
        year=year,
        overall_completion_pct=questionnaire.overall_completion_pct,
        auto_population_rate=questionnaire.auto_population_rate,
        total_questions=questionnaire.total_questions,
        total_answered=questionnaire.total_answered,
        section_progress=section_progress,
        last_updated=questionnaire.generated_at,
    )


@router.post(
    "/questionnaire/{year}/validate",
    response_model=CDPValidation,
    summary="Validate the questionnaire",
    responses={404: {"model": ErrorResponse, "description": "Questionnaire not found"}},
)
async def validate_questionnaire(year: int) -> CDPValidation:
    """
    Run full validation on the CDP questionnaire.

    Checks required question coverage, numeric range constraints,
    conditional question logic, and cross-reference consistency.
    Returns errors and warnings grouped by section.

    Args:
        year: The reporting year.

    Returns:
        CDPValidation with per-section errors and warnings.
    """
    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}.",
        )

    validation = _generator.validate_questionnaire(questionnaire)

    logger.info(
        "Validated CDP questionnaire for year %d: valid=%s, %d errors, %d warnings",
        year, validation.is_valid, validation.total_errors, validation.total_warnings,
    )

    return validation


@router.get(
    "/questionnaire/{year}/gaps",
    response_model=List[DataGap],
    summary="Get data gap analysis",
    responses={404: {"model": ErrorResponse, "description": "Questionnaire not found"}},
)
async def get_data_gaps(year: int) -> List[DataGap]:
    """
    Identify all data gaps in the CDP questionnaire.

    Analyzes each section for unanswered questions, classifying gaps
    by severity (critical for required questions in critical sections,
    warning for other required questions, info for optional questions).

    Args:
        year: The reporting year.

    Returns:
        List of DataGap objects ordered by severity.
    """
    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}.",
        )

    gaps = _generator.identify_data_gaps(questionnaire)

    logger.info("Identified %d data gaps for year %d", len(gaps), year)
    return gaps


@router.get(
    "/questionnaire/{year}/score-prediction",
    response_model=CDPScorePrediction,
    summary="Predict CDP score",
    responses={404: {"model": ErrorResponse, "description": "Questionnaire not found"}},
)
async def get_score_prediction(year: int) -> CDPScorePrediction:
    """
    Predict the CDP score based on current questionnaire completeness.

    Uses the CDP scoring criteria to predict a score from A (Leadership)
    to D- (Disclosure) based on overall completion, section coverage,
    and presence of key features (SBTi targets, verification, etc.).

    Args:
        year: The reporting year.

    Returns:
        CDPScorePrediction with predicted score and improvement actions.
    """
    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}.",
        )

    prediction = _generator.predict_cdp_score(questionnaire)

    logger.info(
        "Score prediction for year %d: %s (%s), confidence %.2f",
        year, prediction.predicted_score, prediction.predicted_band, prediction.confidence,
    )

    return prediction


@router.get(
    "/questionnaire/compare/{year1}/{year2}",
    response_model=YearComparison,
    summary="Compare questionnaires between two years",
    responses={404: {"model": ErrorResponse, "description": "Questionnaire not found for one or both years"}},
)
async def compare_years(year1: int, year2: int) -> YearComparison:
    """
    Compare CDP questionnaires between two reporting years.

    Identifies new answers, removed answers, changed values, and
    per-section completion changes. The first year is treated as
    current, the second as previous.

    Args:
        year1: Current reporting year.
        year2: Previous reporting year.

    Returns:
        YearComparison with detailed change analysis.
    """
    q1 = _questionnaire_store.get(year1)
    q2 = _questionnaire_store.get(year2)

    if q1 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year1}.",
        )
    if q2 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year2}.",
        )

    comparison = _generator.compare_years(q1, q2)

    logger.info(
        "Year comparison %d vs %d: %.1f%% change, %d new, %d removed, %d changed",
        year1, year2, comparison.completion_change_pct,
        len(comparison.new_answers), len(comparison.removed_answers), len(comparison.changed_answers),
    )

    return comparison


@router.post(
    "/questionnaire/{year}/export",
    summary="Export questionnaire",
    responses={
        404: {"model": ErrorResponse, "description": "Questionnaire not found"},
        400: {"model": ErrorResponse, "description": "Invalid export format"},
    },
)
async def export_questionnaire(year: int, body: ExportRequest) -> Response:
    """
    Export the CDP questionnaire in the specified format.

    Supported formats:
    - json: Complete questionnaire as JSON
    - excel: Structured data suitable for Excel import
    - pdf: Document-oriented structure suitable for PDF rendering

    Args:
        year: The reporting year.
        body: Export request with format specification.

    Returns:
        Response with exported content in the requested format.
    """
    questionnaire = _questionnaire_store.get(year)
    if questionnaire is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No CDP questionnaire found for year {year}.",
        )

    export_format = body.format.lower()
    if export_format not in ("json", "excel", "pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid export format '{body.format}'. Supported: json, excel, pdf.",
        )

    try:
        content_bytes = _generator.export_questionnaire(questionnaire, export_format)

        content_type_map = {
            "json": "application/json",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "pdf": "application/pdf",
        }
        filename_ext_map = {
            "json": "json",
            "excel": "xlsx",
            "pdf": "pdf",
        }

        content_type = content_type_map[export_format]
        filename = f"cdp_questionnaire_{year}.{filename_ext_map[export_format]}"

        logger.info("Exported CDP questionnaire for year %d as %s (%d bytes)",
                     year, export_format, len(content_bytes))

        return Response(
            content=content_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        logger.error("Export failed for year %d format %s: %s", year, export_format, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}",
        )


@router.get(
    "/sections",
    response_model=SectionsResponse,
    summary="Get CDP section metadata",
)
async def get_sections() -> SectionsResponse:
    """
    Get metadata for all CDP questionnaire sections.

    Returns section IDs, titles, descriptions, question counts, scoring
    weights, and criticality flags for the CDP Climate Change questionnaire.

    Returns:
        SectionsResponse with complete section metadata.
    """
    sections_list: List[SectionMetadata] = []

    for section_id in get_section_ids():
        section_schema = CDP_QUESTIONNAIRE_SCHEMA["sections"].get(section_id, {})
        sections_list.append(SectionMetadata(
            id=section_id,
            title=section_schema.get("title", ""),
            description=section_schema.get("description", ""),
            question_count=len(section_schema.get("questions", [])),
            weight=section_schema.get("weight", 1.0),
            critical=section_schema.get("critical", False),
        ))

    return SectionsResponse(
        version=CDP_QUESTIONNAIRE_SCHEMA["version"],
        total_sections=len(sections_list),
        total_questions=get_total_question_count(),
        auto_populatable_questions=get_auto_populatable_count(),
        required_questions=get_required_question_count(),
        sections=sections_list,
    )


__all__ = ["router"]
