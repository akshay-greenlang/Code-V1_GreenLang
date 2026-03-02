# -*- coding: utf-8 -*-
"""
Multi-Standard Compliance Scorecard API Routes - GL-VCCI Scope 3 Platform v1.1.0

FastAPI router providing endpoints for the multi-standard compliance scorecard
engine. Assesses emissions data against GHG Protocol, ESRS E1, CDP, IFRS S2,
and ISO 14083 standards with cross-standard gap analysis and prioritized
action items.

Endpoints:
    GET  /api/v1/compliance/scorecard                 - Full multi-standard scorecard
    GET  /api/v1/compliance/standard/{standard}       - Standard-specific coverage
    GET  /api/v1/compliance/gaps                      - Cross-standard gap analysis
    GET  /api/v1/compliance/evidence/{requirement_id} - Evidence trail for requirement
    GET  /api/v1/compliance/action-items              - Action items list
    GET  /api/v1/compliance/trend                     - Compliance trend over time

Version: 1.1.0
Date: 2026-03-01
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..agents.reporting.standards.compliance_scorecard import (
    ActionItem,
    ComplianceGap,
    ComplianceRequirement,
    ComplianceScorecard,
    ComplianceScorecardEngine,
    StandardCompliance,
)

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])

# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

# Current scorecard (last generated)
_current_scorecard: Optional[ComplianceScorecard] = None

# Historical scorecards for trend analysis (keyed by ISO timestamp)
_scorecard_history: List[ComplianceScorecard] = []

# Input data store for regeneration
_input_data_store: Dict[str, Any] = {}

# Engine instance
_engine = ComplianceScorecardEngine()


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class ScorecardRequest(BaseModel):
    """Request body for generating a compliance scorecard."""
    emissions_data: Dict[str, Any] = Field(..., description="Emissions data (Scope 1/2/3)")
    company_info: Dict[str, Any] = Field(..., description="Company information")
    cdp_questionnaire: Optional[Dict[str, Any]] = Field(None, description="CDP questionnaire response data")
    targets_data: Optional[Dict[str, Any]] = Field(None, description="Emission reduction targets")
    risks_data: Optional[Dict[str, Any]] = Field(None, description="Climate risks and opportunities")
    governance_data: Optional[Dict[str, Any]] = Field(None, description="Governance data")
    transport_data: Optional[Dict[str, Any]] = Field(None, description="Transport data for ISO 14083")


class StandardCoverageResponse(BaseModel):
    """Response for standard-specific coverage."""
    standard_name: str = Field(..., description="Standard name")
    standard_code: str = Field(..., description="Standard code")
    version: str = Field(default="", description="Standard version")
    coverage_pct: float = Field(default=0.0, description="Coverage percentage")
    total_requirements: int = Field(default=0, description="Total requirements")
    met_count: int = Field(default=0, description="Met requirements")
    partially_met_count: int = Field(default=0, description="Partially met requirements")
    not_met_count: int = Field(default=0, description="Not met requirements")
    not_applicable_count: int = Field(default=0, description="Not applicable requirements")
    predicted_score: Optional[str] = Field(None, description="Predicted score (CDP only)")
    requirements: List[ComplianceRequirement] = Field(default_factory=list, description="All requirements")
    summary: str = Field(default="", description="Assessment summary")


class EvidenceResponse(BaseModel):
    """Evidence trail for a specific requirement."""
    requirement_id: str = Field(..., description="Requirement ID")
    standard: str = Field(default="", description="Standard name")
    description: str = Field(default="", description="Requirement description")
    status: str = Field(default="not_assessed", description="Compliance status")
    evidence: Optional[str] = Field(None, description="Evidence data")
    data_fields_required: List[str] = Field(default_factory=list, description="Required data fields")
    notes: str = Field(default="", description="Assessment notes")
    criticality: str = Field(default="required", description="Requirement criticality")


class TrendDataPoint(BaseModel):
    """Single data point in the compliance trend."""
    timestamp: str = Field(..., description="Assessment timestamp")
    overall_score: float = Field(default=0.0, description="Overall compliance score")
    overall_grade: str = Field(default="", description="Letter grade")
    standards: Dict[str, float] = Field(default_factory=dict, description="Per-standard coverage")
    gap_count: int = Field(default=0, description="Number of gaps")
    action_item_count: int = Field(default=0, description="Number of action items")


class TrendResponse(BaseModel):
    """Compliance trend over time."""
    data_points: List[TrendDataPoint] = Field(default_factory=list, description="Historical data points")
    trend_direction: str = Field(default="stable", description="improving, declining, or stable")
    avg_score_change: float = Field(default=0.0, description="Average score change per assessment")
    summary: str = Field(default="", description="Trend summary")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional detail")


# ============================================================================
# VALID STANDARD CODES
# ============================================================================

VALID_STANDARDS = {"ghg_protocol", "esrs_e1", "cdp", "ifrs_s2", "iso_14083"}


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/scorecard",
    response_model=ComplianceScorecard,
    summary="Get full multi-standard compliance scorecard",
    responses={404: {"model": ErrorResponse, "description": "No scorecard generated yet"}},
)
async def get_scorecard() -> ComplianceScorecard:
    """
    Retrieve the most recently generated multi-standard compliance scorecard.

    Returns the full scorecard including per-standard compliance results,
    cross-standard gaps, action items, and the overall weighted score.

    Returns:
        ComplianceScorecard with complete multi-standard assessment.
    """
    if _current_scorecard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No compliance scorecard has been generated yet. POST to /scorecard to generate one.",
        )

    logger.info("Retrieved compliance scorecard: overall %.1f%% (%s)",
                _current_scorecard.overall_score, _current_scorecard.overall_grade)
    return _current_scorecard


@router.post(
    "/scorecard",
    response_model=ComplianceScorecard,
    summary="Generate a new compliance scorecard",
    status_code=status.HTTP_201_CREATED,
)
async def generate_scorecard(body: ScorecardRequest) -> ComplianceScorecard:
    """
    Generate a new multi-standard compliance scorecard.

    Assesses the provided data against GHG Protocol, ESRS E1, CDP, IFRS S2,
    and ISO 14083 standards. Identifies cross-standard gaps and generates
    prioritized action items.

    Args:
        body: Input data for compliance assessment.

    Returns:
        ComplianceScorecard with full assessment results.
    """
    global _current_scorecard

    logger.info("Generating new compliance scorecard for %s", body.company_info.get("name", "Unknown"))

    try:
        scorecard = _engine.generate_scorecard(
            emissions_data=body.emissions_data,
            company_info=body.company_info,
            cdp_questionnaire=body.cdp_questionnaire,
            targets_data=body.targets_data,
            risks_data=body.risks_data,
            governance_data=body.governance_data,
            transport_data=body.transport_data,
        )

        # Store as current
        _current_scorecard = scorecard

        # Add to history
        _scorecard_history.append(scorecard)
        # Keep only last 50 entries
        if len(_scorecard_history) > 50:
            _scorecard_history.pop(0)

        # Store input data
        _input_data_store.update(body.dict())

        logger.info(
            "Scorecard generated: overall %.1f%% (%s), %d gaps, %d actions",
            scorecard.overall_score, scorecard.overall_grade,
            len(scorecard.gaps), len(scorecard.action_items),
        )

        return scorecard

    except Exception as e:
        logger.error("Scorecard generation failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scorecard generation failed: {str(e)}",
        )


@router.get(
    "/standard/{standard}",
    response_model=StandardCoverageResponse,
    summary="Get standard-specific compliance coverage",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid standard code"},
        404: {"model": ErrorResponse, "description": "No scorecard generated yet"},
    },
)
async def get_standard_coverage(standard: str) -> StandardCoverageResponse:
    """
    Get compliance coverage for a specific reporting standard.

    Valid standard codes:
    - ghg_protocol (GHG Protocol Corporate Standard)
    - esrs_e1 (ESRS E1 / EU CSRD)
    - cdp (CDP Climate Change)
    - ifrs_s2 (IFRS S2 Climate Disclosures)
    - iso_14083 (ISO 14083 Transport)

    Args:
        standard: Standard code identifier.

    Returns:
        StandardCoverageResponse with per-requirement assessment.
    """
    standard_lower = standard.lower()

    if standard_lower not in VALID_STANDARDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid standard '{standard}'. Valid: {', '.join(sorted(VALID_STANDARDS))}.",
        )

    if _current_scorecard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No compliance scorecard has been generated yet.",
        )

    std_compliance = _current_scorecard.standards.get(standard_lower)
    if std_compliance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Standard '{standard_lower}' not found in current scorecard.",
        )

    return StandardCoverageResponse(
        standard_name=std_compliance.standard_name,
        standard_code=std_compliance.standard_code,
        version=std_compliance.version,
        coverage_pct=std_compliance.coverage_pct,
        total_requirements=std_compliance.total_requirements,
        met_count=std_compliance.met_count,
        partially_met_count=std_compliance.partially_met_count,
        not_met_count=std_compliance.not_met_count,
        not_applicable_count=std_compliance.not_applicable_count,
        predicted_score=std_compliance.predicted_score,
        requirements=std_compliance.requirements,
        summary=std_compliance.summary,
    )


@router.get(
    "/gaps",
    response_model=List[ComplianceGap],
    summary="Get cross-standard gap analysis",
    responses={404: {"model": ErrorResponse, "description": "No scorecard generated yet"}},
)
async def get_gaps(
    severity: Optional[str] = Query(None, description="Filter by severity: critical, high, medium, low"),
) -> List[ComplianceGap]:
    """
    Get cross-standard compliance gap analysis.

    Returns gaps that affect multiple standards, sorted by severity.
    Optionally filter by severity level.

    Args:
        severity: Optional severity filter.

    Returns:
        List of ComplianceGap objects.
    """
    if _current_scorecard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No compliance scorecard has been generated yet.",
        )

    gaps = _current_scorecard.gaps

    if severity:
        severity_lower = severity.lower()
        if severity_lower not in ("critical", "high", "medium", "low"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity '{severity}'. Valid: critical, high, medium, low.",
            )
        gaps = [g for g in gaps if g.severity == severity_lower]

    logger.info("Retrieved %d compliance gaps (filter: %s)", len(gaps), severity or "none")
    return gaps


@router.get(
    "/evidence/{requirement_id}",
    response_model=EvidenceResponse,
    summary="Get evidence trail for a specific requirement",
    responses={
        404: {"model": ErrorResponse, "description": "Requirement not found or no scorecard generated"},
    },
)
async def get_evidence(requirement_id: str) -> EvidenceResponse:
    """
    Get the evidence trail for a specific compliance requirement.

    Returns the assessment status, supporting evidence, data fields
    evaluated, and assessor notes for the specified requirement ID.

    Args:
        requirement_id: The requirement ID (e.g., GHG-001, ESRS-003, CDP-007).

    Returns:
        EvidenceResponse with evidence details.
    """
    if _current_scorecard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No compliance scorecard has been generated yet.",
        )

    # Search across all standards for the requirement
    for std_code, std_compliance in _current_scorecard.standards.items():
        for req in std_compliance.requirements:
            if req.id == requirement_id:
                return EvidenceResponse(
                    requirement_id=req.id,
                    standard=std_compliance.standard_name,
                    description=req.description,
                    status=req.status,
                    evidence=req.evidence,
                    data_fields_required=req.data_fields_required,
                    notes=req.notes,
                    criticality=req.criticality,
                )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Requirement '{requirement_id}' not found in current scorecard.",
    )


@router.get(
    "/action-items",
    response_model=List[ActionItem],
    summary="Get prioritized action items",
    responses={404: {"model": ErrorResponse, "description": "No scorecard generated yet"}},
)
async def get_action_items(
    priority: Optional[str] = Query(None, description="Filter by priority: critical, high, medium, low"),
    standard: Optional[str] = Query(None, description="Filter by standard code"),
) -> List[ActionItem]:
    """
    Get prioritized action items for closing compliance gaps.

    Each action item includes estimated effort, responsible role,
    deadline recommendation, and the gaps it addresses.

    Args:
        priority: Optional priority filter.
        standard: Optional standard filter.

    Returns:
        List of ActionItem objects sorted by priority.
    """
    if _current_scorecard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No compliance scorecard has been generated yet.",
        )

    items = _current_scorecard.action_items

    if priority:
        priority_lower = priority.lower()
        if priority_lower not in ("critical", "high", "medium", "low"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority '{priority}'. Valid: critical, high, medium, low.",
            )
        items = [i for i in items if i.priority == priority_lower]

    if standard:
        standard_lower = standard.lower()
        if standard_lower not in VALID_STANDARDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid standard '{standard}'. Valid: {', '.join(sorted(VALID_STANDARDS))}.",
            )
        items = [i for i in items if standard_lower in i.standards_affected]

    logger.info("Retrieved %d action items (priority=%s, standard=%s)",
                len(items), priority or "all", standard or "all")
    return items


@router.get(
    "/trend",
    response_model=TrendResponse,
    summary="Get compliance trend over time",
)
async def get_trend() -> TrendResponse:
    """
    Get the compliance score trend over time.

    Returns historical data points from previous scorecard generations,
    along with trend direction and average score change.

    Returns:
        TrendResponse with historical data and trend analysis.
    """
    if not _scorecard_history:
        return TrendResponse(
            data_points=[],
            trend_direction="stable",
            avg_score_change=0.0,
            summary="No historical data available. Generate at least two scorecards to see trends.",
        )

    data_points: List[TrendDataPoint] = []
    for scorecard in _scorecard_history:
        std_scores: Dict[str, float] = {}
        for std_code, std_compliance in scorecard.standards.items():
            std_scores[std_code] = std_compliance.coverage_pct

        data_points.append(TrendDataPoint(
            timestamp=scorecard.generated_at,
            overall_score=scorecard.overall_score,
            overall_grade=scorecard.overall_grade,
            standards=std_scores,
            gap_count=len(scorecard.gaps),
            action_item_count=len(scorecard.action_items),
        ))

    # Calculate trend
    if len(data_points) >= 2:
        scores = [dp.overall_score for dp in data_points]
        changes = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        avg_change = sum(changes) / len(changes) if changes else 0.0

        if avg_change > 1.0:
            direction = "improving"
        elif avg_change < -1.0:
            direction = "declining"
        else:
            direction = "stable"

        summary = (
            f"Compliance trend is {direction} across {len(data_points)} assessments. "
            f"Average score change: {avg_change:+.1f} points per assessment. "
            f"Latest score: {data_points[-1].overall_score:.1f}% ({data_points[-1].overall_grade})."
        )
    else:
        direction = "stable"
        avg_change = 0.0
        summary = f"Only one assessment available. Score: {data_points[0].overall_score:.1f}% ({data_points[0].overall_grade})."

    return TrendResponse(
        data_points=data_points,
        trend_direction=direction,
        avg_score_change=round(avg_change, 1),
        summary=summary,
    )


__all__ = ["router"]
