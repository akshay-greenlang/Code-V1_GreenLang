# -*- coding: utf-8 -*-
"""
Compliance Assessment Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for full compliance assessment, single-category checks, listing,
detail retrieval, and historical trend tracking across all EUDR compliance
categories (deforestation-free, legality, due diligence, traceability,
documentation, risk assessment, risk mitigation, monitoring, reporting)
per EUDR Articles 3, 4, 8, 9, 10, 11, 12.

Endpoints:
    POST /compliance/assess                       - Full compliance assessment
    POST /compliance/check-category               - Check single category
    GET  /compliance                              - List assessments (paginated)
    GET  /compliance/{assessment_id}              - Get assessment details
    GET  /compliance/{assessment_id}/history      - Get assessment history

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, ComplianceAssessmentEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_compliance_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    CategoryAssessmentEntry,
    CategoryCheckRequest,
    CategoryCheckResponse,
    ComplianceAssessRequest,
    ComplianceAssessResponse,
    ComplianceCategoryEnum,
    ComplianceDetailResponse,
    ComplianceHistoryEntry,
    ComplianceHistoryResponse,
    ComplianceListEntry,
    ComplianceListResponse,
    ComplianceOutcomeEnum,
    ComplianceRecommendationEntry,
    EUDRCommodityEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance Assessment"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /compliance/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=ComplianceAssessResponse,
    status_code=status.HTTP_200_OK,
    summary="Full compliance assessment",
    description=(
        "Perform a comprehensive EUDR compliance assessment for an operator "
        "or supplier across all compliance categories. Evaluates documentation, "
        "certifications, red flags, and regulatory requirements. Returns "
        "per-category scores, overall score, and recommendations."
    ),
    responses={
        200: {"description": "Compliance assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_compliance(
    request: Request,
    body: ComplianceAssessRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:compliance:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> ComplianceAssessResponse:
    """Perform a full compliance assessment.

    Args:
        body: Assessment request with entity and scope.
        user: Authenticated user with compliance:create permission.

    Returns:
        ComplianceAssessResponse with full assessment results.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.assess(
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            commodity=body.commodity.value if body.commodity else None,
            country_code=body.country_code,
            include_documents=body.include_documents,
            include_certifications=body.include_certifications,
            include_red_flags=body.include_red_flags,
            include_recommendations=body.include_recommendations,
            assessment_scope=[c.value for c in body.assessment_scope]
            if body.assessment_scope else None,
            framework_ids=body.framework_ids,
            assessed_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Compliance assessment failed: insufficient data",
            )

        category_assessments = []
        for ca in result.get("category_assessments", []):
            category_assessments.append(
                CategoryAssessmentEntry(
                    category=ComplianceCategoryEnum(ca.get("category", "legality")),
                    outcome=ComplianceOutcomeEnum(ca.get("outcome", "pending")),
                    score=Decimal(str(ca.get("score", 0))),
                    requirements_met=ca.get("requirements_met", 0),
                    requirements_total=ca.get("requirements_total", 0),
                    issues=ca.get("issues", []),
                    regulatory_reference=ca.get("regulatory_reference"),
                )
            )

        recommendations = []
        for r in result.get("recommendations", []):
            recommendations.append(
                ComplianceRecommendationEntry(
                    recommendation_id=r.get("recommendation_id", ""),
                    category=ComplianceCategoryEnum(r.get("category", "legality")),
                    priority=r.get("priority", "medium"),
                    description=r.get("description", ""),
                    regulatory_reference=r.get("regulatory_reference"),
                    estimated_effort_days=r.get("estimated_effort_days"),
                )
            )

        overall_outcome = ComplianceOutcomeEnum(
            result.get("overall_outcome", "pending")
        )
        overall_score = Decimal(str(result.get("overall_score", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_assess:{body.operator_id}:{body.supplier_id}",
            str(overall_outcome.value),
        )

        logger.info(
            "Compliance assessed: outcome=%s score=%s categories=%d "
            "operator=%s supplier=%s user=%s",
            overall_outcome.value,
            overall_score,
            len(category_assessments),
            body.operator_id,
            body.supplier_id,
            user.user_id,
        )

        return ComplianceAssessResponse(
            assessment_id=result.get("assessment_id", ""),
            overall_outcome=overall_outcome,
            overall_score=overall_score,
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            commodity=body.commodity,
            country_code=body.country_code,
            category_assessments=category_assessments,
            total_requirements_met=result.get("total_requirements_met", 0),
            total_requirements=result.get("total_requirements", 0),
            red_flags_found=result.get("red_flags_found", 0),
            recommendations=recommendations,
            frameworks_assessed=result.get("frameworks_assessed", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "ComplianceAssessmentEngine",
                    "DocumentVerificationEngine",
                    "CertificationValidationEngine",
                    "RedFlagDetectionEngine",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance assessment failed",
        )


# ---------------------------------------------------------------------------
# POST /compliance/check-category
# ---------------------------------------------------------------------------


@router.post(
    "/check-category",
    response_model=CategoryCheckResponse,
    summary="Check single compliance category",
    description=(
        "Check compliance for a single EUDR category (e.g. deforestation-free, "
        "legality, traceability, documentation). Faster than full assessment "
        "for targeted checks."
    ),
    responses={
        200: {"description": "Category check completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def check_category(
    request: Request,
    body: CategoryCheckRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CategoryCheckResponse:
    """Check compliance for a single category.

    Args:
        body: Category check request.
        user: Authenticated user with compliance:read permission.

    Returns:
        CategoryCheckResponse with category assessment result.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.check_category(
            category=body.category.value,
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            commodity=body.commodity.value if body.commodity else None,
            country_code=body.country_code,
        )

        category_assessment = CategoryAssessmentEntry(
            category=body.category,
            outcome=ComplianceOutcomeEnum(result.get("outcome", "pending")),
            score=Decimal(str(result.get("score", 0))),
            requirements_met=result.get("requirements_met", 0),
            requirements_total=result.get("requirements_total", 0),
            issues=result.get("issues", []),
            regulatory_reference=result.get("regulatory_reference"),
        )

        recommendations = []
        for r in result.get("recommendations", []):
            recommendations.append(
                ComplianceRecommendationEntry(
                    recommendation_id=r.get("recommendation_id", ""),
                    category=body.category,
                    priority=r.get("priority", "medium"),
                    description=r.get("description", ""),
                    regulatory_reference=r.get("regulatory_reference"),
                    estimated_effort_days=r.get("estimated_effort_days"),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"category_check:{body.category.value}:{body.operator_id}",
            str(category_assessment.outcome.value),
        )

        logger.info(
            "Category check: category=%s outcome=%s score=%s user=%s",
            body.category.value,
            category_assessment.outcome.value,
            category_assessment.score,
            user.user_id,
        )

        return CategoryCheckResponse(
            category_assessment=category_assessment,
            recommendations=recommendations,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceAssessmentEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Category check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Category check failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ComplianceListResponse,
    summary="List compliance assessments",
    description=(
        "Retrieve a paginated list of compliance assessments with optional "
        "filtering by outcome, operator, supplier, and commodity."
    ),
    responses={
        200: {"description": "Assessments retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_assessments(
    request: Request,
    outcome: Optional[ComplianceOutcomeEnum] = Query(
        None, description="Filter by overall outcome"
    ),
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    supplier_id: Optional[str] = Query(
        None, description="Filter by supplier ID"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by commodity"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceListResponse:
    """List compliance assessments with pagination.

    Args:
        outcome: Optional outcome filter.
        operator_id: Optional operator filter.
        supplier_id: Optional supplier filter.
        commodity: Optional commodity filter.
        pagination: Pagination parameters.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceListResponse with paginated assessments.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.list_assessments(
            outcome=outcome.value if outcome else None,
            operator_id=operator_id,
            supplier_id=supplier_id,
            commodity=commodity.value if commodity else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        assessments = []
        for a in result.get("assessments", []):
            assessments.append(
                ComplianceListEntry(
                    assessment_id=a.get("assessment_id", ""),
                    overall_outcome=ComplianceOutcomeEnum(
                        a.get("overall_outcome", "pending")
                    ),
                    overall_score=Decimal(str(a.get("overall_score", 0))),
                    operator_id=a.get("operator_id"),
                    supplier_id=a.get("supplier_id"),
                    commodity=EUDRCommodityEnum(a["commodity"])
                    if a.get("commodity") else None,
                    assessed_at=a.get("assessed_at"),
                )
            )

        total = result.get("total", len(assessments))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_list:{outcome}:{operator_id}",
            str(total),
        )

        logger.info(
            "Assessments listed: total=%d user=%s",
            total,
            user.user_id,
        )

        return ComplianceListResponse(
            assessments=assessments,
            total_assessments=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceAssessmentEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Assessment listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment listing failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/{assessment_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{assessment_id}",
    response_model=ComplianceDetailResponse,
    summary="Get assessment details",
    description=(
        "Retrieve full details of a compliance assessment including "
        "per-category assessments, recommendations, and frameworks assessed."
    ),
    responses={
        200: {"description": "Assessment details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Assessment not found"},
    },
)
async def get_assessment_detail(
    assessment_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceDetailResponse:
    """Get detailed compliance assessment.

    Args:
        assessment_id: Unique assessment identifier.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceDetailResponse with full assessment details.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.get_detail(assessment_id=assessment_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment not found: {assessment_id}",
            )

        category_assessments = []
        for ca in result.get("category_assessments", []):
            category_assessments.append(
                CategoryAssessmentEntry(
                    category=ComplianceCategoryEnum(ca.get("category", "legality")),
                    outcome=ComplianceOutcomeEnum(ca.get("outcome", "pending")),
                    score=Decimal(str(ca.get("score", 0))),
                    requirements_met=ca.get("requirements_met", 0),
                    requirements_total=ca.get("requirements_total", 0),
                    issues=ca.get("issues", []),
                    regulatory_reference=ca.get("regulatory_reference"),
                )
            )

        recommendations = []
        for r in result.get("recommendations", []):
            recommendations.append(
                ComplianceRecommendationEntry(
                    recommendation_id=r.get("recommendation_id", ""),
                    category=ComplianceCategoryEnum(r.get("category", "legality")),
                    priority=r.get("priority", "medium"),
                    description=r.get("description", ""),
                    regulatory_reference=r.get("regulatory_reference"),
                    estimated_effort_days=r.get("estimated_effort_days"),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_detail:{assessment_id}",
            str(result.get("overall_outcome", "")),
        )

        logger.info(
            "Assessment detail retrieved: id=%s user=%s",
            assessment_id,
            user.user_id,
        )

        return ComplianceDetailResponse(
            assessment_id=result.get("assessment_id", assessment_id),
            overall_outcome=ComplianceOutcomeEnum(
                result.get("overall_outcome", "pending")
            ),
            overall_score=Decimal(str(result.get("overall_score", 0))),
            operator_id=result.get("operator_id"),
            supplier_id=result.get("supplier_id"),
            commodity=EUDRCommodityEnum(result["commodity"])
            if result.get("commodity") else None,
            country_code=result.get("country_code"),
            category_assessments=category_assessments,
            recommendations=recommendations,
            red_flags_found=result.get("red_flags_found", 0),
            frameworks_assessed=result.get("frameworks_assessed", []),
            assessed_at=result.get("assessed_at"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceAssessmentEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Assessment detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/{assessment_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/{assessment_id}/history",
    response_model=ComplianceHistoryResponse,
    summary="Get compliance assessment history",
    description=(
        "Retrieve the historical trend of compliance assessments for the "
        "same entity, including score changes over time and trend direction."
    ),
    responses={
        200: {"description": "Assessment history retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Assessment not found"},
    },
)
async def get_assessment_history(
    assessment_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceHistoryResponse:
    """Get compliance assessment history for an entity.

    Args:
        assessment_id: Assessment to retrieve history for.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceHistoryResponse with historical trend data.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.get_history(assessment_id=assessment_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment not found: {assessment_id}",
            )

        history = []
        for h in result.get("history", []):
            history.append(
                ComplianceHistoryEntry(
                    assessment_id=h.get("assessment_id", ""),
                    overall_outcome=ComplianceOutcomeEnum(
                        h.get("overall_outcome", "pending")
                    ),
                    overall_score=Decimal(str(h.get("overall_score", 0))),
                    assessed_at=h.get("assessed_at"),
                    change_from_previous=Decimal(str(h["change_from_previous"]))
                    if h.get("change_from_previous") is not None else None,
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_history:{assessment_id}",
            str(len(history)),
        )

        logger.info(
            "Assessment history retrieved: id=%s entries=%d user=%s",
            assessment_id,
            len(history),
            user.user_id,
        )

        return ComplianceHistoryResponse(
            assessment_id=assessment_id,
            history=history,
            total_assessments=len(history),
            trend=result.get("trend"),
            average_score=Decimal(str(result["average_score"]))
            if result.get("average_score") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceAssessmentEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Assessment history retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment history retrieval failed",
        )
