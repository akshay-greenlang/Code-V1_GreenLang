# -*- coding: utf-8 -*-
"""
Compliance & Risk Routes - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Endpoints for supplier risk assessment, risk propagation through the
supply chain, and compliance status monitoring with alerting.

Risk Endpoints:
    POST /risk/assess           - Assess supplier risk
    POST /risk/propagate        - Propagate risk through chain
    GET  /risk/{supplier_id}    - Get supplier risk profile
    POST /risk/batch            - Batch risk assessment

Compliance Endpoints:
    POST /compliance/check      - Check supplier compliance
    GET  /compliance/{id}       - Get compliance status
    POST /compliance/batch      - Batch compliance check
    GET  /compliance/alerts     - Get compliance alerts

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from greenlang.agents.eudr.multi_tier_supplier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_pagination,
    get_supplier_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
    validate_severity,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    BatchComplianceRequestSchema,
    BatchComplianceResponseSchema,
    BatchRiskRequestSchema,
    BatchRiskResponseSchema,
    ComplianceAlertsResponseSchema,
    ComplianceCheckRequestSchema,
    ComplianceCheckResponseSchema,
    ComplianceStatusSchema,
    RiskAssessmentRequestSchema,
    RiskPropagationRequestSchema,
    RiskPropagationResponseSchema,
    RiskScoreSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Risk & Compliance"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ==========================================================================
# RISK ENDPOINTS
# ==========================================================================


# ---------------------------------------------------------------------------
# POST /risk/assess
# ---------------------------------------------------------------------------


@router.post(
    "/risk/assess",
    response_model=RiskScoreSchema,
    status_code=status.HTTP_200_OK,
    summary="Assess supplier risk",
    description=(
        "Perform a risk assessment for a specific supplier across "
        "all configured risk categories: deforestation_proximity, "
        "country_risk, certification_gap, compliance_history, "
        "data_quality, and concentration_risk. Returns composite "
        "score and per-category breakdown."
    ),
    responses={
        200: {"description": "Risk assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_risk(
    body: RiskAssessmentRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:risk:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskScoreSchema:
    """Assess risk for a single supplier.

    Evaluates risk across all specified categories using deterministic
    scoring algorithms. No LLM calls for numeric calculations.

    Args:
        body: Risk assessment request with supplier ID and categories.
        request: FastAPI request object.
        user: Authenticated user with risk:write permission.

    Returns:
        RiskScoreSchema with composite and per-category scores.

    Raises:
        HTTPException: 400 on invalid input, 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Risk assess: user=%s supplier_id=%s categories=%s",
        user.user_id,
        body.supplier_id,
        body.risk_categories,
    )

    try:
        service = get_supplier_service()

        result = service.assess_risk(
            supplier_id=body.supplier_id,
            commodity=body.commodity,
            risk_categories=body.risk_categories,
            custom_weights=body.custom_weights,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {body.supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"risk_assess|{body.supplier_id}|"
            f"{result.get('composite_score', 0)}|{elapsed}"
        )

        logger.info(
            "Risk assessed: user=%s supplier_id=%s composite=%.1f "
            "level=%s elapsed_ms=%.1f",
            user.user_id,
            body.supplier_id,
            result.get("composite_score", 0),
            result.get("risk_level", "unknown"),
            elapsed * 1000,
        )

        return RiskScoreSchema(
            supplier_id=body.supplier_id,
            composite_score=result.get("composite_score", 0.0),
            risk_level=result.get("risk_level", "medium"),
            category_scores=result.get("category_scores", []),
            overall_trend=result.get("overall_trend", "stable"),
            next_assessment_due=result.get("next_assessment_due"),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Risk assess validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Risk assessment validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Risk assess failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk assessment failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /risk/propagate
# ---------------------------------------------------------------------------


@router.post(
    "/risk/propagate",
    response_model=RiskPropagationResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Propagate risk through supply chain",
    description=(
        "Propagate risk scores from deep-tier suppliers upstream to "
        "Tier 1 and the operator. Supports three methods: max "
        "(worst-case), weighted_average, and volume_weighted."
    ),
    responses={
        200: {"description": "Risk propagation completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def propagate_risk(
    body: RiskPropagationRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:risk:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskPropagationResponseSchema:
    """Propagate risk through a supply chain.

    Uses deterministic propagation algorithms to flow risk from
    deep-tier suppliers to the root supplier.

    Args:
        body: Propagation request with root supplier and method.
        request: FastAPI request object.
        user: Authenticated user with risk:write permission.

    Returns:
        RiskPropagationResponseSchema with propagation path.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Risk propagate: user=%s root=%s commodity=%s method=%s",
        user.user_id,
        body.root_supplier_id,
        body.commodity,
        body.propagation_method,
    )

    try:
        service = get_supplier_service()

        result = service.propagate_risk(
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            propagation_method=body.propagation_method,
            max_depth=body.max_depth,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"risk_propagate|{body.root_supplier_id}|{body.commodity}|"
            f"{body.propagation_method}|"
            f"{result.get('root_propagated_score', 0)}|{elapsed}"
        )

        logger.info(
            "Risk propagated: user=%s root=%s propagated_score=%.1f "
            "depth=%d suppliers=%d elapsed_ms=%.1f",
            user.user_id,
            body.root_supplier_id,
            result.get("root_propagated_score", 0),
            result.get("max_depth_reached", 0),
            result.get("total_suppliers_assessed", 0),
            elapsed * 1000,
        )

        return RiskPropagationResponseSchema(
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            propagation_method=body.propagation_method,
            root_propagated_score=result.get("root_propagated_score", 0.0),
            max_depth_reached=result.get("max_depth_reached", 0),
            total_suppliers_assessed=result.get("total_suppliers_assessed", 0),
            risk_path=result.get("risk_path", []),
            high_risk_suppliers=result.get("high_risk_suppliers", []),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Risk propagate validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Risk propagation validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Risk propagate failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk propagation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /risk/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/risk/{supplier_id}",
    response_model=RiskScoreSchema,
    status_code=status.HTTP_200_OK,
    summary="Get supplier risk profile",
    description=(
        "Retrieve the most recent risk assessment for a supplier "
        "including composite score, per-category breakdown, trend, "
        "and next assessment date."
    ),
    responses={
        200: {"description": "Risk profile retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_risk_profile(
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskScoreSchema:
    """Get the latest risk profile for a supplier.

    Args:
        supplier_id: Supplier identifier.
        request: FastAPI request object.
        user: Authenticated user with risk:read permission.

    Returns:
        RiskScoreSchema with latest assessment data.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Get risk profile: user=%s supplier_id=%s",
        user.user_id,
        supplier_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_risk_profile(supplier_id=supplier_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier risk profile not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"risk_get|{supplier_id}|{result.get('composite_score', 0)}|{elapsed}"
        )

        logger.info(
            "Risk profile retrieved: user=%s supplier_id=%s "
            "composite=%.1f elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            result.get("composite_score", 0),
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        result["elapsed_ms"] = elapsed * 1000
        return RiskScoreSchema(**result)

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get risk profile failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk profile retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /risk/batch
# ---------------------------------------------------------------------------


@router.post(
    "/risk/batch",
    response_model=BatchRiskResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch risk assessment",
    description=(
        "Assess risk for multiple suppliers in a single batch request. "
        "Each assessment is processed independently with individual "
        "error handling. Maximum 200 assessments per batch."
    ),
    responses={
        200: {"description": "Batch risk assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_risk_assessment(
    body: BatchRiskRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:risk:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchRiskResponseSchema:
    """Batch risk assessment for multiple suppliers.

    Args:
        body: Batch request with list of risk assessment requests.
        request: FastAPI request object.
        user: Authenticated user with risk:write permission.

    Returns:
        BatchRiskResponseSchema with per-supplier results and errors.

    Raises:
        HTTPException: 400 on validation error, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Batch risk assessment: user=%s count=%d",
        user.user_id,
        len(body.assessments),
    )

    try:
        service = get_supplier_service()
        results: List[RiskScoreSchema] = []
        errors: List[Dict[str, Any]] = []

        for idx, assessment in enumerate(body.assessments):
            try:
                result = service.assess_risk(
                    supplier_id=assessment.supplier_id,
                    commodity=assessment.commodity,
                    risk_categories=assessment.risk_categories,
                    custom_weights=assessment.custom_weights,
                )

                if result is not None:
                    item_provenance = _compute_provenance(
                        f"batch_risk|{assessment.supplier_id}|"
                        f"{result.get('composite_score', 0)}"
                    )
                    results.append(RiskScoreSchema(
                        supplier_id=assessment.supplier_id,
                        composite_score=result.get("composite_score", 0.0),
                        risk_level=result.get("risk_level", "medium"),
                        category_scores=result.get("category_scores", []),
                        overall_trend=result.get("overall_trend", "stable"),
                        elapsed_ms=0.0,
                        provenance_hash=item_provenance,
                    ))
                else:
                    errors.append({
                        "index": idx,
                        "supplier_id": assessment.supplier_id,
                        "error": "Supplier not found",
                    })

            except Exception as item_exc:
                errors.append({
                    "index": idx,
                    "supplier_id": assessment.supplier_id,
                    "error": str(item_exc),
                })

        elapsed = time.monotonic() - start
        batch_provenance = _compute_provenance(
            f"batch_risk|{len(results)}|{len(errors)}|{elapsed}"
        )

        logger.info(
            "Batch risk completed: user=%s assessed=%d errors=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            len(results),
            len(errors),
            elapsed * 1000,
        )

        return BatchRiskResponseSchema(
            total_assessed=len(results),
            results=results,
            errors=errors,
            elapsed_ms=elapsed * 1000,
            provenance_hash=batch_provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Batch risk validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch risk assessment validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch risk assessment failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch risk assessment failed due to an internal error",
        )


# ==========================================================================
# COMPLIANCE ENDPOINTS
# ==========================================================================


# ---------------------------------------------------------------------------
# POST /compliance/check
# ---------------------------------------------------------------------------


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Check supplier compliance",
    description=(
        "Perform a compliance check for a supplier across all "
        "configured dimensions: DDS validity, certification status, "
        "geolocation coverage, and deforestation-free verification. "
        "Returns composite score and per-dimension results."
    ),
    responses={
        200: {"description": "Compliance check completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_compliance(
    body: ComplianceCheckRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:compliance:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceCheckResponseSchema:
    """Check compliance status for a supplier.

    Evaluates all specified compliance dimensions using deterministic
    rules-based checking. Returns actionable remediation guidance.

    Args:
        body: Compliance check request with supplier ID and dimensions.
        request: FastAPI request object.
        user: Authenticated user with compliance:write permission.

    Returns:
        ComplianceCheckResponseSchema with status and remediation.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Compliance check: user=%s supplier_id=%s dimensions=%s",
        user.user_id,
        body.supplier_id,
        body.dimensions,
    )

    try:
        service = get_supplier_service()

        result = service.check_compliance(
            supplier_id=body.supplier_id,
            commodity=body.commodity,
            dimensions=body.dimensions,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {body.supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"compliance_check|{body.supplier_id}|"
            f"{result.get('compliance_status', 'unknown')}|"
            f"{result.get('composite_score', 0)}|{elapsed}"
        )

        logger.info(
            "Compliance checked: user=%s supplier_id=%s status=%s "
            "score=%.1f elapsed_ms=%.1f",
            user.user_id,
            body.supplier_id,
            result.get("compliance_status", "unknown"),
            result.get("composite_score", 0),
            elapsed * 1000,
        )

        return ComplianceCheckResponseSchema(
            supplier_id=body.supplier_id,
            compliance_status=result.get("compliance_status", "unverified"),
            composite_score=result.get("composite_score", 0.0),
            dimensions=result.get("dimensions", []),
            dds_impact=result.get("dds_impact", ""),
            remediation_required=result.get("remediation_required", False),
            remediation_actions=result.get("remediation_actions", []),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Compliance check validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Compliance check validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Compliance check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /compliance/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/compliance/{supplier_id}",
    response_model=ComplianceStatusSchema,
    status_code=status.HTTP_200_OK,
    summary="Get compliance status",
    description=(
        "Retrieve the current compliance status for a supplier "
        "including score, per-dimension status, trend, and history."
    ),
    responses={
        200: {"description": "Compliance status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_compliance_status(
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceStatusSchema:
    """Get current compliance status for a supplier.

    Args:
        supplier_id: Supplier identifier.
        request: FastAPI request object.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceStatusSchema with current status and trend.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Get compliance status: user=%s supplier_id=%s",
        user.user_id,
        supplier_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_compliance_status(supplier_id=supplier_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier compliance status not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"compliance_get|{supplier_id}|"
            f"{result.get('compliance_status', 'unknown')}|{elapsed}"
        )

        logger.info(
            "Compliance status retrieved: user=%s supplier_id=%s "
            "status=%s elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            result.get("compliance_status", "unknown"),
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        return ComplianceStatusSchema(**result)

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get compliance status failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance status retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /compliance/batch
# ---------------------------------------------------------------------------


@router.post(
    "/compliance/batch",
    response_model=BatchComplianceResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch compliance check",
    description=(
        "Check compliance for multiple suppliers in a single batch. "
        "Each check is processed independently. Maximum 200 checks per batch."
    ),
    responses={
        200: {"description": "Batch compliance check completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_compliance_check(
    body: BatchComplianceRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:compliance:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchComplianceResponseSchema:
    """Batch compliance check for multiple suppliers.

    Args:
        body: Batch request with list of compliance check requests.
        request: FastAPI request object.
        user: Authenticated user with compliance:write permission.

    Returns:
        BatchComplianceResponseSchema with per-supplier results.

    Raises:
        HTTPException: 400 on validation error, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Batch compliance check: user=%s count=%d",
        user.user_id,
        len(body.checks),
    )

    try:
        service = get_supplier_service()
        results: List[ComplianceCheckResponseSchema] = []
        errors: List[Dict[str, Any]] = []
        compliant_count = 0
        non_compliant_count = 0

        for idx, check_req in enumerate(body.checks):
            try:
                result = service.check_compliance(
                    supplier_id=check_req.supplier_id,
                    commodity=check_req.commodity,
                    dimensions=check_req.dimensions,
                )

                if result is not None:
                    item_provenance = _compute_provenance(
                        f"batch_compliance|{check_req.supplier_id}|"
                        f"{result.get('compliance_status', 'unknown')}"
                    )
                    resp = ComplianceCheckResponseSchema(
                        supplier_id=check_req.supplier_id,
                        compliance_status=result.get(
                            "compliance_status", "unverified"
                        ),
                        composite_score=result.get("composite_score", 0.0),
                        dimensions=result.get("dimensions", []),
                        dds_impact=result.get("dds_impact", ""),
                        remediation_required=result.get(
                            "remediation_required", False
                        ),
                        remediation_actions=result.get(
                            "remediation_actions", []
                        ),
                        elapsed_ms=0.0,
                        provenance_hash=item_provenance,
                    )
                    results.append(resp)

                    status_val = result.get("compliance_status", "unverified")
                    if status_val == "compliant":
                        compliant_count += 1
                    elif status_val in ("non_compliant", "expired"):
                        non_compliant_count += 1
                else:
                    errors.append({
                        "index": idx,
                        "supplier_id": check_req.supplier_id,
                        "error": "Supplier not found",
                    })

            except Exception as item_exc:
                errors.append({
                    "index": idx,
                    "supplier_id": check_req.supplier_id,
                    "error": str(item_exc),
                })

        elapsed = time.monotonic() - start
        batch_provenance = _compute_provenance(
            f"batch_compliance|{len(results)}|{compliant_count}|"
            f"{non_compliant_count}|{elapsed}"
        )

        logger.info(
            "Batch compliance completed: user=%s checked=%d "
            "compliant=%d non_compliant=%d errors=%d elapsed_ms=%.1f",
            user.user_id,
            len(results),
            compliant_count,
            non_compliant_count,
            len(errors),
            elapsed * 1000,
        )

        return BatchComplianceResponseSchema(
            total_checked=len(results),
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            results=results,
            errors=errors,
            elapsed_ms=elapsed * 1000,
            provenance_hash=batch_provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Batch compliance validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch compliance check validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch compliance check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch compliance check failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /compliance/alerts
# ---------------------------------------------------------------------------


@router.get(
    "/compliance/alerts",
    response_model=ComplianceAlertsResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get compliance alerts",
    description=(
        "Retrieve compliance alerts including status changes, "
        "expiry warnings, non-compliance notifications, and data "
        "gap alerts. Supports filtering by severity and pagination."
    ),
    responses={
        200: {"description": "Compliance alerts retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_compliance_alerts(
    severity: Optional[str] = Depends(validate_severity),
    acknowledged: Optional[bool] = Query(
        None, description="Filter by acknowledged status"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceAlertsResponseSchema:
    """Get compliance alerts with filtering.

    Args:
        severity: Optional severity filter.
        acknowledged: Optional acknowledged status filter.
        pagination: Pagination parameters.
        request: FastAPI request object.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceAlertsResponseSchema with alerts.

    Raises:
        HTTPException: 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Get compliance alerts: user=%s severity=%s acknowledged=%s "
        "limit=%d offset=%d",
        user.user_id,
        severity,
        acknowledged,
        pagination.limit,
        pagination.offset,
    )

    try:
        service = get_supplier_service()
        result = service.get_compliance_alerts(
            severity=severity,
            acknowledged=acknowledged,
            limit=pagination.limit,
            offset=pagination.offset,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        alerts = result.get("alerts", [])
        total = result.get("total_alerts", len(alerts))
        critical_count = sum(
            1 for a in alerts if a.get("severity") == "critical"
        )
        high_count = sum(
            1 for a in alerts if a.get("severity") == "high"
        )

        provenance = _compute_provenance(
            f"compliance_alerts|{total}|{critical_count}|{elapsed}"
        )

        logger.info(
            "Compliance alerts retrieved: user=%s total=%d "
            "critical=%d high=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            critical_count,
            high_count,
            elapsed * 1000,
        )

        return ComplianceAlertsResponseSchema(
            total_alerts=total,
            critical_count=critical_count,
            high_count=high_count,
            alerts=alerts,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=(pagination.offset + pagination.limit) < total,
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get compliance alerts failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance alerts retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
