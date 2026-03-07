# -*- coding: utf-8 -*-
"""
Verification Routes - AGENT-EUDR-005 Land Use Change Detector API

Endpoints for EUDR cutoff date compliance verification including
single-plot verification, batch processing, stored verdict retrieval,
evidence package retrieval, and complete verification pipeline.

Endpoints:
    POST /cutoff        - Verify single plot against EUDR cutoff date
    POST /batch         - Batch cutoff verification
    GET  /{plot_id}     - Get stored verification result
    GET  /{plot_id}/evidence - Get evidence package
    POST /complete      - Complete verification pipeline

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.land_use_change.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_cutoff_engine,
    get_land_use_service,
    get_request_id,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    ComplianceVerdict,
    CutoffVerificationResult,
    EvidenceItem,
    EvidencePackage,
    LandUseCategory,
    TransitionType,
    VerificationBatchResponse,
    VerifyBatchRequest,
    VerifyCompleteRequest,
    VerifyCutoffRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cutoff Verification"])

# ---------------------------------------------------------------------------
# In-memory result store (replaced by database in production)
# ---------------------------------------------------------------------------

_verification_store: Dict[str, Dict[str, Any]] = {}


def _get_verification_store() -> Dict[str, Dict[str, Any]]:
    """Return the verification store. Replaceable for testing."""
    return _verification_store


# ---------------------------------------------------------------------------
# Helper: build evidence items from engine result
# ---------------------------------------------------------------------------


def _build_evidence_items(raw_evidence: list) -> List[EvidenceItem]:
    """Convert raw engine evidence objects to API schema.

    Args:
        raw_evidence: List of evidence objects from engine.

    Returns:
        List of EvidenceItem schema instances.
    """
    items = []
    for item in raw_evidence:
        items.append(
            EvidenceItem(
                evidence_type=getattr(item, "evidence_type", ""),
                description=getattr(item, "description", ""),
                date=getattr(item, "date", None),
                source=getattr(item, "source", ""),
                value=getattr(item, "value", None),
                unit=getattr(item, "unit", None),
                confidence=getattr(item, "confidence", 0.0),
            )
        )
    return items


# ---------------------------------------------------------------------------
# POST /cutoff
# ---------------------------------------------------------------------------


@router.post(
    "/cutoff",
    response_model=CutoffVerificationResult,
    status_code=status.HTTP_200_OK,
    summary="Verify EUDR cutoff date compliance",
    description=(
        "Verify that a production plot has not undergone deforestation "
        "or forest degradation since the EUDR cutoff date (31 December "
        "2020, per Article 2(1) of EU 2023/1115). Combines land use "
        "classification, transition detection, and trajectory analysis "
        "to produce a compliance verdict with supporting evidence."
    ),
    responses={
        200: {"description": "Verification verdict with evidence"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_cutoff(
    body: VerifyCutoffRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:verification:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CutoffVerificationResult:
    """Verify EUDR cutoff date compliance for a single plot.

    Args:
        body: Verification request with coordinates and commodity.
        user: Authenticated user with verification:write permission.

    Returns:
        CutoffVerificationResult with verdict and evidence.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-vrf-{uuid.uuid4().hex[:12]}"
    verification_id = f"vrf-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Cutoff verification: user=%s plot=%s lat=%.6f lon=%.6f "
        "commodity=%s include_evidence=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.commodity.value,
        body.include_evidence,
    )

    try:
        engine = get_cutoff_engine()

        result = engine.verify(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity.value,
            polygon_wkt=body.polygon_wkt,
            include_evidence=body.include_evidence,
        )

        elapsed = time.monotonic() - start

        # Build evidence items
        evidence = []
        if body.include_evidence:
            raw_evidence = getattr(result, "evidence", [])
            evidence = _build_evidence_items(raw_evidence)

        verdict = getattr(
            result, "verdict", ComplianceVerdict.INCONCLUSIVE
        )

        response = CutoffVerificationResult(
            request_id=get_request_id(),
            verification_id=getattr(
                result, "verification_id", verification_id
            ),
            plot_id=plot_id,
            verdict=verdict,
            commodity=body.commodity.value,
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            cutoff_classification=getattr(
                result, "cutoff_classification",
                LandUseCategory.OTHER,
            ),
            current_classification=getattr(
                result, "current_classification",
                LandUseCategory.OTHER,
            ),
            transition_detected=getattr(
                result, "transition_detected", False
            ),
            transition_type=getattr(result, "transition_type", None),
            evidence=evidence,
            confidence=getattr(result, "confidence", 0.0),
            engines_used=getattr(result, "engines_used", []),
            data_sources=getattr(result, "data_sources", []),
            latitude=body.latitude,
            longitude=body.longitude,
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_verification_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "verification_id": verification_id,
            "response_data": response.model_dump(mode="json"),
            "evidence_data": [e.model_dump(mode="json") for e in evidence],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Cutoff verification completed: user=%s plot=%s "
            "verdict=%s confidence=%.2f transition=%s elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            getattr(verdict, "value", verdict),
            response.confidence,
            response.transition_detected,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Verification error: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Verification failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cutoff verification failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=VerificationBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch cutoff verification",
    description=(
        "Verify EUDR cutoff date compliance for multiple plots in a "
        "single request. Supports up to 5000 plots per batch. Returns "
        "per-plot verdicts and aggregate compliance statistics."
    ),
    responses={
        200: {"description": "Batch verification results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_batch(
    body: VerifyBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:verification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> VerificationBatchResponse:
    """Batch verify cutoff compliance for multiple plots.

    Args:
        body: Batch request with list of plots.
        user: Authenticated user with verification:write permission.

    Returns:
        VerificationBatchResponse with results and statistics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    total = len(body.plots)

    logger.info(
        "Batch verification: user=%s plots=%d",
        user.user_id,
        total,
    )

    results: List[CutoffVerificationResult] = []
    successful = 0
    failed_count = 0
    compliant_count = 0
    non_compliant_count = 0
    degraded_count = 0
    inconclusive_count = 0
    verdict_counts: Dict[str, int] = {}

    try:
        engine = get_cutoff_engine()
        store = _get_verification_store()

        for plot_req in body.plots:
            plot_id = (
                plot_req.plot_id
                or f"luc-vrf-{uuid.uuid4().hex[:12]}"
            )
            verification_id = f"vrf-{uuid.uuid4().hex[:12]}"

            try:
                result = engine.verify(
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    commodity=plot_req.commodity.value,
                    polygon_wkt=plot_req.polygon_wkt,
                    include_evidence=body.include_evidence,
                )

                verdict = getattr(
                    result, "verdict",
                    ComplianceVerdict.INCONCLUSIVE,
                )
                verdict_val = (
                    verdict.value
                    if hasattr(verdict, "value")
                    else str(verdict)
                )

                evidence = []
                if body.include_evidence:
                    raw_ev = getattr(result, "evidence", [])
                    evidence = _build_evidence_items(raw_ev)

                verification = CutoffVerificationResult(
                    request_id=get_request_id(),
                    verification_id=verification_id,
                    plot_id=plot_id,
                    verdict=verdict,
                    commodity=plot_req.commodity.value,
                    cutoff_date=getattr(
                        result, "cutoff_date", "2020-12-31"
                    ),
                    cutoff_classification=getattr(
                        result, "cutoff_classification",
                        LandUseCategory.OTHER,
                    ),
                    current_classification=getattr(
                        result, "current_classification",
                        LandUseCategory.OTHER,
                    ),
                    transition_detected=getattr(
                        result, "transition_detected", False
                    ),
                    transition_type=getattr(
                        result, "transition_type", None
                    ),
                    evidence=evidence,
                    confidence=getattr(result, "confidence", 0.0),
                    engines_used=getattr(result, "engines_used", []),
                    data_sources=getattr(result, "data_sources", []),
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(verification)
                successful += 1

                verdict_counts[verdict_val] = (
                    verdict_counts.get(verdict_val, 0) + 1
                )
                if verdict_val == "compliant":
                    compliant_count += 1
                elif verdict_val == "non_compliant":
                    non_compliant_count += 1
                elif verdict_val == "degraded":
                    degraded_count += 1
                elif verdict_val == "inconclusive":
                    inconclusive_count += 1

                store[plot_id] = {
                    "plot_id": plot_id,
                    "verification_id": verification_id,
                    "response_data": verification.model_dump(
                        mode="json"
                    ),
                    "evidence_data": [
                        e.model_dump(mode="json") for e in evidence
                    ],
                    "created_at": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "created_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch verification failed for plot %s: %s",
                    plot_id,
                    exc,
                )
                failed_count += 1

        elapsed = time.monotonic() - start

        logger.info(
            "Batch verification completed: user=%s total=%d "
            "successful=%d failed=%d compliant=%d non_compliant=%d "
            "degraded=%d inconclusive=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed_count,
            compliant_count,
            non_compliant_count,
            degraded_count,
            inconclusive_count,
            elapsed * 1000,
        )

        return VerificationBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed_count,
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            degraded_count=degraded_count,
            inconclusive_count=inconclusive_count,
            verdict_distribution=verdict_counts,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch verification failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch verification failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=CutoffVerificationResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored verification result",
    description=(
        "Retrieve a previously computed verification result by plot ID."
    ),
    responses={
        200: {"description": "Verification result"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_verification(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:verification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CutoffVerificationResult:
    """Retrieve a stored verification result by plot ID.

    Args:
        plot_id: Plot identifier to look up.
        user: Authenticated user with verification:read permission.

    Returns:
        CutoffVerificationResult from the store.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_verification_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No verification result found for plot_id '{plot_id}'"
            ),
        )

    record = store[plot_id]
    return CutoffVerificationResult(**record["response_data"])


# ---------------------------------------------------------------------------
# GET /{plot_id}/evidence
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}/evidence",
    response_model=EvidencePackage,
    status_code=status.HTTP_200_OK,
    summary="Get evidence package",
    description=(
        "Retrieve the complete evidence package for a verification "
        "verdict, including classification, transition, and trajectory "
        "evidence with satellite imagery references."
    ),
    responses={
        200: {"description": "Evidence package"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_evidence(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:verification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> EvidencePackage:
    """Retrieve the evidence package for a verification verdict.

    Returns categorized evidence items (classification, transition,
    trajectory) and satellite imagery references used to produce
    the compliance verdict.

    Args:
        plot_id: Plot identifier to look up evidence for.
        user: Authenticated user with verification:read permission.

    Returns:
        EvidencePackage with all evidence items.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_verification_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No evidence found for plot_id '{plot_id}'. "
                "Run verification with include_evidence=true first."
            ),
        )

    record = store[plot_id]
    evidence_data = record.get("evidence_data", [])
    all_evidence = [EvidenceItem(**e) for e in evidence_data]

    # Categorize evidence
    classification_evidence = [
        e for e in all_evidence
        if "classif" in e.evidence_type.lower()
    ]
    transition_evidence = [
        e for e in all_evidence
        if "transition" in e.evidence_type.lower()
    ]
    trajectory_evidence = [
        e for e in all_evidence
        if "trajectory" in e.evidence_type.lower()
    ]

    # Extract satellite imagery references
    satellite_refs = [
        e.source for e in all_evidence
        if e.source and (
            "sentinel" in e.source.lower()
            or "landsat" in e.source.lower()
            or "modis" in e.source.lower()
        )
    ]

    return EvidencePackage(
        plot_id=plot_id,
        evidence_items=all_evidence,
        classification_evidence=classification_evidence,
        transition_evidence=transition_evidence,
        trajectory_evidence=trajectory_evidence,
        satellite_imagery_refs=list(set(satellite_refs)),
        total_evidence_items=len(all_evidence),
        provenance_hash=record.get("response_data", {}).get(
            "provenance_hash", ""
        ),
    )


# ---------------------------------------------------------------------------
# POST /complete
# ---------------------------------------------------------------------------


@router.post(
    "/complete",
    response_model=CutoffVerificationResult,
    status_code=status.HTTP_200_OK,
    summary="Complete verification pipeline",
    description=(
        "Run the full verification pipeline combining land use "
        "classification, transition detection, trajectory analysis, "
        "cutoff verification, and optional risk assessment in a single "
        "request. Returns a comprehensive verdict with all supporting "
        "evidence consolidated."
    ),
    responses={
        200: {"description": "Complete verification result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_complete(
    body: VerifyCompleteRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:verification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> CutoffVerificationResult:
    """Run the complete verification pipeline.

    Orchestrates all analysis engines (classification, transition,
    trajectory, cutoff verification, and optionally risk assessment)
    in a single request for comprehensive EUDR compliance assessment.

    Args:
        body: Complete verification request with all options.
        user: Authenticated user with verification:write permission.

    Returns:
        CutoffVerificationResult with comprehensive verdict.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-vrf-{uuid.uuid4().hex[:12]}"
    verification_id = f"vrf-complete-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Complete verification: user=%s plot=%s lat=%.6f lon=%.6f "
        "commodity=%s include_trajectory=%s include_risk=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.commodity.value,
        body.include_trajectory,
        body.include_risk,
    )

    try:
        service = get_land_use_service()

        result = service.verify_complete(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity.value,
            polygon_wkt=body.polygon_wkt,
            include_evidence=body.include_evidence,
            include_trajectory=body.include_trajectory,
            include_risk=body.include_risk,
        )

        elapsed = time.monotonic() - start

        # Build evidence
        evidence = []
        if body.include_evidence:
            raw_evidence = getattr(result, "evidence", [])
            evidence = _build_evidence_items(raw_evidence)

        # Build risk assessment summary
        risk_assessment = None
        if body.include_risk:
            risk_assessment = getattr(
                result, "risk_assessment", None
            )
            if risk_assessment is not None and hasattr(
                risk_assessment, "__dict__"
            ):
                risk_assessment = {
                    "composite_score": getattr(
                        risk_assessment, "composite_score", 0.0
                    ),
                    "risk_tier": getattr(
                        risk_assessment, "risk_tier", "low"
                    ),
                    "probability_12m": getattr(
                        risk_assessment, "probability_12m", 0.0
                    ),
                }

        verdict = getattr(
            result, "verdict", ComplianceVerdict.INCONCLUSIVE
        )

        response = CutoffVerificationResult(
            request_id=get_request_id(),
            verification_id=getattr(
                result, "verification_id", verification_id
            ),
            plot_id=plot_id,
            verdict=verdict,
            commodity=body.commodity.value,
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            cutoff_classification=getattr(
                result, "cutoff_classification",
                LandUseCategory.OTHER,
            ),
            current_classification=getattr(
                result, "current_classification",
                LandUseCategory.OTHER,
            ),
            transition_detected=getattr(
                result, "transition_detected", False
            ),
            transition_type=getattr(result, "transition_type", None),
            trajectory=getattr(result, "trajectory", None),
            evidence=evidence,
            risk_assessment=risk_assessment,
            confidence=getattr(result, "confidence", 0.0),
            engines_used=getattr(result, "engines_used", []),
            data_sources=getattr(result, "data_sources", []),
            latitude=body.latitude,
            longitude=body.longitude,
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_verification_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "verification_id": verification_id,
            "response_data": response.model_dump(mode="json"),
            "evidence_data": [
                e.model_dump(mode="json") for e in evidence
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Complete verification completed: user=%s plot=%s "
            "verdict=%s confidence=%.2f engines=%d evidence=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            getattr(verdict, "value", verdict),
            response.confidence,
            len(response.engines_used),
            len(evidence),
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Complete verification error: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Complete verification failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Complete verification failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
