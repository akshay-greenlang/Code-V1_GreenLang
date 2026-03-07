# -*- coding: utf-8 -*-
"""
Verification Routes - AGENT-EUDR-004 Forest Cover Analysis API

Endpoints for deforestation-free verification including single-plot
verification, batch processing, stored verdict retrieval, evidence
package retrieval, and complete plot analysis combining all engines.

Endpoints:
    POST /              - Verify deforestation-free status for a plot
    POST /batch         - Batch deforestation-free verification
    GET  /{plot_id}     - Get stored verification verdict
    GET  /{plot_id}/evidence - Get evidence package for a verdict
    POST /complete      - Complete plot analysis (all engines + verdict)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.forest_cover_analysis.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_forest_cover_service,
    get_request_id,
    get_verification_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    BatchVerifyRequest,
    CompletePlotAnalysisRequest,
    DeforestationFreeResponse,
    DeforestationVerdict,
    EvidenceItem,
    PlotProfileResponse,
    VerifyDeforestationFreeRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deforestation-Free Verification"])


# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=DeforestationFreeResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify deforestation-free status",
    description=(
        "Verify that a production plot has been deforestation-free since "
        "the EUDR cutoff date (2020-12-31). Combines multiple analysis "
        "engines (density, classification, historical reconstruction, "
        "height, fragmentation, biomass) to produce a verdict with "
        "supporting evidence. Returns one of: deforestation_free, "
        "deforestation_detected, degradation_detected, inconclusive, "
        "or insufficient_data."
    ),
    responses={
        200: {"description": "Verification verdict with evidence"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_deforestation_free(
    body: VerifyDeforestationFreeRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:verification:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> DeforestationFreeResponse:
    """Verify deforestation-free status for a production plot.

    Coordinates all analysis engines to produce a comprehensive
    deforestation-free verdict with supporting evidence chain.

    Args:
        body: Verification request with plot polygon and commodity.
        user: Authenticated user with verification:write permission.

    Returns:
        DeforestationFreeResponse with verdict and evidence.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    verification_id = f"vrf-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Deforestation verification: user=%s plot_id=%s commodity=%s "
        "include_evidence=%s",
        user.user_id,
        body.plot_id,
        body.commodity.value,
        body.include_evidence,
    )

    try:
        engine = get_verification_engine()

        result = engine.verify(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            commodity=body.commodity.value,
            include_evidence=body.include_evidence,
        )

        elapsed = time.monotonic() - start

        # Build evidence items
        evidence = []
        if body.include_evidence:
            raw_evidence = getattr(result, "evidence", [])
            for item in raw_evidence:
                evidence.append(EvidenceItem(
                    evidence_type=getattr(item, "evidence_type", ""),
                    description=getattr(item, "description", ""),
                    date=getattr(item, "date", None),
                    source=getattr(item, "source", ""),
                    value=getattr(item, "value", None),
                    unit=getattr(item, "unit", None),
                    confidence=getattr(item, "confidence", 0.0),
                ))

        logger.info(
            "Deforestation verification completed: user=%s plot_id=%s "
            "verdict=%s confidence=%.2f engines=%d elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "verdict", "inconclusive"),
            getattr(result, "confidence", 0.0),
            len(getattr(result, "engines_used", [])),
            elapsed * 1000,
        )

        return DeforestationFreeResponse(
            request_id=get_request_id(),
            verification_id=getattr(
                result, "verification_id", verification_id
            ),
            plot_id=body.plot_id,
            commodity=body.commodity.value,
            verdict=getattr(
                result, "verdict", DeforestationVerdict.INCONCLUSIVE
            ),
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            forest_cover_at_cutoff_pct=getattr(
                result, "forest_cover_at_cutoff_pct", 0.0
            ),
            forest_cover_current_pct=getattr(
                result, "forest_cover_current_pct", 0.0
            ),
            forest_cover_change_pct=getattr(
                result, "forest_cover_change_pct", 0.0
            ),
            deforestation_area_ha=getattr(
                result, "deforestation_area_ha", 0.0
            ),
            degradation_area_ha=getattr(result, "degradation_area_ha", 0.0),
            engines_used=getattr(result, "engines_used", []),
            engine_agreement=getattr(result, "engine_agreement", 0.0),
            evidence=evidence,
            risk_level=getattr(result, "risk_level", "unknown"),
            confidence=getattr(result, "confidence", 0.0),
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Verification error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Verification failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deforestation verification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=List[DeforestationFreeResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch deforestation-free verification",
    description=(
        "Verify deforestation-free status for multiple plots in a single "
        "request. Supports up to 5,000 plots per batch."
    ),
    responses={
        200: {"description": "Batch verification results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_verify(
    body: BatchVerifyRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:verification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> List[DeforestationFreeResponse]:
    """Batch deforestation-free verification for multiple plots.

    Args:
        body: Batch request with list of verification requests.
        user: Authenticated user with verification:write permission.

    Returns:
        List of DeforestationFreeResponse for each plot.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Batch verification: user=%s plots=%d",
        user.user_id,
        len(body.plots),
    )

    try:
        engine = get_verification_engine()
        results = []

        for plot_req in body.plots:
            verification_id = f"vrf-{uuid.uuid4().hex[:12]}"
            try:
                result = engine.verify(
                    plot_id=plot_req.plot_id,
                    polygon_wkt=plot_req.polygon_wkt,
                    commodity=plot_req.commodity.value,
                    include_evidence=plot_req.include_evidence,
                )

                evidence = []
                if plot_req.include_evidence:
                    for item in getattr(result, "evidence", []):
                        evidence.append(EvidenceItem(
                            evidence_type=getattr(item, "evidence_type", ""),
                            description=getattr(item, "description", ""),
                            date=getattr(item, "date", None),
                            source=getattr(item, "source", ""),
                            value=getattr(item, "value", None),
                            unit=getattr(item, "unit", None),
                            confidence=getattr(item, "confidence", 0.0),
                        ))

                results.append(DeforestationFreeResponse(
                    request_id=get_request_id(),
                    verification_id=getattr(
                        result, "verification_id", verification_id
                    ),
                    plot_id=plot_req.plot_id,
                    commodity=plot_req.commodity.value,
                    verdict=getattr(
                        result, "verdict",
                        DeforestationVerdict.INCONCLUSIVE,
                    ),
                    evidence=evidence,
                    confidence=getattr(result, "confidence", 0.0),
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(result, "provenance_hash", ""),
                ))
            except Exception as plot_exc:
                logger.warning(
                    "Batch verify: plot %s failed: %s",
                    plot_req.plot_id,
                    plot_exc,
                )
                results.append(DeforestationFreeResponse(
                    request_id=get_request_id(),
                    verification_id=verification_id,
                    plot_id=plot_req.plot_id,
                    commodity=plot_req.commodity.value,
                    verdict=DeforestationVerdict.INSUFFICIENT_DATA,
                    confidence=0.0,
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch verification completed: user=%s plots=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            len(results),
            elapsed * 1000,
        )

        return results

    except Exception as exc:
        logger.error(
            "Batch verification failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch verification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=DeforestationFreeResponse,
    summary="Get stored verification verdict",
    description=(
        "Retrieve the most recent deforestation-free verification "
        "verdict for a production plot."
    ),
    responses={
        200: {"description": "Stored verification verdict"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Verdict not found"},
    },
)
async def get_verdict(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:verification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DeforestationFreeResponse:
    """Get the most recent stored verification verdict for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with verification:read permission.

    Returns:
        DeforestationFreeResponse with stored verdict.

    Raises:
        HTTPException: 404 if verdict not found.
    """
    logger.info(
        "Verdict retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        engine = get_verification_engine()
        result = engine.get_result(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No verification verdict found for plot {plot_id}",
            )

        # Reconstruct evidence items
        evidence = []
        for item in getattr(result, "evidence", []):
            evidence.append(EvidenceItem(
                evidence_type=getattr(item, "evidence_type", ""),
                description=getattr(item, "description", ""),
                date=getattr(item, "date", None),
                source=getattr(item, "source", ""),
                value=getattr(item, "value", None),
                unit=getattr(item, "unit", None),
                confidence=getattr(item, "confidence", 0.0),
            ))

        return DeforestationFreeResponse(
            request_id=get_request_id(),
            verification_id=getattr(result, "verification_id", ""),
            plot_id=plot_id,
            commodity=getattr(result, "commodity", ""),
            verdict=getattr(
                result, "verdict", DeforestationVerdict.INCONCLUSIVE
            ),
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            forest_cover_at_cutoff_pct=getattr(
                result, "forest_cover_at_cutoff_pct", 0.0
            ),
            forest_cover_current_pct=getattr(
                result, "forest_cover_current_pct", 0.0
            ),
            forest_cover_change_pct=getattr(
                result, "forest_cover_change_pct", 0.0
            ),
            deforestation_area_ha=getattr(
                result, "deforestation_area_ha", 0.0
            ),
            degradation_area_ha=getattr(result, "degradation_area_ha", 0.0),
            engines_used=getattr(result, "engines_used", []),
            engine_agreement=getattr(result, "engine_agreement", 0.0),
            evidence=evidence,
            risk_level=getattr(result, "risk_level", "unknown"),
            confidence=getattr(result, "confidence", 0.0),
            data_sources=getattr(result, "data_sources", []),
            timestamp=getattr(result, "timestamp", datetime.now(timezone.utc)),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Verdict retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verdict retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}/evidence
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}/evidence",
    response_model=List[EvidenceItem],
    summary="Get evidence package for a verdict",
    description=(
        "Retrieve the supporting evidence package for the most recent "
        "deforestation-free verification verdict of a plot."
    ),
    responses={
        200: {"description": "Evidence package"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Evidence not found"},
    },
)
async def get_evidence(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:verification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> List[EvidenceItem]:
    """Get the evidence package for a plot's verification verdict.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with verification:read permission.

    Returns:
        List of EvidenceItem supporting the verdict.

    Raises:
        HTTPException: 404 if evidence not found.
    """
    logger.info(
        "Evidence retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        engine = get_verification_engine()
        result = engine.get_result(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No verification evidence found for plot {plot_id}",
            )

        evidence = []
        for item in getattr(result, "evidence", []):
            evidence.append(EvidenceItem(
                evidence_type=getattr(item, "evidence_type", ""),
                description=getattr(item, "description", ""),
                date=getattr(item, "date", None),
                source=getattr(item, "source", ""),
                value=getattr(item, "value", None),
                unit=getattr(item, "unit", None),
                confidence=getattr(item, "confidence", 0.0),
            ))

        if not evidence:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No evidence items found for plot {plot_id}",
            )

        return evidence

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Evidence retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /complete
# ---------------------------------------------------------------------------


@router.post(
    "/complete",
    response_model=PlotProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Complete plot analysis (all engines + verdict)",
    description=(
        "Run a complete plot analysis using all available engines: "
        "canopy density, forest classification, historical reconstruction, "
        "canopy height, fragmentation analysis, biomass estimation, and "
        "deforestation-free verification. Returns a comprehensive plot "
        "profile with all results and the final EUDR verdict."
    ),
    responses={
        200: {"description": "Complete plot profile"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def complete_analysis(
    body: CompletePlotAnalysisRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:verification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> PlotProfileResponse:
    """Run complete plot analysis with all engines.

    Coordinates all analysis engines to produce a comprehensive
    plot profile including density, classification, historical cover,
    height, fragmentation, biomass, and the final deforestation-free
    verification verdict.

    Args:
        body: Complete analysis request with plot polygon and commodity.
        user: Authenticated user with verification:write permission.

    Returns:
        PlotProfileResponse with all analysis results.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Complete plot analysis: user=%s plot_id=%s commodity=%s "
        "include_all=%s",
        user.user_id,
        body.plot_id,
        body.commodity.value,
        body.include_all,
    )

    try:
        service = get_forest_cover_service()

        result = service.analyze_complete(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            commodity=body.commodity.value,
            include_all=body.include_all,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Complete analysis finished: user=%s plot_id=%s "
            "fao_status=%s confidence=%.2f elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "fao_forest_status", "unknown"),
            getattr(result, "overall_confidence", 0.0),
            elapsed * 1000,
        )

        return PlotProfileResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            area_ha=getattr(result, "area_ha", 0.0),
            density=getattr(result, "density", None),
            classification=getattr(result, "classification", None),
            historical=getattr(result, "historical", None),
            height=getattr(result, "height", None),
            fragmentation=getattr(result, "fragmentation", None),
            biomass=getattr(result, "biomass", None),
            verification=getattr(result, "verification", None),
            fao_forest_status=getattr(
                result, "fao_forest_status", "unknown"
            ),
            overall_confidence=getattr(result, "overall_confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Complete analysis error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Complete analysis failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complete plot analysis failed due to an internal error",
        )
