# -*- coding: utf-8 -*-
"""
Cutoff Date Verification Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for EUDR cutoff date verification (31 December 2020 per Article 2(1))
with multi-source temporal evidence, pre/post-cutoff classification, 90-day
grace period handling, and 0.85 confidence threshold.

Endpoints:
    POST /cutoff/verify                       - Verify single detection against cutoff
    POST /cutoff/batch-verify                 - Batch verify multiple detections
    GET  /cutoff/{detection_id}/evidence      - Get temporal evidence for detection
    GET  /cutoff/{detection_id}/timeline      - Get forest state timeline

EUDR Cutoff: 31 December 2020 (Article 2(1))
Grace Period: 90 days pre-cutoff
Confidence Threshold: 0.85

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, CutoffDateVerifier Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    get_cutoff_verifier,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    CutoffBatchResultEntry,
    CutoffBatchVerifyRequest,
    CutoffBatchVerifyResponse,
    CutoffEvidenceResponse,
    CutoffResultEnum,
    CutoffTimelineResponse,
    CutoffVerifyRequest,
    CutoffVerifyResponse,
    ErrorResponse,
    EvidenceQualityEnum,
    ForestStateEntry,
    MetadataSchema,
    ProvenanceInfo,
    SatelliteSourceEnum,
    TemporalEvidenceEntry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cutoff", tags=["Cutoff Date Verification"])

# EUDR cutoff date constant
EUDR_CUTOFF_DATE = date(2020, 12, 31)


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /cutoff/verify
# ---------------------------------------------------------------------------


@router.post(
    "/verify",
    response_model=CutoffVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify detection against EUDR cutoff date",
    description=(
        "Verify whether a satellite detection occurred before or after the "
        "EUDR cutoff date (31 December 2020) using multi-source temporal "
        "evidence. Returns PRE_CUTOFF, POST_CUTOFF, GRACE_PERIOD, or "
        "INCONCLUSIVE classification with confidence score."
    ),
    responses={
        200: {"description": "Cutoff verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Detection not found"},
    },
)
async def verify_cutoff(
    request: Request,
    body: CutoffVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:cutoff:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CutoffVerifyResponse:
    """Verify a detection against the EUDR cutoff date.

    Args:
        body: Verification request with detection details.
        user: Authenticated user with cutoff:create permission.

    Returns:
        CutoffVerifyResponse with verification result.
    """
    start = time.monotonic()

    try:
        engine = get_cutoff_verifier()
        result = engine.verify(
            detection_id=body.detection_id,
            detection_date=body.detection_date,
            latitude=float(body.latitude),
            longitude=float(body.longitude),
            sources=[s.value for s in body.sources] if body.sources else None,
            confidence_threshold=float(body.confidence_threshold)
            if body.confidence_threshold else None,
        )

        cutoff_result = CutoffResultEnum(result.get("cutoff_result", "inconclusive"))
        days_from_cutoff = (body.detection_date - EUDR_CUTOFF_DATE).days

        evidence = []
        for ev in result.get("evidence", []):
            evidence.append(
                TemporalEvidenceEntry(
                    source=SatelliteSourceEnum(ev.get("source", "sentinel2")),
                    observation_date=ev.get("observation_date"),
                    forest_cover_pct=Decimal(str(ev.get("forest_cover_pct", 0)))
                    if ev.get("forest_cover_pct") is not None else None,
                    ndvi_value=Decimal(str(ev.get("ndvi_value", 0)))
                    if ev.get("ndvi_value") is not None else None,
                    quality=EvidenceQualityEnum(ev.get("quality", "medium")),
                    confidence=Decimal(str(ev.get("confidence", 0))),
                )
            )

        confidence = Decimal(str(result.get("confidence", 0)))
        is_compliant = cutoff_result in (
            CutoffResultEnum.PRE_CUTOFF,
            CutoffResultEnum.INCONCLUSIVE,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"verify_cutoff:{body.detection_id}:{body.detection_date}",
            str(cutoff_result.value),
        )

        logger.info(
            "Cutoff verification: detection_id=%s result=%s confidence=%s days=%d operator=%s",
            body.detection_id,
            cutoff_result.value,
            confidence,
            days_from_cutoff,
            user.operator_id or user.user_id,
        )

        return CutoffVerifyResponse(
            detection_id=body.detection_id,
            cutoff_result=cutoff_result,
            cutoff_date=EUDR_CUTOFF_DATE,
            detection_date=body.detection_date,
            days_from_cutoff=days_from_cutoff,
            confidence=confidence,
            evidence=evidence,
            evidence_count=len(evidence),
            is_compliant=is_compliant,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=list({ev.source.value for ev in evidence}),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Cutoff verification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cutoff date verification failed",
        )


# ---------------------------------------------------------------------------
# POST /cutoff/batch-verify
# ---------------------------------------------------------------------------


@router.post(
    "/batch-verify",
    response_model=CutoffBatchVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch verify detections against cutoff date",
    description=(
        "Verify multiple detections against the EUDR cutoff date in a single "
        "batch request. Each verification is independent; partial failures "
        "return per-detection error messages."
    ),
    responses={
        200: {"description": "Batch verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_verify_cutoff(
    request: Request,
    body: CutoffBatchVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:cutoff:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> CutoffBatchVerifyResponse:
    """Batch verify detections against the EUDR cutoff date.

    Args:
        body: Batch verification request.
        user: Authenticated user with cutoff:create permission.

    Returns:
        CutoffBatchVerifyResponse with per-detection results.
    """
    start = time.monotonic()

    try:
        engine = get_cutoff_verifier()
        results: List[CutoffBatchResultEntry] = []
        pre_count = 0
        post_count = 0
        inconclusive_count = 0
        failed_count = 0

        for verification in body.verifications:
            try:
                result = engine.verify(
                    detection_id=verification.detection_id,
                    detection_date=verification.detection_date,
                    latitude=float(verification.latitude),
                    longitude=float(verification.longitude),
                    sources=[s.value for s in verification.sources]
                    if verification.sources else None,
                    confidence_threshold=float(verification.confidence_threshold)
                    if verification.confidence_threshold else None,
                )

                cutoff_result = CutoffResultEnum(
                    result.get("cutoff_result", "inconclusive")
                )
                confidence = Decimal(str(result.get("confidence", 0)))
                is_compliant = cutoff_result in (
                    CutoffResultEnum.PRE_CUTOFF,
                    CutoffResultEnum.INCONCLUSIVE,
                )

                results.append(
                    CutoffBatchResultEntry(
                        detection_id=verification.detection_id,
                        cutoff_result=cutoff_result,
                        confidence=confidence,
                        is_compliant=is_compliant,
                    )
                )

                if cutoff_result == CutoffResultEnum.PRE_CUTOFF:
                    pre_count += 1
                elif cutoff_result == CutoffResultEnum.POST_CUTOFF:
                    post_count += 1
                else:
                    inconclusive_count += 1

            except Exception as exc:
                results.append(
                    CutoffBatchResultEntry(
                        detection_id=verification.detection_id,
                        cutoff_result=CutoffResultEnum.INCONCLUSIVE,
                        confidence=Decimal("0"),
                        is_compliant=False,
                        error=str(exc),
                    )
                )
                failed_count += 1

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"batch_verify:{len(body.verifications)}",
            f"pre={pre_count}/post={post_count}/inc={inconclusive_count}",
        )

        logger.info(
            "Batch cutoff verification: total=%d pre=%d post=%d inc=%d failed=%d operator=%s",
            len(body.verifications),
            pre_count,
            post_count,
            inconclusive_count,
            failed_count,
            user.operator_id or user.user_id,
        )

        return CutoffBatchVerifyResponse(
            results=results,
            total_verified=len(body.verifications),
            pre_cutoff_count=pre_count,
            post_cutoff_count=post_count,
            inconclusive_count=inconclusive_count,
            failed_count=failed_count,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["CutoffDateVerifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch cutoff verification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch cutoff verification failed",
        )


# ---------------------------------------------------------------------------
# GET /cutoff/{detection_id}/evidence
# ---------------------------------------------------------------------------


@router.get(
    "/{detection_id}/evidence",
    response_model=CutoffEvidenceResponse,
    summary="Get temporal evidence for a detection",
    description=(
        "Retrieve all temporal evidence used for cutoff date verification "
        "of a specific detection, including observation dates, forest cover "
        "measurements, and evidence quality classifications."
    ),
    responses={
        200: {"description": "Evidence retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Detection not found"},
    },
)
async def get_cutoff_evidence(
    detection_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:cutoff:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CutoffEvidenceResponse:
    """Get temporal evidence for a detection's cutoff verification.

    Args:
        detection_id: Detection identifier.
        user: Authenticated user with cutoff:read permission.

    Returns:
        CutoffEvidenceResponse with evidence records.
    """
    start = time.monotonic()

    try:
        engine = get_cutoff_verifier()
        result = engine.get_evidence(detection_id=detection_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection not found: {detection_id}",
            )

        evidence = []
        for ev in result.get("evidence", []):
            evidence.append(
                TemporalEvidenceEntry(
                    source=SatelliteSourceEnum(ev.get("source", "sentinel2")),
                    observation_date=ev.get("observation_date"),
                    forest_cover_pct=Decimal(str(ev.get("forest_cover_pct", 0)))
                    if ev.get("forest_cover_pct") is not None else None,
                    ndvi_value=Decimal(str(ev.get("ndvi_value", 0)))
                    if ev.get("ndvi_value") is not None else None,
                    quality=EvidenceQualityEnum(ev.get("quality", "medium")),
                    confidence=Decimal(str(ev.get("confidence", 0))),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"evidence:{detection_id}", str(len(evidence))
        )

        logger.info(
            "Cutoff evidence retrieved: detection_id=%s evidence_count=%d operator=%s",
            detection_id,
            len(evidence),
            user.operator_id or user.user_id,
        )

        return CutoffEvidenceResponse(
            detection_id=detection_id,
            evidence=evidence,
            total_evidence=len(evidence),
            earliest_observation=result.get("earliest_observation"),
            latest_observation=result.get("latest_observation"),
            average_confidence=Decimal(str(result.get("average_confidence", 0)))
            if result.get("average_confidence") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=list({ev.source.value for ev in evidence}),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Cutoff evidence retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cutoff evidence retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /cutoff/{detection_id}/timeline
# ---------------------------------------------------------------------------


@router.get(
    "/{detection_id}/timeline",
    response_model=CutoffTimelineResponse,
    summary="Get forest state timeline for a detection",
    description=(
        "Retrieve a chronological timeline of forest state observations at "
        "a detection location, showing forest cover changes relative to the "
        "EUDR cutoff date (2020-12-31)."
    ),
    responses={
        200: {"description": "Timeline retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Detection not found"},
    },
)
async def get_forest_timeline(
    detection_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:cutoff:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CutoffTimelineResponse:
    """Get forest state timeline for a detection location.

    Args:
        detection_id: Detection identifier.
        user: Authenticated user with cutoff:read permission.

    Returns:
        CutoffTimelineResponse with forest state timeline.
    """
    start = time.monotonic()

    try:
        engine = get_cutoff_verifier()
        result = engine.get_timeline(detection_id=detection_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection not found: {detection_id}",
            )

        timeline = []
        for entry in result.get("timeline", []):
            obs_date = entry.get("observation_date")
            is_cutoff_period = False
            if obs_date:
                days_diff = abs((obs_date - EUDR_CUTOFF_DATE).days)
                is_cutoff_period = days_diff <= 90

            timeline.append(
                ForestStateEntry(
                    observation_date=obs_date,
                    forest_cover_pct=Decimal(str(entry.get("forest_cover_pct", 0))),
                    canopy_density=Decimal(str(entry.get("canopy_density", 0)))
                    if entry.get("canopy_density") is not None else None,
                    change_from_prior=Decimal(str(entry.get("change_from_prior", 0)))
                    if entry.get("change_from_prior") is not None else None,
                    source=SatelliteSourceEnum(entry.get("source", "sentinel2")),
                    is_cutoff_period=is_cutoff_period,
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"timeline:{detection_id}", str(len(timeline))
        )

        logger.info(
            "Forest timeline retrieved: detection_id=%s observations=%d operator=%s",
            detection_id,
            len(timeline),
            user.operator_id or user.user_id,
        )

        return CutoffTimelineResponse(
            detection_id=detection_id,
            timeline=timeline,
            cutoff_date=EUDR_CUTOFF_DATE,
            forest_cover_at_cutoff=Decimal(str(result.get("forest_cover_at_cutoff", 0)))
            if result.get("forest_cover_at_cutoff") is not None else None,
            current_forest_cover=Decimal(str(result.get("current_forest_cover", 0)))
            if result.get("current_forest_cover") is not None else None,
            total_change_pct=Decimal(str(result.get("total_change_pct", 0)))
            if result.get("total_change_pct") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=list({e.source.value for e in timeline}),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Forest timeline retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forest state timeline retrieval failed",
        )
