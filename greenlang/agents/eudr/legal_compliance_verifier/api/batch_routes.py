# -*- coding: utf-8 -*-
"""
Batch Processing Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for batch compliance assessment, batch document verification,
and batch status monitoring for high-volume operations.

Endpoints:
    POST /batch/assess              - Batch compliance assessment
    POST /batch/verify              - Batch document verification
    GET  /batch/{batch_id}/status   - Get batch processing status

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, Batch Processing
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    get_compliance_engine,
    get_document_engine,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    BatchAssessRequest,
    BatchAssessResponse,
    BatchAssessResultEntry,
    BatchStatusEnum,
    BatchStatusResponse,
    BatchVerifyRequest,
    BatchVerifyResponse,
    BatchVerifyResultEntry,
    ComplianceOutcomeEnum,
    DocumentStatusEnum,
    ErrorResponse,
    MetadataSchema,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /batch/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=BatchAssessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch compliance assessment",
    description=(
        "Submit multiple operators or suppliers for batch compliance "
        "assessment. Up to 500 entities per request. Returns individual "
        "assessment results for each entity."
    ),
    responses={
        202: {"description": "Batch assessment started"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_assess(
    request: Request,
    body: BatchAssessRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:batch:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> BatchAssessResponse:
    """Perform batch compliance assessment.

    Args:
        body: Batch assessment request with entity IDs.
        user: Authenticated user with batch:create permission.

    Returns:
        BatchAssessResponse with per-entity results.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.batch_assess(
            entity_ids=body.entity_ids,
            entity_type=body.entity_type,
            commodity=body.commodity.value if body.commodity else None,
            assessment_scope=[c.value for c in body.assessment_scope]
            if body.assessment_scope else None,
            assessed_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch assessment failed: invalid parameters",
            )

        results = []
        for r in result.get("results", []):
            results.append(
                BatchAssessResultEntry(
                    entity_id=r.get("entity_id", ""),
                    assessment_id=r.get("assessment_id"),
                    overall_outcome=ComplianceOutcomeEnum(r["overall_outcome"])
                    if r.get("overall_outcome") else None,
                    overall_score=Decimal(str(r["overall_score"]))
                    if r.get("overall_score") is not None else None,
                    error=r.get("error"),
                )
            )

        completed = sum(1 for r in results if r.assessment_id is not None)
        failed = sum(1 for r in results if r.error is not None)

        batch_status = BatchStatusEnum.COMPLETED
        if failed > 0 and completed > 0:
            batch_status = BatchStatusEnum.PARTIALLY_COMPLETED
        elif failed == len(results):
            batch_status = BatchStatusEnum.FAILED

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"batch_assess:{len(body.entity_ids)}:{body.entity_type}",
            str(completed),
        )

        logger.info(
            "Batch assessment: submitted=%d completed=%d failed=%d user=%s",
            len(body.entity_ids),
            completed,
            failed,
            user.user_id,
        )

        return BatchAssessResponse(
            batch_id=result.get("batch_id", ""),
            status=batch_status,
            total_submitted=len(body.entity_ids),
            completed_count=completed,
            failed_count=failed,
            results=results,
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
        logger.error("Batch assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch assessment failed",
        )


# ---------------------------------------------------------------------------
# POST /batch/verify
# ---------------------------------------------------------------------------


@router.post(
    "/verify",
    response_model=BatchVerifyResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch document verification",
    description=(
        "Submit multiple documents for batch verification. Up to 200 "
        "documents per request. Returns individual verification results "
        "and red flag counts for each document."
    ),
    responses={
        202: {"description": "Batch verification started"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_verify(
    request: Request,
    body: BatchVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:batch:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> BatchVerifyResponse:
    """Perform batch document verification.

    Args:
        body: Batch verification request with documents.
        user: Authenticated user with batch:create permission.

    Returns:
        BatchVerifyResponse with per-document results.
    """
    start = time.monotonic()

    try:
        engine = get_document_engine()
        result = engine.batch_verify(
            documents=[d.model_dump() for d in body.documents],
            verified_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch verification failed: invalid parameters",
            )

        results = []
        for r in result.get("results", []):
            results.append(
                BatchVerifyResultEntry(
                    document_reference=r.get("document_reference", ""),
                    document_id=r.get("document_id"),
                    status=DocumentStatusEnum(r["status"])
                    if r.get("status") else None,
                    verification_score=Decimal(str(r["verification_score"]))
                    if r.get("verification_score") is not None else None,
                    red_flags_detected=r.get("red_flags_detected", 0),
                    error=r.get("error"),
                )
            )

        verified = sum(1 for r in results if r.document_id is not None)
        failed = sum(1 for r in results if r.error is not None)
        total_red_flags = sum(r.red_flags_detected for r in results)

        batch_status = BatchStatusEnum.COMPLETED
        if failed > 0 and verified > 0:
            batch_status = BatchStatusEnum.PARTIALLY_COMPLETED
        elif failed == len(results):
            batch_status = BatchStatusEnum.FAILED

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"batch_verify:{len(body.documents)}",
            str(verified),
        )

        logger.info(
            "Batch verification: submitted=%d verified=%d failed=%d "
            "red_flags=%d user=%s",
            len(body.documents),
            verified,
            failed,
            total_red_flags,
            user.user_id,
        )

        return BatchVerifyResponse(
            batch_id=result.get("batch_id", ""),
            status=batch_status,
            total_submitted=len(body.documents),
            verified_count=verified,
            failed_count=failed,
            total_red_flags=total_red_flags,
            results=results,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DocumentVerificationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch verification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch verification failed",
        )


# ---------------------------------------------------------------------------
# GET /batch/{batch_id}/status
# ---------------------------------------------------------------------------


@router.get(
    "/{batch_id}/status",
    response_model=BatchStatusResponse,
    summary="Get batch processing status",
    description=(
        "Check the current status and progress of a batch processing "
        "operation including completion counts and estimated time remaining."
    ),
    responses={
        200: {"description": "Batch status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch not found"},
    },
)
async def get_batch_status(
    batch_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:batch:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchStatusResponse:
    """Get batch processing status.

    Args:
        batch_id: Unique batch identifier.
        user: Authenticated user with batch:read permission.

    Returns:
        BatchStatusResponse with current batch status.
    """
    start = time.monotonic()

    try:
        # Try compliance engine first, then document engine
        compliance_engine = get_compliance_engine()
        result = compliance_engine.get_batch_status(batch_id=batch_id)

        if result is None:
            document_engine = get_document_engine()
            result = document_engine.get_batch_status(batch_id=batch_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch not found: {batch_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"batch_status:{batch_id}",
            str(result.get("status", "")),
        )

        logger.info(
            "Batch status retrieved: id=%s status=%s progress=%s user=%s",
            batch_id,
            result.get("status", ""),
            result.get("progress_percent", 0),
            user.user_id,
        )

        return BatchStatusResponse(
            batch_id=batch_id,
            status=BatchStatusEnum(result.get("status", "processing")),
            batch_type=result.get("batch_type", "unknown"),
            total_submitted=result.get("total_submitted", 0),
            completed_count=result.get("completed_count", 0),
            failed_count=result.get("failed_count", 0),
            progress_percent=Decimal(str(result.get("progress_percent", 0))),
            started_at=result.get("started_at"),
            completed_at=result.get("completed_at"),
            estimated_remaining_seconds=result.get("estimated_remaining_seconds"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["BatchProcessingEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch status retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch status retrieval failed",
        )
