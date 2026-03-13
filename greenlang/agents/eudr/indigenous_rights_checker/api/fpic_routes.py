# -*- coding: utf-8 -*-
"""
FPIC Verification Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for Free, Prior and Informed Consent (FPIC) verification
including document validation, consent status checking, FPIC document
management, and compliance scoring per UNDRIP and EUDR requirements.

Endpoints:
    POST /fpic/verify                 - Verify FPIC documentation
    GET  /fpic/documents              - List FPIC documents with filters
    GET  /fpic/documents/{doc_id}     - Get FPIC document details
    POST /fpic/score                  - Calculate FPIC compliance score

FPIC is a critical EUDR compliance requirement when supply chain
plots overlap with indigenous territories (EUDR Recital 31, Art. 3).

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, FPICVerifier Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.indigenous_rights_checker.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_fpic_verifier,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    ErrorResponse,
    FPICDocumentEntry,
    FPICDocumentListResponse,
    FPICDocumentResponse,
    FPICDocumentTypeEnum,
    FPICScoreRequest,
    FPICScoreResponse,
    FPICStatusEnum,
    FPICVerifyRequest,
    FPICVerifyResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    SortOrderEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fpic", tags=["FPIC Verification"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /fpic/verify
# ---------------------------------------------------------------------------


@router.post(
    "/verify",
    response_model=FPICVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify FPIC documentation",
    description=(
        "Verify Free, Prior and Informed Consent (FPIC) documentation for "
        "a plot-territory pair. Checks document validity, consent status, "
        "expiry dates, signatory requirements, and language accessibility. "
        "Returns a compliance score and list of identified issues."
    ),
    responses={
        200: {"description": "FPIC verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot or territory not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_fpic(
    request: Request,
    body: FPICVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:fpic:verify")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FPICVerifyResponse:
    """Verify FPIC documentation for a plot-territory pair.

    Args:
        body: FPIC verification request.
        user: Authenticated user with fpic:verify permission.

    Returns:
        FPICVerifyResponse with verification outcome and score.
    """
    start = time.monotonic()

    try:
        engine = get_fpic_verifier()
        result = engine.verify_fpic(
            plot_id=body.plot_id,
            territory_id=body.territory_id,
            supplier_id=body.supplier_id,
            document_ids=body.document_ids,
            commodity=body.commodity.value if body.commodity else None,
            verification_date=str(body.verification_date) if body.verification_date else None,
            verified_by=user.user_id,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"verify_fpic:{body.plot_id}:{body.territory_id}",
            str(result.get("compliance_score", 0)),
        )

        logger.info(
            "FPIC verified: plot_id=%s territory_id=%s status=%s score=%s operator=%s",
            body.plot_id,
            body.territory_id,
            result.get("fpic_status", "unknown"),
            result.get("compliance_score", 0),
            user.operator_id or user.user_id,
        )

        return FPICVerifyResponse(
            verification_id=result.get("verification_id", ""),
            plot_id=body.plot_id,
            territory_id=body.territory_id,
            fpic_status=FPICStatusEnum(result.get("fpic_status", "pending")),
            compliance_score=Decimal(str(result.get("compliance_score", 0))),
            documents_verified=result.get("documents_verified", 0),
            documents_valid=result.get("documents_valid", 0),
            issues=result.get("issues", []),
            recommendations=result.get("recommendations", []),
            expiry_date=result.get("expiry_date"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "FPICVerifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("FPIC verification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FPIC verification failed",
        )


# ---------------------------------------------------------------------------
# GET /fpic/documents
# ---------------------------------------------------------------------------


@router.get(
    "/documents",
    response_model=FPICDocumentListResponse,
    summary="List FPIC documents",
    description=(
        "Retrieve a paginated list of FPIC documents with optional filters "
        "for territory, community, document type, status, and expiry. "
        "Results ordered by creation date descending."
    ),
    responses={
        200: {"description": "FPIC documents listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_fpic_documents(
    request: Request,
    territory_id: Optional[str] = Query(
        None, description="Filter by territory ID"
    ),
    community_id: Optional[str] = Query(
        None, description="Filter by community ID"
    ),
    document_type: Optional[FPICDocumentTypeEnum] = Query(
        None, description="Filter by document type"
    ),
    fpic_status: Optional[FPICStatusEnum] = Query(
        None, alias="status", description="Filter by FPIC status"
    ),
    include_expired: bool = Query(
        False, description="Include expired documents"
    ),
    sort_by: Optional[str] = Query(
        "created_at", description="Sort field (created_at, expiry_date, title)"
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.DESC, description="Sort order"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:fpic:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FPICDocumentListResponse:
    """List FPIC documents with optional filters.

    Args:
        territory_id: Optional territory filter.
        community_id: Optional community filter.
        document_type: Optional document type filter.
        fpic_status: Optional status filter.
        include_expired: Whether to include expired documents.
        sort_by: Sort field.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        FPICDocumentListResponse with paginated document list.
    """
    start = time.monotonic()

    try:
        engine = get_fpic_verifier()
        result = engine.list_documents(
            territory_id=territory_id,
            community_id=community_id,
            document_type=document_type.value if document_type else None,
            status=fpic_status.value if fpic_status else None,
            include_expired=include_expired,
            sort_by=sort_by,
            sort_order=sort_order.value if sort_order else "desc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        documents = []
        for entry in result.get("documents", []):
            documents.append(
                FPICDocumentEntry(
                    document_id=entry.get("document_id", ""),
                    territory_id=entry.get("territory_id", ""),
                    community_id=entry.get("community_id"),
                    document_type=FPICDocumentTypeEnum(
                        entry.get("document_type", "consent_agreement")
                    ),
                    title=entry.get("title", ""),
                    status=FPICStatusEnum(entry.get("status", "pending")),
                    issue_date=entry.get("issue_date"),
                    expiry_date=entry.get("expiry_date"),
                    signatories=entry.get("signatories"),
                    language=entry.get("language"),
                    storage_url=entry.get("storage_url"),
                    verification_notes=entry.get("verification_notes"),
                    created_at=entry.get("created_at"),
                )
            )

        total = result.get("total", len(documents))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_fpic_documents:{territory_id}:{document_type}:{fpic_status}",
            str(total),
        )

        logger.info(
            "FPIC documents listed: total=%d returned=%d operator=%s",
            total,
            len(documents),
            user.operator_id or user.user_id,
        )

        return FPICDocumentListResponse(
            documents=documents,
            total_documents=total,
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
                data_sources=["IndigenousRightsChecker", "FPICVerifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("FPIC document listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FPIC document listing failed",
        )


# ---------------------------------------------------------------------------
# GET /fpic/documents/{doc_id}
# ---------------------------------------------------------------------------


@router.get(
    "/documents/{doc_id}",
    response_model=FPICDocumentResponse,
    summary="Get FPIC document details",
    description=(
        "Retrieve detailed information for a specific FPIC document "
        "including signatories, language, verification status, and "
        "associated territory and community information."
    ),
    responses={
        200: {"description": "FPIC document details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def get_fpic_document(
    doc_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:fpic:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FPICDocumentResponse:
    """Get detailed information for a specific FPIC document.

    Args:
        doc_id: Unique document identifier.
        user: Authenticated user.

    Returns:
        FPICDocumentResponse with full document details.
    """
    start = time.monotonic()

    try:
        engine = get_fpic_verifier()
        result = engine.get_document(document_id=doc_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FPIC document not found: {doc_id}",
            )

        doc_data = result.get("document", {})
        document_entry = FPICDocumentEntry(
            document_id=doc_data.get("document_id", doc_id),
            territory_id=doc_data.get("territory_id", ""),
            community_id=doc_data.get("community_id"),
            document_type=FPICDocumentTypeEnum(
                doc_data.get("document_type", "consent_agreement")
            ),
            title=doc_data.get("title", ""),
            status=FPICStatusEnum(doc_data.get("status", "pending")),
            issue_date=doc_data.get("issue_date"),
            expiry_date=doc_data.get("expiry_date"),
            signatories=doc_data.get("signatories"),
            language=doc_data.get("language"),
            storage_url=doc_data.get("storage_url"),
            verification_notes=doc_data.get("verification_notes"),
            created_at=doc_data.get("created_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"get_fpic_document:{doc_id}",
            document_entry.title,
        )

        logger.info(
            "FPIC document retrieved: doc_id=%s territory=%s operator=%s",
            doc_id,
            document_entry.territory_id,
            user.operator_id or user.user_id,
        )

        return FPICDocumentResponse(
            document=document_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "FPICVerifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("FPIC document retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FPIC document retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /fpic/score
# ---------------------------------------------------------------------------


@router.post(
    "/score",
    response_model=FPICScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate FPIC compliance score",
    description=(
        "Calculate a comprehensive FPIC compliance score for a territory, "
        "considering document validity, consent freshness, signatory "
        "completeness, community representation, and language accessibility. "
        "Optionally includes expired documents in the assessment."
    ),
    responses={
        200: {"description": "FPIC score calculated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Territory not found"},
    },
)
async def calculate_fpic_score(
    request: Request,
    body: FPICScoreRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:fpic:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FPICScoreResponse:
    """Calculate FPIC compliance score for a territory.

    Args:
        body: Score calculation request.
        user: Authenticated user.

    Returns:
        FPICScoreResponse with compliance score and breakdown.
    """
    start = time.monotonic()

    try:
        engine = get_fpic_verifier()
        result = engine.calculate_score(
            territory_id=body.territory_id,
            plot_id=body.plot_id,
            include_expired=body.include_expired,
            weight_document_quality=body.weight_document_quality,
            scored_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Territory not found: {body.territory_id}",
            )

        # Convert score breakdown decimals
        score_breakdown: Optional[Dict[str, Decimal]] = None
        raw_breakdown = result.get("score_breakdown")
        if raw_breakdown:
            score_breakdown = {
                k: Decimal(str(v)) for k, v in raw_breakdown.items()
            }

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"fpic_score:{body.territory_id}:{body.include_expired}",
            str(result.get("overall_score", 0)),
        )

        logger.info(
            "FPIC score calculated: territory_id=%s score=%s status=%s operator=%s",
            body.territory_id,
            result.get("overall_score", 0),
            result.get("fpic_status", "unknown"),
            user.operator_id or user.user_id,
        )

        return FPICScoreResponse(
            territory_id=body.territory_id,
            overall_score=Decimal(str(result.get("overall_score", 0))),
            fpic_status=FPICStatusEnum(result.get("fpic_status", "pending")),
            document_count=result.get("document_count", 0),
            valid_document_count=result.get("valid_document_count", 0),
            expired_document_count=result.get("expired_document_count", 0),
            score_breakdown=score_breakdown,
            risk_factors=result.get("risk_factors", []),
            recommendations=result.get("recommendations", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "FPICVerifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("FPIC score calculation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FPIC score calculation failed",
        )
