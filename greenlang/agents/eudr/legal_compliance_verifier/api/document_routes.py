# -*- coding: utf-8 -*-
"""
Document Verification Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for document verification, validity checking, and expiry monitoring
covering due diligence statements, import/export permits, phytosanitary
certificates, and other EUDR-required documentation per Articles 4, 9, 10.

Endpoints:
    POST /documents/verify          - Verify a legal document
    GET  /documents                 - List documents (paginated)
    GET  /documents/{document_id}   - Get document details
    POST /documents/validity-check  - Batch validity check
    GET  /documents/expiring        - Get expiring documents

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, DocumentVerificationEngine
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
    get_document_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    DocumentDetailResponse,
    DocumentEntry,
    DocumentListResponse,
    DocumentStatusEnum,
    DocumentTypeEnum,
    DocumentVerifyRequest,
    DocumentVerifyResponse,
    EUDRCommodityEnum,
    ErrorResponse,
    ExpiringDocumentEntry,
    ExpiringDocumentsResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    ValidityCheckRequest,
    ValidityCheckResponse,
    DocumentValidityEntry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Verification"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /documents/verify
# ---------------------------------------------------------------------------


@router.post(
    "/verify",
    response_model=DocumentVerifyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Verify a legal document",
    description=(
        "Verify authenticity, validity, and EUDR compliance of a legal "
        "document including due diligence statements, import licenses, "
        "export permits, phytosanitary certificates, and certificates "
        "of origin. Performs automated checks and red flag detection."
    ),
    responses={
        201: {"description": "Document verified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_document(
    request: Request,
    body: DocumentVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:document:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> DocumentVerifyResponse:
    """Verify a legal document for EUDR compliance.

    Args:
        body: Document verification request.
        user: Authenticated user with document:create permission.

    Returns:
        DocumentVerifyResponse with verification results.
    """
    start = time.monotonic()

    try:
        engine = get_document_engine()
        result = engine.verify(
            document_type=body.document_type.value,
            document_reference=body.document_reference,
            issuing_authority=body.issuing_authority,
            issuing_country=body.issuing_country,
            issue_date=body.issue_date,
            expiry_date=body.expiry_date,
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            commodity=body.commodity.value if body.commodity else None,
            file_hash=body.file_hash,
            file_url=body.file_url,
            additional_data=body.additional_data,
            verified_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document verification failed: invalid document data",
            )

        document = DocumentEntry(
            document_id=result.get("document_id", ""),
            document_type=DocumentTypeEnum(result.get("document_type", body.document_type.value)),
            document_reference=result.get("document_reference", body.document_reference),
            status=DocumentStatusEnum(result.get("status", "pending")),
            issuing_authority=result.get("issuing_authority", body.issuing_authority),
            issuing_country=result.get("issuing_country", body.issuing_country),
            issue_date=result.get("issue_date", body.issue_date),
            expiry_date=result.get("expiry_date", body.expiry_date),
            operator_id=result.get("operator_id", body.operator_id),
            supplier_id=result.get("supplier_id", body.supplier_id),
            commodity=EUDRCommodityEnum(result["commodity"]) if result.get("commodity") else body.commodity,
            verification_score=Decimal(str(result.get("verification_score", 0)))
            if result.get("verification_score") is not None else None,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"document_verify:{body.document_reference}",
            document.document_id,
        )

        logger.info(
            "Document verified: id=%s ref=%s type=%s status=%s user=%s",
            document.document_id,
            body.document_reference,
            body.document_type.value,
            document.status.value,
            user.user_id,
        )

        return DocumentVerifyResponse(
            document=document,
            verification_details=result.get("verification_details", {}),
            red_flags_detected=result.get("red_flags_detected", 0),
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
        logger.error("Document verification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document verification failed",
        )


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List verified documents",
    description=(
        "Retrieve a paginated list of verified documents with optional "
        "filtering by type, status, operator, supplier, and commodity."
    ),
    responses={
        200: {"description": "Documents retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_documents(
    request: Request,
    document_type: Optional[DocumentTypeEnum] = Query(
        None, description="Filter by document type"
    ),
    document_status: Optional[DocumentStatusEnum] = Query(
        None, alias="status", description="Filter by verification status"
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
        require_permission("eudr-lcv:document:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DocumentListResponse:
    """List verified documents with pagination.

    Args:
        document_type: Optional type filter.
        document_status: Optional status filter.
        operator_id: Optional operator filter.
        supplier_id: Optional supplier filter.
        commodity: Optional commodity filter.
        pagination: Pagination parameters.
        user: Authenticated user with document:read permission.

    Returns:
        DocumentListResponse with paginated documents.
    """
    start = time.monotonic()

    try:
        engine = get_document_engine()
        result = engine.list_documents(
            document_type=document_type.value if document_type else None,
            status=document_status.value if document_status else None,
            operator_id=operator_id,
            supplier_id=supplier_id,
            commodity=commodity.value if commodity else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        documents = []
        for d in result.get("documents", []):
            documents.append(
                DocumentEntry(
                    document_id=d.get("document_id", ""),
                    document_type=DocumentTypeEnum(d.get("document_type", "other")),
                    document_reference=d.get("document_reference", ""),
                    status=DocumentStatusEnum(d.get("status", "pending")),
                    issuing_authority=d.get("issuing_authority"),
                    issuing_country=d.get("issuing_country"),
                    issue_date=d.get("issue_date"),
                    expiry_date=d.get("expiry_date"),
                    operator_id=d.get("operator_id"),
                    supplier_id=d.get("supplier_id"),
                    commodity=EUDRCommodityEnum(d["commodity"]) if d.get("commodity") else None,
                    verification_score=Decimal(str(d["verification_score"]))
                    if d.get("verification_score") is not None else None,
                )
            )

        total = result.get("total", len(documents))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"document_list:{document_type}:{document_status}",
            str(total),
        )

        logger.info(
            "Documents listed: total=%d user=%s",
            total,
            user.user_id,
        )

        return DocumentListResponse(
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
                data_sources=["DocumentVerificationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document listing failed",
        )


# ---------------------------------------------------------------------------
# GET /documents/{document_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description=(
        "Retrieve full details of a verified document including verification "
        "details, file metadata, related documents, and audit trail."
    ),
    responses={
        200: {"description": "Document details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def get_document_detail(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:document:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DocumentDetailResponse:
    """Get detailed information about a verified document.

    Args:
        document_id: Unique document identifier.
        user: Authenticated user with document:read permission.

    Returns:
        DocumentDetailResponse with full document details.
    """
    start = time.monotonic()

    try:
        engine = get_document_engine()
        result = engine.get_detail(document_id=document_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        document = DocumentEntry(
            document_id=result.get("document_id", document_id),
            document_type=DocumentTypeEnum(result.get("document_type", "other")),
            document_reference=result.get("document_reference", ""),
            status=DocumentStatusEnum(result.get("status", "pending")),
            issuing_authority=result.get("issuing_authority"),
            issuing_country=result.get("issuing_country"),
            issue_date=result.get("issue_date"),
            expiry_date=result.get("expiry_date"),
            operator_id=result.get("operator_id"),
            supplier_id=result.get("supplier_id"),
            commodity=EUDRCommodityEnum(result["commodity"]) if result.get("commodity") else None,
            verification_score=Decimal(str(result["verification_score"]))
            if result.get("verification_score") is not None else None,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"document_detail:{document_id}",
            document.document_reference,
        )

        logger.info(
            "Document detail retrieved: id=%s user=%s",
            document_id,
            user.user_id,
        )

        return DocumentDetailResponse(
            document=document,
            verification_details=result.get("verification_details", {}),
            file_hash=result.get("file_hash"),
            file_url=result.get("file_url"),
            additional_data=result.get("additional_data", {}),
            related_documents=result.get("related_documents", []),
            audit_trail=result.get("audit_trail", []),
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
        logger.error("Document detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /documents/validity-check
# ---------------------------------------------------------------------------


@router.post(
    "/validity-check",
    response_model=ValidityCheckResponse,
    summary="Batch document validity check",
    description=(
        "Check the current validity status of multiple documents. Returns "
        "per-document validity, expiry warnings, and identified issues."
    ),
    responses={
        200: {"description": "Validity check completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def check_validity(
    request: Request,
    body: ValidityCheckRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:document:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ValidityCheckResponse:
    """Check validity status of multiple documents.

    Args:
        body: Validity check request with document IDs.
        user: Authenticated user with document:read permission.

    Returns:
        ValidityCheckResponse with per-document validity results.
    """
    start = time.monotonic()

    try:
        engine = get_document_engine()
        result = engine.check_validity(
            document_ids=body.document_ids,
            check_date=body.check_date,
            include_expiry_warnings=body.include_expiry_warnings,
        )

        results = []
        for r in result.get("results", []):
            results.append(
                DocumentValidityEntry(
                    document_id=r.get("document_id", ""),
                    document_type=DocumentTypeEnum(r.get("document_type", "other")),
                    is_valid=r.get("is_valid", False),
                    status=DocumentStatusEnum(r.get("status", "pending")),
                    expiry_date=r.get("expiry_date"),
                    days_until_expiry=r.get("days_until_expiry"),
                    expiry_warning=r.get("expiry_warning", False),
                    issues=r.get("issues", []),
                )
            )

        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = sum(1 for r in results if not r.is_valid)
        expiring_soon = sum(1 for r in results if r.expiry_warning)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"validity_check:{len(body.document_ids)}",
            str(valid_count),
        )

        logger.info(
            "Validity check: total=%d valid=%d invalid=%d expiring=%d user=%s",
            len(results),
            valid_count,
            invalid_count,
            expiring_soon,
            user.user_id,
        )

        return ValidityCheckResponse(
            results=results,
            total_checked=len(results),
            valid_count=valid_count,
            invalid_count=invalid_count,
            expiring_soon_count=expiring_soon,
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
        logger.error("Validity check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document validity check failed",
        )


# ---------------------------------------------------------------------------
# GET /documents/expiring
# ---------------------------------------------------------------------------


@router.get(
    "/expiring",
    response_model=ExpiringDocumentsResponse,
    summary="Get expiring documents",
    description=(
        "Retrieve documents expiring within a specified number of days "
        "(default: 90). Supports filtering by type, operator, supplier."
    ),
    responses={
        200: {"description": "Expiring documents retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_expiring_documents(
    request: Request,
    days: int = Query(
        default=90, ge=1, le=365,
        description="Number of days to look ahead for expiry (1-365)",
    ),
    document_type: Optional[DocumentTypeEnum] = Query(
        None, description="Filter by document type"
    ),
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    supplier_id: Optional[str] = Query(
        None, description="Filter by supplier ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:document:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ExpiringDocumentsResponse:
    """Get documents expiring within the specified window.

    Args:
        days: Number of days to look ahead.
        document_type: Optional type filter.
        operator_id: Optional operator filter.
        supplier_id: Optional supplier filter.
        pagination: Pagination parameters.
        user: Authenticated user with document:read permission.

    Returns:
        ExpiringDocumentsResponse with expiring documents.
    """
    start = time.monotonic()

    try:
        engine = get_document_engine()
        result = engine.get_expiring(
            days=days,
            document_type=document_type.value if document_type else None,
            operator_id=operator_id,
            supplier_id=supplier_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        documents = []
        for d in result.get("documents", []):
            documents.append(
                ExpiringDocumentEntry(
                    document_id=d.get("document_id", ""),
                    document_type=DocumentTypeEnum(d.get("document_type", "other")),
                    document_reference=d.get("document_reference", ""),
                    operator_id=d.get("operator_id"),
                    supplier_id=d.get("supplier_id"),
                    expiry_date=d.get("expiry_date"),
                    days_until_expiry=d.get("days_until_expiry", 0),
                    commodity=EUDRCommodityEnum(d["commodity"]) if d.get("commodity") else None,
                    status=DocumentStatusEnum(d.get("status", "verified")),
                )
            )

        total = result.get("total", len(documents))
        within_7 = sum(1 for d in documents if d.days_until_expiry <= 7)
        within_30 = sum(1 for d in documents if d.days_until_expiry <= 30)
        within_90 = sum(1 for d in documents if d.days_until_expiry <= 90)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"expiring_documents:{days}",
            str(total),
        )

        logger.info(
            "Expiring documents: total=%d within_7d=%d within_30d=%d user=%s",
            total,
            within_7,
            within_30,
            user.user_id,
        )

        return ExpiringDocumentsResponse(
            documents=documents,
            total_expiring=total,
            expiring_within_7_days=within_7,
            expiring_within_30_days=within_30,
            expiring_within_90_days=within_90,
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
                data_sources=["DocumentVerificationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Expiring documents retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Expiring documents retrieval failed",
        )
