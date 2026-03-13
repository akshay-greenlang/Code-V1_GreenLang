# -*- coding: utf-8 -*-
"""
Certification Scheme Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for certification scheme integration covering certificate listing,
status sync, and EUDR coverage matrix analysis for FSC, PEFC, RSPO,
Rainforest Alliance, and ISCC.

Endpoints (3):
    GET  /schemes/certificates               - List certificates
    POST /schemes/certificates/sync          - Trigger certification sync
    GET  /schemes/coverage/{supplier_id}     - Get EUDR coverage matrix

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024, CertificationIntegrationEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_certification_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    CertificateEntry,
    CertificateListResponse,
    CertSchemeEnum,
    CertStatusEnum,
    CertSyncRequest,
    CertSyncResponse,
    CoverageMatrixEntry,
    CoverageMatrixResponse,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schemes", tags=["Certification Schemes"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GET /schemes/certificates
# ---------------------------------------------------------------------------


@router.get(
    "/certificates",
    response_model=CertificateListResponse,
    summary="List certification certificates",
    description=(
        "Retrieve paginated list of certification certificates with "
        "optional filtering by scheme, supplier, and status."
    ),
    responses={200: {"description": "Certificates retrieved"}},
)
async def list_certificates(
    request: Request,
    scheme: Optional[str] = Query(None, description="Filter by scheme"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier"),
    cert_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-tam:schemes:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CertificateListResponse:
    """Retrieve paginated certificate list.

    Args:
        scheme: Optional scheme filter.
        supplier_id: Optional supplier filter.
        cert_status: Optional status filter.
        pagination: Pagination parameters.
        user: Authenticated user with schemes:read permission.

    Returns:
        CertificateListResponse with certificates and pagination.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.list_certificates(
            scheme=scheme,
            supplier_id=supplier_id,
            status=cert_status,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        certs = []
        for c in result.get("certificates", []):
            certs.append(CertificateEntry(
                certificate_id=c.get("certificate_id", ""),
                supplier_id=c.get("supplier_id", ""),
                scheme=CertSchemeEnum(c.get("scheme", "fsc")),
                certificate_number=c.get("certificate_number", ""),
                status=CertStatusEnum(c.get("status", "active")),
                scope=c.get("scope"),
                issue_date=c.get("issue_date"),
                expiry_date=c.get("expiry_date"),
                certification_body=c.get("certification_body"),
                eudr_coverage_matrix=c.get("eudr_coverage_matrix", {}),
            ))

        total = result.get("total", len(certs))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return CertificateListResponse(
            certificates=certs,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("cert_list", f"total:{total}"),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Certificate list failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve certificates",
        )


# ---------------------------------------------------------------------------
# POST /schemes/certificates/sync
# ---------------------------------------------------------------------------


@router.post(
    "/certificates/sync",
    response_model=CertSyncResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger certification status sync",
    description=(
        "Trigger synchronization of certificate statuses from external "
        "scheme databases (FSC, PEFC, RSPO, RA, ISCC)."
    ),
    responses={
        202: {"description": "Sync triggered"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def sync_certificates(
    request: Request,
    body: CertSyncRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:schemes:sync")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> CertSyncResponse:
    """Trigger certification status synchronization.

    Args:
        body: Sync request with scheme and optional supplier IDs.
        user: Authenticated user with schemes:sync permission.

    Returns:
        CertSyncResponse with sync results.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.sync_certificates(
            scheme=body.scheme.value,
            supplier_ids=body.supplier_ids,
            force=body.force,
            triggered_by=user.user_id,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        logger.info(
            "Cert sync triggered: scheme=%s synced=%d user=%s",
            body.scheme.value,
            result.get("synced_count", 0),
            user.user_id,
        )

        return CertSyncResponse(
            synced_count=result.get("synced_count", 0),
            updated_count=result.get("updated_count", 0),
            new_count=result.get("new_count", 0),
            errors=result.get("errors", []),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(
                    f"cert_sync:{body.scheme.value}", str(result.get("synced_count", 0))
                ),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Certificate sync failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Certificate sync failed",
        )


# ---------------------------------------------------------------------------
# GET /schemes/coverage/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/coverage/{supplier_id}",
    response_model=CoverageMatrixResponse,
    summary="Get EUDR coverage matrix",
    description=(
        "Get the EUDR article coverage matrix for a supplier across "
        "all active certification schemes with gap analysis."
    ),
    responses={
        200: {"description": "Coverage matrix retrieved"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_coverage_matrix(
    supplier_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-tam:schemes:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CoverageMatrixResponse:
    """Get EUDR coverage matrix for a supplier.

    Args:
        supplier_id: UUID of the supplier.
        user: Authenticated user with schemes:read permission.

    Returns:
        CoverageMatrixResponse with coverage analysis.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.get_coverage_matrix(supplier_id=supplier_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No certificates found for supplier: {supplier_id}",
            )

        schemes = []
        for s in result.get("schemes", []):
            schemes.append(CoverageMatrixEntry(
                scheme=CertSchemeEnum(s.get("scheme", "fsc")),
                certificate_number=s.get("certificate_number"),
                covered_articles=s.get("covered_articles", []),
                uncovered_articles=s.get("uncovered_articles", []),
                coverage_percentage=Decimal(str(s.get("coverage_percentage", 0))),
                gaps=s.get("gaps", []),
            ))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return CoverageMatrixResponse(
            supplier_id=supplier_id,
            schemes=schemes,
            overall_coverage=Decimal(str(result.get("overall_coverage", 0))),
            remaining_gaps=result.get("remaining_gaps", []),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(
                    f"coverage:{supplier_id}", str(len(schemes))
                ),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Coverage matrix failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve coverage matrix",
        )
