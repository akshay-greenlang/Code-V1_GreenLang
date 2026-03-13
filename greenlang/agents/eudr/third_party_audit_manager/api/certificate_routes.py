# -*- coding: utf-8 -*-
"""
Certification Integration Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for managing certification scheme integration with FSC, PEFC,
RSPO, Rainforest Alliance, and ISCC for EUDR compliance verification.

Endpoints (4):
    POST /certificates                        - Create/import a certificate
    GET  /certificates                        - List certificates with filters
    GET  /suppliers/{supplier_id}/certificates - Get supplier certificates
    POST /certificates/validate-eudr          - Validate EUDR coverage matrix

RBAC Permissions:
    eudr-tam:certificate:create   - Import/create certificate records
    eudr-tam:certificate:read     - View certificate data
    eudr-tam:certificate:validate - Run EUDR coverage validation

Coverage matrix (pre-coded reference data):
    FSC:  75% EUDR coverage (FULL on Art.3, Art.10, Art.2(40) Cat 1-5)
    PEFC: 70% coverage (FULL on Art.3, Art.10)
    RSPO: 65% coverage (FULL on Art.3, Art.9, Art.10)
    RA:   60% coverage (FULL on Art.3, Art.10)
    ISCC: 55% coverage (FULL on Art.3)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_certification_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    CertificateListResponse,
    CertSchemeEnum,
    CertStatusEnum,
    CoverageMatrixResponse,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Certification Integration"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /certificates
# ---------------------------------------------------------------------------


@router.post(
    "/certificates",
    status_code=status.HTTP_201_CREATED,
    summary="Create or import a certificate",
    description=(
        "Create or import a certification record for a supplier. Supports "
        "FSC, PEFC, RSPO, Rainforest Alliance, and ISCC certificates. "
        "Computes EUDR coverage matrix automatically based on scheme type."
    ),
    responses={
        201: {"description": "Certificate created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid certificate data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def create_certificate(
    request: Request,
    body: Dict[str, Any],
    user: AuthUser = Depends(require_permission("eudr-tam:certificate:create")),
    _rl: None = Depends(rate_limit_write),
    cert_engine: object = Depends(get_certification_engine),
) -> dict:
    """Create or import a certification certificate record.

    Args:
        body: Certificate data (supplier, scheme, number, status, dates).
        user: Authenticated user with certificate:create permission.
        cert_engine: CertificationIntegrationEngine singleton.

    Returns:
        Created certificate with EUDR coverage matrix.
    """
    start = time.monotonic()
    try:
        supplier_id = body.get("supplier_id")
        if not supplier_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="supplier_id is required",
            )

        scheme = body.get("scheme")
        if not scheme:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="scheme is required (fsc, pefc, rspo, rainforest_alliance, iscc)",
            )

        result: Dict[str, Any] = {}
        if hasattr(cert_engine, "create_certificate"):
            result = await cert_engine.create_certificate(body)
        else:
            cert_hash = hashlib.sha256(
                f"{supplier_id}{scheme}{time.time()}".encode()
            ).hexdigest()
            result = {
                "certificate_id": cert_hash[:36],
                "supplier_id": supplier_id,
                "scheme": scheme,
                "status": body.get("status", "active"),
                "certificate_number": body.get("certificate_number", ""),
            }

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(supplier_id, result.get("certificate_id", ""))

        return {
            "certificate_id": result.get("certificate_id", ""),
            "supplier_id": supplier_id,
            "scheme": scheme,
            "status": result.get("status", "active"),
            "eudr_coverage_matrix": result.get("eudr_coverage_matrix", {}),
            "provenance_hash": prov_hash,
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create certificate: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create certificate record",
        )


# ---------------------------------------------------------------------------
# GET /certificates
# ---------------------------------------------------------------------------


@router.get(
    "/certificates",
    response_model=CertificateListResponse,
    summary="List certificates with filters",
    description=(
        "Retrieve a paginated list of certification certificates with "
        "optional filters for scheme, status, supplier, and expiry date."
    ),
    responses={
        200: {"description": "Certificates listed successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_certificates(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:certificate:read")),
    _rl: None = Depends(rate_limit_standard),
    pagination: PaginationParams = Depends(get_pagination),
    cert_engine: object = Depends(get_certification_engine),
    scheme: Optional[CertSchemeEnum] = Query(
        None, description="Filter by certification scheme"
    ),
    cert_status: Optional[CertStatusEnum] = Query(
        None, alias="status", description="Filter by certificate status"
    ),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
) -> CertificateListResponse:
    """List certification certificates with optional filters.

    Args:
        user: Authenticated user with certificate:read permission.
        pagination: Standard limit/offset parameters.
        cert_engine: CertificationIntegrationEngine singleton.

    Returns:
        Paginated list of certificate records.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {}
        if scheme:
            filters["scheme"] = scheme.value
        if cert_status:
            filters["status"] = cert_status.value
        if supplier_id:
            filters["supplier_id"] = supplier_id

        certificates: List[Dict[str, Any]] = []
        total = 0
        if hasattr(cert_engine, "list_certificates"):
            result = await cert_engine.list_certificates(
                filters=filters,
                limit=pagination.limit,
                offset=pagination.offset,
            )
            certificates = result.get("certificates", [])
            total = result.get("total", 0)

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(filters, len(certificates))

        return CertificateListResponse(
            certificates=certificates,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list certificates: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve certificate list",
        )


# ---------------------------------------------------------------------------
# GET /suppliers/{supplier_id}/certificates
# ---------------------------------------------------------------------------


@router.get(
    "/suppliers/{supplier_id}/certificates",
    summary="Get supplier certificates",
    description=(
        "Retrieve all certification certificates for a specific supplier "
        "across all 5 supported schemes (FSC, PEFC, RSPO, RA, ISCC). "
        "Includes EUDR coverage gap analysis per supplier."
    ),
    responses={
        200: {"description": "Supplier certificates retrieved"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_supplier_certificates(
    supplier_id: str,
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:certificate:read")),
    _rl: None = Depends(rate_limit_standard),
    cert_engine: object = Depends(get_certification_engine),
) -> dict:
    """Get all certificates for a supplier with EUDR coverage analysis.

    Args:
        supplier_id: Unique supplier identifier.
        user: Authenticated user with certificate:read permission.
        cert_engine: CertificationIntegrationEngine singleton.

    Returns:
        Supplier's certificates with aggregate EUDR coverage.
    """
    start = time.monotonic()
    try:
        result: Dict[str, Any] = {}
        if hasattr(cert_engine, "get_supplier_certificates"):
            result = await cert_engine.get_supplier_certificates(
                supplier_id=supplier_id
            )
        else:
            result = {
                "supplier_id": supplier_id,
                "certificates": [],
                "aggregate_eudr_coverage": {},
                "coverage_gaps": [],
            }

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(supplier_id, len(result.get("certificates", [])))

        return {
            "supplier_id": supplier_id,
            "certificates": result.get("certificates", []),
            "total_certificates": len(result.get("certificates", [])),
            "aggregate_eudr_coverage": result.get("aggregate_eudr_coverage", {}),
            "coverage_gaps": result.get("coverage_gaps", []),
            "provenance_hash": prov_hash,
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get certificates for %s: %s", supplier_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supplier certificates",
        )


# ---------------------------------------------------------------------------
# POST /certificates/validate-eudr
# ---------------------------------------------------------------------------


@router.post(
    "/certificates/validate-eudr",
    response_model=CoverageMatrixResponse,
    summary="Validate EUDR coverage matrix",
    description=(
        "Validate a supplier's certification portfolio against EUDR "
        "requirements. Computes the aggregate coverage matrix across "
        "all active certificates and identifies EUDR requirement gaps "
        "that require additional audit scope."
    ),
    responses={
        200: {"description": "EUDR coverage validated"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def validate_eudr_coverage(
    request: Request,
    body: Dict[str, Any],
    user: AuthUser = Depends(require_permission("eudr-tam:certificate:validate")),
    _rl: None = Depends(rate_limit_heavy),
    cert_engine: object = Depends(get_certification_engine),
) -> CoverageMatrixResponse:
    """Validate EUDR coverage from certification portfolio.

    Uses pre-coded scheme-to-EUDR coverage matrix (deterministic
    reference data, not LLM-generated) to identify gaps.

    Args:
        body: Validation request with supplier_id.
        user: Authenticated user with certificate:validate permission.
        cert_engine: CertificationIntegrationEngine singleton.

    Returns:
        EUDR coverage matrix with gap analysis.
    """
    start = time.monotonic()
    try:
        supplier_id = body.get("supplier_id")
        if not supplier_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="supplier_id is required",
            )

        result: Dict[str, Any] = {}
        if hasattr(cert_engine, "validate_eudr_coverage"):
            result = await cert_engine.validate_eudr_coverage(
                supplier_id=supplier_id
            )
        else:
            result = {
                "supplier_id": supplier_id,
                "coverage_matrix": {},
                "gaps": [],
                "overall_coverage_pct": "0.00",
            }

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(supplier_id, result.get("overall_coverage_pct", "0"))

        return CoverageMatrixResponse(
            supplier_id=supplier_id,
            coverage_matrix=result.get("coverage_matrix", {}),
            gaps=result.get("gaps", []),
            overall_coverage_pct=Decimal(
                str(result.get("overall_coverage_pct", "0.00"))
            ),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to validate EUDR coverage: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate EUDR coverage",
        )
