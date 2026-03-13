# -*- coding: utf-8 -*-
"""
Profile Routes - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Endpoints for supplier profile CRUD operations, search, and batch
create/update of supplier records.

Endpoints:
    POST   /suppliers            - Create supplier profile
    GET    /suppliers/{id}       - Get supplier profile
    PUT    /suppliers/{id}       - Update supplier profile
    DELETE /suppliers/{id}       - Deactivate supplier
    POST   /suppliers/search     - Search suppliers
    POST   /suppliers/batch      - Batch create/update suppliers

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
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from greenlang.agents.eudr.multi_tier_supplier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_supplier_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    BatchSupplierRequestSchema,
    BatchSupplierResponseSchema,
    CreateSupplierSchema,
    DeactivateResponseSchema,
    SearchCriteriaSchema,
    SupplierProfileSchema,
    SupplierSearchResponseSchema,
    UpdateSupplierSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Supplier Profiles"])


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


# ---------------------------------------------------------------------------
# POST /suppliers
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=SupplierProfileSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new supplier profile",
    description=(
        "Create a comprehensive supplier profile with legal entity, "
        "location, commodities, certifications, contacts, and capacity "
        "information. Automatically calculates profile completeness score."
    ),
    responses={
        201: {"description": "Supplier profile created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Supplier already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_supplier(
    body: CreateSupplierSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:suppliers:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SupplierProfileSchema:
    """Create a new supplier profile.

    Validates all fields, calculates profile completeness, and
    persists the supplier record with full audit trail.

    Args:
        body: Supplier profile data.
        request: FastAPI request object.
        user: Authenticated user with suppliers:write permission.

    Returns:
        Created SupplierProfileSchema with generated ID and scores.

    Raises:
        HTTPException: 400 on validation error, 409 on duplicate.
    """
    start = time.monotonic()
    logger.info(
        "Create supplier request: user=%s name='%s' commodity=%s tier=%d",
        user.user_id,
        body.name[:80],
        body.commodities,
        body.tier,
    )

    try:
        service = get_supplier_service()

        result = service.create_supplier(
            data=body.model_dump(),
            created_by=user.user_id,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"create|{result.get('supplier_id', '')}|{body.name}|{elapsed}"
        )

        logger.info(
            "Supplier created: user=%s supplier_id=%s completeness=%.1f "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("supplier_id", ""),
            result.get("profile_completeness", 0),
            elapsed * 1000,
        )

        return SupplierProfileSchema(
            supplier_id=result.get("supplier_id", str(uuid.uuid4())),
            name=body.name,
            registration_id=body.registration_id,
            tax_id=body.tax_id,
            duns_number=body.duns_number,
            location=body.location,
            commodities=body.commodities,
            tier=body.tier,
            certifications=body.certifications,
            contacts=body.contacts,
            annual_volume_tonnes=body.annual_volume_tonnes,
            processing_capacity_tonnes=body.processing_capacity_tonnes,
            upstream_supplier_count=body.upstream_supplier_count,
            dds_references=body.dds_references,
            profile_completeness=result.get("profile_completeness", 0.0),
            missing_fields=result.get("missing_fields", []),
            compliance_status=result.get("compliance_status", "unverified"),
            risk_score=result.get("risk_score"),
            is_active=True,
            version=1,
            metadata=body.metadata,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Create supplier validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Supplier creation validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Create supplier failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier creation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /suppliers/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{supplier_id}",
    response_model=SupplierProfileSchema,
    status_code=status.HTTP_200_OK,
    summary="Get supplier profile",
    description=(
        "Retrieve a complete supplier profile including location, "
        "certifications, compliance status, risk score, and profile "
        "completeness metrics."
    ),
    responses={
        200: {"description": "Supplier profile retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_supplier(
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:suppliers:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SupplierProfileSchema:
    """Retrieve a supplier profile by ID.

    Args:
        supplier_id: Unique supplier identifier.
        request: FastAPI request object.
        user: Authenticated user with suppliers:read permission.

    Returns:
        SupplierProfileSchema for the requested supplier.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Get supplier request: user=%s supplier_id=%s",
        user.user_id,
        supplier_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_supplier(supplier_id=supplier_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"get|{supplier_id}|{result.get('version', 1)}|{elapsed}"
        )

        logger.info(
            "Supplier retrieved: user=%s supplier_id=%s elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        return SupplierProfileSchema(**result)

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get supplier failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# PUT /suppliers/{supplier_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{supplier_id}",
    response_model=SupplierProfileSchema,
    status_code=status.HTTP_200_OK,
    summary="Update supplier profile",
    description=(
        "Partially update a supplier profile. Only provided fields are "
        "updated; omitted fields retain their current values. Profile "
        "version is incremented and changes are audited."
    ),
    responses={
        200: {"description": "Supplier profile updated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def update_supplier(
    body: UpdateSupplierSchema,
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:suppliers:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SupplierProfileSchema:
    """Update an existing supplier profile.

    Applies partial updates to the supplier record, increments the
    version number, and logs the change to the audit trail.

    Args:
        body: Partial update data.
        supplier_id: Supplier identifier to update.
        request: FastAPI request object.
        user: Authenticated user with suppliers:write permission.

    Returns:
        Updated SupplierProfileSchema.

    Raises:
        HTTPException: 404 if supplier not found, 400 on validation error.
    """
    start = time.monotonic()
    update_fields = body.model_dump(exclude_none=True)
    logger.info(
        "Update supplier request: user=%s supplier_id=%s fields=%s",
        user.user_id,
        supplier_id,
        list(update_fields.keys()),
    )

    try:
        service = get_supplier_service()

        result = service.update_supplier(
            supplier_id=supplier_id,
            updates=update_fields,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"update|{supplier_id}|{result.get('version', 1)}|{elapsed}"
        )

        logger.info(
            "Supplier updated: user=%s supplier_id=%s version=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            result.get("version", 1),
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        return SupplierProfileSchema(**result)

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Update supplier validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Supplier update validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Update supplier failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier update failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# DELETE /suppliers/{supplier_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{supplier_id}",
    response_model=DeactivateResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Deactivate supplier",
    description=(
        "Soft-delete a supplier by marking it as inactive. The supplier "
        "record is retained for audit purposes but excluded from active "
        "queries. Relationships are also marked as terminated."
    ),
    responses={
        200: {"description": "Supplier deactivated successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def deactivate_supplier(
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:suppliers:delete")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DeactivateResponseSchema:
    """Deactivate a supplier (soft delete).

    Sets the supplier to inactive and terminates all active
    relationships. The record is preserved for EUDR Article 14
    5-year retention requirements.

    Args:
        supplier_id: Supplier identifier to deactivate.
        request: FastAPI request object.
        user: Authenticated user with suppliers:delete permission.

    Returns:
        DeactivateResponseSchema confirming deactivation.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Deactivate supplier request: user=%s supplier_id=%s",
        user.user_id,
        supplier_id,
    )

    try:
        service = get_supplier_service()
        result = service.deactivate_supplier(
            supplier_id=supplier_id,
            deactivated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"deactivate|{supplier_id}|{elapsed}"
        )

        logger.info(
            "Supplier deactivated: user=%s supplier_id=%s elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            elapsed * 1000,
        )

        return DeactivateResponseSchema(
            supplier_id=supplier_id,
            status="deactivated",
            message=f"Supplier {supplier_id} has been deactivated",
            provenance_hash=provenance,
        )

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Deactivate supplier failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier deactivation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /suppliers/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SupplierSearchResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Search suppliers",
    description=(
        "Search supplier profiles using multiple filter criteria including "
        "commodity, country, tier level, compliance status, risk score, "
        "certification type, and free-text query. Supports pagination "
        "and configurable sorting."
    ),
    responses={
        200: {"description": "Search results returned"},
        400: {"model": ErrorResponse, "description": "Invalid search criteria"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def search_suppliers(
    body: SearchCriteriaSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:suppliers:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SupplierSearchResponseSchema:
    """Search supplier profiles with filter criteria.

    Supports full-text search, multi-field filtering, pagination,
    and configurable sort order.

    Args:
        body: Search criteria with filters and pagination.
        request: FastAPI request object.
        user: Authenticated user with suppliers:read permission.

    Returns:
        SupplierSearchResponseSchema with matching profiles.

    Raises:
        HTTPException: 400 on invalid criteria, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Search suppliers request: user=%s query='%s' commodity=%s "
        "country=%s tier=%s limit=%d offset=%d",
        user.user_id,
        (body.query or "")[:80],
        body.commodity,
        body.country_iso,
        body.tier,
        body.limit,
        body.offset,
    )

    try:
        service = get_supplier_service()

        result = service.search_suppliers(
            criteria=body.model_dump(exclude_none=True),
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        total = result.get("total", 0)
        suppliers = result.get("suppliers", [])
        provenance = _compute_provenance(
            f"search|{total}|{body.limit}|{body.offset}|{elapsed}"
        )

        logger.info(
            "Search completed: user=%s total=%d returned=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            len(suppliers),
            elapsed * 1000,
        )

        return SupplierSearchResponseSchema(
            total=total,
            suppliers=suppliers,
            limit=body.limit,
            offset=body.offset,
            has_more=(body.offset + body.limit) < total,
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Search validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Search criteria validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Search suppliers failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier search failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /suppliers/batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=BatchSupplierResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch create/update suppliers",
    description=(
        "Create or update multiple supplier profiles in a single batch "
        "request. Supports upsert mode (update existing suppliers) or "
        "insert-only mode (skip duplicates). Maximum 500 suppliers per batch."
    ),
    responses={
        200: {"description": "Batch operation completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_create_update_suppliers(
    body: BatchSupplierRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:suppliers:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchSupplierResponseSchema:
    """Batch create or update supplier profiles.

    Processes each supplier independently. Individual errors do not
    block the entire batch. Returns per-supplier error details.

    Args:
        body: Batch request with list of supplier profiles and upsert flag.
        request: FastAPI request object.
        user: Authenticated user with suppliers:write permission.

    Returns:
        BatchSupplierResponseSchema with creation/update counts.

    Raises:
        HTTPException: 400 on validation error, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Batch suppliers request: user=%s count=%d upsert=%s",
        user.user_id,
        len(body.suppliers),
        body.upsert,
    )

    try:
        service = get_supplier_service()

        result = service.batch_create_update_suppliers(
            suppliers=[s.model_dump() for s in body.suppliers],
            upsert=body.upsert,
            created_by=user.user_id,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"batch|{result.get('created', 0)}|{result.get('updated', 0)}|{elapsed}"
        )

        logger.info(
            "Batch suppliers completed: user=%s submitted=%d created=%d "
            "updated=%d skipped=%d errors=%d elapsed_ms=%.1f",
            user.user_id,
            len(body.suppliers),
            result.get("created", 0),
            result.get("updated", 0),
            result.get("skipped", 0),
            len(result.get("errors", [])),
            elapsed * 1000,
        )

        return BatchSupplierResponseSchema(
            total_submitted=len(body.suppliers),
            created=result.get("created", 0),
            updated=result.get("updated", 0),
            skipped=result.get("skipped", 0),
            errors=result.get("errors", []),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Batch suppliers validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch supplier validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch suppliers failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch supplier operation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
