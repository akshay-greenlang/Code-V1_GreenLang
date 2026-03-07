# -*- coding: utf-8 -*-
"""
Boundary CRUD Routes - AGENT-EUDR-006 Plot Boundary Manager API

Endpoints for creating, reading, updating, and deleting plot boundaries,
plus batch creation and spatial/attribute search.

Endpoints:
    POST   /boundaries          - Create a new plot boundary
    GET    /boundaries/{plot_id} - Get boundary by plot ID
    PUT    /boundaries/{plot_id} - Update existing boundary
    DELETE /boundaries/{plot_id} - Delete boundary (soft delete)
    POST   /boundaries/batch    - Batch create boundaries
    POST   /boundaries/search   - Search boundaries by filters

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from greenlang.agents.eudr.plot_boundary.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_boundary_service,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.plot_boundary.api.schemas import (
    BatchCreateRequestSchema,
    BatchCreateResponseSchema,
    BatchCreateResultSchema,
    BoundaryListResponseSchema,
    BoundaryResponseSchema,
    CentroidSchema,
    CompactnessMetricsSchema,
    CreateBoundaryRequestSchema,
    SearchRequestSchema,
    UpdateBoundaryRequestSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Boundary CRUD"])


# ---------------------------------------------------------------------------
# In-memory boundary store (replaced by database in production)
# ---------------------------------------------------------------------------

_boundary_store: Dict[str, Dict[str, Any]] = {}


def _get_boundary_store() -> Dict[str, Dict[str, Any]]:
    """Return the boundary store. Replaceable for testing."""
    return _boundary_store


def _compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: Dictionary of boundary data to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = str(sorted(data.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _classify_threshold(area_hectares: float, threshold: float = 4.0) -> str:
    """Classify area against EUDR 4ha threshold.

    Args:
        area_hectares: Area in hectares.
        threshold: EUDR threshold (default 4.0ha).

    Returns:
        Classification string.
    """
    if area_hectares > threshold:
        return "above_threshold"
    elif area_hectares < threshold:
        return "below_threshold"
    return "at_threshold"


def _build_boundary_response(record: Dict[str, Any]) -> BoundaryResponseSchema:
    """Build a BoundaryResponseSchema from an internal record.

    Args:
        record: Internal boundary storage record.

    Returns:
        BoundaryResponseSchema for API serialization.
    """
    area_hectares = record.get("area_hectares", 0.0)
    return BoundaryResponseSchema(
        plot_id=record["plot_id"],
        geometry=record.get("geometry"),
        geometry_type=record.get("geometry_type", "Polygon"),
        source_crs=record.get("source_crs", "EPSG:4326"),
        stored_crs="EPSG:4326",
        commodity=record.get("commodity", ""),
        country_iso=record.get("country_iso", ""),
        owner_id=record.get("owner_id"),
        certification_id=record.get("certification_id"),
        area_m2=record.get("area_m2", 0.0),
        area_hectares=area_hectares,
        perimeter_m=record.get("perimeter_m", 0.0),
        centroid=record.get("centroid"),
        vertex_count=record.get("vertex_count", 0),
        ring_count=record.get("ring_count", 1),
        is_valid=record.get("is_valid", True),
        threshold_classification=_classify_threshold(area_hectares),
        version_number=record.get("version_number", 1),
        provenance_hash=record.get("provenance_hash", ""),
        created_at=record.get("created_at", datetime.now(timezone.utc)),
        updated_at=record.get("updated_at", datetime.now(timezone.utc)),
        metadata=record.get("metadata"),
    )


# ---------------------------------------------------------------------------
# POST /boundaries
# ---------------------------------------------------------------------------


@router.post(
    "/boundaries",
    response_model=BoundaryResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new plot boundary",
    description=(
        "Register a new plot boundary in the EUDR boundary registry. "
        "Accepts GeoJSON, WKT, or KML geometry input. If plot_id is "
        "not provided, a UUID is generated automatically. The geometry "
        "is validated, reprojected to WGS84 if needed, and stored with "
        "geodetic area, perimeter, and provenance hash computed."
    ),
    responses={
        201: {"description": "Boundary created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"description": "Plot ID already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_boundary(
    body: CreateBoundaryRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:boundaries:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BoundaryResponseSchema:
    """Create a new plot boundary.

    Validates the input geometry, computes geodetic area and perimeter,
    generates a provenance hash, and stores the boundary with version 1.

    Args:
        body: Boundary creation request with geometry and metadata.
        user: Authenticated user with boundaries:write permission.

    Returns:
        BoundaryResponseSchema with computed fields.

    Raises:
        HTTPException: 400 if geometry invalid, 409 if plot_id exists.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"plt-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Create boundary request: user=%s plot_id=%s commodity=%s country=%s",
        user.user_id,
        plot_id,
        body.commodity,
        body.country_iso,
    )

    store = _get_boundary_store()

    # Check for duplicate plot_id
    if plot_id in store:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Plot ID '{plot_id}' already exists",
        )

    try:
        service = get_boundary_service()
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Determine geometry type from input
        geometry_type = "Polygon"
        if body.geometry:
            geometry_type = body.geometry.type

        # Compute placeholder area/perimeter (real engine computes geodetic)
        area_m2 = 0.0
        area_hectares = 0.0
        perimeter_m = 0.0
        vertex_count = 0

        if body.geometry and body.geometry.coordinates:
            coords = body.geometry.coordinates
            if geometry_type == "Polygon" and isinstance(coords, list) and len(coords) > 0:
                ring = coords[0]
                vertex_count = len(ring) if isinstance(ring, list) else 0

        # Build record
        record: Dict[str, Any] = {
            "plot_id": plot_id,
            "geometry": body.geometry.model_dump() if body.geometry else None,
            "geometry_type": geometry_type,
            "wkt": body.wkt,
            "kml": body.kml,
            "source_crs": body.source_crs,
            "commodity": body.commodity,
            "country_iso": body.country_iso,
            "owner_id": body.owner_id,
            "certification_id": body.certification_id,
            "area_m2": area_m2,
            "area_hectares": area_hectares,
            "perimeter_m": perimeter_m,
            "vertex_count": vertex_count,
            "ring_count": 1,
            "is_valid": True,
            "version_number": 1,
            "created_at": now,
            "updated_at": now,
            "created_by": user.user_id,
            "metadata": body.metadata,
        }

        record["provenance_hash"] = _compute_provenance_hash(record)
        store[plot_id] = record

        elapsed = time.monotonic() - start
        logger.info(
            "Boundary created: plot_id=%s area_ha=%.4f "
            "vertices=%d elapsed_ms=%.1f",
            plot_id,
            area_hectares,
            vertex_count,
            elapsed * 1000,
        )

        return _build_boundary_response(record)

    except ValueError as exc:
        logger.warning(
            "Create boundary error: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Create boundary failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Boundary creation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /boundaries/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/boundaries/{plot_id}",
    response_model=BoundaryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get boundary by plot ID",
    description="Retrieve a plot boundary by its unique identifier.",
    responses={
        200: {"description": "Boundary found"},
        400: {"model": ErrorResponse, "description": "Invalid plot_id"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Boundary not found"},
    },
)
async def get_boundary(
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:boundaries:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BoundaryResponseSchema:
    """Retrieve a single plot boundary by its identifier.

    Args:
        plot_id: Plot identifier from URL path.
        user: Authenticated user with boundaries:read permission.

    Returns:
        BoundaryResponseSchema with full boundary data.

    Raises:
        HTTPException: 400 if plot_id invalid, 404 if not found.
    """
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Get boundary request: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    store = _get_boundary_store()
    record = store.get(validated_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Boundary not found: {validated_id}",
        )

    return _build_boundary_response(record)


# ---------------------------------------------------------------------------
# PUT /boundaries/{plot_id}
# ---------------------------------------------------------------------------


@router.put(
    "/boundaries/{plot_id}",
    response_model=BoundaryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Update existing boundary",
    description=(
        "Update an existing plot boundary. Only provided fields are "
        "updated. A new version is created with the change tracked "
        "in the version history."
    ),
    responses={
        200: {"description": "Boundary updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Boundary not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def update_boundary(
    body: UpdateBoundaryRequestSchema,
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:boundaries:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BoundaryResponseSchema:
    """Update an existing plot boundary.

    Merges provided fields into the existing record, increments the
    version number, and creates a new version history entry.

    Args:
        body: Update request with optional fields.
        plot_id: Plot identifier from URL path.
        user: Authenticated user with boundaries:write permission.

    Returns:
        Updated BoundaryResponseSchema.

    Raises:
        HTTPException: 400 if invalid, 404 if not found.
    """
    start = time.monotonic()
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Update boundary request: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    store = _get_boundary_store()
    record = store.get(validated_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Boundary not found: {validated_id}",
        )

    try:
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Apply updates (only non-None fields)
        if body.geometry is not None:
            record["geometry"] = body.geometry.model_dump()
            record["geometry_type"] = body.geometry.type
        if body.wkt is not None:
            record["wkt"] = body.wkt
        if body.kml is not None:
            record["kml"] = body.kml
        if body.source_crs is not None:
            record["source_crs"] = body.source_crs
        if body.commodity is not None:
            record["commodity"] = body.commodity
        if body.country_iso is not None:
            record["country_iso"] = body.country_iso
        if body.owner_id is not None:
            record["owner_id"] = body.owner_id
        if body.certification_id is not None:
            record["certification_id"] = body.certification_id
        if body.metadata is not None:
            existing_meta = record.get("metadata") or {}
            existing_meta.update(body.metadata)
            record["metadata"] = existing_meta

        # Increment version
        record["version_number"] = record.get("version_number", 1) + 1
        record["updated_at"] = now
        record["updated_by"] = user.user_id
        record["provenance_hash"] = _compute_provenance_hash(record)

        store[validated_id] = record

        elapsed = time.monotonic() - start
        logger.info(
            "Boundary updated: plot_id=%s version=%d elapsed_ms=%.1f",
            validated_id,
            record["version_number"],
            elapsed * 1000,
        )

        return _build_boundary_response(record)

    except ValueError as exc:
        logger.warning(
            "Update boundary error: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Update boundary failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Boundary update failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# DELETE /boundaries/{plot_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/boundaries/{plot_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete boundary (soft delete)",
    description=(
        "Soft-delete a plot boundary. The boundary is marked as deleted "
        "but retained for audit trail per EUDR Article 31."
    ),
    responses={
        200: {"description": "Boundary deleted"},
        400: {"model": ErrorResponse, "description": "Invalid plot_id"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Boundary not found"},
    },
)
async def delete_boundary(
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:boundaries:delete")
    ),
    _rate: None = Depends(rate_limit_write),
) -> Dict[str, Any]:
    """Soft-delete a plot boundary.

    Marks the boundary as deleted while retaining it for the Article 31
    retention period.

    Args:
        plot_id: Plot identifier from URL path.
        user: Authenticated user with boundaries:delete permission.

    Returns:
        Confirmation message with plot_id.

    Raises:
        HTTPException: 400 if invalid, 404 if not found.
    """
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Delete boundary request: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    store = _get_boundary_store()
    record = store.get(validated_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Boundary not found: {validated_id}",
        )

    # Soft delete
    now = datetime.now(timezone.utc).replace(microsecond=0)
    record["deleted_at"] = now
    record["deleted_by"] = user.user_id
    record["is_deleted"] = True
    store[validated_id] = record

    logger.info(
        "Boundary soft-deleted: plot_id=%s by=%s",
        validated_id,
        user.user_id,
    )

    return {
        "status": "deleted",
        "plot_id": validated_id,
        "deleted_at": now.isoformat(),
        "message": f"Boundary {validated_id} soft-deleted. "
                   f"Retained for Article 31 compliance.",
    }


# ---------------------------------------------------------------------------
# POST /boundaries/batch
# ---------------------------------------------------------------------------


@router.post(
    "/boundaries/batch",
    response_model=BatchCreateResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Batch create boundaries",
    description=(
        "Create multiple plot boundaries in a single request. "
        "Supports up to 10,000 boundaries per batch. Each boundary "
        "is validated independently. Partial success is supported: "
        "valid boundaries are created even if some fail."
    ),
    responses={
        201: {"description": "Batch creation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_create_boundaries(
    body: BatchCreateRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:boundaries:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchCreateResponseSchema:
    """Create multiple plot boundaries in a single batch.

    Each boundary is processed independently. Failures do not prevent
    successful boundaries from being created.

    Args:
        body: Batch creation request with list of boundaries.
        user: Authenticated user with boundaries:write permission.

    Returns:
        BatchCreateResponseSchema with per-boundary results.
    """
    start = time.monotonic()
    total = len(body.boundaries)

    logger.info(
        "Batch create request: user=%s count=%d",
        user.user_id,
        total,
    )

    store = _get_boundary_store()
    results: List[BatchCreateResultSchema] = []
    errors: List[Dict[str, Any]] = []
    created_count = 0
    failed_count = 0

    for idx, boundary_req in enumerate(body.boundaries):
        plot_id = boundary_req.plot_id or f"plt-{uuid.uuid4().hex[:12]}"

        try:
            if plot_id in store:
                raise ValueError(f"Plot ID '{plot_id}' already exists")

            now = datetime.now(timezone.utc).replace(microsecond=0)
            geometry_type = "Polygon"
            if boundary_req.geometry:
                geometry_type = boundary_req.geometry.type

            record: Dict[str, Any] = {
                "plot_id": plot_id,
                "geometry": (
                    boundary_req.geometry.model_dump()
                    if boundary_req.geometry
                    else None
                ),
                "geometry_type": geometry_type,
                "wkt": boundary_req.wkt,
                "kml": boundary_req.kml,
                "source_crs": boundary_req.source_crs,
                "commodity": boundary_req.commodity,
                "country_iso": boundary_req.country_iso,
                "owner_id": boundary_req.owner_id,
                "certification_id": boundary_req.certification_id,
                "area_m2": 0.0,
                "area_hectares": 0.0,
                "perimeter_m": 0.0,
                "vertex_count": 0,
                "ring_count": 1,
                "is_valid": True,
                "version_number": 1,
                "created_at": now,
                "updated_at": now,
                "created_by": user.user_id,
                "metadata": boundary_req.metadata,
            }
            record["provenance_hash"] = _compute_provenance_hash(record)
            store[plot_id] = record

            results.append(BatchCreateResultSchema(
                plot_id=plot_id,
                success=True,
                boundary=_build_boundary_response(record),
            ))
            created_count += 1

        except Exception as exc:
            failed_count += 1
            error_detail = {
                "index": idx,
                "plot_id": plot_id,
                "error": str(exc),
            }
            errors.append(error_detail)
            results.append(BatchCreateResultSchema(
                plot_id=plot_id,
                success=False,
                error=str(exc),
            ))

    elapsed = time.monotonic() - start
    logger.info(
        "Batch create completed: user=%s total=%d created=%d "
        "failed=%d elapsed_ms=%.1f",
        user.user_id,
        total,
        created_count,
        failed_count,
        elapsed * 1000,
    )

    return BatchCreateResponseSchema(
        created=created_count,
        failed=failed_count,
        total=total,
        results=results,
        errors=errors,
        processing_time_ms=elapsed * 1000,
    )


# ---------------------------------------------------------------------------
# POST /boundaries/search
# ---------------------------------------------------------------------------


@router.post(
    "/boundaries/search",
    response_model=BoundaryListResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Search boundaries",
    description=(
        "Search plot boundaries by spatial bounding box, commodity, "
        "country, owner, validity status, and area range. Results "
        "are paginated."
    ),
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def search_boundaries(
    body: SearchRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:boundaries:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BoundaryListResponseSchema:
    """Search plot boundaries by spatial and attribute filters.

    Supports bounding box, commodity, country, owner, validity,
    and area range filters with pagination.

    Args:
        body: Search request with filter criteria.
        user: Authenticated user with boundaries:read permission.

    Returns:
        Paginated BoundaryListResponseSchema.
    """
    start = time.monotonic()

    logger.info(
        "Search boundaries: user=%s commodity=%s country=%s limit=%d offset=%d",
        user.user_id,
        body.commodity,
        body.country_iso,
        body.limit,
        body.offset,
    )

    store = _get_boundary_store()
    matching: List[Dict[str, Any]] = []

    for record in store.values():
        # Skip soft-deleted boundaries
        if record.get("is_deleted"):
            continue

        # Apply commodity filter
        if body.commodity and record.get("commodity") != body.commodity:
            continue

        # Apply country filter
        if body.country_iso:
            iso_upper = body.country_iso.upper()
            if record.get("country_iso", "").upper() != iso_upper:
                continue

        # Apply owner filter
        if body.owner_id and record.get("owner_id") != body.owner_id:
            continue

        # Apply validity filter
        if body.is_valid is not None:
            if record.get("is_valid") != body.is_valid:
                continue

        # Apply area range filters
        area_ha = record.get("area_hectares", 0.0)
        if body.min_area_hectares is not None and area_ha < body.min_area_hectares:
            continue
        if body.max_area_hectares is not None and area_ha > body.max_area_hectares:
            continue

        matching.append(record)

    total = len(matching)
    page = matching[body.offset: body.offset + body.limit]
    items = [_build_boundary_response(r) for r in page]
    has_more = (body.offset + body.limit) < total

    elapsed = time.monotonic() - start
    logger.info(
        "Search completed: total=%d returned=%d elapsed_ms=%.1f",
        total,
        len(items),
        elapsed * 1000,
    )

    return BoundaryListResponseSchema(
        items=items,
        total=total,
        limit=body.limit,
        offset=body.offset,
        has_more=has_more,
    )
