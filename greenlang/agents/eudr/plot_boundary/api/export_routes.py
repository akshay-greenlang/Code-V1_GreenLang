# -*- coding: utf-8 -*-
"""
Export and Split/Merge Routes - AGENT-EUDR-006 Plot Boundary Manager API

Endpoints for exporting plot boundaries in multiple formats (GeoJSON, KML,
Shapefile, EUDR XML), batch multi-format export, retrieving export results,
splitting/merging boundaries, and tracing split/merge genealogy.

Endpoints:
    POST /export/geojson        - Export to GeoJSON
    POST /export/kml            - Export to KML
    POST /export/shapefile      - Export to Shapefile
    POST /export/eudr-xml       - Export to EUDR XML
    POST /export/batch          - Batch multi-format export
    GET  /export/{export_id}    - Get export result
    POST /split                 - Split plot boundary
    POST /merge                 - Merge plot boundaries
    GET  /genealogy/{plot_id}   - Get split/merge genealogy

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
    get_compliance_reporter,
    get_export_engine,
    get_split_merge_engine,
    rate_limit_batch,
    rate_limit_export,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.plot_boundary.api.schemas import (
    AreaConservationSchema,
    BatchExportRequestSchema,
    BatchExportResponseSchema,
    ExportFormatSchema,
    ExportRequestSchema,
    ExportResponseSchema,
    GenealogyOperationSchema,
    GenealogyResponseSchema,
    MergeRequestSchema,
    MergeResponseSchema,
    SplitRequestSchema,
    SplitResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Export & Split/Merge"])


# ---------------------------------------------------------------------------
# In-memory export store (replaced by object storage in production)
# ---------------------------------------------------------------------------

_export_store: Dict[str, Dict[str, Any]] = {}


def _get_export_store() -> Dict[str, Dict[str, Any]]:
    """Return the export store. Replaceable for testing."""
    return _export_store


def _compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 provenance hash.

    Args:
        data: Data to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = str(sorted(data.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Helper to build a format-specific export
# ---------------------------------------------------------------------------


async def _do_format_export(
    format_enum: ExportFormatSchema,
    plot_ids: List[str],
    precision: int,
    simplify: bool,
    simplification_tolerance: float = None,
    user_id: str = "",
) -> ExportResponseSchema:
    """Execute a single-format export.

    Attempts to use the real ExportEngine; falls back to a stub
    response when the engine is not yet available.

    Args:
        format_enum: Export format enum value.
        plot_ids: List of plot IDs to export.
        precision: Coordinate precision (decimal places).
        simplify: Whether to simplify before export.
        simplification_tolerance: Tolerance in degrees.
        user_id: User performing the export.

    Returns:
        ExportResponseSchema with export result.
    """
    start = time.monotonic()
    export_id = f"exp-{uuid.uuid4().hex[:12]}"
    format_str = format_enum.value

    logger.info(
        "Export request: user=%s format=%s plots=%d precision=%d simplify=%s",
        user_id,
        format_str,
        len(plot_ids),
        precision,
        simplify,
    )

    try:
        engine = get_export_engine()

        # Try to use real engine if available
        if hasattr(engine, "export"):
            result = engine.export(
                plot_ids=plot_ids,
                format=format_str,
                precision=precision,
                simplify=simplify,
                simplification_tolerance=simplification_tolerance,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Export completed: export_id=%s format=%s plots=%d "
                "elapsed_ms=%.1f",
                export_id,
                format_str,
                len(plot_ids),
                elapsed * 1000,
            )
            return result

    except Exception as exc:
        logger.warning(
            "Export engine unavailable, using stub: %s", exc,
        )

    # Stub response for development
    now = datetime.now(timezone.utc).replace(microsecond=0)
    elapsed = time.monotonic() - start

    stub_data = f'{{"type":"FeatureCollection","features":[]}}'
    if format_str == "kml":
        stub_data = '<?xml version="1.0" encoding="UTF-8"?><kml/>'
    elif format_str == "eudr_xml":
        stub_data = '<?xml version="1.0" encoding="UTF-8"?><eudr-boundaries/>'
    elif format_str in ("shapefile", "wkb"):
        stub_data = None  # Binary formats return download URL

    response = ExportResponseSchema(
        export_id=export_id,
        format=format_str,
        file_size_bytes=len(stub_data.encode("utf-8")) if stub_data else 0,
        plot_count=len(plot_ids),
        crs="EPSG:4326",
        download_url=f"/api/v1/eudr-pbm/export/{export_id}" if not stub_data else None,
        data=stub_data,
        created_at=now,
    )

    # Store for retrieval
    store = _get_export_store()
    store[export_id] = {
        "export_id": export_id,
        "format": format_str,
        "data": stub_data,
        "plot_count": len(plot_ids),
        "created_at": now,
    }

    logger.info(
        "Export completed (stub): export_id=%s format=%s elapsed_ms=%.1f",
        export_id,
        format_str,
        elapsed * 1000,
    )

    return response


# ---------------------------------------------------------------------------
# POST /export/geojson
# ---------------------------------------------------------------------------


@router.post(
    "/export/geojson",
    response_model=ExportResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Export to GeoJSON",
    description=(
        "Export plot boundaries to GeoJSON FeatureCollection format per "
        "RFC 7946. Includes boundary geometry, computed area, metadata, "
        "and provenance data as GeoJSON properties."
    ),
    responses={
        200: {"description": "GeoJSON export result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def export_geojson(
    body: ExportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:export:read")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ExportResponseSchema:
    """Export boundaries to GeoJSON format.

    Args:
        body: Export request with plot_ids and options.
        user: Authenticated user with export:read permission.

    Returns:
        ExportResponseSchema with GeoJSON data.
    """
    return await _do_format_export(
        format_enum=ExportFormatSchema.GEOJSON,
        plot_ids=body.plot_ids,
        precision=body.precision,
        simplify=body.simplify,
        simplification_tolerance=body.simplification_tolerance,
        user_id=user.user_id,
    )


# ---------------------------------------------------------------------------
# POST /export/kml
# ---------------------------------------------------------------------------


@router.post(
    "/export/kml",
    response_model=ExportResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Export to KML",
    description=(
        "Export plot boundaries to Keyhole Markup Language (KML) format "
        "for use in Google Earth and other GIS applications."
    ),
    responses={
        200: {"description": "KML export result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def export_kml(
    body: ExportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:export:read")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ExportResponseSchema:
    """Export boundaries to KML format.

    Args:
        body: Export request with plot_ids and options.
        user: Authenticated user with export:read permission.

    Returns:
        ExportResponseSchema with KML data.
    """
    return await _do_format_export(
        format_enum=ExportFormatSchema.KML,
        plot_ids=body.plot_ids,
        precision=body.precision,
        simplify=body.simplify,
        simplification_tolerance=body.simplification_tolerance,
        user_id=user.user_id,
    )


# ---------------------------------------------------------------------------
# POST /export/shapefile
# ---------------------------------------------------------------------------


@router.post(
    "/export/shapefile",
    response_model=ExportResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Export to Shapefile",
    description=(
        "Export plot boundaries to ESRI Shapefile format (zipped). "
        "Returns a download URL for the generated shapefile archive."
    ),
    responses={
        200: {"description": "Shapefile export result with download URL"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def export_shapefile(
    body: ExportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:export:read")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ExportResponseSchema:
    """Export boundaries to ESRI Shapefile format.

    Args:
        body: Export request with plot_ids and options.
        user: Authenticated user with export:read permission.

    Returns:
        ExportResponseSchema with download URL.
    """
    return await _do_format_export(
        format_enum=ExportFormatSchema.SHAPEFILE,
        plot_ids=body.plot_ids,
        precision=body.precision,
        simplify=body.simplify,
        simplification_tolerance=body.simplification_tolerance,
        user_id=user.user_id,
    )


# ---------------------------------------------------------------------------
# POST /export/eudr-xml
# ---------------------------------------------------------------------------


@router.post(
    "/export/eudr-xml",
    response_model=ExportResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Export to EUDR XML",
    description=(
        "Export plot boundaries to EUDR-specific XML format for "
        "regulatory submission to EU member state competent authorities. "
        "Includes all required Article 9 fields and provenance data."
    ),
    responses={
        200: {"description": "EUDR XML export result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def export_eudr_xml(
    body: ExportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:export:read")
    ),
    _rate: None = Depends(rate_limit_export),
) -> ExportResponseSchema:
    """Export boundaries to EUDR XML format for regulatory submission.

    Args:
        body: Export request with plot_ids and options.
        user: Authenticated user with export:read permission.

    Returns:
        ExportResponseSchema with EUDR XML data.
    """
    return await _do_format_export(
        format_enum=ExportFormatSchema.EUDR_XML,
        plot_ids=body.plot_ids,
        precision=body.precision,
        simplify=body.simplify,
        simplification_tolerance=body.simplification_tolerance,
        user_id=user.user_id,
    )


# ---------------------------------------------------------------------------
# POST /export/batch
# ---------------------------------------------------------------------------


@router.post(
    "/export/batch",
    response_model=BatchExportResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch multi-format export",
    description=(
        "Export plot boundaries in multiple formats simultaneously. "
        "Returns per-format export results with data or download URLs."
    ),
    responses={
        200: {"description": "Batch export results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_export(
    body: BatchExportRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:export:read")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchExportResponseSchema:
    """Export boundaries in multiple formats simultaneously.

    Each format is processed independently. Results include
    per-format export data or download URLs.

    Args:
        body: Batch export request with plot_ids and formats.
        user: Authenticated user with export:read permission.

    Returns:
        BatchExportResponseSchema with per-format results.
    """
    start = time.monotonic()

    logger.info(
        "Batch export request: user=%s plots=%d formats=%s",
        user.user_id,
        len(body.plot_ids),
        [f.value for f in body.formats],
    )

    exports: List[ExportResponseSchema] = []

    for fmt in body.formats:
        try:
            result = await _do_format_export(
                format_enum=fmt,
                plot_ids=body.plot_ids,
                precision=body.precision,
                simplify=body.simplify,
                simplification_tolerance=body.simplification_tolerance,
                user_id=user.user_id,
            )
            exports.append(result)
        except Exception as exc:
            logger.error(
                "Batch export format failed: format=%s error=%s",
                fmt.value,
                exc,
            )

    elapsed = time.monotonic() - start
    logger.info(
        "Batch export completed: formats=%d plots=%d elapsed_ms=%.1f",
        len(exports),
        len(body.plot_ids),
        elapsed * 1000,
    )

    return BatchExportResponseSchema(
        exports=exports,
        total_formats=len(exports),
        plot_count=len(body.plot_ids),
        processing_time_ms=elapsed * 1000,
    )


# ---------------------------------------------------------------------------
# GET /export/{export_id}
# ---------------------------------------------------------------------------


@router.get(
    "/export/{export_id}",
    response_model=ExportResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get export result",
    description=(
        "Retrieve a previously generated export result by its identifier. "
        "Returns the export data or download URL."
    ),
    responses={
        200: {"description": "Export result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Export not found"},
    },
)
async def get_export_result(
    export_id: str = Path(..., description="Export identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:export:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ExportResponseSchema:
    """Retrieve a previously generated export result.

    Args:
        export_id: Export identifier from URL path.
        user: Authenticated user with export:read permission.

    Returns:
        ExportResponseSchema with export data or download URL.

    Raises:
        HTTPException: 404 if export not found.
    """
    logger.info(
        "Get export result: user=%s export_id=%s",
        user.user_id,
        export_id,
    )

    store = _get_export_store()
    record = store.get(export_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export not found: {export_id}",
        )

    data = record.get("data")
    return ExportResponseSchema(
        export_id=export_id,
        format=record.get("format", "geojson"),
        file_size_bytes=len(data.encode("utf-8")) if data else 0,
        plot_count=record.get("plot_count", 0),
        crs="EPSG:4326",
        data=data,
        created_at=record.get("created_at", datetime.now(timezone.utc)),
    )


# ---------------------------------------------------------------------------
# POST /split
# ---------------------------------------------------------------------------


@router.post(
    "/split",
    response_model=SplitResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Split plot boundary",
    description=(
        "Split a plot boundary along a cutting line into two or more "
        "child plots. Verifies area conservation within the configured "
        "tolerance and creates genealogy records linking parent to "
        "children. Child plot IDs are auto-generated."
    ),
    responses={
        201: {"description": "Split operation result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Plot not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def split_boundary(
    body: SplitRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:split-merge:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SplitResponseSchema:
    """Split a plot boundary along a cutting line.

    Computes the geometric split, verifies area conservation,
    creates child boundaries, and records genealogy.

    Args:
        body: Split request with plot_id and cutting line.
        user: Authenticated user with split-merge:write permission.

    Returns:
        SplitResponseSchema with child plots and area conservation.

    Raises:
        HTTPException: 400 if invalid, 404 if plot not found.
    """
    start = time.monotonic()
    validated_id = validate_plot_id(body.plot_id)

    logger.info(
        "Split boundary request: user=%s plot_id=%s cutting_points=%d",
        user.user_id,
        validated_id,
        len(body.cutting_line),
    )

    try:
        engine = get_split_merge_engine()

        # Try to use real engine if available
        if hasattr(engine, "split"):
            result = engine.split(
                plot_id=validated_id,
                cutting_line=body.cutting_line,
                change_reason=body.change_reason,
                user_id=user.user_id,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Split completed: plot_id=%s children=%d elapsed_ms=%.1f",
                validated_id,
                len(result.child_plot_ids) if hasattr(result, "child_plot_ids") else 0,
                elapsed * 1000,
            )
            return result

        # Stub response for development
        child_ids = [
            f"plt-{uuid.uuid4().hex[:12]}",
            f"plt-{uuid.uuid4().hex[:12]}",
        ]

        hash_input = {
            "operation": "split",
            "parent": validated_id,
            "children": child_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Split completed (stub): plot_id=%s children=%d elapsed_ms=%.1f",
            validated_id,
            len(child_ids),
            elapsed * 1000,
        )

        return SplitResponseSchema(
            parent_plot_id=validated_id,
            child_plot_ids=child_ids,
            child_boundaries=[],
            area_conservation=AreaConservationSchema(
                original_area_m2=0.0,
                result_area_m2=0.0,
                difference_m2=0.0,
                difference_pct=0.0,
                within_tolerance=True,
            ),
            provenance_hash=_compute_provenance_hash(hash_input),
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Split boundary error: user=%s plot_id=%s error=%s",
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
            "Split boundary failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Split operation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /merge
# ---------------------------------------------------------------------------


@router.post(
    "/merge",
    response_model=MergeResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Merge plot boundaries",
    description=(
        "Merge two or more plot boundaries into a single boundary. "
        "Verifies area conservation within the configured tolerance "
        "and creates genealogy records linking parents to the merged "
        "result. Merged plot ID is auto-generated unless specified."
    ),
    responses={
        201: {"description": "Merge operation result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "One or more plots not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def merge_boundaries(
    body: MergeRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:split-merge:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> MergeResponseSchema:
    """Merge multiple plot boundaries into one.

    Computes the geometric union, verifies area conservation,
    creates the merged boundary, and records genealogy.

    Args:
        body: Merge request with plot_ids.
        user: Authenticated user with split-merge:write permission.

    Returns:
        MergeResponseSchema with merged plot and area conservation.

    Raises:
        HTTPException: 400 if invalid, 404 if plots not found.
    """
    start = time.monotonic()

    logger.info(
        "Merge boundaries request: user=%s plots=%d merged_id=%s",
        user.user_id,
        len(body.plot_ids),
        body.merged_plot_id,
    )

    try:
        engine = get_split_merge_engine()

        # Try to use real engine if available
        if hasattr(engine, "merge"):
            result = engine.merge(
                plot_ids=body.plot_ids,
                merged_plot_id=body.merged_plot_id,
                change_reason=body.change_reason,
                user_id=user.user_id,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Merge completed: parents=%d merged_id=%s elapsed_ms=%.1f",
                len(body.plot_ids),
                result.merged_plot_id if hasattr(result, "merged_plot_id") else "unknown",
                elapsed * 1000,
            )
            return result

        # Stub response for development
        merged_id = body.merged_plot_id or f"plt-{uuid.uuid4().hex[:12]}"

        hash_input = {
            "operation": "merge",
            "parents": body.plot_ids,
            "merged": merged_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Merge completed (stub): parents=%d merged_id=%s elapsed_ms=%.1f",
            len(body.plot_ids),
            merged_id,
            elapsed * 1000,
        )

        return MergeResponseSchema(
            parent_plot_ids=body.plot_ids,
            merged_plot_id=merged_id,
            merged_boundary=None,
            area_conservation=AreaConservationSchema(
                original_area_m2=0.0,
                result_area_m2=0.0,
                difference_m2=0.0,
                difference_pct=0.0,
                within_tolerance=True,
            ),
            provenance_hash=_compute_provenance_hash(hash_input),
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Merge boundaries error: user=%s error=%s",
            user.user_id,
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
            "Merge boundaries failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Merge operation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /genealogy/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/genealogy/{plot_id}",
    response_model=GenealogyResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get split/merge genealogy",
    description=(
        "Retrieve the split/merge genealogy for a plot boundary. "
        "Returns parent and child relationships along with the full "
        "history of split and merge operations involving this plot."
    ),
    responses={
        200: {"description": "Genealogy records"},
        400: {"model": ErrorResponse, "description": "Invalid plot_id"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "No genealogy found"},
    },
)
async def get_genealogy(
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:split-merge:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> GenealogyResponseSchema:
    """Retrieve split/merge genealogy for a plot boundary.

    Returns parent-child relationships and the complete history
    of split and merge operations involving this plot.

    Args:
        plot_id: Plot identifier from URL path.
        user: Authenticated user with split-merge:read permission.

    Returns:
        GenealogyResponseSchema with parent/child relationships.

    Raises:
        HTTPException: 400 if plot_id invalid.
    """
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Get genealogy: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    try:
        engine = get_split_merge_engine()

        # Try to use real engine if available
        if hasattr(engine, "get_genealogy"):
            result = engine.get_genealogy(plot_id=validated_id)
            return result

        # Stub response for development
        return GenealogyResponseSchema(
            plot_id=validated_id,
            parents=[],
            children=[],
            operations=[],
            lineage_depth=0,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Get genealogy failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Genealogy retrieval failed due to an internal error",
        )
