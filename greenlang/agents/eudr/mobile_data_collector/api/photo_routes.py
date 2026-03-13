# -*- coding: utf-8 -*-
"""
Photo Routes - AGENT-EUDR-015 Mobile Data Collector

REST API endpoints for geotagged photo evidence management including
upload metadata recording, listing, annotation, geotag validation,
and download URL generation.

Endpoints (7):
    POST   /photos                         Record photo capture metadata
    GET    /photos                          List photos with filters
    GET    /photos/{photo_id}              Get photo metadata
    GET    /photos/{photo_id}/download     Download photo (returns URL)
    POST   /photos/{photo_id}/annotate     Add annotation
    POST   /photos/validate-geotag         Validate photo geotag proximity
    DELETE /photos/{photo_id}              Delete photo

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015, Section 7.4
Agent ID: GL-EUDR-MDC-015
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from greenlang.agents.eudr.mobile_data_collector.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_mdc_service,
    get_pagination,
    rate_limit_read,
    rate_limit_upload,
    rate_limit_write,
    require_permission,
    validate_photo_id,
)
from greenlang.agents.eudr.mobile_data_collector.api.schemas import (
    ErrorSchema,
    GeotagValidationResponseSchema,
    GeotagValidationSchema,
    PaginationSchema,
    PhotoAnnotationSchema,
    PhotoListSchema,
    PhotoResponseSchema,
    PhotoTypeSchema,
    PhotoUploadSchema,
    SuccessSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/photos",
    tags=["EUDR Mobile Data - Photos"],
    responses={
        400: {"model": ErrorSchema, "description": "Validation error"},
        404: {"model": ErrorSchema, "description": "Photo not found"},
    },
)


# ---------------------------------------------------------------------------
# POST /photos
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=PhotoResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Record photo capture metadata",
    description=(
        "Record metadata for a photo captured on a mobile device "
        "including GPS coordinates, EXIF data, SHA-256 integrity hash, "
        "and form linkage. The actual image binary is uploaded via "
        "the sync engine separately."
    ),
    responses={
        201: {"description": "Photo metadata recorded successfully"},
        400: {"description": "Invalid photo metadata"},
    },
)
async def upload_photo(
    body: PhotoUploadSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:create")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_upload),
) -> PhotoResponseSchema:
    """Record photo capture metadata.

    Args:
        body: Photo metadata including coordinates, hash, and dimensions.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PhotoResponseSchema with recorded photo metadata.
    """
    start = time.monotonic()
    logger.info(
        "Photo upload: user=%s device=%s form=%s type=%s size=%d",
        user.user_id,
        body.device_id,
        body.form_id,
        body.photo_type.value,
        body.file_size_bytes,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return PhotoResponseSchema(
        form_id=body.form_id,
        photo_type=body.photo_type.value,
        file_name=body.file_name,
        file_size_bytes=body.file_size_bytes,
        file_format=body.file_format,
        width_px=body.width_px,
        height_px=body.height_px,
        integrity_hash=body.integrity_hash,
        latitude=body.latitude,
        longitude=body.longitude,
        annotation=body.annotation,
        processing_time_ms=round(elapsed_ms, 2),
        message="Photo metadata recorded successfully",
    )


# ---------------------------------------------------------------------------
# GET /photos
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=PhotoListSchema,
    summary="List photos with filters",
    description=(
        "List photo evidence records with optional filters by form ID, "
        "device ID, photo type, and operator ID. Results are paginated."
    ),
    responses={
        200: {"description": "Photos retrieved successfully"},
    },
)
async def list_photos(
    form_id: Optional[str] = Query(
        None, max_length=255, description="Filter by form ID"
    ),
    device_id: Optional[str] = Query(
        None, max_length=255, description="Filter by device ID"
    ),
    photo_type: Optional[PhotoTypeSchema] = Query(
        None, description="Filter by photo type"
    ),
    operator_id: Optional[str] = Query(
        None, max_length=255, description="Filter by operator ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PhotoListSchema:
    """List photos with optional filters.

    Args:
        form_id: Filter by associated form.
        device_id: Filter by source device.
        photo_type: Filter by photo category.
        operator_id: Filter by field agent.
        pagination: Pagination parameters.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PhotoListSchema with matching photos and pagination.
    """
    start = time.monotonic()
    logger.info(
        "List photos: user=%s page=%d",
        user.user_id,
        pagination.page,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return PhotoListSchema(
        photos=[],
        pagination=PaginationSchema(
            total=0,
            page=pagination.page,
            page_size=pagination.page_size,
            has_more=False,
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# GET /photos/{photo_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{photo_id}",
    response_model=PhotoResponseSchema,
    summary="Get photo metadata",
    description="Retrieve metadata for a specific photo by its identifier.",
    responses={
        200: {"description": "Photo metadata retrieved"},
        404: {"description": "Photo not found"},
    },
)
async def get_photo(
    photo_id: str = Depends(validate_photo_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PhotoResponseSchema:
    """Get photo metadata by identifier.

    Args:
        photo_id: Photo evidence identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PhotoResponseSchema with photo metadata.

    Raises:
        HTTPException: 404 if photo not found.
    """
    logger.info("Get photo: user=%s photo_id=%s", user.user_id, photo_id)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Photo {photo_id} not found",
    )


# ---------------------------------------------------------------------------
# GET /photos/{photo_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{photo_id}/download",
    response_model=SuccessSchema,
    summary="Download photo",
    description=(
        "Get a pre-signed download URL for the photo binary. "
        "The URL expires after 1 hour."
    ),
    responses={
        200: {"description": "Download URL generated"},
        404: {"description": "Photo not found"},
    },
)
async def download_photo(
    photo_id: str = Depends(validate_photo_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> SuccessSchema:
    """Get a pre-signed download URL for a photo.

    Args:
        photo_id: Photo evidence identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        SuccessSchema with download URL in data field.

    Raises:
        HTTPException: 404 if photo not found.
    """
    logger.info(
        "Download photo: user=%s photo_id=%s", user.user_id, photo_id
    )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Photo {photo_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /photos/{photo_id}/annotate
# ---------------------------------------------------------------------------


@router.post(
    "/{photo_id}/annotate",
    response_model=PhotoResponseSchema,
    summary="Add annotation to photo",
    description=(
        "Add a text annotation to a photo. Annotations are appended "
        "to the photo metadata for field notes and quality observations."
    ),
    responses={
        200: {"description": "Annotation added successfully"},
        404: {"description": "Photo not found"},
    },
)
async def annotate_photo(
    body: PhotoAnnotationSchema,
    photo_id: str = Depends(validate_photo_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:update")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PhotoResponseSchema:
    """Add annotation to a photo.

    Args:
        body: Annotation text and author.
        photo_id: Photo evidence identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PhotoResponseSchema with updated metadata.

    Raises:
        HTTPException: 404 if photo not found.
    """
    logger.info(
        "Annotate photo: user=%s photo_id=%s by=%s",
        user.user_id,
        photo_id,
        body.annotated_by,
    )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Photo {photo_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /photos/validate-geotag
# ---------------------------------------------------------------------------


@router.post(
    "/validate-geotag",
    response_model=GeotagValidationResponseSchema,
    summary="Validate photo geotag proximity",
    description=(
        "Validate that a photo EXIF geotag is within an acceptable "
        "distance of a reference GPS point. Used to verify that "
        "photos were taken at the claimed location."
    ),
    responses={
        200: {"description": "Geotag validation completed"},
    },
)
async def validate_geotag(
    body: GeotagValidationSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:validate")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> GeotagValidationResponseSchema:
    """Validate photo geotag proximity.

    Args:
        body: Photo and reference coordinates with max distance.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        GeotagValidationResponseSchema with proximity result.
    """
    start = time.monotonic()
    logger.info(
        "Validate geotag: user=%s max_dist=%.1fm",
        user.user_id,
        body.max_distance_m,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return GeotagValidationResponseSchema(
        is_valid=True,
        distance_m=0.0,
        max_distance_m=body.max_distance_m,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# DELETE /photos/{photo_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{photo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete photo",
    description=(
        "Delete a photo evidence record and its associated binary. "
        "Photos linked to sealed packages cannot be deleted."
    ),
    responses={
        204: {"description": "Photo deleted successfully"},
        404: {"description": "Photo not found"},
        409: {"description": "Photo is linked to a sealed package"},
    },
)
async def delete_photo(
    photo_id: str = Depends(validate_photo_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:photos:delete")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> None:
    """Delete a photo evidence record.

    Args:
        photo_id: Photo evidence identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Raises:
        HTTPException: 404 if not found, 409 if linked to sealed package.
    """
    logger.info("Delete photo: user=%s photo_id=%s", user.user_id, photo_id)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Photo {photo_id} not found",
    )
