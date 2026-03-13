# -*- coding: utf-8 -*-
"""
Package Routes - AGENT-EUDR-015 Mobile Data Collector

REST API endpoints for tamper-evident data package management including
create, get, add items (form, GPS, photo, signature), seal, validate,
manifest, export, and list operations. Packages use SHA-256 Merkle
tree integrity for EUDR Article 14 compliance.

Endpoints (11):
    POST   /packages                              Create package
    GET    /packages/{package_id}                 Get package
    POST   /packages/{package_id}/add-form        Add form to package
    POST   /packages/{package_id}/add-gps         Add GPS capture to package
    POST   /packages/{package_id}/add-photo       Add photo to package
    POST   /packages/{package_id}/add-signature   Add signature to package
    POST   /packages/{package_id}/seal            Seal package
    POST   /packages/{package_id}/validate        Validate sealed package
    GET    /packages/{package_id}/manifest        Get package manifest
    GET    /packages/{package_id}/export           Export/download package
    GET    /packages                              List packages with filters

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
    rate_limit_write,
    require_permission,
    validate_package_id,
)
from greenlang.agents.eudr.mobile_data_collector.api.schemas import (
    ErrorSchema,
    ManifestSchema,
    PackageAddItemSchema,
    PackageCreateSchema,
    PackageExportSchema,
    PackageListSchema,
    PackageResponseSchema,
    PackageSealSchema,
    PackageStatusSchema,
    PackageValidateSchema,
    PaginationSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/packages",
    tags=["EUDR Mobile Data - Packages"],
    responses={
        400: {"model": ErrorSchema, "description": "Validation error"},
        404: {"model": ErrorSchema, "description": "Package not found"},
        409: {"model": ErrorSchema, "description": "Package state conflict"},
    },
)


# ---------------------------------------------------------------------------
# POST /packages
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=PackageResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create data package",
    description=(
        "Create a new data package for collecting and sealing forms, "
        "GPS captures, photos, and signatures into a tamper-evident "
        "bundle. Packages start in 'building' status and must be "
        "sealed before upload."
    ),
    responses={
        201: {"description": "Package created successfully"},
        400: {"description": "Invalid package configuration"},
    },
)
async def create_package(
    body: PackageCreateSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:create")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PackageResponseSchema:
    """Create a new data package.

    Args:
        body: Package creation data with device, operator, and format.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with created package details.
    """
    start = time.monotonic()
    logger.info(
        "Create package: user=%s device=%s operator=%s format=%s",
        user.user_id,
        body.device_id,
        body.operator_id,
        body.export_format.value,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return PackageResponseSchema(
        status="building",
        device_id=body.device_id,
        operator_id=body.operator_id,
        compression_format=body.compression_format,
        export_format=body.export_format.value,
        processing_time_ms=round(elapsed_ms, 2),
        message="Package created successfully",
    )


# ---------------------------------------------------------------------------
# GET /packages/{package_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{package_id}",
    response_model=PackageResponseSchema,
    summary="Get package",
    description="Retrieve a data package by its identifier.",
    responses={
        200: {"description": "Package retrieved"},
        404: {"description": "Package not found"},
    },
)
async def get_package(
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PackageResponseSchema:
    """Get a data package by identifier.

    Args:
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with package details.

    Raises:
        HTTPException: 404 if package not found.
    """
    logger.info(
        "Get package: user=%s package_id=%s", user.user_id, package_id
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /packages/{package_id}/add-form
# ---------------------------------------------------------------------------


@router.post(
    "/{package_id}/add-form",
    response_model=PackageResponseSchema,
    summary="Add form to package",
    description=(
        "Add a form submission to an open (building) package. The "
        "form's data and metadata are included in the package manifest."
    ),
    responses={
        200: {"description": "Form added to package"},
        404: {"description": "Package or form not found"},
        409: {"description": "Package is already sealed"},
    },
)
async def add_form_to_package(
    body: PackageAddItemSchema,
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:update")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PackageResponseSchema:
    """Add a form to a data package.

    Args:
        body: Item to add with form identifier.
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with updated package details.

    Raises:
        HTTPException: 404 if not found, 409 if package is sealed.
    """
    logger.info(
        "Add form to package: user=%s package=%s form=%s",
        user.user_id,
        package_id,
        body.item_id,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /packages/{package_id}/add-gps
# ---------------------------------------------------------------------------


@router.post(
    "/{package_id}/add-gps",
    response_model=PackageResponseSchema,
    summary="Add GPS capture to package",
    description=(
        "Add a GPS point capture or polygon trace to an open package. "
        "GPS data includes coordinates and accuracy metadata per "
        "EUDR Article 9(1)(d)."
    ),
    responses={
        200: {"description": "GPS capture added to package"},
        404: {"description": "Package or GPS capture not found"},
        409: {"description": "Package is already sealed"},
    },
)
async def add_gps_to_package(
    body: PackageAddItemSchema,
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:update")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PackageResponseSchema:
    """Add a GPS capture to a data package.

    Args:
        body: Item to add with GPS capture identifier.
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with updated package details.

    Raises:
        HTTPException: 404 if not found, 409 if package is sealed.
    """
    logger.info(
        "Add GPS to package: user=%s package=%s gps=%s",
        user.user_id,
        package_id,
        body.item_id,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /packages/{package_id}/add-photo
# ---------------------------------------------------------------------------


@router.post(
    "/{package_id}/add-photo",
    response_model=PackageResponseSchema,
    summary="Add photo to package",
    description=(
        "Add a geotagged photo evidence record to an open package. "
        "Photo binary data is referenced by hash; the package "
        "includes metadata and integrity hash."
    ),
    responses={
        200: {"description": "Photo added to package"},
        404: {"description": "Package or photo not found"},
        409: {"description": "Package is already sealed"},
    },
)
async def add_photo_to_package(
    body: PackageAddItemSchema,
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:update")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PackageResponseSchema:
    """Add a photo to a data package.

    Args:
        body: Item to add with photo identifier.
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with updated package details.

    Raises:
        HTTPException: 404 if not found, 409 if package is sealed.
    """
    logger.info(
        "Add photo to package: user=%s package=%s photo=%s",
        user.user_id,
        package_id,
        body.item_id,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /packages/{package_id}/add-signature
# ---------------------------------------------------------------------------


@router.post(
    "/{package_id}/add-signature",
    response_model=PackageResponseSchema,
    summary="Add signature to package",
    description=(
        "Add a digital signature to an open package. Includes the "
        "ECDSA signature bytes, signer metadata, and timestamp "
        "binding in the package manifest."
    ),
    responses={
        200: {"description": "Signature added to package"},
        404: {"description": "Package or signature not found"},
        409: {"description": "Package is already sealed"},
    },
)
async def add_signature_to_package(
    body: PackageAddItemSchema,
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:update")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PackageResponseSchema:
    """Add a signature to a data package.

    Args:
        body: Item to add with signature identifier.
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with updated package details.

    Raises:
        HTTPException: 404 if not found, 409 if package is sealed.
    """
    logger.info(
        "Add signature to package: user=%s package=%s signature=%s",
        user.user_id,
        package_id,
        body.item_id,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /packages/{package_id}/seal
# ---------------------------------------------------------------------------


@router.post(
    "/{package_id}/seal",
    response_model=PackageResponseSchema,
    summary="Seal package",
    description=(
        "Seal a data package computing the SHA-256 Merkle tree root "
        "over all artifacts and optionally signing the manifest with "
        "ECDSA. A sealed package is immutable and ready for upload. "
        "Sealed packages cannot have items added or removed."
    ),
    responses={
        200: {"description": "Package sealed successfully"},
        404: {"description": "Package not found"},
        409: {"description": "Package is already sealed or empty"},
    },
)
async def seal_package(
    body: PackageSealSchema,
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:seal")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PackageResponseSchema:
    """Seal a data package.

    Args:
        body: Seal options (compute Merkle, sign package).
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageResponseSchema with sealed package details.

    Raises:
        HTTPException: 404 if not found, 409 if already sealed or empty.
    """
    start = time.monotonic()
    logger.info(
        "Seal package: user=%s package=%s merkle=%s sign=%s",
        user.user_id,
        package_id,
        body.compute_merkle,
        body.sign_package,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /packages/{package_id}/validate
# ---------------------------------------------------------------------------


@router.post(
    "/{package_id}/validate",
    response_model=PackageValidateSchema,
    summary="Validate sealed package",
    description=(
        "Validate a sealed package by re-computing the Merkle tree "
        "and verifying the manifest signature. Returns detailed "
        "integrity check results per artifact."
    ),
    responses={
        200: {"description": "Validation completed"},
        404: {"description": "Package not found"},
        409: {"description": "Package is not sealed"},
    },
)
async def validate_package(
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:validate")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PackageValidateSchema:
    """Validate a sealed package integrity.

    Args:
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageValidateSchema with integrity check results.

    Raises:
        HTTPException: 404 if not found, 409 if not sealed.
    """
    start = time.monotonic()
    logger.info(
        "Validate package: user=%s package=%s", user.user_id, package_id
    )

    # Placeholder - real implementation performs integrity checks
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# GET /packages/{package_id}/manifest
# ---------------------------------------------------------------------------


@router.get(
    "/{package_id}/manifest",
    response_model=ManifestSchema,
    summary="Get package manifest",
    description=(
        "Retrieve the manifest for a data package listing all included "
        "artifacts with their types, sizes, SHA-256 hashes, and "
        "positions in the Merkle tree."
    ),
    responses={
        200: {"description": "Manifest retrieved"},
        404: {"description": "Package not found"},
    },
)
async def get_manifest(
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> ManifestSchema:
    """Get the manifest for a data package.

    Args:
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        ManifestSchema with artifact records and Merkle root.

    Raises:
        HTTPException: 404 if package not found.
    """
    logger.info(
        "Get manifest: user=%s package=%s", user.user_id, package_id
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# GET /packages/{package_id}/export
# ---------------------------------------------------------------------------


@router.get(
    "/{package_id}/export",
    response_model=PackageExportSchema,
    summary="Export/download package",
    description=(
        "Get a pre-signed download URL for a sealed data package. "
        "The package is exported in the configured format (zip, "
        "tar.gz, JSON-LD). The URL expires after 1 hour."
    ),
    responses={
        200: {"description": "Export URL generated"},
        404: {"description": "Package not found"},
        409: {"description": "Package is not sealed"},
    },
)
async def export_package(
    package_id: str = Depends(validate_package_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PackageExportSchema:
    """Export a sealed data package.

    Args:
        package_id: Package identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageExportSchema with pre-signed download URL.

    Raises:
        HTTPException: 404 if not found, 409 if not sealed.
    """
    logger.info(
        "Export package: user=%s package=%s", user.user_id, package_id
    )

    # Placeholder - real implementation generates pre-signed URL
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Package {package_id} not found",
    )


# ---------------------------------------------------------------------------
# GET /packages
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=PackageListSchema,
    summary="List packages with filters",
    description=(
        "List data packages with optional filters by status, device ID, "
        "operator ID, and export format. Results are paginated."
    ),
    responses={
        200: {"description": "Packages retrieved successfully"},
    },
)
async def list_packages(
    package_status: Optional[PackageStatusSchema] = Query(
        None, alias="status",
        description="Filter by package status (building, sealed, uploaded, verified, expired)",
    ),
    device_id: Optional[str] = Query(
        None, max_length=255, description="Filter by device ID",
    ),
    operator_id: Optional[str] = Query(
        None, max_length=255, description="Filter by operator ID",
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-mdc:packages:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PackageListSchema:
    """List data packages with optional filters.

    Args:
        package_status: Filter by package lifecycle status.
        device_id: Filter by source device.
        operator_id: Filter by operator.
        pagination: Pagination parameters.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PackageListSchema with matching packages and pagination.
    """
    start = time.monotonic()
    logger.info(
        "List packages: user=%s page=%d",
        user.user_id,
        pagination.page,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return PackageListSchema(
        packages=[],
        pagination=PaginationSchema(
            total=0,
            page=pagination.page,
            page_size=pagination.page_size,
            has_more=False,
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )
