# -*- coding: utf-8 -*-
"""
Batch & Geocoding Routes - AGENT-EUDR-007 GPS Coordinate Validator API

Endpoints for reverse geocoding, country lookup, datum transformation,
and large-scale batch job management. Groups operations that involve
external reference data lookups or long-running batch processing.

Endpoints:
    POST /geo/reverse           - Reverse geocode a single coordinate
    POST /geo/reverse/batch     - Batch reverse geocode
    POST /geo/country           - Country lookup for a coordinate
    POST /geo/datum/transform   - Single datum transformation
    POST /geo/datum/batch       - Batch datum transformation
    GET  /geo/datum/list        - List supported datums
    POST /batch/submit          - Submit large batch processing job
    GET  /batch/{job_id}        - Get batch job status
    DELETE /batch/{job_id}      - Cancel batch job

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.gps_coordinate_validator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_gps_validator_service,
    rate_limit_batch,
    rate_limit_geocode,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.schemas import (
    BatchDatumTransformRequestSchema,
    BatchDatumTransformResponseSchema,
    BatchJobCancelResponseSchema,
    BatchJobRequestSchema,
    BatchJobResponseSchema,
    BatchReverseGeocodeRequestSchema,
    BatchReverseGeocodeResponseSchema,
    CountryLookupResponseSchema,
    DatumInfoSchema,
    DatumListResponseSchema,
    DatumTransformRequestSchema,
    DatumTransformResponseSchema,
    ReverseGeocodeRequestSchema,
    ReverseGeocodeResponseSchema,
    SUPPORTED_DATUMS,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Batch & Geocoding"])

# ---------------------------------------------------------------------------
# In-memory batch job store (replaced by database in production)
# ---------------------------------------------------------------------------

_batch_job_store: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Datum metadata for GET /datums endpoint
# ---------------------------------------------------------------------------

_DATUM_METADATA: Dict[str, Dict[str, Any]] = {
    "wgs84": {"name": "World Geodetic System 1984", "epsg": 4326, "region": "Global"},
    "nad27": {"name": "North American Datum 1927", "epsg": 4267, "region": "North America"},
    "nad83": {"name": "North American Datum 1983", "epsg": 4269, "region": "North America"},
    "ed50": {"name": "European Datum 1950", "epsg": 4230, "region": "Europe"},
    "etrs89": {"name": "European Terrestrial Reference System 1989", "epsg": 4258, "region": "Europe"},
    "osgb36": {"name": "Ordnance Survey Great Britain 1936", "epsg": 4277, "region": "United Kingdom"},
    "tokyo": {"name": "Tokyo Datum", "epsg": 4301, "region": "Japan"},
    "indian_1975": {"name": "Indian Datum 1975", "epsg": 4240, "region": "South/SE Asia"},
    "pulkovo_1942": {"name": "Pulkovo 1942", "epsg": 4284, "region": "Russia/Eastern Europe"},
    "agd66": {"name": "Australian Geodetic Datum 1966", "epsg": 4202, "region": "Australia"},
    "agd84": {"name": "Australian Geodetic Datum 1984", "epsg": 4203, "region": "Australia"},
    "gda94": {"name": "Geocentric Datum of Australia 1994", "epsg": 4283, "region": "Australia"},
    "gda2020": {"name": "Geocentric Datum of Australia 2020", "epsg": 7844, "region": "Australia"},
    "sad69": {"name": "South American Datum 1969", "epsg": 4618, "region": "South America"},
    "sirgas2000": {"name": "SIRGAS 2000", "epsg": 4674, "region": "South America"},
    "hartebeesthoek94": {"name": "Hartebeesthoek 94", "epsg": 4148, "region": "South Africa"},
    "arc1960": {"name": "Arc 1960", "epsg": 4210, "region": "East Africa"},
    "cape": {"name": "Cape Datum", "epsg": 4222, "region": "South Africa"},
    "adindan": {"name": "Adindan", "epsg": 4201, "region": "North Africa"},
    "minna": {"name": "Minna", "epsg": 4263, "region": "Nigeria"},
    "camacupa": {"name": "Camacupa", "epsg": 4220, "region": "Angola"},
    "schwarzeck": {"name": "Schwarzeck", "epsg": 4293, "region": "Namibia"},
    "massawa": {"name": "Massawa", "epsg": 4262, "region": "Eritrea"},
    "merchich": {"name": "Merchich", "epsg": 4261, "region": "Morocco"},
    "egypt_1907": {"name": "Egypt 1907", "epsg": 4229, "region": "Egypt"},
    "lome": {"name": "Lome", "epsg": None, "region": "Togo/Ghana"},
    "accra": {"name": "Accra", "epsg": 4168, "region": "Ghana"},
    "jakarta": {"name": "Jakarta", "epsg": 4804, "region": "Indonesia"},
    "kalianpur": {"name": "Kalianpur", "epsg": 4243, "region": "India"},
    "kertau": {"name": "Kertau", "epsg": 4245, "region": "Malaysia"},
    "luzon_1911": {"name": "Luzon 1911", "epsg": 4253, "region": "Philippines"},
    "timbalai_1948": {"name": "Timbalai 1948", "epsg": 4298, "region": "Brunei/Malaysia"},
    "nzgd49": {"name": "New Zealand Geodetic Datum 1949", "epsg": 4272, "region": "New Zealand"},
}

# ---------------------------------------------------------------------------
# POST /geo/reverse
# ---------------------------------------------------------------------------

@router.post(
    "/geo/reverse",
    response_model=ReverseGeocodeResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Reverse geocode a coordinate",
    description=(
        "Perform reverse geocoding on a GPS coordinate to determine "
        "country, administrative region, nearest place, land use, "
        "elevation, and EUDR commodity zone. Uses reference boundary "
        "and land classification data."
    ),
    responses={
        200: {"description": "Reverse geocoding result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def reverse_geocode(
    body: ReverseGeocodeRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:geocode:read")
    ),
    _rate: None = Depends(rate_limit_geocode),
) -> ReverseGeocodeResponseSchema:
    """Reverse geocode a coordinate to location metadata.

    Args:
        body: Request with latitude and longitude.
        request: FastAPI request object.
        user: Authenticated user with geocode:read permission.

    Returns:
        ReverseGeocodeResponseSchema with location details.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Reverse geocode: user=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.latitude,
        body.longitude,
    )

    try:
        service = get_gps_validator_service()

        result = service.reverse_geocode(
            latitude=body.latitude,
            longitude=body.longitude,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"geocode|{body.latitude}|{body.longitude}|"
            f"{result.get('country_iso', '')}"
        )

        logger.info(
            "Reverse geocode completed: user=%s country=%s "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("country_iso"),
            elapsed * 1000,
        )

        return ReverseGeocodeResponseSchema(
            country_iso=result.get("country_iso"),
            country_name=result.get("country_name"),
            admin_region=result.get("admin_region"),
            nearest_place=result.get("nearest_place"),
            land_use=result.get("land_use"),
            distance_to_coast_km=result.get("distance_to_coast_km"),
            commodity_zone=result.get("commodity_zone"),
            elevation_m=result.get("elevation_m"),
            provenance_hash=provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Reverse geocode failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reverse geocoding failed due to an internal error",
        )

# ---------------------------------------------------------------------------
# POST /geo/reverse/batch
# ---------------------------------------------------------------------------

@router.post(
    "/geo/reverse/batch",
    response_model=BatchReverseGeocodeResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch reverse geocode",
    description=(
        "Reverse geocode multiple coordinates in a single batch request. "
        "Maximum 5,000 coordinates per batch."
    ),
    responses={
        200: {"description": "Batch reverse geocoding results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def reverse_geocode_batch(
    body: BatchReverseGeocodeRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:geocode:read")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchReverseGeocodeResponseSchema:
    """Batch reverse geocode multiple coordinates.

    Args:
        body: Batch request with list of coordinates.
        request: FastAPI request object.
        user: Authenticated user with geocode:read permission.

    Returns:
        BatchReverseGeocodeResponseSchema with per-coordinate results.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Batch reverse geocode: user=%s total=%d",
        user.user_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        results: List[ReverseGeocodeResponseSchema] = []
        for coord in body.coordinates:
            try:
                result = service.reverse_geocode(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                )
                provenance = _compute_provenance(
                    f"geocode|{coord.latitude}|{coord.longitude}"
                )
                results.append(ReverseGeocodeResponseSchema(
                    country_iso=result.get("country_iso"),
                    country_name=result.get("country_name"),
                    admin_region=result.get("admin_region"),
                    nearest_place=result.get("nearest_place"),
                    land_use=result.get("land_use"),
                    distance_to_coast_km=result.get("distance_to_coast_km"),
                    commodity_zone=result.get("commodity_zone"),
                    elevation_m=result.get("elevation_m"),
                    provenance_hash=provenance,
                ))
            except Exception:
                results.append(ReverseGeocodeResponseSchema(
                    country_iso=None,
                    country_name=None,
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch reverse geocode completed: user=%s total=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            total,
            elapsed * 1000,
        )

        return BatchReverseGeocodeResponseSchema(
            total=total,
            results=results,
            processing_time_ms=elapsed * 1000,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch reverse geocode failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch reverse geocoding failed",
        )

# ---------------------------------------------------------------------------
# POST /geo/country
# ---------------------------------------------------------------------------

@router.post(
    "/geo/country",
    response_model=CountryLookupResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Country lookup for coordinate",
    description=(
        "Look up the country for a GPS coordinate. Returns the ISO "
        "country code, full name, land status, and administrative "
        "region hierarchy."
    ),
    responses={
        200: {"description": "Country lookup result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def country_lookup(
    body: ReverseGeocodeRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:geocode:read")
    ),
    _rate: None = Depends(rate_limit_geocode),
) -> CountryLookupResponseSchema:
    """Look up the country for a coordinate.

    Args:
        body: Request with latitude and longitude.
        request: FastAPI request object.
        user: Authenticated user with geocode:read permission.

    Returns:
        CountryLookupResponseSchema with country details.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Country lookup: user=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.latitude,
        body.longitude,
    )

    try:
        service = get_gps_validator_service()

        result = service.lookup_country(
            latitude=body.latitude,
            longitude=body.longitude,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Country lookup completed: user=%s country=%s elapsed_ms=%.1f",
            user.user_id,
            result.get("country_iso"),
            elapsed * 1000,
        )

        return CountryLookupResponseSchema(
            latitude=body.latitude,
            longitude=body.longitude,
            country_iso=result.get("country_iso"),
            country_name=result.get("country_name"),
            is_on_land=result.get("is_on_land", True),
            admin_regions=result.get("admin_regions", []),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Country lookup failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Country lookup failed due to an internal error",
        )

# ---------------------------------------------------------------------------
# POST /geo/datum/transform
# ---------------------------------------------------------------------------

@router.post(
    "/geo/datum/transform",
    response_model=DatumTransformResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Transform coordinate between datums",
    description=(
        "Transform a coordinate from one geodetic datum to another "
        "using Helmert 7-parameter transformation. Returns the "
        "transformed coordinate with displacement in metres."
    ),
    responses={
        200: {"description": "Datum transformation result"},
        400: {"model": ErrorResponse, "description": "Invalid or unsupported datum"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def datum_transform(
    body: DatumTransformRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:datum:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DatumTransformResponseSchema:
    """Transform a coordinate between geodetic datums.

    Args:
        body: Request with coordinate and source/target datums.
        request: FastAPI request object.
        user: Authenticated user with datum:write permission.

    Returns:
        DatumTransformResponseSchema with transformed coordinate.

    Raises:
        HTTPException: 400 if datum unsupported, 500 on error.
    """
    start = time.monotonic()
    logger.info(
        "Datum transform: user=%s lat=%.6f lon=%.6f %s -> %s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.source_datum,
        body.target_datum,
    )

    try:
        service = get_gps_validator_service()

        result = service.transform_datum(
            latitude=body.latitude,
            longitude=body.longitude,
            source_datum=body.source_datum,
            target_datum=body.target_datum,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"datum|{body.latitude}|{body.longitude}|"
            f"{body.source_datum}|{body.target_datum}"
        )

        logger.info(
            "Datum transform completed: user=%s displacement_m=%.3f "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("displacement_m", 0.0),
            elapsed * 1000,
        )

        return DatumTransformResponseSchema(
            latitude=result["latitude"],
            longitude=result["longitude"],
            source_datum=body.source_datum,
            target_datum=body.target_datum,
            displacement_m=result.get("displacement_m", 0.0),
            transformation_method=result.get(
                "transformation_method", "helmert_7_param"
            ),
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Datum transform error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Datum transform failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Datum transformation failed due to an internal error",
        )

# ---------------------------------------------------------------------------
# POST /geo/datum/batch
# ---------------------------------------------------------------------------

@router.post(
    "/geo/datum/batch",
    response_model=BatchDatumTransformResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch datum transformation",
    description=(
        "Transform multiple coordinates between datums in a single "
        "request. Maximum 10,000 coordinates per batch."
    ),
    responses={
        200: {"description": "Batch datum transformation results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def datum_transform_batch(
    body: BatchDatumTransformRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:datum:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchDatumTransformResponseSchema:
    """Batch transform coordinates between datums.

    Args:
        body: Batch request with coordinates and datums.
        request: FastAPI request object.
        user: Authenticated user with datum:write permission.

    Returns:
        BatchDatumTransformResponseSchema with results.

    Raises:
        HTTPException: 400 if request invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Batch datum transform: user=%s total=%d",
        user.user_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        results: List[DatumTransformResponseSchema] = []
        for coord in body.coordinates:
            try:
                result = service.transform_datum(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    source_datum=coord.source_datum,
                    target_datum=coord.target_datum,
                )
                provenance = _compute_provenance(
                    f"datum|{coord.latitude}|{coord.longitude}|"
                    f"{coord.source_datum}|{coord.target_datum}"
                )
                results.append(DatumTransformResponseSchema(
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                    source_datum=coord.source_datum,
                    target_datum=coord.target_datum,
                    displacement_m=result.get("displacement_m", 0.0),
                    transformation_method=result.get(
                        "transformation_method", "helmert_7_param"
                    ),
                    provenance_hash=provenance,
                ))
            except (ValueError, KeyError) as exc:
                # Append original unchanged for failed transforms
                results.append(DatumTransformResponseSchema(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    source_datum=coord.source_datum,
                    target_datum=coord.target_datum,
                    displacement_m=0.0,
                    transformation_method="failed",
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch datum transform completed: user=%s total=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            total,
            elapsed * 1000,
        )

        return BatchDatumTransformResponseSchema(
            total=total,
            results=results,
            processing_time_ms=elapsed * 1000,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch datum transform failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch datum transformation failed",
        )

# ---------------------------------------------------------------------------
# GET /geo/datum/list
# ---------------------------------------------------------------------------

@router.get(
    "/geo/datum/list",
    response_model=DatumListResponseSchema,
    summary="List supported datums",
    description=(
        "Return the complete list of supported geodetic datums with "
        "names, EPSG codes, and geographic regions."
    ),
    responses={
        200: {"description": "List of supported datums"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
    },
)
async def list_datums(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:datum:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DatumListResponseSchema:
    """Return the list of all supported geodetic datums.

    Args:
        request: FastAPI request object.
        user: Authenticated user with datum:read permission.

    Returns:
        DatumListResponseSchema with all supported datums.
    """
    datums: List[DatumInfoSchema] = []
    for code in SUPPORTED_DATUMS:
        meta = _DATUM_METADATA.get(code, {})
        datums.append(DatumInfoSchema(
            code=code,
            name=meta.get("name", code),
            epsg=meta.get("epsg"),
            region=meta.get("region", ""),
            description=f"{meta.get('name', code)} geodetic datum",
        ))

    return DatumListResponseSchema(
        datums=datums,
        total=len(datums),
    )

# ---------------------------------------------------------------------------
# POST /batch/submit
# ---------------------------------------------------------------------------

@router.post(
    "/batch/submit",
    response_model=BatchJobResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit large batch processing job",
    description=(
        "Submit a large batch of coordinates for asynchronous processing. "
        "Supports up to 50,000 coordinates with configurable operations "
        "(validate, plausibility, precision, assess) and priority levels. "
        "Returns a job_id for status tracking."
    ),
    responses={
        202: {"description": "Batch job accepted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def submit_batch_job(
    body: BatchJobRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:batch:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchJobResponseSchema:
    """Submit a large batch processing job.

    Creates a job for asynchronous processing of up to 50,000
    coordinates. Use GET /batch/{job_id} to track progress.

    Args:
        body: Batch job request with coordinates and operations.
        request: FastAPI request object.
        user: Authenticated user with batch:write permission.

    Returns:
        BatchJobResponseSchema with job_id for tracking.

    Raises:
        HTTPException: 400 if request invalid, 500 on error.
    """
    start = time.monotonic()
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    total = len(body.coordinates)

    logger.info(
        "Batch job submitted: user=%s job_id=%s total=%d ops=%s "
        "priority=%s",
        user.user_id,
        job_id,
        total,
        body.operations,
        body.priority,
    )

    # Estimate completion time
    seconds_per_coord: Dict[str, float] = {
        "validate": 0.1,
        "plausibility": 0.5,
        "precision": 0.05,
        "assess": 1.0,
    }
    estimated = total * sum(
        seconds_per_coord.get(op, 0.5) for op in body.operations
    )

    now = utcnow()
    _batch_job_store[job_id] = {
        "job_id": job_id,
        "user_id": user.user_id,
        "status": "accepted",
        "total_coordinates": total,
        "completed_coordinates": 0,
        "operations": body.operations,
        "priority": body.priority,
        "submitted_at": now.isoformat(),
        "started_at": None,
        "completed_at": None,
    }

    # Submit to pipeline for async processing
    try:
        service = get_gps_validator_service()
        service.submit_batch_job(
            job_id=job_id,
            coordinates=[c.model_dump() for c in body.coordinates],
            operations=body.operations,
            priority=body.priority,
        )
    except (NotImplementedError, AttributeError):
        # Service not fully implemented yet, job stored for polling
        logger.info("Batch job stored for deferred processing: %s", job_id)
    except Exception as exc:
        logger.error(
            "Batch job submission error: job_id=%s error=%s",
            job_id,
            exc,
            exc_info=True,
        )

    elapsed = time.monotonic() - start
    logger.info(
        "Batch job accepted: job_id=%s estimated_s=%.0f elapsed_ms=%.1f",
        job_id,
        estimated,
        elapsed * 1000,
    )

    return BatchJobResponseSchema(
        job_id=job_id,
        status="accepted",
        total_coordinates=total,
        operations=body.operations,
        priority=body.priority,
        submitted_at=now,
        estimated_completion_seconds=estimated,
    )

# ---------------------------------------------------------------------------
# GET /batch/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/batch/{job_id}",
    response_model=BatchJobResponseSchema,
    summary="Get batch job status",
    description=(
        "Retrieve the current status of a batch processing job "
        "including progress percentage and completion timestamp."
    ),
    responses={
        200: {"description": "Batch job status"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_batch_job_status(
    job_id: str = Path(..., description="Batch job identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:batch:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobResponseSchema:
    """Get the status of a batch processing job.

    Args:
        job_id: Batch job identifier.
        request: FastAPI request object.
        user: Authenticated user with batch:read permission.

    Returns:
        BatchJobResponseSchema with current status.

    Raises:
        HTTPException: 404 if job not found, 403 if unauthorized.
    """
    logger.info(
        "Batch job status: user=%s job_id=%s",
        user.user_id,
        job_id,
    )

    job = _batch_job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )

    # Authorization check
    if job.get("user_id") != user.user_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this batch job",
        )

    total = job.get("total_coordinates", 0)
    completed = job.get("completed_coordinates", 0)
    progress = (completed / total * 100.0) if total > 0 else 0.0

    return BatchJobResponseSchema(
        job_id=job_id,
        status=job.get("status", "unknown"),
        total_coordinates=total,
        operations=job.get("operations", []),
        priority=job.get("priority", "normal"),
        submitted_at=job.get("submitted_at"),
        progress_percent=round(progress, 2),
        completed_at=job.get("completed_at"),
    )

# ---------------------------------------------------------------------------
# DELETE /batch/{job_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/batch/{job_id}",
    response_model=BatchJobCancelResponseSchema,
    summary="Cancel batch job",
    description=(
        "Cancel a running or pending batch processing job. Coordinates "
        "already processed retain their results."
    ),
    responses={
        200: {"description": "Batch job cancelled"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job already completed/cancelled"},
    },
)
async def cancel_batch_job(
    job_id: str = Path(..., description="Batch job identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:batch:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobCancelResponseSchema:
    """Cancel a batch processing job.

    Args:
        job_id: Batch job identifier.
        request: FastAPI request object.
        user: Authenticated user with batch:write permission.

    Returns:
        BatchJobCancelResponseSchema confirming cancellation.

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized,
            409 if already complete/cancelled.
    """
    logger.info(
        "Batch job cancel: user=%s job_id=%s",
        user.user_id,
        job_id,
    )

    job = _batch_job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )

    # Authorization check
    if job.get("user_id") != user.user_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this batch job",
        )

    current_status = job.get("status", "")
    if current_status in ("completed", "cancelled", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Batch job {job_id} is already {current_status}",
        )

    now = utcnow()
    job["status"] = "cancelled"
    job["completed_at"] = now.isoformat()

    completed = job.get("completed_coordinates", 0)
    total = job.get("total_coordinates", 0)

    logger.info(
        "Batch job cancelled: job_id=%s completed_before_cancel=%d/%d",
        job_id,
        completed,
        total,
    )

    return BatchJobCancelResponseSchema(
        job_id=job_id,
        status="cancelled",
        completed_coordinates=completed,
        total_coordinates=total,
        cancelled_at=now,
    )
