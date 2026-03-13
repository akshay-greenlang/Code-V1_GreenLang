# -*- coding: utf-8 -*-
"""
Satellite Detection Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for multi-source satellite change detection covering Sentinel-2,
Landsat 8/9, GLAD alerts, Hansen Global Forest Change, and RADD radar alerts
for deforestation monitoring near EUDR supply chain plots.

Endpoints:
    POST /satellite/detect                  - Trigger change detection for area
    POST /satellite/scan                    - Scan specific area with specific source
    GET  /satellite/sources                 - List available satellite sources
    GET  /satellite/{detection_id}/imagery  - Get imagery metadata for detection

Spectral Indices: NDVI, EVI, NBR, NDMI, SAVI
Sources: Sentinel-2 (10m/5d), Landsat (30m/8d), GLAD (weekly), Hansen GFC (annual), RADD (SAR)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, SatelliteChangeDetector Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    get_das_config,
    get_satellite_detector,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    DetectionEntry,
    ErrorResponse,
    ImageryMetadata,
    MetadataSchema,
    ProvenanceInfo,
    SatelliteDetectionRequest,
    SatelliteDetectionResponse,
    SatelliteImageryResponse,
    SatelliteScanRequest,
    SatelliteScanResponse,
    SatelliteSourceEnum,
    SatelliteSourceInfo,
    SatelliteSourcesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/satellite", tags=["Satellite Detection"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /satellite/detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=SatelliteDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger satellite change detection for area",
    description=(
        "Trigger multi-source satellite change detection for a geographic area. "
        "Analyzes spectral indices (NDVI, EVI, NBR, NDMI, SAVI) across enabled "
        "satellite sources to detect deforestation, degradation, and disturbance "
        "events. Returns a list of detected changes with confidence scores."
    ),
    responses={
        200: {"description": "Change detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_changes(
    request: Request,
    body: SatelliteDetectionRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:satellite:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> SatelliteDetectionResponse:
    """Trigger satellite change detection for a geographic area.

    Args:
        body: Detection request with coordinates, radius, and parameters.
        user: Authenticated user with satellite:create permission.

    Returns:
        SatelliteDetectionResponse with detected changes.
    """
    start = time.monotonic()

    try:
        engine = get_satellite_detector()
        result = engine.detect_changes(
            latitude=float(body.center.latitude),
            longitude=float(body.center.longitude),
            radius_km=float(body.radius_km),
            sources=[s.value for s in body.sources] if body.sources else None,
            spectral_indices=[si.value for si in body.spectral_indices] if body.spectral_indices else None,
            start_date=body.start_date,
            end_date=body.end_date,
            min_confidence=float(body.min_confidence) if body.min_confidence else None,
            max_cloud_cover_pct=body.max_cloud_cover_pct,
            plot_ids=body.plot_ids,
        )

        detections = []
        for det in result.get("detections", []):
            detections.append(
                DetectionEntry(
                    detection_id=det.get("detection_id", ""),
                    source=SatelliteSourceEnum(det.get("source", "sentinel2")),
                    latitude=Decimal(str(det.get("latitude", 0))),
                    longitude=Decimal(str(det.get("longitude", 0))),
                    area_ha=Decimal(str(det.get("area_ha", 0))),
                    change_type=det.get("change_type", "deforestation"),
                    confidence=Decimal(str(det.get("confidence", 0))),
                    detection_date=det.get("detection_date"),
                    spectral_changes=det.get("spectral_changes"),
                    cloud_cover_pct=det.get("cloud_cover_pct"),
                    resolution_m=det.get("resolution_m"),
                )
            )

        has_deforestation = any(
            d.change_type.value == "deforestation" for d in detections
        ) if detections else False

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"detect:{body.center.latitude},{body.center.longitude}:{body.radius_km}",
            str(len(detections)),
        )

        logger.info(
            "Satellite detection completed: lat=%s lon=%s radius_km=%s detections=%d operator=%s",
            body.center.latitude,
            body.center.longitude,
            body.radius_km,
            len(detections),
            user.operator_id or user.user_id,
        )

        return SatelliteDetectionResponse(
            detection_id=result.get("detection_id", ""),
            detections=detections,
            total_detections=len(detections),
            sources_queried=[
                SatelliteSourceEnum(s) for s in result.get("sources_queried", [])
            ],
            area_scanned_km2=Decimal(str(result.get("area_scanned_km2", 0)))
            if result.get("area_scanned_km2") else None,
            deforestation_detected=has_deforestation,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=result.get("data_sources", ["Sentinel-2", "Landsat"]),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Satellite change detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Satellite change detection failed",
        )


# ---------------------------------------------------------------------------
# POST /satellite/scan
# ---------------------------------------------------------------------------


@router.post(
    "/scan",
    response_model=SatelliteScanResponse,
    status_code=status.HTTP_200_OK,
    summary="Scan specific area with specific satellite source",
    description=(
        "Perform a targeted scan of a specific area using a single satellite "
        "source. Returns detailed detection results with spectral analysis "
        "for the specified source and date range."
    ),
    responses={
        200: {"description": "Scan completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def scan_area(
    request: Request,
    body: SatelliteScanRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:satellite:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> SatelliteScanResponse:
    """Perform a targeted satellite scan of a specific area.

    Args:
        body: Scan request with source, area, and date range.
        user: Authenticated user with satellite:create permission.

    Returns:
        SatelliteScanResponse with scan results.
    """
    start = time.monotonic()

    try:
        engine = get_satellite_detector()

        scan_params: Dict[str, Any] = {
            "source": body.source.value,
            "date_range_start": body.date_range_start,
            "date_range_end": body.date_range_end,
            "spectral_indices": [si.value for si in body.spectral_indices],
            "max_cloud_cover_pct": body.max_cloud_cover_pct,
        }

        if body.polygon:
            scan_params["polygon"] = [
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.polygon.coordinates
            ]
        elif body.center:
            scan_params["latitude"] = float(body.center.latitude)
            scan_params["longitude"] = float(body.center.longitude)
            scan_params["radius_km"] = float(body.radius_km) if body.radius_km else 10.0

        result = engine.scan_area(**scan_params)

        detections = []
        for det in result.get("detections", []):
            detections.append(
                DetectionEntry(
                    detection_id=det.get("detection_id", ""),
                    source=body.source,
                    latitude=Decimal(str(det.get("latitude", 0))),
                    longitude=Decimal(str(det.get("longitude", 0))),
                    area_ha=Decimal(str(det.get("area_ha", 0))),
                    change_type=det.get("change_type", "deforestation"),
                    confidence=Decimal(str(det.get("confidence", 0))),
                    detection_date=det.get("detection_date"),
                    spectral_changes=det.get("spectral_changes"),
                    cloud_cover_pct=det.get("cloud_cover_pct"),
                    resolution_m=det.get("resolution_m"),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"scan:{body.source.value}:{body.date_range_start}-{body.date_range_end}",
            str(len(detections)),
        )

        logger.info(
            "Satellite scan completed: source=%s scenes=%d detections=%d operator=%s",
            body.source.value,
            result.get("scenes_analyzed", 0),
            len(detections),
            user.operator_id or user.user_id,
        )

        return SatelliteScanResponse(
            scan_id=result.get("scan_id", ""),
            source=body.source,
            scenes_analyzed=result.get("scenes_analyzed", 0),
            detections=detections,
            total_detections=len(detections),
            area_scanned_km2=Decimal(str(result.get("area_scanned_km2", 0))),
            cloud_cover_avg_pct=Decimal(str(result.get("cloud_cover_avg_pct", 0)))
            if result.get("cloud_cover_avg_pct") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[body.source.value],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Satellite scan failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Satellite scan failed",
        )


# ---------------------------------------------------------------------------
# GET /satellite/sources
# ---------------------------------------------------------------------------


@router.get(
    "/sources",
    response_model=SatelliteSourcesResponse,
    summary="List available satellite data sources",
    description=(
        "List all configured satellite data sources with resolution, revisit "
        "period, enabled status, and coverage information."
    ),
    responses={
        200: {"description": "Sources listed successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_sources(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:satellite:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SatelliteSourcesResponse:
    """List available satellite data sources.

    Args:
        user: Authenticated user with satellite:read permission.

    Returns:
        SatelliteSourcesResponse with source information.
    """
    start = time.monotonic()

    try:
        config = get_das_config()

        sources = [
            SatelliteSourceInfo(
                source=SatelliteSourceEnum.SENTINEL2,
                name="Sentinel-2 (ESA Copernicus)",
                resolution_m=config.sentinel2_resolution_m,
                revisit_days=config.sentinel2_revisit_days,
                enabled=config.sentinel2_enabled,
                coverage="global",
                data_type="optical",
            ),
            SatelliteSourceInfo(
                source=SatelliteSourceEnum.LANDSAT,
                name="Landsat 8/9 (USGS/NASA)",
                resolution_m=config.landsat_resolution_m,
                revisit_days=config.landsat_revisit_days,
                enabled=config.landsat_enabled,
                coverage="global",
                data_type="optical",
            ),
            SatelliteSourceInfo(
                source=SatelliteSourceEnum.GLAD,
                name="GLAD Alerts (University of Maryland)",
                resolution_m=30,
                revisit_days=7,
                enabled=config.glad_enabled,
                coverage="tropical",
                data_type="derived_alert",
            ),
            SatelliteSourceInfo(
                source=SatelliteSourceEnum.HANSEN_GFC,
                name="Hansen Global Forest Change",
                resolution_m=30,
                revisit_days=365,
                enabled=config.hansen_gfc_enabled,
                coverage="global",
                data_type="annual_composite",
            ),
            SatelliteSourceInfo(
                source=SatelliteSourceEnum.RADD,
                name="RADD (Sentinel-1 SAR)",
                resolution_m=10,
                revisit_days=6,
                enabled=config.radd_enabled,
                coverage="tropical",
                data_type="radar",
            ),
        ]

        enabled_count = sum(1 for s in sources if s.enabled)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "satellite_sources",
            str(len(sources)),
        )

        logger.info(
            "Satellite sources listed: total=%d enabled=%d operator=%s",
            len(sources),
            enabled_count,
            user.operator_id or user.user_id,
        )

        return SatelliteSourcesResponse(
            sources=sources,
            total_sources=len(sources),
            enabled_count=enabled_count,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DeforestationAlertSystemConfig"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Satellite source listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Satellite source listing failed",
        )


# ---------------------------------------------------------------------------
# GET /satellite/{detection_id}/imagery
# ---------------------------------------------------------------------------


@router.get(
    "/{detection_id}/imagery",
    response_model=SatelliteImageryResponse,
    summary="Get imagery metadata for a detection",
    description=(
        "Retrieve satellite imagery metadata associated with a specific "
        "detection including scene IDs, acquisition dates, cloud cover, "
        "and available spectral bands."
    ),
    responses={
        200: {"description": "Imagery metadata retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Detection not found"},
    },
)
async def get_detection_imagery(
    detection_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:satellite:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SatelliteImageryResponse:
    """Get imagery metadata for a satellite detection.

    Args:
        detection_id: Unique detection identifier.
        user: Authenticated user with satellite:read permission.

    Returns:
        SatelliteImageryResponse with imagery metadata.
    """
    start = time.monotonic()

    try:
        engine = get_satellite_detector()
        result = engine.get_imagery_metadata(detection_id=detection_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection not found: {detection_id}",
            )

        imagery = []
        for scene in result.get("imagery", []):
            imagery.append(
                ImageryMetadata(
                    scene_id=scene.get("scene_id", ""),
                    source=SatelliteSourceEnum(scene.get("source", "sentinel2")),
                    acquisition_date=scene.get("acquisition_date"),
                    cloud_cover_pct=Decimal(str(scene.get("cloud_cover_pct", 0))),
                    resolution_m=scene.get("resolution_m", 10),
                    bands=scene.get("bands"),
                    thumbnail_url=scene.get("thumbnail_url"),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"imagery:{detection_id}",
            str(len(imagery)),
        )

        logger.info(
            "Detection imagery retrieved: detection_id=%s scenes=%d operator=%s",
            detection_id,
            len(imagery),
            user.operator_id or user.user_id,
        )

        return SatelliteImageryResponse(
            detection_id=detection_id,
            imagery=imagery,
            total_scenes=len(imagery),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=list({s.source.value for s in imagery}),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Imagery retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Imagery metadata retrieval failed",
        )
