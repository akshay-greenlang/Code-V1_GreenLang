# -*- coding: utf-8 -*-
"""
GIS Connector Service REST API Router - AGENT-DATA-006 (GL-DATA-GEO-001)

FastAPI router providing 20 REST API endpoints for GIS connector
operations including format parsing, CRS transformation, spatial
analysis, land cover classification, boundary resolution, geocoding,
and layer management.

Endpoints:
    1.  POST /v1/gis/parse              - Parse geospatial data
    2.  POST /v1/gis/parse/batch        - Batch parse multiple datasets
    3.  GET  /v1/gis/parse/{parse_id}   - Get parse result by ID
    4.  POST /v1/gis/transform          - Transform CRS coordinates
    5.  POST /v1/gis/transform/geometry - Transform full geometry CRS
    6.  GET  /v1/gis/crs                - List available CRS definitions
    7.  GET  /v1/gis/crs/{epsg_code}    - Get CRS info
    8.  POST /v1/gis/spatial/distance   - Calculate distance
    9.  POST /v1/gis/spatial/area       - Calculate polygon area
    10. POST /v1/gis/spatial/contains   - Test containment
    11. POST /v1/gis/land-cover/classify - Classify land cover
    12. POST /v1/gis/land-cover/carbon  - Estimate carbon stock
    13. POST /v1/gis/boundary/country   - Resolve country
    14. POST /v1/gis/boundary/climate   - Resolve climate zone
    15. POST /v1/gis/geocode/forward    - Forward geocode
    16. POST /v1/gis/geocode/reverse    - Reverse geocode
    17. POST /v1/gis/layers             - Create layer
    18. GET  /v1/gis/layers             - List layers
    19. GET  /v1/gis/health             - Health check
    20. GET  /v1/gis/statistics         - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/gis", tags=["GIS Connector"])


# =============================================================================
# Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    agent_id: str = "GL-DATA-GEO-001"
    agent_name: str = "GIS/Mapping Connector"
    version: str = "1.0.0"
    timestamp: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""


class ParseResultResponse(BaseModel):
    """Parse result response."""
    parse_id: str
    source_format: str = ""
    geometry_type: str = ""
    feature_count: int = 0
    is_valid: bool = True
    errors: List[str] = []


class TransformResultResponse(BaseModel):
    """Transform result response."""
    transform_id: str
    source_crs: str = ""
    target_crs: str = ""
    output_coordinates: Any = None
    method: str = ""


class SpatialResultResponse(BaseModel):
    """Spatial analysis result response."""
    result_id: str
    operation: str = ""
    output_data: Dict[str, Any] = {}
    unit: str = ""


class GeocodingResultResponse(BaseModel):
    """Geocoding result response."""
    result_id: str
    direction: str = ""
    results: List[Dict[str, Any]] = []
    total_results: int = 0


class LayerResponse(BaseModel):
    """Layer response."""
    layer_id: str
    name: str = ""
    geometry_type: str = ""
    crs: str = "EPSG:4326"
    feature_count: int = 0
    status: str = "active"


# =============================================================================
# Service Dependency
# =============================================================================


def _get_service():
    """Get or create the GIS Connector Service singleton.

    Returns:
        GISConnectorService instance.
    """
    from greenlang.gis_connector.setup import GISConnectorService
    if not hasattr(_get_service, "_instance"):
        _get_service._instance = GISConnectorService()
    return _get_service._instance


# =============================================================================
# 1. POST /v1/gis/parse - Parse geospatial data
# =============================================================================


@router.post("/parse", tags=["Parsing"])
async def parse_data(request: Dict[str, Any]):
    """Parse geospatial data from GeoJSON, WKT, CSV, or KML format.

    Auto-detects format if not specified.

    Args:
        request: Dictionary with 'data', optional 'format', 'source_crs'.

    Returns:
        ParseResult dictionary.
    """
    try:
        service = _get_service()
        data = request.get("data", "")
        fmt = request.get("format")
        source_crs = request.get("source_crs")
        result = service.parse_data(data, format=fmt, source_crs=source_crs)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error parsing data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 2. POST /v1/gis/parse/batch - Batch parse multiple datasets
# =============================================================================


@router.post("/parse/batch", tags=["Parsing"])
async def parse_batch(request: Dict[str, Any]):
    """Batch parse multiple geospatial datasets.

    Args:
        request: Dictionary with 'items' list of parse requests.

    Returns:
        List of ParseResult dictionaries.
    """
    try:
        service = _get_service()
        items = request.get("items", [])
        if not items:
            raise ValueError("No items provided in batch request")
        results = []
        for item in items:
            data = item.get("data", "")
            fmt = item.get("format")
            source_crs = item.get("source_crs")
            result = service.parse_data(data, format=fmt, source_crs=source_crs)
            results.append(result)
        return {"results": results, "total": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error in batch parse: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 3. GET /v1/gis/parse/{parse_id} - Get parse result by ID
# =============================================================================


@router.get("/parse/{parse_id}", tags=["Parsing"])
async def get_parse_result(parse_id: str):
    """Get a previously parsed result by ID.

    Args:
        parse_id: Parse result identifier.

    Returns:
        ParseResult dictionary.
    """
    service = _get_service()
    result = service.format_parser.get_result(parse_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Parse result not found: {parse_id}"
        )
    return result


# =============================================================================
# 4. POST /v1/gis/transform - Transform CRS coordinates
# =============================================================================


@router.post("/transform", tags=["CRS"])
async def transform_coordinates(request: Dict[str, Any]):
    """Transform coordinates between coordinate reference systems.

    Args:
        request: Dictionary with 'coordinates', 'source_crs', 'target_crs'.

    Returns:
        TransformResult dictionary.
    """
    try:
        service = _get_service()
        coordinates = request.get("coordinates", [])
        source_crs = request.get("source_crs", "EPSG:4326")
        target_crs = request.get("target_crs", "EPSG:3857")
        result = service.transform_coordinates(coordinates, source_crs, target_crs)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error transforming coordinates: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 5. POST /v1/gis/transform/geometry - Transform full geometry CRS
# =============================================================================


@router.post("/transform/geometry", tags=["CRS"])
async def transform_geometry(request: Dict[str, Any]):
    """Transform all coordinates in a geometry to a new CRS.

    Args:
        request: Dictionary with 'geometry', 'source_crs', 'target_crs'.

    Returns:
        Transformed geometry dictionary.
    """
    try:
        service = _get_service()
        geometry = request.get("geometry", {})
        source_crs = request.get("source_crs", "EPSG:4326")
        target_crs = request.get("target_crs", "EPSG:3857")
        result = service.transform_geometry(geometry, source_crs, target_crs)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error transforming geometry: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 6. GET /v1/gis/crs - List available CRS definitions
# =============================================================================


@router.get("/crs", tags=["CRS"])
async def list_crs(
    filter: Optional[str] = Query(
        None, description="Filter by type: geographic or projected"
    ),
):
    """List all available CRS definitions.

    Args:
        filter: Optional type filter (geographic, projected).

    Returns:
        List of CRS info dictionaries.
    """
    try:
        service = _get_service()
        results = service.crs_transformer.list_crs(filter=filter)
        return {"crs_definitions": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 7. GET /v1/gis/crs/{epsg_code} - Get CRS info
# =============================================================================


@router.get("/crs/{epsg_code}", tags=["CRS"])
async def get_crs_info(epsg_code: str):
    """Get CRS metadata for an EPSG code.

    Args:
        epsg_code: EPSG identifier (e.g., "EPSG:4326" or "4326").

    Returns:
        CRS info dictionary.
    """
    service = _get_service()
    info = service.crs_transformer.get_crs_info(epsg_code)
    if info is None:
        raise HTTPException(
            status_code=404, detail=f"CRS not found: {epsg_code}"
        )
    return info


# =============================================================================
# 8. POST /v1/gis/spatial/distance - Calculate distance
# =============================================================================


@router.post("/spatial/distance", tags=["Spatial"])
async def calculate_distance(request: Dict[str, Any]):
    """Calculate distance between two points.

    Args:
        request: Dictionary with 'point_a', 'point_b', optional 'method'.

    Returns:
        SpatialResult with distance.
    """
    try:
        service = _get_service()
        point_a = request.get("point_a", [])
        point_b = request.get("point_b", [])
        method = request.get("method", "haversine")
        result = service.calculate_distance(point_a, point_b, method)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error calculating distance: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 9. POST /v1/gis/spatial/area - Calculate polygon area
# =============================================================================


@router.post("/spatial/area", tags=["Spatial"])
async def calculate_area(request: Dict[str, Any]):
    """Calculate the geodesic area of a polygon.

    Args:
        request: Dictionary with 'geometry' (Polygon type).

    Returns:
        SpatialResult with area.
    """
    try:
        service = _get_service()
        geometry = request.get("geometry", {})
        result = service.calculate_area(geometry)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error calculating area: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 10. POST /v1/gis/spatial/contains - Test containment
# =============================================================================


@router.post("/spatial/contains", tags=["Spatial"])
async def test_contains(request: Dict[str, Any]):
    """Test if geometry A contains geometry B.

    Args:
        request: Dictionary with 'geom_a' and 'geom_b'.

    Returns:
        SpatialResult with contains boolean.
    """
    try:
        service = _get_service()
        geom_a = request.get("geom_a", {})
        geom_b = request.get("geom_b", {})
        result = service.spatial_analyzer.contains(geom_a, geom_b)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error testing containment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 11. POST /v1/gis/land-cover/classify - Classify land cover
# =============================================================================


@router.post("/land-cover/classify", tags=["Land Cover"])
async def classify_land_cover(request: Dict[str, Any]):
    """Classify land cover at a geographic coordinate.

    Args:
        request: Dictionary with 'coordinate' [lon, lat], optional 'corine_code'.

    Returns:
        LandCoverClassification dictionary.
    """
    try:
        service = _get_service()
        coordinate = request.get("coordinate", [])
        corine_code = request.get("corine_code")
        result = service.classify_land_cover(coordinate, corine_code)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error classifying land cover: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 12. POST /v1/gis/land-cover/carbon - Estimate carbon stock
# =============================================================================


@router.post("/land-cover/carbon", tags=["Land Cover"])
async def estimate_carbon_stock(request: Dict[str, Any]):
    """Estimate carbon stock for a land cover type.

    Args:
        request: Dictionary with 'land_cover_type'.

    Returns:
        Carbon stock estimate dictionary.
    """
    try:
        service = _get_service()
        land_cover_type = request.get("land_cover_type", "unknown")
        result = service.estimate_carbon_stock(land_cover_type)
        return {"land_cover_type": land_cover_type, "carbon_stock": result}
    except Exception as e:
        logger.error("Error estimating carbon stock: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 13. POST /v1/gis/boundary/country - Resolve country
# =============================================================================


@router.post("/boundary/country", tags=["Boundaries"])
async def resolve_country(request: Dict[str, Any]):
    """Resolve country from geographic coordinates.

    Args:
        request: Dictionary with 'coordinate' [lon, lat].

    Returns:
        BoundaryResult with country information.
    """
    try:
        service = _get_service()
        coordinate = request.get("coordinate", [])
        result = service.resolve_country(coordinate)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error resolving country: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 14. POST /v1/gis/boundary/climate - Resolve climate zone
# =============================================================================


@router.post("/boundary/climate", tags=["Boundaries"])
async def resolve_climate_zone(request: Dict[str, Any]):
    """Resolve Koppen-Geiger climate zone for coordinates.

    Args:
        request: Dictionary with 'coordinate' [lon, lat].

    Returns:
        BoundaryResult with climate zone information.
    """
    try:
        service = _get_service()
        coordinate = request.get("coordinate", [])
        result = service.resolve_climate_zone(coordinate)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error resolving climate zone: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 15. POST /v1/gis/geocode/forward - Forward geocode
# =============================================================================


@router.post("/geocode/forward", tags=["Geocoding"])
async def forward_geocode(request: Dict[str, Any]):
    """Forward geocode: address to coordinates.

    Args:
        request: Dictionary with 'address', optional 'country_hint', 'limit'.

    Returns:
        GeocodingResult dictionary.
    """
    try:
        service = _get_service()
        address = request.get("address", "")
        country_hint = request.get("country_hint")
        limit = request.get("limit", 5)
        result = service.forward_geocode(address, country_hint, limit)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error forward geocoding: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 16. POST /v1/gis/geocode/reverse - Reverse geocode
# =============================================================================


@router.post("/geocode/reverse", tags=["Geocoding"])
async def reverse_geocode(request: Dict[str, Any]):
    """Reverse geocode: coordinates to address.

    Args:
        request: Dictionary with 'coordinate' [lon, lat], optional 'limit'.

    Returns:
        GeocodingResult dictionary.
    """
    try:
        service = _get_service()
        coordinate = request.get("coordinate", [])
        limit = request.get("limit", 1)
        result = service.reverse_geocode(coordinate, limit)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error reverse geocoding: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 17. POST /v1/gis/layers - Create layer
# =============================================================================


@router.post("/layers", tags=["Layers"])
async def create_layer(request: Dict[str, Any]):
    """Create a new geospatial layer.

    Args:
        request: Dictionary with 'name', 'geometry_type', optional 'crs',
                 'features', 'description', 'tags'.

    Returns:
        GeoLayer dictionary.
    """
    try:
        service = _get_service()
        name = request.get("name", "")
        geometry_type = request.get("geometry_type", "Point")
        crs = request.get("crs", "EPSG:4326")
        features = request.get("features")
        description = request.get("description", "")
        tags = request.get("tags")
        result = service.layer_manager.create_layer(
            name=name,
            geometry_type=geometry_type,
            crs=crs,
            features=features,
            description=description,
            tags=tags,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error creating layer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 18. GET /v1/gis/layers - List layers
# =============================================================================


@router.get("/layers", tags=["Layers"])
async def list_layers(
    status: Optional[str] = Query(None, description="Filter by status"),
    geometry_type: Optional[str] = Query(
        None, description="Filter by geometry type"
    ),
):
    """List all geospatial layers with optional filters.

    Args:
        status: Optional status filter (active, inactive).
        geometry_type: Optional geometry type filter.

    Returns:
        List of GeoLayer dictionaries.
    """
    try:
        service = _get_service()
        results = service.layer_manager.list_layers(
            status=status,
            geometry_type=geometry_type,
        )
        return {"layers": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 19. GET /v1/gis/health - Health check
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check endpoint.

    Returns:
        Health status with agent metadata.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
    )


# =============================================================================
# 20. GET /v1/gis/statistics - Service statistics
# =============================================================================


@router.get("/statistics", tags=["Statistics"])
async def get_statistics():
    """Get comprehensive service statistics.

    Returns:
        Statistics from all 7 engines.
    """
    try:
        service = _get_service()
        return service.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Utility
# =============================================================================


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model or dataclass to dictionary.

    Args:
        obj: Object to convert.

    Returns:
        Dictionary representation.
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
