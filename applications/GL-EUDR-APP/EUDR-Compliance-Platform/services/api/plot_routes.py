"""
Plot Management API Routes for GL-EUDR-APP v1.0

Manages geospatial plot data for EUDR compliance. Each plot represents
a production area linked to a supplier and a regulated commodity.
Supports GeoJSON polygon geometry, coordinate validation, area
calculation, and overlap detection.

Prefix: /api/v1/plots
Tags: Plots
"""

import uuid
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/plots", tags=["Plots"])

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class GeoJSONPolygon(BaseModel):
    """GeoJSON Polygon geometry.

    Coordinates are a list of linear rings. The first ring is the
    exterior boundary; subsequent rings are holes. Each ring is a list
    of [longitude, latitude] pairs.

    Example::

        {
            "type": "Polygon",
            "coordinates": [[
                [-60.0, -3.0], [-60.0, -2.0], [-59.0, -2.0],
                [-59.0, -3.0], [-60.0, -3.0]
            ]]
        }
    """

    type: str = Field("Polygon", description="GeoJSON geometry type")
    coordinates: List[List[List[float]]] = Field(
        ..., description="List of linear rings; each ring is [[lon, lat], ...]"
    )

    @field_validator("type")
    @classmethod
    def must_be_polygon(cls, v: str) -> str:
        if v != "Polygon":
            raise ValueError("Geometry type must be 'Polygon'")
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, rings: List[List[List[float]]]) -> List[List[List[float]]]:
        if not rings or not rings[0]:
            raise ValueError("At least one linear ring with coordinates required")
        for ring in rings:
            if len(ring) < 4:
                raise ValueError("A polygon ring must have at least 4 coordinate pairs (closed)")
            if ring[0] != ring[-1]:
                raise ValueError("Polygon ring must be closed (first == last coordinate)")
            for point in ring:
                if len(point) < 2:
                    raise ValueError("Each coordinate must have [longitude, latitude]")
                lon, lat = point[0], point[1]
                if not (-180.0 <= lon <= 180.0):
                    raise ValueError(f"Longitude {lon} out of WGS84 range [-180, 180]")
                if not (-90.0 <= lat <= 90.0):
                    raise ValueError(f"Latitude {lat} out of WGS84 range [-90, 90]")
        return rings


class PlotCreateRequest(BaseModel):
    """Request body for creating a production plot.

    Example::

        {
            "name": "Plot Alpha",
            "supplier_id": "sup_abc123",
            "coordinates": {
                "type": "Polygon",
                "coordinates": [[
                    [-60.0, -3.0], [-60.0, -2.0], [-59.0, -2.0],
                    [-59.0, -3.0], [-60.0, -3.0]
                ]]
            },
            "commodity": "soya",
            "country_iso3": "BRA"
        }
    """

    name: str = Field(..., min_length=1, max_length=255, description="Human-readable plot name")
    supplier_id: str = Field(..., description="ID of the owning supplier")
    coordinates: GeoJSONPolygon = Field(..., description="Plot boundary as GeoJSON Polygon")
    commodity: str = Field(..., description="EUDR-regulated commodity produced on this plot")
    country_iso3: str = Field(
        ..., min_length=3, max_length=3, description="ISO 3166-1 alpha-3 country code"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        allowed = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid commodity '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class PlotUpdateRequest(BaseModel):
    """Request body for updating a plot."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    coordinates: Optional[GeoJSONPolygon] = None
    commodity: Optional[str] = None
    country_iso3: Optional[str] = Field(None, min_length=3, max_length=3)

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid commodity '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class PlotResponse(BaseModel):
    """Response model for a single plot."""

    plot_id: str = Field(..., description="Unique plot identifier")
    name: str
    supplier_id: str
    coordinates: GeoJSONPolygon
    commodity: str
    country_iso3: str
    area_hectares: Optional[float] = Field(None, description="Calculated area in hectares")
    risk_level: str = Field("unknown", description="low | medium | high | critical | unknown")
    validation_status: str = Field(
        "pending", description="pending | valid | invalid"
    )
    created_at: datetime
    updated_at: datetime


class PlotListResponse(BaseModel):
    """Paginated list of plots."""

    items: List[PlotResponse]
    page: int
    limit: int
    total: int
    total_pages: int


class PlotValidationResult(BaseModel):
    """Result of plot coordinate validation."""

    plot_id: str
    is_valid: bool
    area_hectares: float = Field(..., description="Calculated area in hectares")
    polygon_closed: bool
    coordinates_in_bounds: bool
    country_match: bool = Field(
        ..., description="Whether centroid falls within declared country (simulated)"
    )
    issues: List[str] = Field(default_factory=list)


class OverlapCheckResult(BaseModel):
    """Result of overlap detection for a plot."""

    plot_id: str
    overlapping_plots: List[str] = Field(
        default_factory=list, description="IDs of plots that overlap"
    )
    overlap_count: int
    message: str


class GeoJSONFeature(BaseModel):
    """A GeoJSON Feature for bulk import."""

    type: str = "Feature"
    geometry: GeoJSONPolygon
    properties: Dict = Field(
        default_factory=dict,
        description="Properties: name, supplier_id, commodity, country_iso3",
    )


class GeoJSONFeatureCollection(BaseModel):
    """GeoJSON FeatureCollection for bulk plot import.

    Example::

        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [...]},
                    "properties": {
                        "name": "Plot A",
                        "supplier_id": "sup_abc",
                        "commodity": "soya",
                        "country_iso3": "BRA"
                    }
                }
            ]
        }
    """

    type: str = Field("FeatureCollection")
    features: List[GeoJSONFeature] = Field(..., min_length=1, max_length=500)


class BulkPlotImportResponse(BaseModel):
    """Response for bulk plot import."""

    total_submitted: int
    total_created: int
    total_failed: int
    created_ids: List[str]
    errors: List[Dict]


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_plots: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _shoelace_area_degrees(ring: List[List[float]]) -> float:
    """
    Compute the approximate area of a polygon in hectares using the
    Shoelace formula on WGS84 coordinates. This is a rough spherical
    approximation suitable for v1.0; production would use proper geodetic
    calculations.
    """
    n = len(ring) - 1  # last == first
    if n < 3:
        return 0.0

    area_deg2 = 0.0
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        area_deg2 += x1 * y2 - x2 * y1
    area_deg2 = abs(area_deg2) / 2.0

    # Approximate conversion: 1 degree latitude ~ 111 km, longitude varies
    avg_lat = sum(p[1] for p in ring[:-1]) / n
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(avg_lat))
    area_km2 = area_deg2 * km_per_deg_lat * km_per_deg_lon
    area_hectares = area_km2 * 100.0  # 1 km2 = 100 ha
    return round(area_hectares, 2)


def _bounding_boxes_overlap(ring_a: List[List[float]], ring_b: List[List[float]]) -> bool:
    """Quick bounding-box overlap check between two polygon rings."""
    min_lon_a = min(p[0] for p in ring_a)
    max_lon_a = max(p[0] for p in ring_a)
    min_lat_a = min(p[1] for p in ring_a)
    max_lat_a = max(p[1] for p in ring_a)

    min_lon_b = min(p[0] for p in ring_b)
    max_lon_b = max(p[0] for p in ring_b)
    min_lat_b = min(p[1] for p in ring_b)
    max_lat_b = max(p[1] for p in ring_b)

    if max_lon_a < min_lon_b or max_lon_b < min_lon_a:
        return False
    if max_lat_a < min_lat_b or max_lat_b < min_lat_a:
        return False
    return True


def _build_plot_response(data: dict) -> PlotResponse:
    return PlotResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=PlotResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create plot",
    description="Register a new production plot with GeoJSON polygon boundary.",
)
async def create_plot(body: PlotCreateRequest) -> PlotResponse:
    """
    Create a production plot linked to a supplier.

    Calculates area from polygon coordinates and stores the plot for
    subsequent validation and overlap checks.

    Returns:
        201 with created plot record.
    """
    now = datetime.now(timezone.utc)
    plot_id = f"plot_{uuid.uuid4().hex[:12]}"

    area = _shoelace_area_degrees(body.coordinates.coordinates[0])

    record = {
        "plot_id": plot_id,
        "name": body.name,
        "supplier_id": body.supplier_id,
        "coordinates": body.coordinates.model_dump(),
        "commodity": body.commodity,
        "country_iso3": body.country_iso3.upper(),
        "area_hectares": area,
        "risk_level": "unknown",
        "validation_status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    _plots[plot_id] = record
    logger.info("Plot created: %s (%s), area=%.2f ha", plot_id, body.name, area)
    return _build_plot_response(record)


@router.get(
    "/{plot_id}",
    response_model=PlotResponse,
    summary="Get plot",
    description="Retrieve a single plot by its identifier.",
)
async def get_plot(plot_id: str) -> PlotResponse:
    """
    Fetch plot details by ID.

    Returns:
        200 with plot record.

    Raises:
        404 if plot not found.
    """
    record = _plots.get(plot_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_id}' not found",
        )
    return _build_plot_response(record)


@router.put(
    "/{plot_id}",
    response_model=PlotResponse,
    summary="Update plot",
    description="Update fields on an existing plot.",
)
async def update_plot(plot_id: str, body: PlotUpdateRequest) -> PlotResponse:
    """
    Partially update a plot. Recalculates area if coordinates change.

    Returns:
        200 with updated plot record.

    Raises:
        404 if plot not found.
    """
    record = _plots.get(plot_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_id}' not found",
        )

    update_data = body.model_dump(exclude_unset=True)

    if "coordinates" in update_data and update_data["coordinates"] is not None:
        update_data["coordinates"] = body.coordinates.model_dump()
        record["area_hectares"] = _shoelace_area_degrees(
            body.coordinates.coordinates[0]
        )
        # Reset validation when geometry changes
        record["validation_status"] = "pending"

    for key, value in update_data.items():
        record[key] = value

    if "country_iso3" in update_data and update_data["country_iso3"]:
        record["country_iso3"] = record["country_iso3"].upper()

    record["updated_at"] = datetime.now(timezone.utc)
    logger.info("Plot updated: %s", plot_id)
    return _build_plot_response(record)


@router.get(
    "/",
    response_model=PlotListResponse,
    summary="List plots",
    description="List plots with filtering and pagination.",
)
async def list_plots(
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    country: Optional[str] = Query(None, description="Filter by country ISO3"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> PlotListResponse:
    """
    Retrieve a paginated list of plots with optional filters.

    Returns:
        200 with paginated plot list.
    """
    results = list(_plots.values())

    if supplier_id:
        results = [p for p in results if p["supplier_id"] == supplier_id]
    if commodity:
        results = [p for p in results if p["commodity"] == commodity.lower()]
    if country:
        results = [p for p in results if p["country_iso3"] == country.upper()]
    if risk_level:
        results = [p for p in results if p["risk_level"] == risk_level.lower()]

    results.sort(key=lambda p: p["created_at"], reverse=True)

    total = len(results)
    total_pages = max(1, math.ceil(total / limit))
    start = (page - 1) * limit
    page_items = results[start : start + limit]

    return PlotListResponse(
        items=[_build_plot_response(p) for p in page_items],
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
    )


@router.post(
    "/{plot_id}/validate",
    response_model=PlotValidationResult,
    summary="Validate plot coordinates",
    description="Run coordinate validation checks: polygon closure, WGS84 bounds, area calculation, country verification.",
)
async def validate_plot(plot_id: str) -> PlotValidationResult:
    """
    Validate a plot's geometry and metadata.

    Checks performed:
    - Polygon ring closure
    - All coordinates within WGS84 bounds
    - Area calculation (non-zero)
    - Country centroid match (simulated in v1.0)

    Updates the plot's validation_status based on results.

    Returns:
        200 with validation results.

    Raises:
        404 if plot not found.
    """
    record = _plots.get(plot_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_id}' not found",
        )

    issues: List[str] = []
    coords = record["coordinates"]["coordinates"]
    exterior_ring = coords[0] if coords else []

    # Check polygon closure
    polygon_closed = len(exterior_ring) >= 4 and exterior_ring[0] == exterior_ring[-1]
    if not polygon_closed:
        issues.append("Exterior ring is not properly closed")

    # Check WGS84 bounds
    in_bounds = True
    for point in exterior_ring:
        lon, lat = point[0], point[1]
        if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
            in_bounds = False
            issues.append(f"Coordinate [{lon}, {lat}] outside WGS84 bounds")

    # Calculate area
    area = _shoelace_area_degrees(exterior_ring)
    if area <= 0:
        issues.append("Calculated area is zero or negative")
    record["area_hectares"] = area

    # Country match (simulated -- always True in v1.0)
    country_match = True

    is_valid = polygon_closed and in_bounds and area > 0 and country_match
    record["validation_status"] = "valid" if is_valid else "invalid"
    record["updated_at"] = datetime.now(timezone.utc)

    logger.info(
        "Plot %s validation: %s (area=%.2f ha, issues=%d)",
        plot_id,
        record["validation_status"],
        area,
        len(issues),
    )

    return PlotValidationResult(
        plot_id=plot_id,
        is_valid=is_valid,
        area_hectares=area,
        polygon_closed=polygon_closed,
        coordinates_in_bounds=in_bounds,
        country_match=country_match,
        issues=issues,
    )


@router.post(
    "/{plot_id}/overlaps",
    response_model=OverlapCheckResult,
    summary="Check plot overlaps",
    description="Detect overlapping plots using bounding-box intersection.",
)
async def check_overlaps(plot_id: str) -> OverlapCheckResult:
    """
    Check whether the given plot's bounding box overlaps with any other
    registered plots. Uses a fast bounding-box test (production would use
    full polygon intersection via Shapely or PostGIS).

    Returns:
        200 with overlap check results.

    Raises:
        404 if plot not found.
    """
    record = _plots.get(plot_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_id}' not found",
        )

    target_ring = record["coordinates"]["coordinates"][0]
    overlapping: List[str] = []

    for other_id, other in _plots.items():
        if other_id == plot_id:
            continue
        other_ring = other["coordinates"]["coordinates"][0]
        if _bounding_boxes_overlap(target_ring, other_ring):
            overlapping.append(other_id)

    message = (
        f"Found {len(overlapping)} overlapping plot(s)"
        if overlapping
        else "No overlapping plots detected"
    )

    return OverlapCheckResult(
        plot_id=plot_id,
        overlapping_plots=overlapping,
        overlap_count=len(overlapping),
        message=message,
    )


@router.post(
    "/bulk-import",
    response_model=BulkPlotImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk import plots",
    description="Import multiple plots from a GeoJSON FeatureCollection.",
)
async def bulk_import_plots(body: GeoJSONFeatureCollection) -> BulkPlotImportResponse:
    """
    Bulk import plots from a GeoJSON FeatureCollection.

    Each Feature's ``properties`` must contain: ``name``, ``supplier_id``,
    ``commodity``, ``country_iso3``.

    Returns:
        201 with import summary.
    """
    created_ids: List[str] = []
    errors: List[Dict] = []
    now = datetime.now(timezone.utc)

    required_props = {"name", "supplier_id", "commodity", "country_iso3"}

    for idx, feature in enumerate(body.features):
        try:
            props = feature.properties
            missing = required_props - set(props.keys())
            if missing:
                raise ValueError(f"Missing required properties: {missing}")

            plot_id = f"plot_{uuid.uuid4().hex[:12]}"
            area = _shoelace_area_degrees(feature.geometry.coordinates[0])

            record = {
                "plot_id": plot_id,
                "name": props["name"],
                "supplier_id": props["supplier_id"],
                "coordinates": feature.geometry.model_dump(),
                "commodity": props["commodity"].lower(),
                "country_iso3": props["country_iso3"].upper(),
                "area_hectares": area,
                "risk_level": "unknown",
                "validation_status": "pending",
                "created_at": now,
                "updated_at": now,
            }
            _plots[plot_id] = record
            created_ids.append(plot_id)

        except Exception as exc:
            errors.append({"index": idx, "error": str(exc)})

    logger.info(
        "Bulk plot import: %d created, %d failed out of %d",
        len(created_ids),
        len(errors),
        len(body.features),
    )

    return BulkPlotImportResponse(
        total_submitted=len(body.features),
        total_created=len(created_ids),
        total_failed=len(errors),
        created_ids=created_ids,
        errors=errors,
    )
