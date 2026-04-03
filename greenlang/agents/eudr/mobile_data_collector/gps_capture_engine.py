# -*- coding: utf-8 -*-
"""
GPS Capture Engine - AGENT-EUDR-015 Mobile Data Collector (Engine 2)

Production-grade high-accuracy GPS coordinate and polygon capture engine
for EUDR compliance covering point capture with accuracy metadata (HDOP,
satellite count, fix type: 2D/3D/DGPS/RTK), polygon boundary tracing
with configurable minimum vertices, area calculation using the Shoelace
formula with geodesic correction (WGS84 ellipsoid), accuracy tier
classification, WGS84 coordinate validation, polygon self-intersection
checking, distance calculation (Haversine formula), centroid calculation,
buffer zone generation, and EUDR plot size validation.

Zero-Hallucination Guarantees:
    - Area calculation uses deterministic Shoelace formula with WGS84
      geodesic correction factor (no LLM, no ML)
    - Distance calculation uses the Haversine formula with standard
      Earth radius (6371008.8 meters)
    - Accuracy classification uses fixed threshold tables
    - All coordinate validation is pure arithmetic
    - SHA-256 provenance recorded for every capture

PRD: PRD-AGENT-EUDR-015 Feature F2 (GPS/Geolocation Capture)
Agent ID: GL-EUDR-MDC-015
Regulation: EU 2023/1115 (EUDR) Article 9(1)(d)

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mobile_data_collector.config import get_config
from greenlang.agents.eudr.mobile_data_collector.metrics import (
    observe_gps_capture_duration,
    record_api_error,
    record_gps_capture,
)
from greenlang.agents.eudr.mobile_data_collector.models import (
    CaptureAccuracyTier,
    GPSCapture,
    GPSResponse,
    PolygonResponse,
    PolygonTrace,
    WGS84_SRID,
)
from greenlang.agents.eudr.mobile_data_collector.provenance import (
    get_provenance_tracker,
)
from greenlang.utilities.exceptions.compliance import ComplianceException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mean Earth radius in meters (WGS84 mean radius).
EARTH_RADIUS_M: float = 6371008.8

#: WGS84 semi-major axis in meters.
WGS84_SEMI_MAJOR_M: float = 6378137.0

#: WGS84 semi-minor axis in meters.
WGS84_SEMI_MINOR_M: float = 6356752.314245

#: Conversion factor: square meters to hectares.
SQM_TO_HA: float = 1e-4

#: Maximum allowed plot area in hectares.
MAX_PLOT_AREA_HA: float = 10000.0


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GPSCaptureEngineError(ComplianceException):
    """Base exception for GPS capture engine operations."""


class CoordinateValidationError(GPSCaptureEngineError):
    """Raised when GPS coordinates fail validation."""


class PolygonValidationError(GPSCaptureEngineError):
    """Raised when a polygon trace fails validation."""


class AccuracyError(GPSCaptureEngineError):
    """Raised when GPS accuracy does not meet requirements."""


class PlotSizeError(GPSCaptureEngineError):
    """Raised when plot area is outside EUDR bounds."""


# ---------------------------------------------------------------------------
# GPSCaptureEngine
# ---------------------------------------------------------------------------


class GPSCaptureEngine:
    """High-accuracy GPS coordinate and polygon capture engine.

    Provides EUDR Article 9(1)(d) compliant GPS data capture with
    accuracy metadata, polygon boundary tracing, geodesic area
    calculation, and coordinate validation against the WGS84 datum.

    Thread Safety:
        All public methods are protected by a reentrant lock for
        concurrent access from multiple API handlers.

    Attributes:
        _config: Agent configuration instance.
        _captures: In-memory GPS capture store keyed by capture_id.
        _polygons: In-memory polygon store keyed by polygon_id.
        _provenance: Provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = GPSCaptureEngine()
        >>> result = engine.capture_point(
        ...     device_id="dev-001",
        ...     operator_id="op-001",
        ...     latitude=5.603717,
        ...     longitude=-0.186964,
        ...     horizontal_accuracy_m=2.5,
        ...     hdop=1.2,
        ...     satellite_count=10,
        ... )
        >>> assert result.accuracy_tier == CaptureAccuracyTier.GOOD
    """

    __slots__ = (
        "_config",
        "_captures",
        "_polygons",
        "_provenance",
        "_lock",
    )

    def __init__(self) -> None:
        """Initialize the GPSCaptureEngine with empty stores."""
        self._config = get_config()
        self._captures: Dict[str, GPSCapture] = {}
        self._polygons: Dict[str, PolygonTrace] = {}
        self._provenance = get_provenance_tracker()
        self._lock = threading.RLock()
        logger.info(
            "GPSCaptureEngine initialized: min_accuracy=%sm, "
            "hdop_threshold=%s, sats=%d, crs=%s, "
            "polygon_verts=[%d,%d]",
            self._config.min_accuracy_meters,
            self._config.hdop_threshold,
            self._config.satellite_count_threshold,
            self._config.default_crs,
            self._config.polygon_min_vertices,
            self._config.polygon_max_vertices,
        )

    # ------------------------------------------------------------------
    # Public API: Point Capture
    # ------------------------------------------------------------------

    def capture_point(
        self,
        device_id: str,
        operator_id: str,
        latitude: float,
        longitude: float,
        horizontal_accuracy_m: float,
        hdop: float,
        satellite_count: int,
        form_id: Optional[str] = None,
        altitude_m: Optional[float] = None,
        vertical_accuracy_m: Optional[float] = None,
        fix_type: str = "GPS",
        augmentation: Optional[str] = None,
        capture_timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GPSResponse:
        """Capture a GPS point coordinate with accuracy metadata.

        Validates WGS84 coordinates, classifies the accuracy tier,
        stores the capture, and records provenance.

        Args:
            device_id: Source device identifier.
            operator_id: Field agent identifier.
            latitude: WGS84 latitude in decimal degrees (-90 to 90).
            longitude: WGS84 longitude in decimal degrees (-180 to 180).
            horizontal_accuracy_m: Estimated horizontal accuracy in meters.
            hdop: Horizontal Dilution of Precision.
            satellite_count: Number of satellites used in fix.
            form_id: Associated form submission identifier.
            altitude_m: Altitude above sea level in meters.
            vertical_accuracy_m: Estimated vertical accuracy in meters.
            fix_type: GPS constellation type.
            augmentation: SBAS augmentation source.
            capture_timestamp: Device timestamp at capture.
            metadata: Additional capture metadata.

        Returns:
            GPSResponse with capture_id, accuracy_tier, and provenance.

        Raises:
            CoordinateValidationError: If coordinates are invalid.
            GPSCaptureEngineError: If capture processing fails.
        """
        start_time = time.monotonic()
        try:
            # Validate coordinates
            self.validate_coordinates(latitude, longitude)

            # Round coordinates
            decimal_places = self._config.coordinate_decimal_places
            lat = round(latitude, decimal_places)
            lon = round(longitude, decimal_places)

            # Classify accuracy
            accuracy_tier = self.classify_accuracy(
                horizontal_accuracy_m, hdop, satellite_count,
            )

            # Build capture record
            now = datetime.now(timezone.utc).replace(microsecond=0)
            capture = GPSCapture(
                form_id=form_id,
                device_id=device_id,
                operator_id=operator_id,
                latitude=lat,
                longitude=lon,
                altitude_m=altitude_m,
                horizontal_accuracy_m=horizontal_accuracy_m,
                vertical_accuracy_m=vertical_accuracy_m,
                hdop=hdop,
                satellite_count=satellite_count,
                fix_type=fix_type,
                augmentation=augmentation,
                accuracy_tier=accuracy_tier,
                capture_timestamp=capture_timestamp or now,
                srid=WGS84_SRID,
                metadata=metadata or {},
                created_at=now,
            )

            # Store capture
            with self._lock:
                self._captures[capture.capture_id] = capture

            # Record provenance
            provenance_entry = self._provenance.record(
                entity_type="gps_capture",
                action="capture",
                entity_id=capture.capture_id,
                data={
                    "latitude": lat,
                    "longitude": lon,
                    "accuracy_m": horizontal_accuracy_m,
                    "hdop": hdop,
                    "satellites": satellite_count,
                    "tier": accuracy_tier.value,
                },
                metadata={
                    "device_id": device_id,
                    "form_id": form_id,
                },
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            observe_gps_capture_duration(elapsed_ms / 1000)
            record_gps_capture(accuracy_tier.value)

            logger.info(
                "GPS point captured: id=%s lat=%.6f lon=%.6f "
                "accuracy=%.1fm tier=%s elapsed=%.1fms",
                capture.capture_id, lat, lon,
                horizontal_accuracy_m, accuracy_tier.value,
                elapsed_ms,
            )

            return GPSResponse(
                capture_id=capture.capture_id,
                accuracy_tier=accuracy_tier,
                provenance_hash=provenance_entry.hash_value,
                processing_time_ms=elapsed_ms,
                message=f"GPS point captured with {accuracy_tier.value} accuracy",
                capture=capture,
            )

        except CoordinateValidationError:
            record_api_error("capture")
            raise
        except Exception as e:
            record_api_error("capture")
            logger.error(
                "GPS capture failed: %s", str(e), exc_info=True,
            )
            raise GPSCaptureEngineError(
                f"GPS capture failed: {str(e)}"
            ) from e

    # ------------------------------------------------------------------
    # Public API: Polygon Capture
    # ------------------------------------------------------------------

    def capture_polygon(
        self,
        device_id: str,
        operator_id: str,
        vertices: List[List[float]],
        form_id: Optional[str] = None,
        vertex_accuracies_m: Optional[List[float]] = None,
        capture_start: Optional[datetime] = None,
        capture_end: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PolygonResponse:
        """Capture a plot boundary polygon via GPS vertex tracing.

        Validates all vertices, checks self-intersection, closes the
        polygon if needed, calculates area using the Shoelace formula
        with geodesic correction, validates plot size against EUDR
        bounds, and records provenance.

        Args:
            device_id: Source device identifier.
            operator_id: Field agent who performed the trace.
            vertices: List of [latitude, longitude] coordinate pairs.
            form_id: Associated form submission identifier.
            vertex_accuracies_m: Per-vertex horizontal accuracy in meters.
            capture_start: Timestamp when tracing started.
            capture_end: Timestamp when tracing completed.
            metadata: Additional polygon metadata.

        Returns:
            PolygonResponse with polygon_id, area_ha, validity, and
            provenance hash.

        Raises:
            PolygonValidationError: If polygon geometry is invalid.
            PlotSizeError: If the calculated area is outside EUDR bounds.
            GPSCaptureEngineError: If polygon capture fails.
        """
        start_time = time.monotonic()
        try:
            # Validate vertex count
            min_verts = self._config.polygon_min_vertices
            max_verts = self._config.polygon_max_vertices
            if len(vertices) < min_verts:
                raise PolygonValidationError(
                    f"Polygon requires at least {min_verts} vertices, "
                    f"got {len(vertices)}"
                )
            if len(vertices) > max_verts:
                raise PolygonValidationError(
                    f"Polygon exceeds max {max_verts} vertices, "
                    f"got {len(vertices)}"
                )

            # Validate each vertex
            decimal_places = self._config.coordinate_decimal_places
            rounded_vertices: List[List[float]] = []
            for i, vertex in enumerate(vertices):
                if len(vertex) < 2:
                    raise PolygonValidationError(
                        f"Vertex[{i}] must have [lat, lon], "
                        f"got {vertex}"
                    )
                self.validate_coordinates(vertex[0], vertex[1])
                rounded_vertices.append([
                    round(vertex[0], decimal_places),
                    round(vertex[1], decimal_places),
                ])

            # Close the polygon if needed
            is_closed = self._are_vertices_equal(
                rounded_vertices[0], rounded_vertices[-1],
            )
            if not is_closed:
                rounded_vertices.append(list(rounded_vertices[0]))
                is_closed = True

            # Check self-intersection
            is_valid = not self.check_self_intersection(
                rounded_vertices
            )

            # Calculate area
            area_ha = self.calculate_area(rounded_vertices)

            # Calculate perimeter
            perimeter_m = self._calculate_perimeter(rounded_vertices)

            # Validate plot size
            self.validate_plot_size(area_ha)

            # Build polygon record
            now = datetime.now(timezone.utc).replace(microsecond=0)
            polygon = PolygonTrace(
                form_id=form_id,
                device_id=device_id,
                operator_id=operator_id,
                vertices=rounded_vertices,
                vertex_accuracies_m=vertex_accuracies_m or [],
                vertex_count=len(rounded_vertices),
                area_ha=area_ha,
                perimeter_m=perimeter_m,
                is_closed=is_closed,
                is_valid=is_valid,
                capture_start=capture_start,
                capture_end=capture_end,
                srid=WGS84_SRID,
                metadata=metadata or {},
                created_at=now,
            )

            # Store polygon
            with self._lock:
                self._polygons[polygon.polygon_id] = polygon

            # Record provenance
            provenance_entry = self._provenance.record(
                entity_type="polygon_trace",
                action="capture",
                entity_id=polygon.polygon_id,
                data={
                    "vertex_count": len(rounded_vertices),
                    "area_ha": area_ha,
                    "perimeter_m": perimeter_m,
                    "is_valid": is_valid,
                },
                metadata={
                    "device_id": device_id,
                    "form_id": form_id,
                },
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            observe_gps_capture_duration(elapsed_ms / 1000)
            record_gps_capture("polygon")

            logger.info(
                "Polygon captured: id=%s vertices=%d area=%.4fha "
                "perimeter=%.1fm valid=%s elapsed=%.1fms",
                polygon.polygon_id, len(rounded_vertices),
                area_ha, perimeter_m, is_valid, elapsed_ms,
            )

            return PolygonResponse(
                polygon_id=polygon.polygon_id,
                area_ha=area_ha,
                vertex_count=len(rounded_vertices),
                is_valid=is_valid,
                provenance_hash=provenance_entry.hash_value,
                processing_time_ms=elapsed_ms,
                message=(
                    f"Polygon captured: {area_ha:.4f} ha, "
                    f"{len(rounded_vertices)} vertices"
                ),
                polygon=polygon,
            )

        except (
            PolygonValidationError,
            CoordinateValidationError,
            PlotSizeError,
        ):
            record_api_error("capture")
            raise
        except Exception as e:
            record_api_error("capture")
            logger.error(
                "Polygon capture failed: %s", str(e), exc_info=True,
            )
            raise GPSCaptureEngineError(
                f"Polygon capture failed: {str(e)}"
            ) from e

    # ------------------------------------------------------------------
    # Public API: Validation and Calculation
    # ------------------------------------------------------------------

    def validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> bool:
        """Validate WGS84 coordinates are within valid bounds.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            True if coordinates are valid.

        Raises:
            CoordinateValidationError: If coordinates are out of bounds.
        """
        if not isinstance(latitude, (int, float)):
            raise CoordinateValidationError(
                f"Latitude must be numeric, got {type(latitude).__name__}"
            )
        if not isinstance(longitude, (int, float)):
            raise CoordinateValidationError(
                f"Longitude must be numeric, got {type(longitude).__name__}"
            )
        if math.isnan(latitude) or math.isinf(latitude):
            raise CoordinateValidationError(
                f"Latitude must be finite, got {latitude}"
            )
        if math.isnan(longitude) or math.isinf(longitude):
            raise CoordinateValidationError(
                f"Longitude must be finite, got {longitude}"
            )
        if latitude < -90.0 or latitude > 90.0:
            raise CoordinateValidationError(
                f"Latitude must be in [-90, 90], got {latitude}"
            )
        if longitude < -180.0 or longitude > 180.0:
            raise CoordinateValidationError(
                f"Longitude must be in [-180, 180], got {longitude}"
            )
        return True

    def calculate_area(
        self,
        vertices: List[List[float]],
    ) -> float:
        """Calculate polygon area using Shoelace formula with geodesic correction.

        Applies the Shoelace formula on projected coordinates using a
        latitude-dependent geodesic correction factor to convert from
        angular to metric area on the WGS84 ellipsoid.

        The formula accounts for the convergence of meridians at higher
        latitudes by multiplying the longitude difference by
        cos(latitude).

        Args:
            vertices: List of [latitude, longitude] pairs. Must form
                a closed polygon (first vertex == last vertex).

        Returns:
            Area in hectares (always non-negative).
        """
        if len(vertices) < 4:
            return 0.0

        # Compute centroid latitude for geodesic correction
        n = len(vertices) - 1  # exclude closing vertex
        avg_lat = sum(v[0] for v in vertices[:n]) / n

        # Degrees to radians
        avg_lat_rad = math.radians(avg_lat)

        # Meters per degree at this latitude
        m_per_deg_lat = (
            111132.92
            - 559.82 * math.cos(2 * avg_lat_rad)
            + 1.175 * math.cos(4 * avg_lat_rad)
            - 0.0023 * math.cos(6 * avg_lat_rad)
        )
        m_per_deg_lon = (
            111412.84 * math.cos(avg_lat_rad)
            - 93.5 * math.cos(3 * avg_lat_rad)
            + 0.118 * math.cos(5 * avg_lat_rad)
        )

        # Shoelace formula in projected coordinates
        area_sqm = 0.0
        for i in range(n):
            j = (i + 1) % n
            xi = vertices[i][1] * m_per_deg_lon
            yi = vertices[i][0] * m_per_deg_lat
            xj = vertices[j][1] * m_per_deg_lon
            yj = vertices[j][0] * m_per_deg_lat
            area_sqm += xi * yj - xj * yi

        area_sqm = abs(area_sqm) / 2.0
        area_ha = area_sqm * SQM_TO_HA

        return round(area_ha, 6)

    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two GPS points using Haversine formula.

        Args:
            lat1: Latitude of first point in decimal degrees.
            lon1: Longitude of first point in decimal degrees.
            lat2: Latitude of second point in decimal degrees.
            lon2: Longitude of second point in decimal degrees.

        Returns:
            Distance in meters.

        Raises:
            CoordinateValidationError: If any coordinate is invalid.
        """
        self.validate_coordinates(lat1, lon1)
        self.validate_coordinates(lat2, lon2)

        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r)
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return round(EARTH_RADIUS_M * c, 3)

    def get_centroid(
        self,
        vertices: List[List[float]],
    ) -> Tuple[float, float]:
        """Calculate the centroid of a polygon.

        Uses the signed area method for computing the centroid of a
        simple polygon from its vertex coordinates.

        Args:
            vertices: List of [latitude, longitude] pairs forming
                a closed polygon.

        Returns:
            Tuple of (centroid_latitude, centroid_longitude).

        Raises:
            PolygonValidationError: If vertices list is too short.
        """
        if len(vertices) < 4:
            raise PolygonValidationError(
                f"Centroid requires at least 4 vertices (3 + closing), "
                f"got {len(vertices)}"
            )

        n = len(vertices) - 1
        signed_area = 0.0
        cx = 0.0
        cy = 0.0

        for i in range(n):
            j = (i + 1) % n
            cross = (
                vertices[i][0] * vertices[j][1]
                - vertices[j][0] * vertices[i][1]
            )
            signed_area += cross
            cx += (vertices[i][0] + vertices[j][0]) * cross
            cy += (vertices[i][1] + vertices[j][1]) * cross

        signed_area *= 0.5
        if abs(signed_area) < 1e-15:
            # Degenerate polygon: return arithmetic mean
            avg_lat = sum(v[0] for v in vertices[:n]) / n
            avg_lon = sum(v[1] for v in vertices[:n]) / n
            return (round(avg_lat, 6), round(avg_lon, 6))

        factor = 1.0 / (6.0 * signed_area)
        cx *= factor
        cy *= factor

        decimal_places = self._config.coordinate_decimal_places
        return (round(cx, decimal_places), round(cy, decimal_places))

    def check_self_intersection(
        self,
        vertices: List[List[float]],
    ) -> bool:
        """Check if a polygon has any self-intersections.

        Uses a simplified brute-force pairwise segment intersection
        test. For each pair of non-adjacent edges, checks if they
        cross using the cross-product orientation method.

        Args:
            vertices: List of [latitude, longitude] pairs forming
                a closed polygon.

        Returns:
            True if self-intersections exist, False if polygon is
            simple (no crossings).
        """
        n = len(vertices) - 1
        if n < 3:
            return False

        edges: List[Tuple[List[float], List[float]]] = []
        for i in range(n):
            edges.append((vertices[i], vertices[i + 1]))

        for i in range(len(edges)):
            for j in range(i + 2, len(edges)):
                # Skip adjacent edges
                if i == 0 and j == len(edges) - 1:
                    continue
                if self._segments_intersect(
                    edges[i][0], edges[i][1],
                    edges[j][0], edges[j][1],
                ):
                    return True

        return False

    def classify_accuracy(
        self,
        horizontal_accuracy_m: float,
        hdop: float,
        satellite_count: int,
    ) -> CaptureAccuracyTier:
        """Classify GPS capture accuracy into a tier.

        Uses a combination of horizontal accuracy, HDOP, and satellite
        count to determine the accuracy tier per PRD Appendix B.

        Args:
            horizontal_accuracy_m: Estimated horizontal accuracy in meters.
            hdop: Horizontal Dilution of Precision.
            satellite_count: Number of satellites used.

        Returns:
            CaptureAccuracyTier classification.
        """
        if (
            horizontal_accuracy_m < 1.0
            and hdop < 1.0
            and satellite_count >= 12
        ):
            return CaptureAccuracyTier.EXCELLENT

        if (
            horizontal_accuracy_m <= 3.0
            and hdop <= 2.0
            and satellite_count >= 8
        ):
            return CaptureAccuracyTier.GOOD

        if (
            horizontal_accuracy_m <= 5.0
            and hdop <= 3.0
            and satellite_count >= 6
        ):
            return CaptureAccuracyTier.ACCEPTABLE

        if (
            horizontal_accuracy_m <= 10.0
            and hdop <= 5.0
            and satellite_count >= 4
        ):
            return CaptureAccuracyTier.POOR

        return CaptureAccuracyTier.REJECTED

    def validate_plot_size(
        self,
        area_ha: float,
    ) -> bool:
        """Validate that plot area is within EUDR bounds.

        Args:
            area_ha: Plot area in hectares.

        Returns:
            True if area is within bounds.

        Raises:
            PlotSizeError: If area is outside EUDR bounds.
        """
        min_ha = self._config.min_plot_area_ha
        if area_ha < min_ha:
            raise PlotSizeError(
                f"Plot area {area_ha:.6f} ha is below minimum "
                f"{min_ha} ha"
            )
        if area_ha > MAX_PLOT_AREA_HA:
            raise PlotSizeError(
                f"Plot area {area_ha:.6f} ha exceeds maximum "
                f"{MAX_PLOT_AREA_HA} ha"
            )
        return True

    def generate_buffer_zone(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
        num_points: int = 36,
    ) -> List[List[float]]:
        """Generate a circular buffer zone approximation around a point.

        Creates a polygon of regularly-spaced points forming a circle
        at the given radius around the center point.

        Args:
            latitude: Center point latitude in decimal degrees.
            longitude: Center point longitude in decimal degrees.
            radius_m: Buffer radius in meters.
            num_points: Number of points in the approximation circle.

        Returns:
            List of [latitude, longitude] pairs forming a closed polygon.

        Raises:
            CoordinateValidationError: If center coordinates are invalid.
        """
        self.validate_coordinates(latitude, longitude)

        if radius_m <= 0:
            raise GPSCaptureEngineError(
                f"Buffer radius must be > 0, got {radius_m}"
            )
        if num_points < 8:
            raise GPSCaptureEngineError(
                f"Buffer requires at least 8 points, got {num_points}"
            )

        decimal_places = self._config.coordinate_decimal_places
        buffer_points: List[List[float]] = []

        # Approximate degrees per meter at this latitude
        lat_rad = math.radians(latitude)
        deg_per_m_lat = 1.0 / 111132.92
        deg_per_m_lon = 1.0 / (
            111412.84 * math.cos(lat_rad)
        )

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dlat = radius_m * math.cos(angle) * deg_per_m_lat
            dlon = radius_m * math.sin(angle) * deg_per_m_lon
            buffer_points.append([
                round(latitude + dlat, decimal_places),
                round(longitude + dlon, decimal_places),
            ])

        # Close the polygon
        buffer_points.append(list(buffer_points[0]))

        return buffer_points

    # ------------------------------------------------------------------
    # Retrieval methods
    # ------------------------------------------------------------------

    def get_capture(self, capture_id: str) -> GPSCapture:
        """Retrieve a GPS capture by its identifier.

        Args:
            capture_id: GPS capture identifier.

        Returns:
            GPSCapture instance.

        Raises:
            GPSCaptureEngineError: If capture is not found.
        """
        with self._lock:
            capture = self._captures.get(capture_id)
        if capture is None:
            raise GPSCaptureEngineError(
                f"GPS capture not found: capture_id={capture_id}"
            )
        return capture

    def get_polygon(self, polygon_id: str) -> PolygonTrace:
        """Retrieve a polygon trace by its identifier.

        Args:
            polygon_id: Polygon trace identifier.

        Returns:
            PolygonTrace instance.

        Raises:
            GPSCaptureEngineError: If polygon is not found.
        """
        with self._lock:
            polygon = self._polygons.get(polygon_id)
        if polygon is None:
            raise GPSCaptureEngineError(
                f"Polygon not found: polygon_id={polygon_id}"
            )
        return polygon

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capture_count(self) -> int:
        """Return the total number of stored GPS captures."""
        with self._lock:
            return len(self._captures)

    @property
    def polygon_count(self) -> int:
        """Return the total number of stored polygon traces."""
        with self._lock:
            return len(self._polygons)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_perimeter(
        self,
        vertices: List[List[float]],
    ) -> float:
        """Calculate the perimeter of a closed polygon in meters.

        Args:
            vertices: List of [latitude, longitude] pairs.

        Returns:
            Perimeter in meters.
        """
        perimeter = 0.0
        for i in range(len(vertices) - 1):
            perimeter += self.calculate_distance(
                vertices[i][0], vertices[i][1],
                vertices[i + 1][0], vertices[i + 1][1],
            )
        return round(perimeter, 3)

    def _are_vertices_equal(
        self,
        v1: List[float],
        v2: List[float],
    ) -> bool:
        """Check if two vertex coordinates are approximately equal.

        Args:
            v1: First vertex [lat, lon].
            v2: Second vertex [lat, lon].

        Returns:
            True if vertices are within floating-point tolerance.
        """
        tol = 10 ** (-self._config.coordinate_decimal_places)
        return (
            abs(v1[0] - v2[0]) < tol
            and abs(v1[1] - v2[1]) < tol
        )

    @staticmethod
    def _cross_product_2d(
        o: List[float],
        a: List[float],
        b: List[float],
    ) -> float:
        """Compute the 2D cross product of vectors OA and OB.

        Args:
            o: Origin point [lat, lon].
            a: First point [lat, lon].
            b: Second point [lat, lon].

        Returns:
            Cross product value. Positive if counter-clockwise,
            negative if clockwise, zero if collinear.
        """
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    @staticmethod
    def _on_segment(
        p: List[float],
        q: List[float],
        r: List[float],
    ) -> bool:
        """Check if point q lies on segment pr.

        Args:
            p: Segment start point [lat, lon].
            q: Point to test [lat, lon].
            r: Segment end point [lat, lon].

        Returns:
            True if q lies on segment pr.
        """
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    @classmethod
    def _segments_intersect(
        cls,
        p1: List[float],
        q1: List[float],
        p2: List[float],
        q2: List[float],
    ) -> bool:
        """Check if two line segments (p1-q1) and (p2-q2) intersect.

        Uses the cross-product orientation method.

        Args:
            p1: Start of first segment [lat, lon].
            q1: End of first segment [lat, lon].
            p2: Start of second segment [lat, lon].
            q2: End of second segment [lat, lon].

        Returns:
            True if segments properly intersect.
        """
        d1 = cls._cross_product_2d(p2, q2, p1)
        d2 = cls._cross_product_2d(p2, q2, q1)
        d3 = cls._cross_product_2d(p1, q1, p2)
        d4 = cls._cross_product_2d(p1, q1, q2)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        if abs(d1) < 1e-15 and cls._on_segment(p2, p1, q2):
            return True
        if abs(d2) < 1e-15 and cls._on_segment(p2, q1, q2):
            return True
        if abs(d3) < 1e-15 and cls._on_segment(p1, p2, q1):
            return True
        if abs(d4) < 1e-15 and cls._on_segment(p1, q2, q1):
            return True

        return False

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"GPSCaptureEngine(captures={self.capture_count}, "
            f"polygons={self.polygon_count})"
        )

    def __len__(self) -> int:
        """Return the total number of captures and polygons."""
        return self.capture_count + self.polygon_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GPSCaptureEngine",
    "GPSCaptureEngineError",
    "CoordinateValidationError",
    "PolygonValidationError",
    "AccuracyError",
    "PlotSizeError",
    "EARTH_RADIUS_M",
    "WGS84_SEMI_MAJOR_M",
    "WGS84_SEMI_MINOR_M",
    "SQM_TO_HA",
    "MAX_PLOT_AREA_HA",
]
