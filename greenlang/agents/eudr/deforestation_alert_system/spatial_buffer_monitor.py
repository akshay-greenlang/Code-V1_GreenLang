# -*- coding: utf-8 -*-
"""
AGENT-EUDR-020: Deforestation Alert System - Spatial Buffer Monitor

Geofencing engine maintaining configurable buffer zones around supply chain
plots for real-time deforestation proximity monitoring. Performs point-in-polygon
testing (ray casting algorithm), circular buffer intersection, nearest-neighbor
analysis, and buffer overlap calculation. Supports circular, polygon-based, and
adaptive buffer geometries at configurable distances from 1-50 km.

Zero-Hallucination Guarantees:
    - All spatial calculations use deterministic Decimal/float arithmetic
    - Point-in-polygon via ray casting algorithm (no ML/LLM)
    - Haversine distance formula for great-circle distances
    - Circular buffer polygon generation via trigonometric functions
    - Buffer overlap via simplified polygon intersection
    - SHA-256 provenance hashes on all result objects
    - No LLM/ML in the spatial computation path

Supported Buffer Types:
    - CIRCULAR: Fixed-radius circle around plot center (64-point polygon)
    - POLYGON: User-defined polygon boundary
    - ADAPTIVE: Variable radius based on commodity and country risk

Spatial Algorithms:
    - Haversine Distance: Great-circle distance between two points
      R = 6371 km, a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
      d = 2*R*arcsin(sqrt(a))
    - Ray Casting: Point-in-polygon test by counting edge crossings
    - Buffer Generation: N-point polygon approximation of circle on sphere

Performance Targets:
    - Single point-in-buffer check: <5ms
    - Check against 1000 active buffers: <500ms
    - Buffer creation: <10ms
    - Polygon buffer generation: <20ms (64 points)

Regulatory References:
    - EUDR Article 9(1)(d): Geolocation precision requirements
    - EUDR Article 10(2): Spatial risk assessment
    - EUDR Article 11: Proportionate monitoring intensity
    - EUDR Article 31: 5-year spatial record retention

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020 (Engine 4: Spatial Buffer Monitor)
Agent ID: GL-EUDR-DAS-020
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Config import (thread-safe singleton)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Provenance import
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.provenance import (
        ProvenanceTracker,
        get_tracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    get_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.metrics import (
        PROMETHEUS_AVAILABLE,
        record_buffer_check,
        set_active_buffers,
        record_api_error,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    record_buffer_check = None  # type: ignore[misc,assignment]
    set_active_buffers = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a unique identifier using UUID4.

    Returns:
        String representation of a new UUID4.
    """
    return str(uuid.uuid4())


def _safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Value to convert.
        default: Default Decimal if conversion fails.

    Returns:
        Decimal representation of value or default.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start.

    Args:
        start: time.perf_counter() start value.

    Returns:
        Elapsed time in milliseconds.
    """
    return round((time.perf_counter() - start) * 1000, 2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Earth's mean radius in kilometers for Haversine calculation.
EARTH_RADIUS_KM: Decimal = Decimal("6371")

#: Minimum allowed buffer radius in kilometers.
MIN_BUFFER_RADIUS_KM: Decimal = Decimal("1")

#: Maximum allowed buffer radius in kilometers.
MAX_BUFFER_RADIUS_KM: Decimal = Decimal("50")

#: Default buffer radius in kilometers.
DEFAULT_BUFFER_RADIUS_KM: Decimal = Decimal("10")

#: Default number of points for circular buffer polygon generation.
DEFAULT_BUFFER_RESOLUTION: int = 64

#: Degrees per radian (for coordinate conversion).
DEG_PER_RAD: float = 180.0 / math.pi

#: Radians per degree (for coordinate conversion).
RAD_PER_DEG: float = math.pi / 180.0

#: Commodity-based adaptive buffer radii (km).
COMMODITY_BUFFER_RADII: Dict[str, Decimal] = {
    "cattle": Decimal("15"),
    "cocoa": Decimal("8"),
    "coffee": Decimal("8"),
    "palm_oil": Decimal("12"),
    "rubber": Decimal("10"),
    "soya": Decimal("15"),
    "wood": Decimal("12"),
}

#: Country risk-based buffer multipliers.
COUNTRY_RISK_MULTIPLIERS: Dict[str, Decimal] = {
    "BR": Decimal("1.2"),   # High deforestation risk
    "ID": Decimal("1.2"),   # High deforestation risk
    "CD": Decimal("1.3"),   # Very high risk
    "CG": Decimal("1.3"),   # Very high risk
    "CO": Decimal("1.1"),
    "PE": Decimal("1.1"),
    "MY": Decimal("1.1"),
    "GH": Decimal("1.0"),
    "CI": Decimal("1.0"),
    "CM": Decimal("1.1"),
    "PY": Decimal("1.2"),
    "BO": Decimal("1.2"),
    "_DEFAULT": Decimal("1.0"),
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BufferType(str, Enum):
    """Types of spatial buffer geometries.

    CIRCULAR: Fixed-radius circle approximated by N-point polygon.
    POLYGON: User-defined polygon boundary.
    ADAPTIVE: Variable radius based on commodity and country risk.
    """

    CIRCULAR = "circular"
    POLYGON = "polygon"
    ADAPTIVE = "adaptive"


class ViolationSeverity(str, Enum):
    """Severity levels for buffer zone violations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class GeoPoint:
    """A geographic point with latitude and longitude.

    Attributes:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).
    """

    lat: Decimal = Decimal("0")
    lon: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary.

        Returns:
            Dictionary with lat and lon as strings.
        """
        return {"lat": str(self.lat), "lon": str(self.lon)}

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to float tuple (lat, lon).

        Returns:
            Tuple of (latitude, longitude) as floats.
        """
        return (float(self.lat), float(self.lon))


@dataclass
class BufferZone:
    """Spatial buffer zone around a supply chain plot.

    Defines the monitoring perimeter around a registered production
    plot for deforestation proximity detection.

    Attributes:
        buffer_id: Unique buffer identifier (UUID).
        plot_id: Associated plot identifier.
        center_lat: Buffer center latitude.
        center_lon: Buffer center longitude.
        radius_km: Buffer radius in kilometers.
        buffer_type: Type of buffer geometry.
        polygon_points: List of (lat, lon) tuples defining the polygon.
        is_active: Whether the buffer is actively monitored.
        commodities: EUDR commodities associated with the plot.
        country_code: ISO 3166-1 alpha-2 country code.
        created_at: Buffer creation timestamp.
        updated_at: Last update timestamp.
        metadata: Additional metadata.
    """

    buffer_id: str = ""
    plot_id: str = ""
    center_lat: Decimal = Decimal("0")
    center_lon: Decimal = Decimal("0")
    radius_km: Decimal = DEFAULT_BUFFER_RADIUS_KM
    buffer_type: str = BufferType.CIRCULAR.value
    polygon_points: List[Tuple[float, float]] = field(default_factory=list)
    is_active: bool = True
    commodities: List[str] = field(default_factory=list)
    country_code: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for unset fields."""
        if not self.buffer_id:
            self.buffer_id = _generate_id()
        if not self.created_at:
            self.created_at = _utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize buffer zone to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "buffer_id": self.buffer_id,
            "plot_id": self.plot_id,
            "center_lat": str(self.center_lat),
            "center_lon": str(self.center_lon),
            "radius_km": str(self.radius_km),
            "buffer_type": self.buffer_type,
            "polygon_point_count": len(self.polygon_points),
            "is_active": self.is_active,
            "commodities": self.commodities,
            "country_code": self.country_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class BufferViolation:
    """Record of a detection event that violates a buffer zone.

    Captures the spatial relationship between a deforestation detection
    and a supply chain plot's buffer zone.

    Attributes:
        violation_id: Unique violation identifier (UUID).
        buffer_id: Buffer zone that was violated.
        detection_id: Detection event identifier.
        distance_km: Distance from detection to buffer center.
        overlap_area_ha: Estimated overlap area in hectares.
        violation_time: Timestamp of violation detection.
        inside_buffer: Whether detection point is inside the buffer.
        violation_severity: Severity classification.
        plot_id: Affected plot identifier.
        commodity: Affected commodity.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional metadata.
    """

    violation_id: str = ""
    buffer_id: str = ""
    detection_id: str = ""
    distance_km: Decimal = Decimal("0")
    overlap_area_ha: Decimal = Decimal("0")
    violation_time: str = ""
    inside_buffer: bool = False
    violation_severity: str = ViolationSeverity.MEDIUM.value
    plot_id: str = ""
    commodity: str = ""
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for unset fields."""
        if not self.violation_id:
            self.violation_id = _generate_id()
        if not self.violation_time:
            self.violation_time = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize violation to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "violation_id": self.violation_id,
            "buffer_id": self.buffer_id,
            "detection_id": self.detection_id,
            "distance_km": str(self.distance_km),
            "overlap_area_ha": str(self.overlap_area_ha),
            "violation_time": self.violation_time,
            "inside_buffer": self.inside_buffer,
            "violation_severity": self.violation_severity,
            "plot_id": self.plot_id,
            "commodity": self.commodity,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
        }


@dataclass
class BufferResult:
    """Result of a buffer creation or update operation.

    Attributes:
        buffer: Created or updated BufferZone.
        polygon_generated: Whether polygon points were generated.
        processing_time_ms: Processing time in milliseconds.
        warnings: List of warning messages.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Operation timestamp.
    """

    buffer: Optional[BufferZone] = None
    polygon_generated: bool = False
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "buffer": self.buffer.to_dict() if self.buffer else None,
            "polygon_generated": self.polygon_generated,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class CheckResult:
    """Result of checking a detection against all active buffers.

    Attributes:
        detection_lat: Detection latitude.
        detection_lon: Detection longitude.
        detection_area_ha: Detection area in hectares.
        buffers_checked: Number of buffers checked.
        violations_found: Number of buffer violations.
        violations: List of BufferViolation objects.
        nearest_buffer_distance_km: Distance to nearest buffer center.
        nearest_buffer_id: ID of nearest buffer.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Check timestamp.
    """

    detection_lat: Decimal = Decimal("0")
    detection_lon: Decimal = Decimal("0")
    detection_area_ha: Decimal = Decimal("0")
    buffers_checked: int = 0
    violations_found: int = 0
    violations: List[BufferViolation] = field(default_factory=list)
    nearest_buffer_distance_km: Decimal = Decimal("-1")
    nearest_buffer_id: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "detection_lat": str(self.detection_lat),
            "detection_lon": str(self.detection_lon),
            "detection_area_ha": str(self.detection_area_ha),
            "buffers_checked": self.buffers_checked,
            "violations_found": self.violations_found,
            "violation_ids": [v.violation_id for v in self.violations],
            "nearest_buffer_distance_km": str(self.nearest_buffer_distance_km),
            "nearest_buffer_id": self.nearest_buffer_id,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class ViolationsResult:
    """List of buffer violations with metadata.

    Attributes:
        violations: List of BufferViolation objects.
        total: Total count of violations.
        buffer_id_filter: Buffer ID used for filtering (if any).
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    violations: List[BufferViolation] = field(default_factory=list)
    total: int = 0
    buffer_id_filter: str = ""
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total": self.total,
            "buffer_id_filter": self.buffer_id_filter,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class ZonesResult:
    """List of buffer zones with metadata.

    Attributes:
        zones: List of BufferZone objects.
        total: Total count of zones.
        active_count: Count of active zones.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    zones: List[BufferZone] = field(default_factory=list)
    total: int = 0
    active_count: int = 0
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total": self.total,
            "active_count": self.active_count,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


# ---------------------------------------------------------------------------
# SpatialBufferMonitor Engine
# ---------------------------------------------------------------------------


class SpatialBufferMonitor:
    """Production-grade geofencing engine for supply chain plot monitoring.

    Maintains configurable buffer zones around supply chain production
    plots and performs real-time checking of satellite detections against
    these zones. Supports circular, polygon, and adaptive buffer types
    with configurable radii from 1-50 km.

    Spatial algorithms use deterministic float/Decimal arithmetic:
    - Haversine formula for great-circle distance
    - Ray casting algorithm for point-in-polygon testing
    - Trigonometric circular buffer polygon generation
    - Simplified polygon overlap estimation

    No LLM/ML in any spatial computation path.

    Attributes:
        _config: Agent configuration from get_config().
        _tracker: ProvenanceTracker instance for audit trails.
        _buffer_store: In-memory buffer storage keyed by buffer_id.
        _violation_store: In-memory violation storage keyed by ID.
        _default_resolution: Default polygon resolution (point count).

    Example:
        >>> monitor = SpatialBufferMonitor()
        >>> result = monitor.create_buffer(
        ...     plot_id="PLOT-001",
        ...     lat=-3.12, lon=28.57, radius_km=10.0,
        ... )
        >>> assert result.buffer is not None
        >>> check = monitor.check_detection(
        ...     detection_lat=-3.15, detection_lon=28.60,
        ... )
        >>> assert check.provenance_hash != ""
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the SpatialBufferMonitor.

        Args:
            config: Optional configuration object. If None, loads from
                get_config() singleton.
        """
        self._config = config
        if self._config is None and get_config is not None:
            try:
                self._config = get_config()
            except Exception:
                logger.warning(
                    "Failed to load config via get_config(), "
                    "using hardcoded defaults"
                )
                self._config = None

        self._tracker: Optional[Any] = None
        if get_tracker is not None:
            try:
                self._tracker = get_tracker()
            except Exception:
                logger.debug("ProvenanceTracker not available")

        self._buffer_store: Dict[str, BufferZone] = {}
        self._violation_store: Dict[str, BufferViolation] = {}
        self._default_resolution = self._load_resolution()

        logger.info(
            "SpatialBufferMonitor initialized: config=%s, "
            "provenance=%s, resolution=%d",
            "loaded" if self._config else "defaults",
            "enabled" if self._tracker else "disabled",
            self._default_resolution,
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _load_resolution(self) -> int:
        """Load buffer polygon resolution from config or default.

        Returns:
            Number of points for circular buffer polygon.
        """
        if self._config and hasattr(self._config, "buffer_resolution_points"):
            return max(4, int(self._config.buffer_resolution_points))
        return DEFAULT_BUFFER_RESOLUTION

    def _get_min_buffer_km(self) -> Decimal:
        """Return minimum buffer radius from config or default.

        Returns:
            Minimum buffer radius in kilometers.
        """
        if self._config and hasattr(self._config, "min_buffer_km"):
            return _safe_decimal(
                self._config.min_buffer_km, MIN_BUFFER_RADIUS_KM
            )
        return MIN_BUFFER_RADIUS_KM

    def _get_max_buffer_km(self) -> Decimal:
        """Return maximum buffer radius from config or default.

        Returns:
            Maximum buffer radius in kilometers.
        """
        if self._config and hasattr(self._config, "max_buffer_km"):
            return _safe_decimal(
                self._config.max_buffer_km, MAX_BUFFER_RADIUS_KM
            )
        return MAX_BUFFER_RADIUS_KM

    # ------------------------------------------------------------------
    # Public API: Buffer management
    # ------------------------------------------------------------------

    def create_buffer(
        self,
        plot_id: str,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        buffer_type: str = "circular",
        commodities: Optional[List[str]] = None,
        country_code: str = "",
        polygon_points: Optional[List[Tuple[float, float]]] = None,
    ) -> BufferResult:
        """Create a new buffer zone around a supply chain plot.

        Validates parameters, generates the buffer polygon, and
        stores the zone for subsequent detection checking.

        Args:
            plot_id: Unique plot identifier.
            lat: Plot center latitude (-90 to 90).
            lon: Plot center longitude (-180 to 180).
            radius_km: Buffer radius in kilometers (1-50).
            buffer_type: Buffer type ('circular', 'polygon', 'adaptive').
            commodities: Optional list of EUDR commodities.
            country_code: ISO 3166-1 alpha-2 country code.
            polygon_points: Custom polygon points for POLYGON type.

        Returns:
            BufferResult with created BufferZone.

        Raises:
            ValueError: If parameters are invalid.
        """
        op_start = time.perf_counter()

        # Validate parameters
        self._validate_coordinates(lat, lon)
        radius_dec = _safe_decimal(radius_km, DEFAULT_BUFFER_RADIUS_KM)

        # Validate radius
        min_km = self._get_min_buffer_km()
        max_km = self._get_max_buffer_km()
        if radius_dec < min_km or radius_dec > max_km:
            raise ValueError(
                f"radius_km must be between {min_km} and {max_km}, "
                f"got {radius_dec}"
            )

        if not plot_id:
            raise ValueError("plot_id must not be empty")

        # Validate buffer type
        try:
            bt = BufferType(buffer_type)
        except ValueError:
            raise ValueError(
                f"buffer_type must be one of {[t.value for t in BufferType]}, "
                f"got '{buffer_type}'"
            )

        center_lat = _safe_decimal(lat)
        center_lon = _safe_decimal(lon)
        warnings: List[str] = []

        # Handle adaptive buffer
        if bt == BufferType.ADAPTIVE:
            radius_dec = self._calculate_adaptive_radius(
                commodities or [], country_code
            )
            warnings.append(
                f"Adaptive radius calculated: {radius_dec} km"
            )

        # Generate polygon points
        polygon_generated = False
        if bt == BufferType.POLYGON and polygon_points:
            generated_points = polygon_points
        else:
            generated_points = self._generate_circular_buffer_polygon(
                center_lat, center_lon, radius_dec, self._default_resolution
            )
            polygon_generated = True

        # Create buffer zone
        zone = BufferZone(
            plot_id=plot_id,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_dec,
            buffer_type=bt.value,
            polygon_points=generated_points,
            is_active=True,
            commodities=commodities or [],
            country_code=country_code,
        )

        # Store buffer
        self._buffer_store[zone.buffer_id] = zone

        # Update active buffer gauge
        if set_active_buffers:
            try:
                active = sum(
                    1 for b in self._buffer_store.values() if b.is_active
                )
                set_active_buffers(active)
            except Exception:
                pass

        # Record provenance
        if self._tracker:
            try:
                self._tracker.record(
                    entity_type="alert",
                    action="create",
                    entity_id=zone.buffer_id,
                    data=zone.to_dict(),
                    metadata={
                        "plot_id": plot_id,
                        "radius_km": str(radius_dec),
                        "buffer_type": bt.value,
                        "polygon_points": len(generated_points),
                    },
                )
            except Exception:
                logger.debug("Failed to record provenance for create_buffer")

        elapsed = _elapsed_ms(op_start)
        result = BufferResult(
            buffer=zone,
            polygon_generated=polygon_generated,
            processing_time_ms=elapsed,
            warnings=warnings,
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "create_buffer: buffer_id=%s, plot=%s, radius=%.1f km, "
            "type=%s, %d polygon points in %.1fms",
            zone.buffer_id[:12], plot_id, float(radius_dec),
            bt.value, len(generated_points), elapsed,
        )
        return result

    def update_buffer(
        self,
        buffer_id: str,
        radius_km: Optional[float] = None,
        is_active: Optional[bool] = None,
    ) -> BufferResult:
        """Update an existing buffer zone.

        Args:
            buffer_id: Buffer identifier to update.
            radius_km: Optional new radius in kilometers.
            is_active: Optional new active status.

        Returns:
            BufferResult with updated BufferZone.

        Raises:
            ValueError: If buffer_id is empty.
            KeyError: If buffer not found.
        """
        op_start = time.perf_counter()

        if not buffer_id:
            raise ValueError("buffer_id must not be empty")

        zone = self._buffer_store.get(buffer_id)
        if zone is None:
            raise KeyError(f"Buffer {buffer_id} not found")

        warnings: List[str] = []
        polygon_regenerated = False

        # Update radius
        if radius_km is not None:
            new_radius = _safe_decimal(radius_km)
            min_km = self._get_min_buffer_km()
            max_km = self._get_max_buffer_km()
            if new_radius < min_km or new_radius > max_km:
                raise ValueError(
                    f"radius_km must be between {min_km} and {max_km}, "
                    f"got {new_radius}"
                )
            zone.radius_km = new_radius
            # Regenerate polygon
            zone.polygon_points = self._generate_circular_buffer_polygon(
                zone.center_lat, zone.center_lon,
                new_radius, self._default_resolution,
            )
            polygon_regenerated = True
            warnings.append(
                f"Radius updated to {new_radius} km; polygon regenerated"
            )

        # Update active status
        if is_active is not None:
            zone.is_active = is_active
            warnings.append(
                f"Active status updated to {is_active}"
            )

        zone.updated_at = _utcnow().isoformat()

        # Update gauge
        if set_active_buffers:
            try:
                active = sum(
                    1 for b in self._buffer_store.values() if b.is_active
                )
                set_active_buffers(active)
            except Exception:
                pass

        elapsed = _elapsed_ms(op_start)
        result = BufferResult(
            buffer=zone,
            polygon_generated=polygon_regenerated,
            processing_time_ms=elapsed,
            warnings=warnings,
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "update_buffer: buffer_id=%s updated in %.1fms",
            buffer_id[:12], elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Detection checking
    # ------------------------------------------------------------------

    def check_detection(
        self,
        detection_lat: float,
        detection_lon: float,
        area_ha: Optional[float] = None,
        detection_id: str = "",
    ) -> CheckResult:
        """Check a detection point against all active buffer zones.

        Tests whether the detection coordinates fall within any active
        buffer zone and calculates proximity to each buffer center.

        Args:
            detection_lat: Detection latitude (-90 to 90).
            detection_lon: Detection longitude (-180 to 180).
            area_ha: Optional detection area in hectares.
            detection_id: Optional detection identifier.

        Returns:
            CheckResult with violations and proximity information.

        Raises:
            ValueError: If coordinates are out of range.
        """
        op_start = time.perf_counter()

        self._validate_coordinates(detection_lat, detection_lon)

        det_lat = _safe_decimal(detection_lat)
        det_lon = _safe_decimal(detection_lon)
        det_area = _safe_decimal(area_ha or 0)

        # Get active buffers
        active_buffers = [
            b for b in self._buffer_store.values() if b.is_active
        ]

        violations: List[BufferViolation] = []
        nearest_distance = Decimal("-1")
        nearest_buffer_id = ""

        for zone in active_buffers:
            # Calculate distance to buffer center
            distance = self._haversine_distance(
                det_lat, det_lon, zone.center_lat, zone.center_lon
            )

            # Track nearest buffer
            if nearest_distance < Decimal("0") or distance < nearest_distance:
                nearest_distance = distance
                nearest_buffer_id = zone.buffer_id

            # Check if point is inside buffer
            inside = False
            if zone.polygon_points:
                inside = self._point_in_polygon(
                    float(det_lat), float(det_lon), zone.polygon_points
                )
            else:
                inside = self._point_in_circular_buffer(
                    det_lat, det_lon,
                    zone.center_lat, zone.center_lon,
                    zone.radius_km,
                )

            if inside:
                # Calculate overlap area
                overlap_ha = self._calculate_buffer_overlap(
                    det_area, zone.radius_km
                )

                # Determine violation severity
                severity = self._classify_violation_severity(
                    distance, zone.radius_km
                )

                violation = BufferViolation(
                    buffer_id=zone.buffer_id,
                    detection_id=detection_id or _generate_id(),
                    distance_km=distance,
                    overlap_area_ha=overlap_ha,
                    inside_buffer=True,
                    violation_severity=severity.value,
                    plot_id=zone.plot_id,
                    commodity=(
                        zone.commodities[0] if zone.commodities else ""
                    ),
                )
                violation.provenance_hash = _compute_hash(
                    violation.to_dict()
                )

                violations.append(violation)
                self._violation_store[violation.violation_id] = violation

        # Record metrics
        if record_buffer_check:
            try:
                record_buffer_check(
                    result="violation" if violations else "clear",
                )
            except Exception:
                pass

        elapsed = _elapsed_ms(op_start)
        result = CheckResult(
            detection_lat=det_lat,
            detection_lon=det_lon,
            detection_area_ha=det_area,
            buffers_checked=len(active_buffers),
            violations_found=len(violations),
            violations=violations,
            nearest_buffer_distance_km=nearest_distance,
            nearest_buffer_id=nearest_buffer_id,
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.debug(
            "check_detection: (%.4f, %.4f), %d buffers checked, "
            "%d violations in %.1fms",
            detection_lat, detection_lon,
            len(active_buffers), len(violations), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Violations and zones queries
    # ------------------------------------------------------------------

    def get_violations(
        self,
        buffer_id: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> ViolationsResult:
        """Get buffer violations with optional filtering.

        Args:
            buffer_id: Optional buffer ID to filter by.
            date_range: Optional (start, end) ISO date strings.

        Returns:
            ViolationsResult with matching violations.
        """
        violations = list(self._violation_store.values())

        if buffer_id:
            violations = [
                v for v in violations if v.buffer_id == buffer_id
            ]

        if date_range and len(date_range) == 2:
            start_str, end_str = date_range
            violations = [
                v for v in violations
                if start_str <= v.violation_time[:10] <= end_str
            ]

        result = ViolationsResult(
            violations=violations,
            total=len(violations),
            buffer_id_filter=buffer_id or "",
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    def get_zones(
        self,
        country_code: Optional[str] = None,
        commodity: Optional[str] = None,
        active_only: bool = True,
    ) -> ZonesResult:
        """Get buffer zones with optional filtering.

        Args:
            country_code: Optional country code filter.
            commodity: Optional commodity filter.
            active_only: Whether to return only active zones.

        Returns:
            ZonesResult with matching buffer zones.
        """
        zones = list(self._buffer_store.values())

        if active_only:
            zones = [z for z in zones if z.is_active]

        if country_code:
            zones = [z for z in zones if z.country_code == country_code]

        if commodity:
            zones = [
                z for z in zones
                if commodity in z.commodities
            ]

        active_count = sum(1 for z in zones if z.is_active)

        result = ZonesResult(
            zones=zones,
            total=len(zones),
            active_count=active_count,
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Spatial algorithms: Haversine distance
    # ------------------------------------------------------------------

    def _haversine_distance(
        self,
        lat1: Decimal,
        lon1: Decimal,
        lat2: Decimal,
        lon2: Decimal,
    ) -> Decimal:
        """Calculate great-circle distance using the Haversine formula.

        R = 6371 km (Earth's mean radius)
        a = sin^2(delta_lat/2) + cos(lat1)*cos(lat2)*sin^2(delta_lon/2)
        d = 2 * R * arcsin(sqrt(a))

        ZERO-HALLUCINATION: Deterministic trigonometric formula.

        Args:
            lat1: First point latitude (decimal degrees).
            lon1: First point longitude (decimal degrees).
            lat2: Second point latitude (decimal degrees).
            lon2: Second point longitude (decimal degrees).

        Returns:
            Distance in kilometers as Decimal (3 decimal places).
        """
        # Convert to radians
        rlat1 = float(lat1) * RAD_PER_DEG
        rlon1 = float(lon1) * RAD_PER_DEG
        rlat2 = float(lat2) * RAD_PER_DEG
        rlon2 = float(lon2) * RAD_PER_DEG

        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
        )
        a = max(0.0, min(1.0, a))

        c = 2 * math.asin(math.sqrt(a))
        distance = float(EARTH_RADIUS_KM) * c

        return _safe_decimal(distance).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Spatial algorithms: Point-in-circular-buffer
    # ------------------------------------------------------------------

    def _point_in_circular_buffer(
        self,
        point_lat: Decimal,
        point_lon: Decimal,
        center_lat: Decimal,
        center_lon: Decimal,
        radius_km: Decimal,
    ) -> bool:
        """Check if a point falls within a circular buffer.

        Calculates Haversine distance and compares against radius.

        Args:
            point_lat: Point latitude.
            point_lon: Point longitude.
            center_lat: Buffer center latitude.
            center_lon: Buffer center longitude.
            radius_km: Buffer radius in kilometers.

        Returns:
            True if the point is within the circular buffer.
        """
        distance = self._haversine_distance(
            point_lat, point_lon, center_lat, center_lon
        )
        return distance <= radius_km

    # ------------------------------------------------------------------
    # Spatial algorithms: Point-in-polygon (ray casting)
    # ------------------------------------------------------------------

    def _point_in_polygon(
        self,
        point_lat: float,
        point_lon: float,
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """Test if a point is inside a polygon using ray casting.

        Casts a horizontal ray from the test point to the right and
        counts the number of polygon edge crossings. An odd count
        means the point is inside; even means outside.

        ZERO-HALLUCINATION: Deterministic geometric algorithm.

        Args:
            point_lat: Point latitude (y coordinate).
            point_lon: Point longitude (x coordinate).
            polygon: List of (lat, lon) tuples defining the polygon
                vertices in order. The polygon is automatically closed.

        Returns:
            True if the point is inside the polygon.
        """
        if len(polygon) < 3:
            return False

        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            yi, xi = polygon[i]
            yj, xj = polygon[j]

            # Check if the ray crosses this edge
            if ((yi > point_lat) != (yj > point_lat)) and \
               (point_lon < (xj - xi) * (point_lat - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    # ------------------------------------------------------------------
    # Spatial algorithms: Circular buffer polygon generation
    # ------------------------------------------------------------------

    def _generate_circular_buffer_polygon(
        self,
        center_lat: Decimal,
        center_lon: Decimal,
        radius_km: Decimal,
        num_points: int = 64,
    ) -> List[Tuple[float, float]]:
        """Generate a polygon approximation of a circular buffer.

        Distributes num_points evenly around the circle at the given
        radius, accounting for latitude-dependent longitude scaling.

        ZERO-HALLUCINATION: Deterministic trigonometric generation.

        Args:
            center_lat: Circle center latitude.
            center_lon: Circle center longitude.
            radius_km: Circle radius in kilometers.
            num_points: Number of polygon vertices (default 64).

        Returns:
            List of (lat, lon) tuples forming the polygon boundary.
        """
        num_points = max(4, num_points)
        points: List[Tuple[float, float]] = []

        c_lat = float(center_lat)
        c_lon = float(center_lon)
        r_km = float(radius_km)

        # Radius in degrees (approximate)
        lat_deg_per_km = 1.0 / 111.32
        lon_deg_per_km = 1.0 / (111.32 * math.cos(c_lat * RAD_PER_DEG))

        # Prevent division by zero at poles
        if abs(math.cos(c_lat * RAD_PER_DEG)) < 0.0001:
            lon_deg_per_km = lat_deg_per_km

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dlat = r_km * lat_deg_per_km * math.sin(angle)
            dlon = r_km * lon_deg_per_km * math.cos(angle)

            p_lat = c_lat + dlat
            p_lon = c_lon + dlon

            # Clamp to valid ranges
            p_lat = max(-90.0, min(90.0, p_lat))
            p_lon = max(-180.0, min(180.0, p_lon))

            points.append((
                round(p_lat, 6),
                round(p_lon, 6),
            ))

        return points

    # ------------------------------------------------------------------
    # Spatial algorithms: Buffer overlap calculation
    # ------------------------------------------------------------------

    def _calculate_buffer_overlap(
        self,
        detection_area_ha: Decimal,
        buffer_radius_km: Decimal,
    ) -> Decimal:
        """Calculate estimated overlap between detection and buffer.

        Uses a simplified estimation based on the detection area and
        buffer radius. Production systems would use actual polygon
        intersection via spatial libraries.

        Args:
            detection_area_ha: Detection event area in hectares.
            buffer_radius_km: Buffer zone radius in kilometers.

        Returns:
            Estimated overlap area in hectares.
        """
        if detection_area_ha <= Decimal("0"):
            return Decimal("0")

        # Buffer area in hectares (pi * r^2 km^2 * 100 ha/km^2)
        buffer_area_ha = (
            Decimal("3.141593") * buffer_radius_km * buffer_radius_km
            * Decimal("100")
        )

        # Overlap is bounded by the smaller of detection and buffer areas
        overlap = min(detection_area_ha, buffer_area_ha)

        # Scale by estimated intersection fraction (conservative: 50%)
        overlap = (overlap * Decimal("0.5")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return max(Decimal("0"), overlap)

    # ------------------------------------------------------------------
    # Internal: Adaptive buffer radius
    # ------------------------------------------------------------------

    def _calculate_adaptive_radius(
        self,
        commodities: List[str],
        country_code: str,
    ) -> Decimal:
        """Calculate adaptive buffer radius from commodity and country risk.

        Combines commodity-specific base radius with country risk
        multiplier for proportionate monitoring intensity per
        EUDR Article 11.

        Args:
            commodities: List of EUDR commodities at the plot.
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Adaptive radius in kilometers as Decimal.
        """
        # Base radius from commodity (use largest if multiple)
        base_radius = DEFAULT_BUFFER_RADIUS_KM
        if commodities:
            commodity_radii = [
                COMMODITY_BUFFER_RADII.get(c, DEFAULT_BUFFER_RADIUS_KM)
                for c in commodities
            ]
            base_radius = max(commodity_radii)

        # Country risk multiplier
        multiplier = COUNTRY_RISK_MULTIPLIERS.get(
            country_code,
            COUNTRY_RISK_MULTIPLIERS["_DEFAULT"],
        )

        # Calculate adapted radius
        adapted = (base_radius * multiplier).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        # Clamp to valid range
        min_km = self._get_min_buffer_km()
        max_km = self._get_max_buffer_km()
        adapted = max(min_km, min(max_km, adapted))

        logger.debug(
            "_calculate_adaptive_radius: commodities=%s, country=%s, "
            "base=%.1f, multiplier=%.1f, adapted=%.1f km",
            commodities, country_code,
            float(base_radius), float(multiplier), float(adapted),
        )
        return adapted

    # ------------------------------------------------------------------
    # Internal: Violation severity classification
    # ------------------------------------------------------------------

    def _classify_violation_severity(
        self,
        distance_km: Decimal,
        buffer_radius_km: Decimal,
    ) -> ViolationSeverity:
        """Classify violation severity from distance and radius.

        Args:
            distance_km: Distance from detection to buffer center.
            buffer_radius_km: Buffer radius in kilometers.

        Returns:
            ViolationSeverity level.
        """
        if buffer_radius_km <= Decimal("0"):
            return ViolationSeverity.MEDIUM

        # Ratio of distance to radius (0 = at center, 1 = at edge)
        ratio = distance_km / buffer_radius_km

        if ratio <= Decimal("0.25"):
            return ViolationSeverity.CRITICAL
        elif ratio <= Decimal("0.5"):
            return ViolationSeverity.HIGH
        elif ratio <= Decimal("0.75"):
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_coordinates(
        self, lat: float, lon: float,
    ) -> None:
        """Validate geographic coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Raises:
            ValueError: If coordinates are out of range.
        """
        if not (-90 <= lat <= 90):
            raise ValueError(
                f"Latitude must be between -90 and 90, got {lat}"
            )
        if not (-180 <= lon <= 180):
            raise ValueError(
                f"Longitude must be between -180 and 180, got {lon}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "BufferType",
    "ViolationSeverity",
    # Constants
    "EARTH_RADIUS_KM",
    "MIN_BUFFER_RADIUS_KM",
    "MAX_BUFFER_RADIUS_KM",
    "DEFAULT_BUFFER_RADIUS_KM",
    "DEFAULT_BUFFER_RESOLUTION",
    "COMMODITY_BUFFER_RADII",
    "COUNTRY_RISK_MULTIPLIERS",
    # Data classes
    "GeoPoint",
    "BufferZone",
    "BufferViolation",
    "BufferResult",
    "CheckResult",
    "ViolationsResult",
    "ZonesResult",
    # Engine class
    "SpatialBufferMonitor",
]
