# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-006 Plot Boundary Manager test suite.

Provides reusable fixtures for configuration objects, engine instances,
sample coordinates, sample polygons (valid and invalid), boundary objects,
overlap pairs, and shared helper functions used across all test modules.

Fixture count: 40+ fixtures
Helper count: 8 utility functions
Dataclass count: 2 (SampleCoordinate, SamplePolygon)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Provenance hash helper
# ---------------------------------------------------------------------------


def compute_test_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH = 64

EUDR_DEFORESTATION_CUTOFF = "2020-12-31"
EUDR_CUTOFF_DATE = date(2020, 12, 31)

EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

# EUDR Article 9 area threshold: 4 hectares
EUDR_AREA_THRESHOLD_HA = 4.0

# Supported CRS identifiers
SUPPORTED_CRS = [
    "EPSG:4326",   # WGS84
    "EPSG:3857",   # Web Mercator
    "EPSG:32721",  # UTM zone 21S (Brazil)
    "EPSG:32748",  # UTM zone 48S (Indonesia)
    "EPSG:32630",  # UTM zone 30N (Ghana)
    "EPSG:4674",   # SIRGAS 2000 (Brazil)
]

# GeoJSON geometry types
GEOJSON_TYPES = ["Polygon", "MultiPolygon", "Point"]

# Export formats
EXPORT_FORMATS = [
    "geojson", "wkt", "wkb", "kml", "gpx", "gml", "shapefile", "eudr_xml",
]

# Overlap severity levels
OVERLAP_SEVERITIES = ["none", "minor", "moderate", "major", "critical"]

# Simplification algorithms
SIMPLIFICATION_ALGORITHMS = ["douglas_peucker", "visvalingam_whyatt"]

# Resolution levels
RESOLUTION_LEVELS = ["high", "medium", "low", "ultra_low"]

# Validation error types
VALIDATION_ERROR_TYPES = [
    "self_intersection",
    "unclosed_ring",
    "duplicate_vertices",
    "spike_vertex",
    "sliver_polygon",
    "wrong_orientation",
    "invalid_coordinates",
    "too_few_vertices",
    "hole_outside_shell",
    "overlapping_holes",
    "nested_shells",
    "zero_area",
]

# Countries with EUDR relevance
EUDR_COUNTRIES = {
    "BR": "Brazil",
    "ID": "Indonesia",
    "GH": "Ghana",
    "CI": "Cote d'Ivoire",
    "CO": "Colombia",
    "MY": "Malaysia",
    "PY": "Paraguay",
    "CM": "Cameroon",
    "NG": "Nigeria",
    "PE": "Peru",
}


# ---------------------------------------------------------------------------
# Test-only Dataclass Models
# ---------------------------------------------------------------------------


@dataclass
class SampleCoordinate:
    """Test-only sample coordinate with metadata."""

    lat: float = 0.0
    lon: float = 0.0
    altitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    description: str = ""


@dataclass
class SamplePolygon:
    """Test-only sample polygon definition with expected properties."""

    name: str = ""
    coordinates: List[List[Tuple[float, float]]] = field(default_factory=list)
    commodity: str = "cocoa"
    country: str = "BR"
    expected_area_ha: Optional[float] = None
    is_valid: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Test-only Boundary / Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Coordinate:
    """Test-only coordinate model."""

    lat: float = 0.0
    lon: float = 0.0
    altitude: Optional[float] = None


@dataclass
class BoundingBox:
    """Test-only bounding box."""

    min_lat: float = 0.0
    max_lat: float = 0.0
    min_lon: float = 0.0
    max_lon: float = 0.0

    @property
    def width(self) -> float:
        return self.max_lon - self.min_lon

    @property
    def height(self) -> float:
        return self.max_lat - self.min_lat

    def contains_point(self, lat: float, lon: float) -> bool:
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            self.max_lat < other.min_lat
            or self.min_lat > other.max_lat
            or self.max_lon < other.min_lon
            or self.min_lon > other.max_lon
        )

    @property
    def area_degrees_sq(self) -> float:
        return self.width * self.height


@dataclass
class Ring:
    """Test-only polygon ring."""

    coords: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        if len(self.coords) < 2:
            return False
        return self.coords[0] == self.coords[-1]

    @property
    def is_ccw(self) -> bool:
        """Check counter-clockwise orientation using shoelace formula sign."""
        return self.signed_area > 0

    @property
    def signed_area(self) -> float:
        """Compute signed area (positive = CCW, negative = CW)."""
        n = len(self.coords)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.coords[i][0] * self.coords[j][1]
            area -= self.coords[j][0] * self.coords[i][1]
        return area / 2.0

    @property
    def vertex_count(self) -> int:
        return len(self.coords)


@dataclass
class PlotBoundary:
    """Test-only plot boundary model."""

    plot_id: str = ""
    exterior_ring: List[Tuple[float, float]] = field(default_factory=list)
    interior_rings: List[List[Tuple[float, float]]] = field(default_factory=list)
    commodity: str = ""
    country: str = ""
    declared_area_ha: Optional[float] = None
    calculated_area_ha: Optional[float] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    bbox: Optional[BoundingBox] = None
    crs: str = "EPSG:4326"
    vertex_count: int = 0
    is_valid: bool = True
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.plot_id:
            self.plot_id = f"PLOT-{uuid.uuid4().hex[:8].upper()}"
        if not self.vertex_count and self.exterior_ring:
            self.vertex_count = len(self.exterior_ring)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class ValidationResult:
    """Test-only validation result."""

    is_valid: bool = True
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    repairs_applied: List[Dict[str, Any]] = field(default_factory=list)
    repaired_exterior: Optional[List[Tuple[float, float]]] = None
    confidence: float = 1.0
    ogc_compliant: bool = True

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def total_issue_count(self) -> int:
        return self.error_count + self.warning_count


@dataclass
class AreaResult:
    """Test-only area calculation result."""

    area_m2: float = 0.0
    area_ha: float = 0.0
    area_acres: float = 0.0
    area_km2: float = 0.0
    perimeter_m: float = 0.0
    method: str = "karney"
    compactness_polsby_popper: float = 0.0
    compactness_schwartzberg: float = 0.0
    convex_hull_ratio: float = 0.0
    uncertainty_m2: float = 0.0
    requires_polygon: bool = False
    provenance_hash: str = ""

    def __post_init__(self):
        if self.area_m2 > 0 and self.area_ha == 0:
            self.area_ha = self.area_m2 / 10000.0
        if self.area_m2 > 0 and self.area_acres == 0:
            self.area_acres = self.area_m2 / 4046.8564224
        if self.area_m2 > 0 and self.area_km2 == 0:
            self.area_km2 = self.area_m2 / 1_000_000.0
        self.requires_polygon = self.area_ha >= EUDR_AREA_THRESHOLD_HA


@dataclass
class OverlapRecord:
    """Test-only overlap record between two plots."""

    plot_a_id: str = ""
    plot_b_id: str = ""
    overlap_area_m2: float = 0.0
    overlap_area_ha: float = 0.0
    overlap_pct_a: float = 0.0
    overlap_pct_b: float = 0.0
    severity: str = "none"
    intersection_geometry: Optional[List[Tuple[float, float]]] = None
    resolution_suggestion: str = ""
    detected_at: str = ""

    def __post_init__(self):
        if self.overlap_area_m2 > 0 and self.overlap_area_ha == 0:
            self.overlap_area_ha = self.overlap_area_m2 / 10000.0
        if not self.severity or self.severity == "none":
            max_pct = max(self.overlap_pct_a, self.overlap_pct_b)
            if max_pct <= 0:
                self.severity = "none"
            elif max_pct < 1.0:
                self.severity = "minor"
            elif max_pct < 10.0:
                self.severity = "moderate"
            elif max_pct < 50.0:
                self.severity = "major"
            else:
                self.severity = "critical"


@dataclass
class BoundaryVersion:
    """Test-only boundary version tracking."""

    plot_id: str = ""
    version: int = 1
    exterior_ring: List[Tuple[float, float]] = field(default_factory=list)
    area_ha: float = 0.0
    modified_at: str = ""
    modified_by: str = ""
    change_type: str = "created"
    provenance_hash: str = ""

    def compute_hash(self) -> str:
        data = {
            "plot_id": self.plot_id,
            "version": self.version,
            "exterior_ring": self.exterior_ring,
            "area_ha": self.area_ha,
        }
        return compute_test_hash(data)


@dataclass
class SimplificationResult:
    """Test-only simplification result."""

    original_vertex_count: int = 0
    simplified_vertex_count: int = 0
    reduction_ratio: float = 0.0
    area_deviation_pct: float = 0.0
    hausdorff_distance_m: float = 0.0
    algorithm: str = "douglas_peucker"
    tolerance: float = 0.0
    topology_preserved: bool = True
    simplified_coords: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def quality_ok(self) -> bool:
        return self.area_deviation_pct < 1.0 and self.topology_preserved


@dataclass
class SplitResult:
    """Test-only split operation result."""

    parent_id: str = ""
    child_ids: List[str] = field(default_factory=list)
    child_boundaries: List[List[Tuple[float, float]]] = field(default_factory=list)
    child_areas_ha: List[float] = field(default_factory=list)
    parent_area_ha: float = 0.0
    area_conservation_ok: bool = True
    cutting_line: Optional[List[Tuple[float, float]]] = None

    @property
    def area_sum(self) -> float:
        return sum(self.child_areas_ha)


@dataclass
class MergeResult:
    """Test-only merge operation result."""

    parent_ids: List[str] = field(default_factory=list)
    child_id: str = ""
    child_boundary: List[Tuple[float, float]] = field(default_factory=list)
    child_area_ha: float = 0.0
    parent_areas_sum_ha: float = 0.0
    area_conservation_ok: bool = True

    @property
    def area_difference(self) -> float:
        return abs(self.child_area_ha - self.parent_areas_sum_ha)


@dataclass
class ExportResult:
    """Test-only export result."""

    format: str = ""
    content: str = ""
    content_bytes: Optional[bytes] = None
    boundary_count: int = 0
    file_size_bytes: int = 0
    is_valid: bool = True
    coordinate_precision: int = 7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlotBoundaryConfig:
    """Test-only configuration for Plot Boundary Manager."""

    database_url: str = "postgresql://localhost:5432/greenlang_test"
    redis_url: str = "redis://localhost:6379/1"
    log_level: str = "DEBUG"
    default_crs: str = "EPSG:4326"
    max_vertices: int = 100_000
    min_vertices: int = 4
    area_tolerance_pct: float = 10.0
    sliver_ratio_threshold: float = 0.001
    spike_angle_threshold_degrees: float = 1.0
    duplicate_vertex_tolerance_m: float = 0.01
    simplification_area_max_deviation_pct: float = 1.0
    overlap_minor_threshold_pct: float = 1.0
    overlap_moderate_threshold_pct: float = 10.0
    overlap_major_threshold_pct: float = 50.0
    max_batch_size: int = 10_000
    coordinate_precision: int = 7
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-PBM-006-TEST-GENESIS"
    enable_metrics: bool = False
    pool_size: int = 5
    eudr_area_threshold_ha: float = 4.0
    max_polygon_area_ha: float = 50_000.0
    split_area_tolerance_pct: float = 0.5
    merge_gap_tolerance_m: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary, redacting credentials."""
        d = {}
        for k, v in self.__dict__.items():
            if "url" in k or "password" in k or "secret" in k:
                d[k] = "***REDACTED***"
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Predefined Sample Coordinates (10+)
# ---------------------------------------------------------------------------


COORD_BRAZIL_AMAZON = SampleCoordinate(
    lat=-3.1234567, lon=-60.0234567,
    altitude=80.0, accuracy_m=3.5,
    description="Para, Brazil - Amazon region",
)

COORD_INDONESIA_KALIMANTAN = SampleCoordinate(
    lat=-2.5678901, lon=111.7654321,
    altitude=25.0, accuracy_m=5.0,
    description="Kalimantan, Indonesia - palm oil region",
)

COORD_GHANA_ASHANTI = SampleCoordinate(
    lat=6.1234567, lon=-1.6234567,
    altitude=250.0, accuracy_m=4.0,
    description="Ashanti, Ghana - cocoa region",
)

COORD_COLOMBIA_CAQUETA = SampleCoordinate(
    lat=1.6139, lon=-75.6062,
    altitude=230.0, accuracy_m=6.0,
    description="Caqueta, Colombia - cattle region",
)

COORD_MALAYSIA_SABAH = SampleCoordinate(
    lat=5.9804, lon=116.0735,
    altitude=15.0, accuracy_m=3.0,
    description="Sabah, Malaysia - palm oil region",
)

COORD_COTE_DIVOIRE_SOUBRE = SampleCoordinate(
    lat=5.7857, lon=-6.5932,
    altitude=200.0, accuracy_m=7.0,
    description="Soubre, Cote d'Ivoire - cocoa region",
)

COORD_PARAGUAY_CHACO = SampleCoordinate(
    lat=-22.3500, lon=-59.9500,
    altitude=130.0, accuracy_m=8.0,
    description="Chaco, Paraguay - soya region",
)

COORD_CAMEROON_SOUTHWEST = SampleCoordinate(
    lat=4.9700, lon=9.1500,
    altitude=400.0, accuracy_m=5.0,
    description="Southwest, Cameroon - cocoa region",
)

COORD_EQUATOR_PRIMEMERIDIAN = SampleCoordinate(
    lat=0.0, lon=0.0,
    altitude=0.0, accuracy_m=10.0,
    description="Equator/Prime Meridian - null island",
)

COORD_HIGH_LATITUDE = SampleCoordinate(
    lat=81.5, lon=25.0,
    altitude=5.0, accuracy_m=15.0,
    description="Svalbard - high latitude test",
)

COORD_ANTI_MERIDIAN = SampleCoordinate(
    lat=0.0, lon=179.9999,
    altitude=0.0, accuracy_m=5.0,
    description="Near anti-meridian - date line test",
)

ALL_SAMPLE_COORDINATES = [
    COORD_BRAZIL_AMAZON,
    COORD_INDONESIA_KALIMANTAN,
    COORD_GHANA_ASHANTI,
    COORD_COLOMBIA_CAQUETA,
    COORD_MALAYSIA_SABAH,
    COORD_COTE_DIVOIRE_SOUBRE,
    COORD_PARAGUAY_CHACO,
    COORD_CAMEROON_SOUTHWEST,
    COORD_EQUATOR_PRIMEMERIDIAN,
    COORD_HIGH_LATITUDE,
    COORD_ANTI_MERIDIAN,
]


# ---------------------------------------------------------------------------
# Predefined Sample Polygons (15+)
# ---------------------------------------------------------------------------


def _make_square_coords(
    center_lat: float,
    center_lon: float,
    half_side_deg: float,
) -> List[Tuple[float, float]]:
    """Generate a closed square polygon (CCW) from center and half-side size."""
    sw = (center_lat - half_side_deg, center_lon - half_side_deg)
    se = (center_lat - half_side_deg, center_lon + half_side_deg)
    ne = (center_lat + half_side_deg, center_lon + half_side_deg)
    nw = (center_lat + half_side_deg, center_lon - half_side_deg)
    return [sw, se, ne, nw, sw]


def _make_circle_coords(
    center_lat: float,
    center_lon: float,
    radius_deg: float,
    n_points: int = 36,
) -> List[Tuple[float, float]]:
    """Generate a closed circular polygon from center and radius."""
    coords = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        lat = center_lat + radius_deg * math.cos(angle)
        lon = center_lon + radius_deg * math.sin(angle)
        coords.append((round(lat, 7), round(lon, 7)))
    coords.append(coords[0])
    return coords


# ~1 km x 1 km square in Brazil (~100 ha at equator, ~0.009 degrees)
SIMPLE_SQUARE = SamplePolygon(
    name="SIMPLE_SQUARE",
    coordinates=[_make_square_coords(-3.12, -60.02, 0.0045)],
    commodity="cocoa",
    country="BR",
    expected_area_ha=100.0,
    is_valid=True,
    description="1km x 1km square in Brazil, approx 100 ha",
)

# 500 hectare palm oil plantation in Indonesia
# ~sqrt(5_000_000 / 10000) ~ 22.36 ha side => ~0.0224 degrees at equator
LARGE_PLANTATION = SamplePolygon(
    name="LARGE_PLANTATION",
    coordinates=[_make_square_coords(-2.57, 111.77, 0.0335)],
    commodity="oil_palm",
    country="ID",
    expected_area_ha=500.0,
    is_valid=True,
    description="500 hectare palm oil plantation in Indonesia",
)

# 2 hectare cocoa farm in Ghana (below 4ha threshold)
SMALL_FARM = SamplePolygon(
    name="SMALL_FARM",
    coordinates=[_make_square_coords(6.12, -1.62, 0.0021)],
    commodity="cocoa",
    country="GH",
    expected_area_ha=2.0,
    is_valid=True,
    description="2 hectare cocoa farm in Ghana (< 4ha threshold)",
)

# Complex irregular boundary with 50+ vertices
_irregular_center = (-3.15, -60.05)
_irregular_coords = []
for _i in range(50):
    _angle = 2 * math.pi * _i / 50
    _r = 0.005 + 0.002 * math.sin(3 * _angle)
    _lat = _irregular_center[0] + _r * math.cos(_angle)
    _lon = _irregular_center[1] + _r * math.sin(_angle)
    _irregular_coords.append((round(_lat, 7), round(_lon, 7)))
_irregular_coords.append(_irregular_coords[0])

IRREGULAR_SHAPE = SamplePolygon(
    name="IRREGULAR_SHAPE",
    coordinates=[_irregular_coords],
    commodity="cocoa",
    country="BR",
    expected_area_ha=80.0,
    is_valid=True,
    description="Complex irregular boundary with 50+ vertices",
)

# Polygon with 2 interior holes (excluded areas)
_shell_with_holes = _make_square_coords(-3.12, -60.02, 0.01)
_hole1 = list(reversed(_make_square_coords(-3.117, -60.017, 0.001)))
_hole2 = list(reversed(_make_square_coords(-3.123, -60.023, 0.001)))

WITH_HOLES = SamplePolygon(
    name="WITH_HOLES",
    coordinates=[_shell_with_holes, _hole1, _hole2],
    commodity="cocoa",
    country="BR",
    expected_area_ha=380.0,
    is_valid=True,
    description="Polygon with 2 interior holes (excluded areas)",
)

# Multi-polygon: 2 separate non-contiguous areas
_multi_part1 = _make_square_coords(-3.10, -60.00, 0.003)
_multi_part2 = _make_square_coords(-3.15, -60.10, 0.003)

MULTI_POLYGON = SamplePolygon(
    name="MULTI_POLYGON",
    coordinates=[_multi_part1, _multi_part2],
    commodity="cocoa",
    country="BR",
    expected_area_ha=70.0,
    is_valid=True,
    description="Non-contiguous plot with 2 separate areas",
)

# Anti-meridian crossing polygon
ANTI_MERIDIAN = SamplePolygon(
    name="ANTI_MERIDIAN",
    coordinates=[[
        (0.0, 179.995),
        (0.0, -179.995),
        (0.005, -179.995),
        (0.005, 179.995),
        (0.0, 179.995),
    ]],
    commodity="wood",
    country="FJ",
    expected_area_ha=6.0,
    is_valid=True,
    description="Polygon crossing the 180th meridian near Fiji",
)

# Near-pole polygon (> 80 degrees latitude)
NEAR_POLE = SamplePolygon(
    name="NEAR_POLE",
    coordinates=[_make_square_coords(81.5, 25.0, 0.005)],
    commodity="wood",
    country="NO",
    expected_area_ha=5.0,
    is_valid=True,
    description="High latitude polygon (> 80 degrees) near Svalbard",
)

# Very large 10,000 hectare cattle ranch
VERY_LARGE = SamplePolygon(
    name="VERY_LARGE",
    coordinates=[_make_square_coords(-22.35, -59.95, 0.15)],
    commodity="cattle",
    country="PY",
    expected_area_ha=10000.0,
    is_valid=True,
    description="10,000 hectare cattle ranch in Paraguay",
)

# Tiny 0.01 hectare garden plot
TINY_PLOT = SamplePolygon(
    name="TINY_PLOT",
    coordinates=[_make_square_coords(6.12, -1.62, 0.00005)],
    commodity="cocoa",
    country="GH",
    expected_area_ha=0.01,
    is_valid=True,
    description="0.01 hectare garden plot in Ghana",
)

# Self-intersecting bowtie polygon (INVALID)
SELF_INTERSECTING = SamplePolygon(
    name="SELF_INTERSECTING",
    coordinates=[[
        (-3.12, -60.02),
        (-3.12, -60.01),
        (-3.13, -60.02),
        (-3.13, -60.01),
        (-3.12, -60.02),
    ]],
    commodity="cocoa",
    country="BR",
    expected_area_ha=None,
    is_valid=False,
    description="Bowtie polygon with self-intersecting edges (invalid)",
)

# Unclosed ring (INVALID)
UNCLOSED = SamplePolygon(
    name="UNCLOSED",
    coordinates=[[
        (-3.12, -60.02),
        (-3.12, -60.01),
        (-3.13, -60.015),
        # missing closure vertex
    ]],
    commodity="cocoa",
    country="BR",
    expected_area_ha=None,
    is_valid=False,
    description="Ring not closed - first != last vertex (invalid)",
)

# Duplicate consecutive vertices (INVALID)
DUPLICATE_VERTICES = SamplePolygon(
    name="DUPLICATE_VERTICES",
    coordinates=[[
        (-3.12, -60.02),
        (-3.12, -60.02),  # duplicate
        (-3.12, -60.01),
        (-3.13, -60.015),
        (-3.12, -60.02),
    ]],
    commodity="cocoa",
    country="BR",
    expected_area_ha=None,
    is_valid=False,
    description="Consecutive duplicate vertices (invalid)",
)

# Spike polygon with narrow protrusion (INVALID)
SPIKE_POLYGON = SamplePolygon(
    name="SPIKE_POLYGON",
    coordinates=[[
        (-3.12, -60.02),
        (-3.12, -60.01),
        (-3.125, -60.015),
        (-3.200, -60.0150001),  # extreme spike
        (-3.125, -60.0150002),
        (-3.13, -60.02),
        (-3.12, -60.02),
    ]],
    commodity="cocoa",
    country="BR",
    expected_area_ha=None,
    is_valid=False,
    description="Polygon with a narrow spike protrusion (invalid)",
)

# Zero-area degenerate polygon (INVALID)
ZERO_AREA = SamplePolygon(
    name="ZERO_AREA",
    coordinates=[[
        (-3.12, -60.02),
        (-3.13, -60.03),
        (-3.12, -60.02),
    ]],
    commodity="cocoa",
    country="BR",
    expected_area_ha=0.0,
    is_valid=False,
    description="Degenerate polygon collapsing to a line (invalid)",
)

ALL_SAMPLE_POLYGONS = [
    SIMPLE_SQUARE,
    LARGE_PLANTATION,
    SMALL_FARM,
    IRREGULAR_SHAPE,
    WITH_HOLES,
    MULTI_POLYGON,
    ANTI_MERIDIAN,
    NEAR_POLE,
    VERY_LARGE,
    TINY_PLOT,
    SELF_INTERSECTING,
    UNCLOSED,
    DUPLICATE_VERTICES,
    SPIKE_POLYGON,
    ZERO_AREA,
]

VALID_POLYGONS = [p for p in ALL_SAMPLE_POLYGONS if p.is_valid]
INVALID_POLYGONS = [p for p in ALL_SAMPLE_POLYGONS if not p.is_valid]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def make_boundary(
    coords: List[Tuple[float, float]],
    commodity: str = "cocoa",
    country: str = "BR",
    plot_id: Optional[str] = None,
) -> PlotBoundary:
    """Quick boundary creation from exterior ring coordinates."""
    pid = plot_id or f"PLOT-{uuid.uuid4().hex[:8].upper()}"
    n = len(coords)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    centroid_lat = sum(lats) / n if n > 0 else 0.0
    centroid_lon = sum(lons) / n if n > 0 else 0.0
    bbox = BoundingBox(
        min_lat=min(lats) if lats else 0.0,
        max_lat=max(lats) if lats else 0.0,
        min_lon=min(lons) if lons else 0.0,
        max_lon=max(lons) if lons else 0.0,
    )
    return PlotBoundary(
        plot_id=pid,
        exterior_ring=coords,
        commodity=commodity,
        country=country,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        bbox=bbox,
        vertex_count=n,
    )


def make_ring(coords: List[Tuple[float, float]]) -> Ring:
    """Quick ring creation from coordinate tuples."""
    return Ring(coords=coords)


def make_square(
    center_lat: float,
    center_lon: float,
    size_degrees: float,
) -> List[Tuple[float, float]]:
    """Generate a closed square polygon from center and size."""
    half = size_degrees / 2.0
    return _make_square_coords(center_lat, center_lon, half)


def make_circle(
    center_lat: float,
    center_lon: float,
    radius_degrees: float,
    n_points: int = 36,
) -> List[Tuple[float, float]]:
    """Generate a closed circular polygon from center and radius."""
    return _make_circle_coords(center_lat, center_lon, radius_degrees, n_points)


def assert_valid_boundary(boundary: PlotBoundary) -> None:
    """Assert that a boundary passes basic validation checks."""
    assert boundary.plot_id, "Boundary must have a plot_id"
    assert len(boundary.exterior_ring) >= 4, "Exterior ring must have >= 4 points"
    assert boundary.exterior_ring[0] == boundary.exterior_ring[-1], (
        "Exterior ring must be closed"
    )
    assert boundary.commodity, "Boundary must have a commodity"
    assert boundary.country, "Boundary must have a country code"
    assert boundary.vertex_count >= 4, "Vertex count must be >= 4"


def assert_area_close(
    actual: float,
    expected: float,
    tolerance: float = 0.01,
) -> None:
    """Assert that an area value is within tolerance of expected."""
    if expected == 0:
        assert abs(actual) < tolerance, (
            f"Expected area ~0, got {actual}"
        )
    else:
        relative_error = abs(actual - expected) / expected
        assert relative_error <= tolerance, (
            f"Area {actual} differs from expected {expected} "
            f"by {relative_error:.4f} (tolerance: {tolerance})"
        )


def assert_overlap_found(
    overlaps: List[OverlapRecord],
    plot_a: str,
    plot_b: str,
) -> None:
    """Assert that a specific overlap exists between two plots."""
    found = any(
        (o.plot_a_id == plot_a and o.plot_b_id == plot_b)
        or (o.plot_a_id == plot_b and o.plot_b_id == plot_a)
        for o in overlaps
    )
    assert found, (
        f"Expected overlap between {plot_a} and {plot_b} not found "
        f"in {len(overlaps)} overlap records"
    )


def geodesic_area_simple(coords: List[Tuple[float, float]]) -> float:
    """Simple geodesic area approximation in hectares using spherical excess.

    Uses the shoelace formula with a cos(latitude) correction. This is
    deterministic and suitable for test assertions on small polygons.
    Not accurate for very large polygons or polar regions.
    """
    n = len(coords)
    if n < 3:
        return 0.0
    R = 6371000.0  # Earth radius in meters
    area_rad = 0.0
    for i in range(n):
        j = (i + 1) % n
        lat1 = math.radians(coords[i][0])
        lon1 = math.radians(coords[i][1])
        lat2 = math.radians(coords[j][0])
        lon2 = math.radians(coords[j][1])
        area_rad += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    area_m2 = abs(area_rad * R * R / 2.0)
    return area_m2 / 10000.0  # convert to hectares


# ---------------------------------------------------------------------------
# Stub classes for engine instantiation
# ---------------------------------------------------------------------------


class PolygonManager:
    """Stub PolygonManager for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config
        self._boundaries: Dict[str, PlotBoundary] = {}

    def create_boundary(self, boundary: PlotBoundary) -> PlotBoundary:
        self._boundaries[boundary.plot_id] = boundary
        return boundary

    def get_boundary(self, plot_id: str) -> Optional[PlotBoundary]:
        return self._boundaries.get(plot_id)

    def update_boundary(self, boundary: PlotBoundary) -> PlotBoundary:
        self._boundaries[boundary.plot_id] = boundary
        return boundary

    def delete_boundary(self, plot_id: str) -> bool:
        return self._boundaries.pop(plot_id, None) is not None

    def search_by_bbox(self, bbox: BoundingBox) -> List[PlotBoundary]:
        results = []
        for b in self._boundaries.values():
            if b.bbox and bbox.intersects(b.bbox):
                results.append(b)
        return results

    def search_by_commodity(self, commodity: str) -> List[PlotBoundary]:
        return [b for b in self._boundaries.values() if b.commodity == commodity]

    def search_by_country(self, country: str) -> List[PlotBoundary]:
        return [b for b in self._boundaries.values() if b.country == country]


class BoundaryValidator:
    """Stub BoundaryValidator for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config

    def validate(self, boundary: PlotBoundary) -> ValidationResult:
        result = ValidationResult()
        ring = boundary.exterior_ring
        if len(ring) < self.config.min_vertices:
            result.is_valid = False
            result.errors.append({
                "type": "too_few_vertices",
                "message": f"Need >= {self.config.min_vertices} vertices, got {len(ring)}",
            })
        if ring and ring[0] != ring[-1]:
            result.is_valid = False
            result.errors.append({
                "type": "unclosed_ring",
                "message": "First vertex != last vertex",
            })
        return result


class AreaCalculator:
    """Stub AreaCalculator for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config

    def calculate(self, coords: List[Tuple[float, float]]) -> AreaResult:
        area_ha = geodesic_area_simple(coords)
        area_m2 = area_ha * 10000.0
        perimeter = self._perimeter(coords)
        pp = 0.0
        if perimeter > 0:
            pp = (4 * math.pi * area_m2) / (perimeter * perimeter)
        return AreaResult(
            area_m2=area_m2,
            area_ha=area_ha,
            perimeter_m=perimeter,
            compactness_polsby_popper=pp,
            method="karney",
        )

    def _perimeter(self, coords: List[Tuple[float, float]]) -> float:
        total = 0.0
        R = 6371000.0
        for i in range(len(coords) - 1):
            lat1 = math.radians(coords[i][0])
            lon1 = math.radians(coords[i][1])
            lat2 = math.radians(coords[i + 1][0])
            lon2 = math.radians(coords[i + 1][1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            total += R * c
        return total


class OverlapDetector:
    """Stub OverlapDetector for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config


class BoundaryVersioner:
    """Stub BoundaryVersioner for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config
        self._versions: Dict[str, List[BoundaryVersion]] = {}


class SimplificationEngine:
    """Stub SimplificationEngine for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config


class SplitMergeEngine:
    """Stub SplitMergeEngine for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config


class ComplianceReporter:
    """Stub ComplianceReporter for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config


class PlotBoundaryService:
    """Stub PlotBoundaryService (facade) for test fixture instantiation."""

    def __init__(self, config: PlotBoundaryConfig):
        self.config = config
        self.polygon_manager = PolygonManager(config)
        self.boundary_validator = BoundaryValidator(config)
        self.area_calculator = AreaCalculator(config)
        self.overlap_detector = OverlapDetector(config)
        self.boundary_versioner = BoundaryVersioner(config)
        self.simplification_engine = SimplificationEngine(config)
        self.split_merge_engine = SplitMergeEngine(config)
        self.compliance_reporter = ComplianceReporter(config)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> PlotBoundaryConfig:
    """Create a PlotBoundaryConfig with test defaults."""
    return PlotBoundaryConfig()


@pytest.fixture
def uuid_gen() -> DeterministicUUID:
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def polygon_manager(config: PlotBoundaryConfig) -> PolygonManager:
    """Create a PolygonManager instance for testing."""
    return PolygonManager(config=config)


@pytest.fixture
def boundary_validator(config: PlotBoundaryConfig) -> BoundaryValidator:
    """Create a BoundaryValidator instance for testing."""
    return BoundaryValidator(config=config)


@pytest.fixture
def area_calculator(config: PlotBoundaryConfig) -> AreaCalculator:
    """Create an AreaCalculator instance for testing."""
    return AreaCalculator(config=config)


@pytest.fixture
def overlap_detector(config: PlotBoundaryConfig) -> OverlapDetector:
    """Create an OverlapDetector instance for testing."""
    return OverlapDetector(config=config)


@pytest.fixture
def boundary_versioner(config: PlotBoundaryConfig) -> BoundaryVersioner:
    """Create a BoundaryVersioner instance for testing."""
    return BoundaryVersioner(config=config)


@pytest.fixture
def simplification_engine(config: PlotBoundaryConfig) -> SimplificationEngine:
    """Create a SimplificationEngine instance for testing."""
    return SimplificationEngine(config=config)


@pytest.fixture
def split_merge_engine(config: PlotBoundaryConfig) -> SplitMergeEngine:
    """Create a SplitMergeEngine instance for testing."""
    return SplitMergeEngine(config=config)


@pytest.fixture
def compliance_reporter(config: PlotBoundaryConfig) -> ComplianceReporter:
    """Create a ComplianceReporter instance for testing."""
    return ComplianceReporter(config=config)


@pytest.fixture
def service(config: PlotBoundaryConfig) -> PlotBoundaryService:
    """Create a PlotBoundaryService instance for testing."""
    return PlotBoundaryService(config=config)


# ---------------------------------------------------------------------------
# Pre-built Boundary Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_square_boundary() -> PlotBoundary:
    """Ready-to-use valid square boundary (~100 ha in Brazil)."""
    coords = SIMPLE_SQUARE.coordinates[0]
    return make_boundary(coords, "cocoa", "BR", plot_id="PLOT-SQ-001")


@pytest.fixture
def large_polygon_boundary() -> PlotBoundary:
    """Boundary exceeding 4ha threshold (500 ha plantation)."""
    coords = LARGE_PLANTATION.coordinates[0]
    return make_boundary(coords, "oil_palm", "ID", plot_id="PLOT-LG-001")


@pytest.fixture
def small_polygon_boundary() -> PlotBoundary:
    """Boundary below 4ha threshold (2 ha farm)."""
    coords = SMALL_FARM.coordinates[0]
    return make_boundary(coords, "cocoa", "GH", plot_id="PLOT-SM-001")


@pytest.fixture
def invalid_boundary() -> PlotBoundary:
    """Self-intersecting boundary (invalid)."""
    coords = SELF_INTERSECTING.coordinates[0]
    return make_boundary(coords, "cocoa", "BR", plot_id="PLOT-INV-001")


@pytest.fixture
def polygon_with_holes() -> PlotBoundary:
    """Valid boundary with interior holes."""
    shell = WITH_HOLES.coordinates[0]
    holes = WITH_HOLES.coordinates[1:]
    boundary = make_boundary(shell, "cocoa", "BR", plot_id="PLOT-HOLES-001")
    boundary.interior_rings = holes
    return boundary


@pytest.fixture
def overlapping_pair() -> Tuple[PlotBoundary, PlotBoundary]:
    """Two overlapping boundaries for overlap testing."""
    coords_a = _make_square_coords(-3.12, -60.02, 0.005)
    coords_b = _make_square_coords(-3.12, -60.017, 0.005)
    boundary_a = make_boundary(coords_a, "cocoa", "BR", plot_id="PLOT-OVL-A")
    boundary_b = make_boundary(coords_b, "cocoa", "BR", plot_id="PLOT-OVL-B")
    return (boundary_a, boundary_b)


@pytest.fixture
def non_overlapping_pair() -> Tuple[PlotBoundary, PlotBoundary]:
    """Two non-overlapping boundaries."""
    coords_a = _make_square_coords(-3.12, -60.02, 0.003)
    coords_b = _make_square_coords(-3.20, -60.10, 0.003)
    boundary_a = make_boundary(coords_a, "cocoa", "BR", plot_id="PLOT-SEP-A")
    boundary_b = make_boundary(coords_b, "cocoa", "BR", plot_id="PLOT-SEP-B")
    return (boundary_a, boundary_b)


@pytest.fixture
def batch_boundaries() -> List[PlotBoundary]:
    """Batch of 10 boundaries for batch operation testing."""
    boundaries = []
    for i in range(10):
        lat = -3.12 + i * 0.02
        lon = -60.02 + i * 0.02
        coords = _make_square_coords(lat, lon, 0.003)
        b = make_boundary(
            coords, "cocoa", "BR", plot_id=f"PLOT-BATCH-{i+1:03d}",
        )
        boundaries.append(b)
    return boundaries


@pytest.fixture
def boundary_version_history() -> List[BoundaryVersion]:
    """Version history for a single plot (5 versions)."""
    plot_id = "PLOT-VER-001"
    versions = []
    base_lat, base_lon = -3.12, -60.02
    for v in range(1, 6):
        offset = v * 0.0001
        coords = _make_square_coords(base_lat + offset, base_lon, 0.004 + offset)
        bv = BoundaryVersion(
            plot_id=plot_id,
            version=v,
            exterior_ring=coords,
            area_ha=50.0 + v * 2.0,
            modified_at=datetime(2024, 1 + v, 1, tzinfo=timezone.utc).isoformat(),
            modified_by=f"user-{v:03d}",
            change_type="created" if v == 1 else "updated",
        )
        bv.provenance_hash = bv.compute_hash()
        versions.append(bv)
    return versions
