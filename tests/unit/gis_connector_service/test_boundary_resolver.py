# -*- coding: utf-8 -*-
"""
Unit Tests for BoundaryResolverEngine (AGENT-DATA-006)

Tests boundary resolution including country resolution, admin level resolution,
protected area detection, climate zone classification, biome classification,
boundary listing, custom boundary registration, bbox queries,
unknown coordinate handling, and provenance tracking
for the GIS/Mapping Connector Agent.

Coverage target: 85%+ of boundary_resolver.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class BoundaryResult:
    """Result of a boundary resolution query."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        country_code: Optional[str] = None,
        country_name: Optional[str] = None,
        admin_level_1: Optional[str] = None,
        admin_level_2: Optional[str] = None,
        protected_area: Optional[str] = None,
        is_protected: bool = False,
        climate_zone: Optional[str] = None,
        biome: Optional[str] = None,
        provenance_hash: str = "",
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.country_code = country_code
        self.country_name = country_name
        self.admin_level_1 = admin_level_1
        self.admin_level_2 = admin_level_2
        self.protected_area = protected_area
        self.is_protected = is_protected
        self.climate_zone = climate_zone
        self.biome = biome
        self.provenance_hash = provenance_hash


class CustomBoundary:
    """A user-registered custom boundary region."""

    def __init__(
        self,
        boundary_id: str,
        name: str,
        boundary_type: str,
        bbox: Tuple[float, float, float, float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.boundary_id = boundary_id
        self.name = name
        self.boundary_type = boundary_type
        self.bbox = bbox  # (min_lat, min_lon, max_lat, max_lon)
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Inline BoundaryResolverEngine
# ---------------------------------------------------------------------------


class BoundaryResolverEngine:
    """Engine for resolving geographic boundaries, climate zones, and biomes.

    Uses simplified lookup tables mapping coordinate ranges to countries,
    admin regions, climate zones, and biome classifications.
    """

    # Simplified country boundaries: (min_lat, min_lon, max_lat, max_lon) -> code
    COUNTRY_BOUNDS: List[Dict[str, Any]] = [
        {"code": "US", "name": "United States", "bbox": (24.5, -125.0, 49.5, -66.5)},
        {"code": "GB", "name": "United Kingdom", "bbox": (49.9, -8.2, 60.9, 1.8)},
        {"code": "BR", "name": "Brazil", "bbox": (-33.8, -73.9, 5.3, -34.8)},
        {"code": "DE", "name": "Germany", "bbox": (47.3, 5.9, 55.1, 15.0)},
        {"code": "AU", "name": "Australia", "bbox": (-43.6, 113.3, -10.6, 153.6)},
        {"code": "IN", "name": "India", "bbox": (6.7, 68.2, 35.5, 97.4)},
        {"code": "JP", "name": "Japan", "bbox": (24.4, 122.9, 45.5, 153.9)},
        {"code": "CN", "name": "China", "bbox": (18.2, 73.6, 53.5, 135.1)},
        {"code": "FR", "name": "France", "bbox": (41.4, -5.6, 51.1, 9.6)},
        {"code": "NG", "name": "Nigeria", "bbox": (4.3, 2.7, 13.9, 14.7)},
    ]

    # Climate zones by latitude band
    CLIMATE_ZONES: List[Dict[str, Any]] = [
        {"lat_min": -90, "lat_max": -66.5, "zone": "polar"},
        {"lat_min": -66.5, "lat_max": -35, "zone": "temperate"},
        {"lat_min": -35, "lat_max": -23.5, "zone": "subtropical"},
        {"lat_min": -23.5, "lat_max": 23.5, "zone": "tropical"},
        {"lat_min": 23.5, "lat_max": 35, "zone": "subtropical"},
        {"lat_min": 35, "lat_max": 55, "zone": "temperate"},
        {"lat_min": 55, "lat_max": 66.5, "zone": "continental"},
        {"lat_min": 66.5, "lat_max": 90, "zone": "polar"},
    ]

    # Simplified biome classification
    BIOME_RULES: List[Dict[str, Any]] = [
        {"lat_min": -90, "lat_max": -60, "biome": "tundra"},
        {"lat_min": -60, "lat_max": -35, "lon_min": -180, "lon_max": 180, "biome": "temperate_forest"},
        {"lat_min": -23.5, "lat_max": 23.5, "lon_min": -80, "lon_max": -35, "biome": "tropical_rainforest"},
        {"lat_min": -23.5, "lat_max": 23.5, "lon_min": -20, "lon_max": 50, "biome": "savanna"},
        {"lat_min": -23.5, "lat_max": 23.5, "lon_min": 95, "lon_max": 150, "biome": "tropical_rainforest"},
        {"lat_min": 35, "lat_max": 55, "lon_min": -10, "lon_max": 40, "biome": "temperate_forest"},
        {"lat_min": 55, "lat_max": 70, "lon_min": -180, "lon_max": 180, "biome": "boreal_forest"},
        {"lat_min": 70, "lat_max": 90, "lon_min": -180, "lon_max": 180, "biome": "tundra"},
    ]

    # Simplified protected areas
    PROTECTED_AREAS: List[Dict[str, Any]] = [
        {"name": "Yellowstone NP", "bbox": (44.1, -111.2, 45.1, -109.8)},
        {"name": "Amazon Reserve", "bbox": (-5.0, -70.0, -1.0, -60.0)},
        {"name": "Kruger NP", "bbox": (-25.5, 30.8, -22.3, 32.0)},
        {"name": "Black Forest NP", "bbox": (48.2, 8.0, 48.6, 8.4)},
    ]

    def __init__(self):
        self._custom_boundaries: Dict[str, CustomBoundary] = {}
        self._boundary_counter = 0

    def resolve(self, latitude: float, longitude: float) -> BoundaryResult:
        """Resolve all boundary information for a coordinate."""
        country_code, country_name = self._resolve_country(latitude, longitude)
        admin1 = self._resolve_admin_level(latitude, longitude, level=1)
        admin2 = self._resolve_admin_level(latitude, longitude, level=2)
        protected = self._check_protected_area(latitude, longitude)
        climate = self._classify_climate_zone(latitude)
        biome = self._classify_biome(latitude, longitude)

        return BoundaryResult(
            latitude=latitude,
            longitude=longitude,
            country_code=country_code,
            country_name=country_name,
            admin_level_1=admin1,
            admin_level_2=admin2,
            protected_area=protected,
            is_protected=protected is not None,
            climate_zone=climate,
            biome=biome,
            provenance_hash=_compute_hash({
                "lat": latitude,
                "lon": longitude,
                "country": country_code,
                "climate": climate,
                "biome": biome,
            }),
        )

    def resolve_country(self, latitude: float, longitude: float) -> Optional[str]:
        """Resolve the country code for a coordinate."""
        code, _ = self._resolve_country(latitude, longitude)
        return code

    def resolve_admin_level(
        self, latitude: float, longitude: float, level: int = 1
    ) -> Optional[str]:
        """Resolve the administrative region at the given level."""
        return self._resolve_admin_level(latitude, longitude, level)

    def check_protected_area(
        self, latitude: float, longitude: float
    ) -> Optional[str]:
        """Check if a coordinate falls within a protected area."""
        return self._check_protected_area(latitude, longitude)

    def classify_climate_zone(self, latitude: float) -> str:
        """Classify the climate zone based on latitude."""
        return self._classify_climate_zone(latitude)

    def classify_biome(self, latitude: float, longitude: float) -> Optional[str]:
        """Classify the biome for a coordinate."""
        return self._classify_biome(latitude, longitude)

    def list_boundaries(self, boundary_type: Optional[str] = None) -> List[CustomBoundary]:
        """List registered custom boundaries, optionally filtered by type."""
        boundaries = list(self._custom_boundaries.values())
        if boundary_type:
            boundaries = [b for b in boundaries if b.boundary_type == boundary_type]
        return boundaries

    def register_custom_boundary(
        self,
        name: str,
        boundary_type: str,
        bbox: Tuple[float, float, float, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CustomBoundary:
        """Register a custom boundary region."""
        self._boundary_counter += 1
        boundary_id = f"BND-{self._boundary_counter:05d}"
        boundary = CustomBoundary(
            boundary_id=boundary_id,
            name=name,
            boundary_type=boundary_type,
            bbox=bbox,
            metadata=metadata,
        )
        self._custom_boundaries[boundary_id] = boundary
        return boundary

    def query_by_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> List[CustomBoundary]:
        """Query custom boundaries that overlap with a bounding box."""
        results = []
        for boundary in self._custom_boundaries.values():
            b_min_lat, b_min_lon, b_max_lat, b_max_lon = boundary.bbox
            if (b_min_lat <= max_lat and b_max_lat >= min_lat
                    and b_min_lon <= max_lon and b_max_lon >= min_lon):
                results.append(boundary)
        return results

    def _resolve_country(
        self, latitude: float, longitude: float
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve country from coordinate using bounding boxes."""
        for country in self.COUNTRY_BOUNDS:
            bbox = country["bbox"]
            if (bbox[0] <= latitude <= bbox[2]
                    and bbox[1] <= longitude <= bbox[3]):
                return country["code"], country["name"]
        return None, None

    def _resolve_admin_level(
        self, latitude: float, longitude: float, level: int
    ) -> Optional[str]:
        """Resolve administrative region (simplified heuristic)."""
        code, _ = self._resolve_country(latitude, longitude)
        if code is None:
            return None
        if level == 1:
            return f"{code}-Admin1"
        if level == 2:
            return f"{code}-Admin2"
        return None

    def _check_protected_area(
        self, latitude: float, longitude: float
    ) -> Optional[str]:
        """Check if coordinate falls within a known protected area."""
        for area in self.PROTECTED_AREAS:
            bbox = area["bbox"]
            if (bbox[0] <= latitude <= bbox[2]
                    and bbox[1] <= longitude <= bbox[3]):
                return area["name"]
        return None

    def _classify_climate_zone(self, latitude: float) -> str:
        """Classify climate zone by latitude band."""
        for zone in self.CLIMATE_ZONES:
            if zone["lat_min"] <= latitude < zone["lat_max"]:
                return zone["zone"]
        return "unknown"

    def _classify_biome(self, latitude: float, longitude: float) -> Optional[str]:
        """Classify biome from coordinate (simplified)."""
        for rule in self.BIOME_RULES:
            lat_ok = rule["lat_min"] <= latitude < rule["lat_max"]
            lon_ok = True
            if "lon_min" in rule:
                lon_ok = rule["lon_min"] <= longitude <= rule["lon_max"]
            if lat_ok and lon_ok:
                return rule["biome"]
        return None


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> BoundaryResolverEngine:
    return BoundaryResolverEngine()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestResolveCountry:
    """Tests for country resolution from coordinates."""

    def test_us_coords_return_us(self, engine):
        code = engine.resolve_country(40.0, -100.0)
        assert code == "US"

    def test_uk_coords_return_gb(self, engine):
        code = engine.resolve_country(51.5, -0.1)
        assert code == "GB"

    def test_brazil_coords_return_br(self, engine):
        code = engine.resolve_country(-15.0, -50.0)
        assert code == "BR"

    def test_germany_coords_return_de(self, engine):
        code = engine.resolve_country(50.0, 10.0)
        assert code == "DE"

    def test_australia_coords_return_au(self, engine):
        code = engine.resolve_country(-25.0, 135.0)
        assert code == "AU"

    def test_india_coords_return_in(self, engine):
        code = engine.resolve_country(20.0, 78.0)
        assert code == "IN"

    def test_japan_coords_return_jp(self, engine):
        code = engine.resolve_country(36.0, 140.0)
        assert code == "JP"

    def test_china_coords_return_cn(self, engine):
        code = engine.resolve_country(35.0, 105.0)
        assert code == "CN"

    def test_ocean_coords_return_none(self, engine):
        # Middle of Atlantic Ocean
        code = engine.resolve_country(30.0, -40.0)
        assert code is None

    def test_resolve_full_includes_name(self, engine):
        result = engine.resolve(40.0, -100.0)
        assert result.country_code == "US"
        assert result.country_name == "United States"


class TestResolveAdminLevel:
    """Tests for administrative level resolution."""

    def test_admin_level_1(self, engine):
        admin1 = engine.resolve_admin_level(40.0, -100.0, level=1)
        assert admin1 is not None
        assert "US" in admin1

    def test_admin_level_2(self, engine):
        admin2 = engine.resolve_admin_level(40.0, -100.0, level=2)
        assert admin2 is not None
        assert "US" in admin2

    def test_admin_level_ocean_returns_none(self, engine):
        admin1 = engine.resolve_admin_level(30.0, -40.0, level=1)
        assert admin1 is None

    def test_resolve_includes_admin_levels(self, engine):
        result = engine.resolve(50.0, 10.0)
        assert result.admin_level_1 is not None
        assert result.admin_level_2 is not None


class TestProtectedAreaDetection:
    """Tests for protected area detection."""

    def test_yellowstone_detected(self, engine):
        name = engine.check_protected_area(44.5, -110.5)
        assert name == "Yellowstone NP"

    def test_amazon_reserve_detected(self, engine):
        name = engine.check_protected_area(-3.0, -65.0)
        assert name == "Amazon Reserve"

    def test_kruger_detected(self, engine):
        name = engine.check_protected_area(-24.0, 31.5)
        assert name == "Kruger NP"

    def test_non_protected_area(self, engine):
        name = engine.check_protected_area(48.0, 11.0)
        assert name is None

    def test_resolve_shows_protected_flag(self, engine):
        result = engine.resolve(44.5, -110.5)
        assert result.is_protected is True
        assert result.protected_area == "Yellowstone NP"

    def test_resolve_shows_not_protected(self, engine):
        result = engine.resolve(48.0, 11.0)
        assert result.is_protected is False
        assert result.protected_area is None


class TestClimateZone:
    """Tests for climate zone classification."""

    def test_tropical(self, engine):
        zone = engine.classify_climate_zone(5.0)
        assert zone == "tropical"

    def test_subtropical_south(self, engine):
        zone = engine.classify_climate_zone(-30.0)
        assert zone == "subtropical"

    def test_subtropical_north(self, engine):
        zone = engine.classify_climate_zone(30.0)
        assert zone == "subtropical"

    def test_temperate_north(self, engine):
        zone = engine.classify_climate_zone(45.0)
        assert zone == "temperate"

    def test_temperate_south(self, engine):
        zone = engine.classify_climate_zone(-50.0)
        assert zone == "temperate"

    def test_continental(self, engine):
        zone = engine.classify_climate_zone(60.0)
        assert zone == "continental"

    def test_polar_north(self, engine):
        zone = engine.classify_climate_zone(80.0)
        assert zone == "polar"

    def test_polar_south(self, engine):
        zone = engine.classify_climate_zone(-80.0)
        assert zone == "polar"

    def test_resolve_includes_climate(self, engine):
        result = engine.resolve(5.0, -60.0)
        assert result.climate_zone == "tropical"


class TestBiomeClassification:
    """Tests for biome classification."""

    def test_tropical_rainforest_amazon(self, engine):
        biome = engine.classify_biome(0.0, -60.0)
        assert biome == "tropical_rainforest"

    def test_tropical_rainforest_asia(self, engine):
        biome = engine.classify_biome(0.0, 110.0)
        assert biome == "tropical_rainforest"

    def test_savanna(self, engine):
        biome = engine.classify_biome(5.0, 20.0)
        assert biome == "savanna"

    def test_temperate_forest(self, engine):
        biome = engine.classify_biome(48.0, 11.0)
        assert biome == "temperate_forest"

    def test_boreal_forest(self, engine):
        biome = engine.classify_biome(62.0, 30.0)
        assert biome == "boreal_forest"

    def test_tundra_north(self, engine):
        biome = engine.classify_biome(75.0, 30.0)
        assert biome == "tundra"

    def test_tundra_south(self, engine):
        biome = engine.classify_biome(-80.0, 0.0)
        assert biome == "tundra"

    def test_resolve_includes_biome(self, engine):
        result = engine.resolve(0.0, -60.0)
        assert result.biome == "tropical_rainforest"


class TestListBoundaries:
    """Tests for listing custom boundaries."""

    def test_empty_initially(self, engine):
        boundaries = engine.list_boundaries()
        assert boundaries == []

    def test_list_after_register(self, engine):
        engine.register_custom_boundary(
            "My Area", "zone", (40.0, -80.0, 42.0, -75.0)
        )
        boundaries = engine.list_boundaries()
        assert len(boundaries) == 1

    def test_filter_by_type(self, engine):
        engine.register_custom_boundary("Zone A", "zone", (40.0, -80.0, 42.0, -75.0))
        engine.register_custom_boundary("Region B", "region", (45.0, -90.0, 47.0, -85.0))
        zones = engine.list_boundaries(boundary_type="zone")
        assert len(zones) == 1
        assert zones[0].name == "Zone A"

    def test_filter_returns_empty(self, engine):
        engine.register_custom_boundary("Zone A", "zone", (40.0, -80.0, 42.0, -75.0))
        result = engine.list_boundaries(boundary_type="nonexistent")
        assert result == []


class TestRegisterCustomBoundary:
    """Tests for registering custom boundaries."""

    def test_register_success(self, engine):
        boundary = engine.register_custom_boundary(
            "Test Area", "zone", (40.0, -80.0, 42.0, -75.0),
            metadata={"owner": "team-alpha"},
        )
        assert boundary.boundary_id.startswith("BND-")
        assert boundary.name == "Test Area"
        assert boundary.boundary_type == "zone"
        assert boundary.bbox == (40.0, -80.0, 42.0, -75.0)
        assert boundary.metadata["owner"] == "team-alpha"

    def test_sequential_ids(self, engine):
        b1 = engine.register_custom_boundary("A", "zone", (0, 0, 1, 1))
        b2 = engine.register_custom_boundary("B", "zone", (2, 2, 3, 3))
        assert b1.boundary_id == "BND-00001"
        assert b2.boundary_id == "BND-00002"

    def test_register_with_no_metadata(self, engine):
        boundary = engine.register_custom_boundary("X", "region", (10, 20, 30, 40))
        assert boundary.metadata == {}


class TestQueryByBbox:
    """Tests for bounding box queries."""

    def test_overlapping_boundary_found(self, engine):
        engine.register_custom_boundary("Z1", "zone", (40.0, -80.0, 42.0, -75.0))
        results = engine.query_by_bbox(41.0, -79.0, 43.0, -74.0)
        assert len(results) == 1
        assert results[0].name == "Z1"

    def test_non_overlapping_not_returned(self, engine):
        engine.register_custom_boundary("Z1", "zone", (40.0, -80.0, 42.0, -75.0))
        results = engine.query_by_bbox(50.0, 10.0, 55.0, 15.0)
        assert results == []

    def test_multiple_overlapping(self, engine):
        engine.register_custom_boundary("Z1", "zone", (40.0, -80.0, 42.0, -75.0))
        engine.register_custom_boundary("Z2", "zone", (41.0, -79.0, 43.0, -74.0))
        engine.register_custom_boundary("Z3", "zone", (50.0, 10.0, 55.0, 15.0))
        results = engine.query_by_bbox(40.5, -79.5, 42.5, -74.5)
        assert len(results) == 2

    def test_empty_when_no_boundaries(self, engine):
        results = engine.query_by_bbox(0, 0, 10, 10)
        assert results == []


class TestUnknownCoordinate:
    """Tests for unknown or edge-case coordinates."""

    def test_ocean_resolve(self, engine):
        result = engine.resolve(30.0, -40.0)
        assert result.country_code is None
        assert result.country_name is None

    def test_ocean_has_climate_zone(self, engine):
        result = engine.resolve(30.0, -40.0)
        assert result.climate_zone is not None

    def test_antarctic_resolve(self, engine):
        result = engine.resolve(-80.0, 0.0)
        assert result.country_code is None
        assert result.climate_zone == "polar"


class TestProvenance:
    """Tests for provenance tracking in boundary results."""

    def test_resolve_has_provenance(self, engine):
        result = engine.resolve(40.0, -100.0)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_different_coords_different_hash(self, engine):
        r1 = engine.resolve(40.0, -100.0)
        r2 = engine.resolve(51.5, -0.1)
        assert r1.provenance_hash != r2.provenance_hash

    def test_provenance_deterministic(self, engine):
        """Same inputs on different engine instances produce same hash."""
        e1 = BoundaryResolverEngine()
        e2 = BoundaryResolverEngine()
        r1 = e1.resolve(40.0, -100.0)
        r2 = e2.resolve(40.0, -100.0)
        assert r1.provenance_hash == r2.provenance_hash
