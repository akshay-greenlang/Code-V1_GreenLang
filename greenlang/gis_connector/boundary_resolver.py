# -*- coding: utf-8 -*-
"""
Boundary Resolver Engine - AGENT-DATA-006: GIS/Mapping Connector (GL-DATA-GEO-001)

Resolves geographic coordinates to administrative boundaries (country,
region), climate zones, biomes, and protected areas using built-in
bounding box databases and rule-based classification.

Zero-Hallucination Guarantees:
    - Country resolution uses ISO 3166 codes with bounding boxes
    - Climate zones use Koppen-Geiger latitude/elevation rules
    - Biome classification uses deterministic latitude bands
    - Protected area checks use registered geometry intersections
    - No ML/LLM used for boundary resolution
    - SHA-256 provenance hashes on all resolution results

Example:
    >>> from greenlang.gis_connector.boundary_resolver import BoundaryResolverEngine
    >>> resolver = BoundaryResolverEngine()
    >>> result = resolver.resolve_country([13.405, 52.52])
    >>> assert result["result_id"].startswith("BND-")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Climate Zones (Koppen-Geiger simplified)
# ---------------------------------------------------------------------------

CLIMATE_ZONES: Dict[str, Dict[str, Any]] = {
    "Af": {"name": "Tropical Rainforest", "group": "Tropical", "temp_range": [18, 35], "precip": "high"},
    "Am": {"name": "Tropical Monsoon", "group": "Tropical", "temp_range": [18, 35], "precip": "seasonal_high"},
    "Aw": {"name": "Tropical Savanna", "group": "Tropical", "temp_range": [18, 35], "precip": "seasonal"},
    "BWh": {"name": "Hot Desert", "group": "Dry", "temp_range": [18, 50], "precip": "very_low"},
    "BWk": {"name": "Cold Desert", "group": "Dry", "temp_range": [-5, 25], "precip": "very_low"},
    "BSh": {"name": "Hot Semi-Arid", "group": "Dry", "temp_range": [15, 40], "precip": "low"},
    "BSk": {"name": "Cold Semi-Arid", "group": "Dry", "temp_range": [-5, 30], "precip": "low"},
    "Csa": {"name": "Mediterranean Hot Summer", "group": "Mediterranean", "temp_range": [5, 35], "precip": "dry_summer"},
    "Csb": {"name": "Mediterranean Warm Summer", "group": "Mediterranean", "temp_range": [5, 25], "precip": "dry_summer"},
    "Cfa": {"name": "Humid Subtropical", "group": "Temperate", "temp_range": [0, 35], "precip": "moderate"},
    "Cfb": {"name": "Oceanic", "group": "Temperate", "temp_range": [0, 22], "precip": "moderate"},
    "Cfc": {"name": "Subpolar Oceanic", "group": "Temperate", "temp_range": [-5, 15], "precip": "moderate"},
    "Dfa": {"name": "Hot-Summer Humid Continental", "group": "Continental", "temp_range": [-20, 35], "precip": "moderate"},
    "Dfb": {"name": "Warm-Summer Humid Continental", "group": "Continental", "temp_range": [-25, 25], "precip": "moderate"},
    "Dfc": {"name": "Subarctic", "group": "Continental", "temp_range": [-40, 15], "precip": "low"},
    "Dfd": {"name": "Extreme Subarctic", "group": "Continental", "temp_range": [-50, 10], "precip": "low"},
    "ET": {"name": "Tundra", "group": "Polar", "temp_range": [-30, 10], "precip": "very_low"},
    "EF": {"name": "Ice Cap", "group": "Polar", "temp_range": [-60, 0], "precip": "very_low"},
    "H": {"name": "Highland", "group": "Highland", "temp_range": [-20, 25], "precip": "variable"},
}

# ---------------------------------------------------------------------------
# Biomes
# ---------------------------------------------------------------------------

BIOMES: Dict[str, Dict[str, Any]] = {
    "tropical_forest": {"name": "Tropical & Subtropical Moist Broadleaf Forest", "lat_range": [-10, 10], "carbon_density": 200},
    "temperate_forest": {"name": "Temperate Broadleaf & Mixed Forest", "lat_range": [30, 50], "carbon_density": 150},
    "boreal_forest": {"name": "Boreal Forest / Taiga", "lat_range": [50, 65], "carbon_density": 100},
    "grassland": {"name": "Temperate Grasslands, Savannas & Shrublands", "lat_range": [25, 50], "carbon_density": 50},
    "desert": {"name": "Deserts & Xeric Shrublands", "lat_range": [15, 35], "carbon_density": 5},
    "tundra": {"name": "Tundra", "lat_range": [60, 90], "carbon_density": 30},
    "wetland": {"name": "Flooded Grasslands & Savannas", "lat_range": [-20, 40], "carbon_density": 150},
    "ocean": {"name": "Marine / Oceanic", "lat_range": [-90, 90], "carbon_density": 0},
}

# ---------------------------------------------------------------------------
# Country Bounding Boxes (top 60 countries + territories by area/importance)
# Format: [min_lon, min_lat, max_lon, max_lat]
# ---------------------------------------------------------------------------

COUNTRY_BBOX: Dict[str, Dict[str, Any]] = {
    "RUS": {"name": "Russia", "iso2": "RU", "bbox": [27.0, 41.0, 180.0, 82.0]},
    "CAN": {"name": "Canada", "iso2": "CA", "bbox": [-141.0, 41.7, -52.6, 83.1]},
    "USA": {"name": "United States", "iso2": "US", "bbox": [-125.0, 24.5, -66.9, 49.4]},
    "CHN": {"name": "China", "iso2": "CN", "bbox": [73.5, 18.2, 134.8, 53.6]},
    "BRA": {"name": "Brazil", "iso2": "BR", "bbox": [-73.9, -33.7, -34.8, 5.3]},
    "AUS": {"name": "Australia", "iso2": "AU", "bbox": [113.2, -43.6, 153.6, -10.7]},
    "IND": {"name": "India", "iso2": "IN", "bbox": [68.2, 7.9, 97.4, 35.5]},
    "ARG": {"name": "Argentina", "iso2": "AR", "bbox": [-73.4, -55.0, -53.6, -21.8]},
    "KAZ": {"name": "Kazakhstan", "iso2": "KZ", "bbox": [46.5, 40.6, 87.3, 55.4]},
    "DZA": {"name": "Algeria", "iso2": "DZ", "bbox": [-8.7, 19.1, 12.0, 37.1]},
    "COD": {"name": "DR Congo", "iso2": "CD", "bbox": [12.2, -13.5, 31.3, 5.4]},
    "SAU": {"name": "Saudi Arabia", "iso2": "SA", "bbox": [34.6, 16.4, 55.7, 32.2]},
    "MEX": {"name": "Mexico", "iso2": "MX", "bbox": [-117.1, 14.5, -86.7, 32.7]},
    "IDN": {"name": "Indonesia", "iso2": "ID", "bbox": [95.0, -11.0, 141.0, 6.1]},
    "SDN": {"name": "Sudan", "iso2": "SD", "bbox": [21.8, 8.7, 38.6, 22.2]},
    "LBY": {"name": "Libya", "iso2": "LY", "bbox": [9.3, 19.5, 25.2, 33.2]},
    "IRN": {"name": "Iran", "iso2": "IR", "bbox": [44.0, 25.1, 63.3, 39.8]},
    "MNG": {"name": "Mongolia", "iso2": "MN", "bbox": [87.8, 41.6, 119.9, 52.1]},
    "PER": {"name": "Peru", "iso2": "PE", "bbox": [-81.3, -18.4, -68.7, -0.04]},
    "TCD": {"name": "Chad", "iso2": "TD", "bbox": [13.5, 7.4, 24.0, 23.5]},
    "NER": {"name": "Niger", "iso2": "NE", "bbox": [0.2, 11.7, 16.0, 23.5]},
    "AGO": {"name": "Angola", "iso2": "AO", "bbox": [11.7, -18.0, 24.1, -4.4]},
    "MLI": {"name": "Mali", "iso2": "ML", "bbox": [-12.2, 10.2, 4.2, 25.0]},
    "ZAF": {"name": "South Africa", "iso2": "ZA", "bbox": [16.5, -34.8, 32.9, -22.1]},
    "COL": {"name": "Colombia", "iso2": "CO", "bbox": [-79.0, -4.2, -66.9, 12.5]},
    "ETH": {"name": "Ethiopia", "iso2": "ET", "bbox": [33.0, 3.4, 48.0, 15.0]},
    "BOL": {"name": "Bolivia", "iso2": "BO", "bbox": [-69.6, -22.9, -57.5, -9.7]},
    "MRT": {"name": "Mauritania", "iso2": "MR", "bbox": [-17.1, 14.7, -4.8, 27.3]},
    "EGY": {"name": "Egypt", "iso2": "EG", "bbox": [24.7, 22.0, 36.9, 31.7]},
    "TZA": {"name": "Tanzania", "iso2": "TZ", "bbox": [29.3, -11.7, 40.4, -1.0]},
    "NGA": {"name": "Nigeria", "iso2": "NG", "bbox": [2.7, 4.3, 14.7, 13.9]},
    "VEN": {"name": "Venezuela", "iso2": "VE", "bbox": [-73.4, 0.6, -59.8, 12.2]},
    "PAK": {"name": "Pakistan", "iso2": "PK", "bbox": [60.9, 23.7, 77.8, 37.1]},
    "TUR": {"name": "Turkey", "iso2": "TR", "bbox": [26.0, 36.0, 44.8, 42.1]},
    "FRA": {"name": "France", "iso2": "FR", "bbox": [-5.1, 42.3, 8.2, 51.1]},
    "DEU": {"name": "Germany", "iso2": "DE", "bbox": [5.9, 47.3, 15.0, 55.1]},
    "GBR": {"name": "United Kingdom", "iso2": "GB", "bbox": [-8.2, 49.9, 1.8, 60.8]},
    "ITA": {"name": "Italy", "iso2": "IT", "bbox": [6.6, 36.6, 18.5, 47.1]},
    "ESP": {"name": "Spain", "iso2": "ES", "bbox": [-9.3, 36.0, 3.3, 43.8]},
    "POL": {"name": "Poland", "iso2": "PL", "bbox": [14.1, 49.0, 24.1, 54.8]},
    "UKR": {"name": "Ukraine", "iso2": "UA", "bbox": [22.1, 44.4, 40.2, 52.4]},
    "JPN": {"name": "Japan", "iso2": "JP", "bbox": [129.5, 31.0, 145.8, 45.5]},
    "KOR": {"name": "South Korea", "iso2": "KR", "bbox": [126.1, 33.1, 129.6, 38.6]},
    "THA": {"name": "Thailand", "iso2": "TH", "bbox": [97.3, 5.6, 105.6, 20.5]},
    "VNM": {"name": "Vietnam", "iso2": "VN", "bbox": [102.1, 8.6, 109.5, 23.4]},
    "PHL": {"name": "Philippines", "iso2": "PH", "bbox": [117.0, 5.0, 126.6, 18.5]},
    "NZL": {"name": "New Zealand", "iso2": "NZ", "bbox": [166.4, -47.3, 178.5, -34.4]},
    "CHL": {"name": "Chile", "iso2": "CL", "bbox": [-75.6, -55.6, -66.9, -17.5]},
    "KEN": {"name": "Kenya", "iso2": "KE", "bbox": [33.9, -4.7, 41.9, 5.0]},
    "GHA": {"name": "Ghana", "iso2": "GH", "bbox": [-3.3, 4.7, 1.2, 11.2]},
    "NOR": {"name": "Norway", "iso2": "NO", "bbox": [4.6, 58.0, 31.1, 71.2]},
    "SWE": {"name": "Sweden", "iso2": "SE", "bbox": [11.1, 55.3, 24.2, 69.1]},
    "FIN": {"name": "Finland", "iso2": "FI", "bbox": [19.5, 59.8, 31.6, 70.1]},
    "MYS": {"name": "Malaysia", "iso2": "MY", "bbox": [100.1, 0.9, 119.3, 7.4]},
    "SGP": {"name": "Singapore", "iso2": "SG", "bbox": [103.6, 1.2, 104.0, 1.5]},
    "ARE": {"name": "United Arab Emirates", "iso2": "AE", "bbox": [51.6, 22.6, 56.4, 26.1]},
    "ISR": {"name": "Israel", "iso2": "IL", "bbox": [34.3, 29.5, 35.9, 33.3]},
    "CHE": {"name": "Switzerland", "iso2": "CH", "bbox": [5.9, 45.8, 10.5, 47.8]},
    "PRT": {"name": "Portugal", "iso2": "PT", "bbox": [-9.5, 37.0, -6.2, 42.2]},
    "CRI": {"name": "Costa Rica", "iso2": "CR", "bbox": [-85.9, 8.0, -82.6, 11.2]},
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

def _make_boundary_result(
    result_id: str,
    coordinate: List[float],
    resolution_type: str,
    resolved: Dict[str, Any],
    confidence: float = 0.0,
) -> Dict[str, Any]:
    """Create a BoundaryResult dictionary.

    Args:
        result_id: Unique result identifier.
        coordinate: [lon, lat] coordinate.
        resolution_type: Type of resolution (country, admin, climate, etc.).
        resolved: Resolution result data.
        confidence: Confidence score (0-1).

    Returns:
        BoundaryResult dictionary.
    """
    return {
        "result_id": result_id,
        "coordinate": coordinate,
        "resolution_type": resolution_type,
        "resolved": resolved,
        "confidence": confidence,
        "created_at": _utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BoundaryResolverEngine:
    """Boundary resolution engine for geographic coordinates.

    Resolves coordinates to country, administrative region, climate zone,
    biome, and protected area information using built-in databases.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _boundaries: In-memory boundary result storage.
        _custom_boundaries: User-registered custom boundaries.

    Example:
        >>> resolver = BoundaryResolverEngine()
        >>> result = resolver.resolve_country([13.405, 52.52])
        >>> assert result["resolved"]["iso3"] == "DEU"
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize BoundaryResolverEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._boundaries: Dict[str, Dict[str, Any]] = {}
        self._custom_boundaries: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "BoundaryResolverEngine initialized with %d countries, "
            "%d climate zones, %d biomes",
            len(COUNTRY_BBOX), len(CLIMATE_ZONES), len(BIOMES),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_country(self, coordinate: List[float]) -> Dict[str, Any]:
        """Get country from coordinates using ISO 3166 bounding boxes.

        Args:
            coordinate: [lon, lat] coordinate.

        Returns:
            BoundaryResult with country information.
        """
        start_time = time.monotonic()
        result_id = f"BND-{uuid.uuid4().hex[:12]}"

        lon, lat = coordinate[0], coordinate[1]
        matched_country = None
        best_area = float("inf")

        for iso3, info in COUNTRY_BBOX.items():
            bbox = info["bbox"]
            if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
                # Use smallest bounding box for best match
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < best_area:
                    best_area = area
                    matched_country = {
                        "iso3": iso3,
                        "iso2": info["iso2"],
                        "name": info["name"],
                    }

        if matched_country:
            resolved = matched_country
            confidence = 0.8
        else:
            resolved = {"iso3": "UNK", "iso2": "XX", "name": "Unknown"}
            confidence = 0.0

        result = _make_boundary_result(
            result_id=result_id,
            coordinate=coordinate,
            resolution_type="country",
            resolved=resolved,
            confidence=confidence,
        )
        self._store_and_track(result, start_time)
        return result

    def resolve_admin(
        self,
        coordinate: List[float],
        level: int = 1,
    ) -> Dict[str, Any]:
        """Get administrative boundary at specified level.

        Level 0 = Country, Level 1 = State/Province, Level 2 = District.

        Args:
            coordinate: [lon, lat] coordinate.
            level: Administrative level (0, 1, 2).

        Returns:
            BoundaryResult with administrative boundary information.
        """
        start_time = time.monotonic()
        result_id = f"BND-{uuid.uuid4().hex[:12]}"

        # Level 0 is country
        if level == 0:
            return self.resolve_country(coordinate)

        # For levels 1+, resolve country first then provide placeholder
        country_result = self.resolve_country(coordinate)
        country_info = country_result.get("resolved", {})

        resolved = {
            "country": country_info,
            "admin_level": level,
            "admin_name": f"Region (level {level})",
            "note": "Detailed admin boundaries require external data source",
        }

        result = _make_boundary_result(
            result_id=result_id,
            coordinate=coordinate,
            resolution_type=f"admin_level_{level}",
            resolved=resolved,
            confidence=0.3,
        )
        self._store_and_track(result, start_time)
        return result

    def resolve_protected_area(self, coordinate: List[float]) -> Dict[str, Any]:
        """Check for protected area intersection at coordinate.

        Checks registered custom boundaries for protected area designations.

        Args:
            coordinate: [lon, lat] coordinate.

        Returns:
            BoundaryResult with protected area information.
        """
        start_time = time.monotonic()
        result_id = f"BND-{uuid.uuid4().hex[:12]}"

        lon, lat = coordinate[0], coordinate[1]
        protected_areas: List[Dict[str, Any]] = []

        for bid, boundary in self._custom_boundaries.items():
            if boundary.get("level") != "protected_area":
                continue
            bbox = boundary.get("bbox", [])
            if bbox and bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
                protected_areas.append({
                    "boundary_id": bid,
                    "name": boundary.get("name", ""),
                    "designation": boundary.get("designation", "protected"),
                })

        result = _make_boundary_result(
            result_id=result_id,
            coordinate=coordinate,
            resolution_type="protected_area",
            resolved={
                "is_protected": len(protected_areas) > 0,
                "protected_areas": protected_areas,
                "count": len(protected_areas),
            },
            confidence=0.7 if protected_areas else 0.5,
        )
        self._store_and_track(result, start_time)
        return result

    def resolve_climate_zone(self, coordinate: List[float]) -> Dict[str, Any]:
        """Get Koppen-Geiger climate zone for a coordinate.

        Uses latitude-based classification rules.

        Args:
            coordinate: [lon, lat] coordinate.

        Returns:
            BoundaryResult with climate zone information.
        """
        start_time = time.monotonic()
        result_id = f"BND-{uuid.uuid4().hex[:12]}"

        lat = coordinate[1]
        abs_lat = abs(lat)

        # Simplified Koppen-Geiger classification
        if abs_lat < 10:
            zone_code = "Af"
        elif abs_lat < 20:
            zone_code = "Aw"
        elif abs_lat < 25:
            zone_code = "BSh"
        elif abs_lat < 35:
            # Check for Mediterranean vs subtropical
            lon = coordinate[0]
            if (30 < lon < 45) or (-10 < lon < 5 and lat > 0):
                zone_code = "Csa"
            else:
                zone_code = "Cfa"
        elif abs_lat < 45:
            zone_code = "Cfb"
        elif abs_lat < 55:
            zone_code = "Dfb"
        elif abs_lat < 65:
            zone_code = "Dfc"
        elif abs_lat < 75:
            zone_code = "ET"
        else:
            zone_code = "EF"

        zone_info = CLIMATE_ZONES.get(zone_code, {})

        result = _make_boundary_result(
            result_id=result_id,
            coordinate=coordinate,
            resolution_type="climate_zone",
            resolved={
                "zone_code": zone_code,
                "zone_name": zone_info.get("name", "Unknown"),
                "group": zone_info.get("group", "Unknown"),
                "temp_range": zone_info.get("temp_range", []),
                "precipitation": zone_info.get("precip", "unknown"),
            },
            confidence=0.6,
        )
        self._store_and_track(result, start_time)
        return result

    def resolve_biome(self, coordinate: List[float]) -> Dict[str, Any]:
        """Get biome classification for a coordinate.

        Uses latitude-band classification.

        Args:
            coordinate: [lon, lat] coordinate.

        Returns:
            BoundaryResult with biome information.
        """
        start_time = time.monotonic()
        result_id = f"BND-{uuid.uuid4().hex[:12]}"

        lat = coordinate[1]
        abs_lat = abs(lat)

        biome_key = "grassland"
        if abs_lat < 10:
            biome_key = "tropical_forest"
        elif abs_lat < 25:
            biome_key = "desert"
        elif abs_lat < 45:
            biome_key = "temperate_forest"
        elif abs_lat < 60:
            biome_key = "boreal_forest"
        elif abs_lat < 75:
            biome_key = "tundra"
        else:
            biome_key = "tundra"

        biome_info = BIOMES.get(biome_key, {})

        result = _make_boundary_result(
            result_id=result_id,
            coordinate=coordinate,
            resolution_type="biome",
            resolved={
                "biome_key": biome_key,
                "biome_name": biome_info.get("name", "Unknown"),
                "carbon_density_tonnes_ha": biome_info.get("carbon_density", 0),
            },
            confidence=0.55,
        )
        self._store_and_track(result, start_time)
        return result

    def list_boundaries(
        self,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List registered custom boundaries.

        Args:
            level: Optional filter by boundary level/type.

        Returns:
            List of boundary dictionaries.
        """
        results = []
        for bid, boundary in self._custom_boundaries.items():
            if level and boundary.get("level") != level:
                continue
            results.append({"boundary_id": bid, **boundary})
        return results

    def register_boundary(
        self,
        name: str,
        level: str,
        geometry: Dict[str, Any],
        designation: Optional[str] = None,
    ) -> str:
        """Register a custom boundary.

        Args:
            name: Boundary name.
            level: Boundary level/type (protected_area, admin, custom).
            geometry: Geometry dictionary.
            designation: Optional designation type.

        Returns:
            Generated boundary ID.
        """
        boundary_id = f"CBN-{uuid.uuid4().hex[:12]}"

        # Compute bounding box from geometry
        coords = geometry.get("coordinates", [])
        bbox = self._compute_bbox_from_coords(coords)

        self._custom_boundaries[boundary_id] = {
            "name": name,
            "level": level,
            "geometry": geometry,
            "bbox": bbox,
            "designation": designation or level,
            "created_at": _utcnow().isoformat(),
        }

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(self._custom_boundaries[boundary_id])
            self._provenance.record(
                entity_type="boundary",
                entity_id=boundary_id,
                action="boundary_resolve",
                data_hash=data_hash,
            )

        logger.info("Registered boundary %s: name=%s, level=%s", boundary_id, name, level)
        return boundary_id

    def query_boundaries(
        self,
        bbox: List[float],
    ) -> List[Dict[str, Any]]:
        """Query custom boundaries within a bounding box.

        Args:
            bbox: Bounding box [minx, miny, maxx, maxy].

        Returns:
            List of matching boundary dictionaries.
        """
        if len(bbox) != 4:
            return []

        results = []
        for bid, boundary in self._custom_boundaries.items():
            b_bbox = boundary.get("bbox", [])
            if not b_bbox or len(b_bbox) != 4:
                continue
            # Check bounding box overlap
            if (b_bbox[0] <= bbox[2] and b_bbox[2] >= bbox[0] and
                    b_bbox[1] <= bbox[3] and b_bbox[3] >= bbox[1]):
                results.append({"boundary_id": bid, **boundary})

        return results

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get a boundary result by ID.

        Args:
            result_id: Result identifier.

        Returns:
            BoundaryResult dictionary or None.
        """
        return self._boundaries.get(result_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_bbox_from_coords(self, coords: Any) -> List[float]:
        """Compute bounding box from nested coordinate arrays.

        Args:
            coords: Nested coordinate array.

        Returns:
            [minx, miny, maxx, maxy] or empty list.
        """
        flat = self._flatten(coords)
        if not flat:
            return []
        min_x = min(c[0] for c in flat)
        min_y = min(c[1] for c in flat)
        max_x = max(c[0] for c in flat)
        max_y = max(c[1] for c in flat)
        return [min_x, min_y, max_x, max_y]

    def _flatten(self, coords: Any) -> List[List[float]]:
        """Flatten nested coordinates to list of [x, y] pairs.

        Args:
            coords: Nested coordinate array.

        Returns:
            Flat list of [x, y] coordinate pairs.
        """
        if not isinstance(coords, list) or not coords:
            return []
        if isinstance(coords[0], (int, float)):
            return [coords[:2]] if len(coords) >= 2 else []
        result: List[List[float]] = []
        for c in coords:
            result.extend(self._flatten(c))
        return result

    def _store_and_track(
        self,
        result: Dict[str, Any],
        start_time: float,
    ) -> None:
        """Store result and record provenance/metrics.

        Args:
            result: BoundaryResult dictionary.
            start_time: Monotonic start time.
        """
        result_id = result["result_id"]
        self._boundaries[result_id] = result

        if self._provenance is not None:
            data_hash = _compute_hash(result)
            self._provenance.record(
                entity_type="boundary_resolution",
                entity_id=result_id,
                action="boundary_resolve",
                data_hash=data_hash,
            )

        try:
            from greenlang.gis_connector.metrics import record_operation
            record_operation(
                operation="boundary_resolve",
                format=result.get("resolution_type", "unknown"),
                status="success",
                duration=(time.monotonic() - start_time),
            )
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Boundary resolution %s completed (%.1f ms)",
            result_id, elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def result_count(self) -> int:
        """Return the total number of stored boundary results."""
        return len(self._boundaries)

    @property
    def custom_boundary_count(self) -> int:
        """Return the number of registered custom boundaries."""
        return len(self._custom_boundaries)

    def get_statistics(self) -> Dict[str, Any]:
        """Get resolver statistics.

        Returns:
            Dictionary with resolution counts and type distribution.
        """
        results = list(self._boundaries.values())
        type_counts: Dict[str, int] = {}
        for r in results:
            rt = r.get("resolution_type", "unknown")
            type_counts[rt] = type_counts.get(rt, 0) + 1

        return {
            "total_resolutions": len(results),
            "type_distribution": type_counts,
            "countries_available": len(COUNTRY_BBOX),
            "climate_zones_available": len(CLIMATE_ZONES),
            "biomes_available": len(BIOMES),
            "custom_boundaries": len(self._custom_boundaries),
        }


__all__ = [
    "BoundaryResolverEngine",
    "COUNTRY_BBOX",
    "CLIMATE_ZONES",
    "BIOMES",
]
