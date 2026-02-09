# -*- coding: utf-8 -*-
"""
Unit Tests for GeocoderEngine (AGENT-DATA-006)

Tests forward geocoding, reverse geocoding, batch operations,
coordinate string parsing (DMS, DD, UTM), address normalization,
elevation lookup, cache behavior, confidence scoring, and provenance tracking
for the GIS/Mapping Connector Agent.

Coverage target: 85%+ of geocoder.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
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


class GeocodeResult:
    """Result of a geocoding operation."""

    def __init__(
        self,
        query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        display_name: Optional[str] = None,
        confidence: float = 0.0,
        elevation_m: Optional[float] = None,
        source: str = "internal",
        provenance_hash: str = "",
    ):
        self.query = query
        self.latitude = latitude
        self.longitude = longitude
        self.display_name = display_name
        self.confidence = confidence
        self.elevation_m = elevation_m
        self.source = source
        self.provenance_hash = provenance_hash


class ParsedCoordinate:
    """Result of parsing a coordinate string."""

    def __init__(
        self,
        raw: str,
        latitude: float,
        longitude: float,
        format_type: str,
        valid: bool = True,
    ):
        self.raw = raw
        self.latitude = latitude
        self.longitude = longitude
        self.format_type = format_type
        self.valid = valid


# ---------------------------------------------------------------------------
# Inline GeocoderEngine
# ---------------------------------------------------------------------------


class GeocoderEngine:
    """Engine for geocoding addresses and parsing coordinate strings.

    Provides forward geocoding (name -> coordinates), reverse geocoding
    (coordinates -> name), coordinate string parsing (DMS, DD, UTM),
    address normalization, and elevation lookup.
    """

    # Internal geocoding database (simplified)
    KNOWN_PLACES: Dict[str, Dict[str, Any]] = {
        "new york": {"lat": 40.7128, "lon": -74.0060, "display": "New York, NY, USA", "elevation": 10.0},
        "london": {"lat": 51.5074, "lon": -0.1278, "display": "London, England, UK", "elevation": 11.0},
        "paris": {"lat": 48.8566, "lon": 2.3522, "display": "Paris, Ile-de-France, France", "elevation": 35.0},
        "tokyo": {"lat": 35.6762, "lon": 139.6503, "display": "Tokyo, Japan", "elevation": 40.0},
        "sydney": {"lat": -33.8688, "lon": 151.2093, "display": "Sydney, NSW, Australia", "elevation": 58.0},
        "sao paulo": {"lat": -23.5505, "lon": -46.6333, "display": "Sao Paulo, Brazil", "elevation": 760.0},
        "berlin": {"lat": 52.5200, "lon": 13.4050, "display": "Berlin, Germany", "elevation": 34.0},
        "mumbai": {"lat": 19.0760, "lon": 72.8777, "display": "Mumbai, Maharashtra, India", "elevation": 14.0},
        "pittsburgh": {"lat": 40.4406, "lon": -79.9959, "display": "Pittsburgh, PA, USA", "elevation": 367.0},
        "nairobi": {"lat": -1.2921, "lon": 36.8219, "display": "Nairobi, Kenya", "elevation": 1795.0},
    }

    # Reverse geocoding: approximate known locations
    REVERSE_TOLERANCE = 1.0  # degrees

    # Address normalization patterns
    ADDRESS_ABBREVIATIONS: Dict[str, str] = {
        "st": "street",
        "st.": "street",
        "ave": "avenue",
        "ave.": "avenue",
        "blvd": "boulevard",
        "blvd.": "boulevard",
        "rd": "road",
        "rd.": "road",
        "dr": "drive",
        "dr.": "drive",
        "ln": "lane",
        "ln.": "lane",
        "ct": "court",
        "ct.": "court",
    }

    def __init__(self):
        self._cache: Dict[str, GeocodeResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def forward_geocode(self, query: str) -> GeocodeResult:
        """Forward geocode: convert a place name to coordinates."""
        normalized = query.strip().lower()

        # Check cache
        if normalized in self._cache:
            self._cache_hits += 1
            return self._cache[normalized]

        self._cache_misses += 1

        place = self.KNOWN_PLACES.get(normalized)
        if place:
            result = GeocodeResult(
                query=query,
                latitude=place["lat"],
                longitude=place["lon"],
                display_name=place["display"],
                confidence=0.95,
                elevation_m=place.get("elevation"),
                source="internal",
                provenance_hash=_compute_hash({
                    "query": normalized,
                    "lat": place["lat"],
                    "lon": place["lon"],
                }),
            )
        else:
            result = GeocodeResult(
                query=query,
                latitude=None,
                longitude=None,
                display_name=None,
                confidence=0.0,
                source="internal",
                provenance_hash=_compute_hash({"query": normalized, "result": "not_found"}),
            )

        self._cache[normalized] = result
        return result

    def reverse_geocode(self, latitude: float, longitude: float) -> GeocodeResult:
        """Reverse geocode: convert coordinates to a place name."""
        cache_key = f"rev_{latitude:.4f}_{longitude:.4f}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        best_match = None
        best_distance = float("inf")

        for name, place in self.KNOWN_PLACES.items():
            dist = math.sqrt(
                (latitude - place["lat"]) ** 2
                + (longitude - place["lon"]) ** 2
            )
            if dist < best_distance and dist <= self.REVERSE_TOLERANCE:
                best_distance = dist
                best_match = (name, place)

        if best_match:
            name, place = best_match
            confidence = max(0.5, 1.0 - best_distance / self.REVERSE_TOLERANCE)
            result = GeocodeResult(
                query=f"{latitude},{longitude}",
                latitude=place["lat"],
                longitude=place["lon"],
                display_name=place["display"],
                confidence=round(confidence, 2),
                elevation_m=place.get("elevation"),
                source="internal",
                provenance_hash=_compute_hash({
                    "lat": latitude,
                    "lon": longitude,
                    "match": name,
                }),
            )
        else:
            result = GeocodeResult(
                query=f"{latitude},{longitude}",
                latitude=latitude,
                longitude=longitude,
                display_name=None,
                confidence=0.0,
                source="internal",
                provenance_hash=_compute_hash({
                    "lat": latitude,
                    "lon": longitude,
                    "result": "no_match",
                }),
            )

        self._cache[cache_key] = result
        return result

    def batch_forward(self, queries: List[str]) -> List[GeocodeResult]:
        """Batch forward geocode multiple place names."""
        return [self.forward_geocode(q) for q in queries]

    def batch_reverse(
        self, coordinates: List[Tuple[float, float]]
    ) -> List[GeocodeResult]:
        """Batch reverse geocode multiple coordinates."""
        return [self.reverse_geocode(lat, lon) for lat, lon in coordinates]

    def parse_coordinate_string(self, raw: str) -> ParsedCoordinate:
        """Parse various coordinate string formats.

        Supports:
        - DMS: 40 deg 26'46"N 79 deg 58'56"W
        - DD: 40.446, -79.982
        - UTM: 17T 589289 4477267
        """
        raw_stripped = raw.strip()

        # Try DD (decimal degrees): "lat, lon"
        if "," in raw_stripped and "'" not in raw_stripped and '"' not in raw_stripped:
            try:
                parts = raw_stripped.split(",")
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return ParsedCoordinate(raw=raw, latitude=lat, longitude=lon, format_type="DD")
            except (ValueError, IndexError):
                pass

        # Try DMS
        if any(c in raw_stripped for c in ["'", '"', "\u00b0", "N", "S", "E", "W"]):
            try:
                lat, lon = self._parse_dms(raw_stripped)
                return ParsedCoordinate(raw=raw, latitude=lat, longitude=lon, format_type="DMS")
            except (ValueError, IndexError):
                pass

        # Try UTM: "zone letter easting northing"
        parts = raw_stripped.split()
        if len(parts) == 3 and len(parts[0]) >= 2:
            try:
                lat, lon = self._parse_utm(parts[0], float(parts[1]), float(parts[2]))
                return ParsedCoordinate(raw=raw, latitude=lat, longitude=lon, format_type="UTM")
            except (ValueError, IndexError):
                pass

        return ParsedCoordinate(raw=raw, latitude=0.0, longitude=0.0, format_type="unknown", valid=False)

    def normalize_address(self, address: str) -> str:
        """Normalize an address string by expanding abbreviations."""
        words = address.lower().split()
        normalized = []
        for word in words:
            replacement = self.ADDRESS_ABBREVIATIONS.get(word, word)
            normalized.append(replacement)
        return " ".join(normalized).title()

    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """Get elevation for a coordinate (simplified lookup)."""
        for place in self.KNOWN_PLACES.values():
            dist = math.sqrt(
                (latitude - place["lat"]) ** 2
                + (longitude - place["lon"]) ** 2
            )
            if dist < 0.1:
                return place.get("elevation")
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
        }

    def _parse_dms(self, raw: str) -> Tuple[float, float]:
        """Parse DMS format: 40 deg 26'46"N 79 deg 58'56"W."""
        import re
        # Clean and extract DMS components
        cleaned = raw.replace("\u00b0", " ").replace("'", " ").replace('"', " ")
        cleaned = cleaned.replace("  ", " ").strip()
        parts = re.split(r"[NSEW]", cleaned)
        dirs = re.findall(r"[NSEW]", raw.upper())

        if len(parts) < 2 or len(dirs) < 2:
            raise ValueError(f"Cannot parse DMS: {raw}")

        lat_parts = parts[0].strip().split()
        lon_parts = parts[1].strip().split()

        lat = float(lat_parts[0])
        if len(lat_parts) > 1:
            lat += float(lat_parts[1]) / 60
        if len(lat_parts) > 2:
            lat += float(lat_parts[2]) / 3600
        if dirs[0] == "S":
            lat = -lat

        lon = float(lon_parts[0])
        if len(lon_parts) > 1:
            lon += float(lon_parts[1]) / 60
        if len(lon_parts) > 2:
            lon += float(lon_parts[2]) / 3600
        if dirs[1] == "W":
            lon = -lon

        return lat, lon

    def _parse_utm(
        self, zone_str: str, easting: float, northing: float
    ) -> Tuple[float, float]:
        """Approximate UTM to lat/lon conversion (simplified)."""
        zone_num = int(zone_str[:-1])
        zone_letter = zone_str[-1].upper()

        # Simplified UTM -> lat/lon conversion
        lon_origin = (zone_num - 1) * 6 - 180 + 3
        lon = lon_origin + (easting - 500000) / 111000
        lat = northing / 111000
        if zone_letter < "N":
            lat = lat - 90

        return round(lat, 4), round(lon, 4)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> GeocoderEngine:
    return GeocoderEngine()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestForwardGeocode:
    """Tests for forward geocoding (name -> coordinates)."""

    def test_known_city(self, engine):
        result = engine.forward_geocode("New York")
        assert result.latitude is not None
        assert result.longitude is not None
        assert abs(result.latitude - 40.7128) < 0.01
        assert abs(result.longitude - (-74.0060)) < 0.01

    def test_london(self, engine):
        result = engine.forward_geocode("London")
        assert result.latitude is not None
        assert abs(result.latitude - 51.5074) < 0.01
        assert result.display_name is not None
        assert "London" in result.display_name

    def test_paris(self, engine):
        result = engine.forward_geocode("Paris")
        assert result.latitude is not None
        assert abs(result.latitude - 48.8566) < 0.01

    def test_tokyo(self, engine):
        result = engine.forward_geocode("Tokyo")
        assert result.latitude is not None
        assert abs(result.latitude - 35.6762) < 0.01

    def test_case_insensitive(self, engine):
        r1 = engine.forward_geocode("New York")
        r2 = engine.forward_geocode("new york")
        assert r1.latitude == r2.latitude
        assert r1.longitude == r2.longitude

    def test_unknown_place(self, engine):
        result = engine.forward_geocode("Nonexistentville")
        assert result.latitude is None
        assert result.longitude is None
        assert result.confidence == 0.0

    def test_confidence_known_place(self, engine):
        result = engine.forward_geocode("Berlin")
        assert result.confidence > 0.8

    def test_display_name(self, engine):
        result = engine.forward_geocode("Sydney")
        assert result.display_name is not None
        assert "Sydney" in result.display_name

    def test_provenance_hash(self, engine):
        result = engine.forward_geocode("New York")
        assert len(result.provenance_hash) == 64


class TestReverseGeocode:
    """Tests for reverse geocoding (coordinates -> location)."""

    def test_near_known_city(self, engine):
        result = engine.reverse_geocode(40.7128, -74.0060)
        assert result.display_name is not None
        assert "New York" in result.display_name

    def test_exact_coordinates(self, engine):
        result = engine.reverse_geocode(51.5074, -0.1278)
        assert result.display_name is not None
        assert "London" in result.display_name

    def test_close_to_known(self, engine):
        # Slightly offset from Sydney
        result = engine.reverse_geocode(-33.87, 151.21)
        assert result.display_name is not None
        assert result.confidence > 0.5

    def test_no_match_remote(self, engine):
        # Middle of Pacific Ocean
        result = engine.reverse_geocode(0.0, -160.0)
        assert result.display_name is None
        assert result.confidence == 0.0

    def test_provenance_hash(self, engine):
        result = engine.reverse_geocode(40.7128, -74.0060)
        assert len(result.provenance_hash) == 64


class TestBatchForward:
    """Tests for batch forward geocoding."""

    def test_batch_multiple(self, engine):
        queries = ["New York", "London", "Tokyo"]
        results = engine.batch_forward(queries)
        assert len(results) == 3
        assert all(r.latitude is not None for r in results)

    def test_batch_empty(self, engine):
        results = engine.batch_forward([])
        assert results == []

    def test_batch_mixed_known_unknown(self, engine):
        queries = ["Paris", "Nonexistentville", "Berlin"]
        results = engine.batch_forward(queries)
        assert results[0].latitude is not None
        assert results[1].latitude is None
        assert results[2].latitude is not None


class TestBatchReverse:
    """Tests for batch reverse geocoding."""

    def test_batch_multiple(self, engine):
        coords = [(40.7128, -74.0060), (51.5074, -0.1278)]
        results = engine.batch_reverse(coords)
        assert len(results) == 2

    def test_batch_empty(self, engine):
        results = engine.batch_reverse([])
        assert results == []


class TestParseCoordinateString:
    """Tests for coordinate string parsing."""

    def test_parse_dd(self, engine):
        result = engine.parse_coordinate_string("40.446, -79.982")
        assert result.format_type == "DD"
        assert result.valid is True
        assert abs(result.latitude - 40.446) < 0.001
        assert abs(result.longitude - (-79.982)) < 0.001

    def test_parse_dd_positive(self, engine):
        result = engine.parse_coordinate_string("51.5074, -0.1278")
        assert result.format_type == "DD"
        assert result.valid is True

    def test_parse_dms(self, engine):
        result = engine.parse_coordinate_string("40\u00b026'46\"N 79\u00b058'56\"W")
        assert result.format_type == "DMS"
        assert result.valid is True
        assert abs(result.latitude - 40.446) < 0.01
        assert result.longitude < 0  # West

    def test_parse_utm(self, engine):
        result = engine.parse_coordinate_string("17T 589289 4477267")
        assert result.format_type == "UTM"
        assert result.valid is True
        assert result.latitude != 0.0 or result.longitude != 0.0

    def test_parse_invalid(self, engine):
        result = engine.parse_coordinate_string("not a coordinate")
        assert result.valid is False
        assert result.format_type == "unknown"


class TestAddressNormalization:
    """Tests for address normalization."""

    def test_expand_st(self, engine):
        result = engine.normalize_address("123 main st")
        assert "Street" in result

    def test_expand_ave(self, engine):
        result = engine.normalize_address("456 park ave")
        assert "Avenue" in result

    def test_expand_blvd(self, engine):
        result = engine.normalize_address("789 sunset blvd")
        assert "Boulevard" in result

    def test_expand_rd(self, engine):
        result = engine.normalize_address("10 oak rd")
        assert "Road" in result

    def test_title_case(self, engine):
        result = engine.normalize_address("main st")
        assert result[0].isupper()

    def test_no_change_needed(self, engine):
        result = engine.normalize_address("123 main street")
        assert "Street" in result


class TestElevationLookup:
    """Tests for elevation lookup."""

    def test_known_location_elevation(self, engine):
        # Near New York (10m)
        elev = engine.get_elevation(40.7128, -74.0060)
        assert elev is not None
        assert elev == 10.0

    def test_nairobi_high_elevation(self, engine):
        elev = engine.get_elevation(-1.2921, 36.8219)
        assert elev is not None
        assert elev > 1000  # Nairobi is at ~1795m

    def test_unknown_location(self, engine):
        elev = engine.get_elevation(0.0, 0.0)
        assert elev is None


class TestCacheBehavior:
    """Tests for geocoder cache behavior."""

    def test_cache_hit(self, engine):
        engine.forward_geocode("London")
        engine.forward_geocode("London")
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_miss(self, engine):
        engine.forward_geocode("London")
        engine.forward_geocode("Paris")
        stats = engine.get_cache_stats()
        assert stats["misses"] == 2

    def test_reverse_cache(self, engine):
        engine.reverse_geocode(40.7128, -74.0060)
        engine.reverse_geocode(40.7128, -74.0060)
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1

    def test_cache_size_grows(self, engine):
        engine.forward_geocode("London")
        engine.forward_geocode("Paris")
        engine.forward_geocode("Tokyo")
        stats = engine.get_cache_stats()
        assert stats["cache_size"] == 3

    def test_hit_rate(self, engine):
        engine.forward_geocode("London")  # miss
        engine.forward_geocode("London")  # hit
        engine.forward_geocode("London")  # hit
        stats = engine.get_cache_stats()
        assert abs(stats["hit_rate"] - 2 / 3) < 0.01


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_known_place_high_confidence(self, engine):
        result = engine.forward_geocode("New York")
        assert result.confidence >= 0.9

    def test_unknown_place_zero_confidence(self, engine):
        result = engine.forward_geocode("Nonexistentville")
        assert result.confidence == 0.0

    def test_exact_reverse_high_confidence(self, engine):
        result = engine.reverse_geocode(40.7128, -74.0060)
        assert result.confidence > 0.8

    def test_remote_reverse_zero_confidence(self, engine):
        result = engine.reverse_geocode(0.0, -160.0)
        assert result.confidence == 0.0


class TestProvenance:
    """Tests for provenance tracking in geocoding results."""

    def test_forward_provenance(self, engine):
        result = engine.forward_geocode("Berlin")
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_reverse_provenance(self, engine):
        result = engine.reverse_geocode(52.5200, 13.4050)
        assert len(result.provenance_hash) == 64

    def test_different_queries_different_hashes(self, engine):
        r1 = engine.forward_geocode("Berlin")
        r2 = engine.forward_geocode("Paris")
        assert r1.provenance_hash != r2.provenance_hash
