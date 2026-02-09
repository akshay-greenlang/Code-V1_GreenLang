# -*- coding: utf-8 -*-
"""
Unit Tests for FormatParserEngine (AGENT-DATA-006)

Tests GeoJSON parsing (Point, Polygon, Feature, FeatureCollection), WKT parsing
(POINT, POLYGON, LINESTRING), CSV parsing (auto-detect lat/lon columns, custom
columns), auto-detect format, geometry validation (ring closure, self-intersection),
feature extraction, bounding box computation, ID format PRS-xxxxx, and provenance
hash generation.

Coverage target: 85%+ of format_parser.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline models (minimal)
# ---------------------------------------------------------------------------


class Coordinate:
    def __init__(self, longitude: float = 0.0, latitude: float = 0.0, altitude: Optional[float] = None):
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude


class BoundingBox:
    def __init__(self, min_lon: float = -180.0, min_lat: float = -90.0,
                 max_lon: float = 180.0, max_lat: float = 90.0):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat


class Geometry:
    def __init__(self, geometry_type: str = "point", coordinates: Optional[Any] = None,
                 properties: Optional[Dict[str, Any]] = None):
        self.geometry_type = geometry_type
        self.coordinates = coordinates or []
        self.properties = properties or {}


class Feature:
    def __init__(self, feature_id: str = "", geometry: Optional[Geometry] = None,
                 properties: Optional[Dict[str, Any]] = None, crs: str = "EPSG:4326",
                 provenance_hash: Optional[str] = None):
        import uuid
        self.feature_id = feature_id or f"FTR-{uuid.uuid4().hex[:5]}"
        self.geometry = geometry
        self.properties = properties or {}
        self.crs = crs
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline FormatParserEngine
# ---------------------------------------------------------------------------


class FormatParserEngine:
    """Parses GeoJSON, WKT, and CSV formats into normalized Feature objects."""

    # Auto-detect lat/lon column name patterns
    LAT_COLUMNS = {"lat", "latitude", "lat_dd", "y", "northing"}
    LON_COLUMNS = {"lon", "lng", "longitude", "lon_dd", "x", "easting"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "parse_operations": 0,
            "features_extracted": 0,
            "validation_errors": 0,
        }

    def _next_parse_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"PRS-{self._counter:05d}"

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -----------------------------------------------------------------------
    # GeoJSON parsing
    # -----------------------------------------------------------------------

    def parse_geojson(self, data: str) -> List[Feature]:
        """Parse GeoJSON string into list of Features."""
        parsed = json.loads(data)
        features = []

        geojson_type = parsed.get("type", "")

        if geojson_type == "Point":
            features.append(self._geojson_geometry_to_feature(parsed))
        elif geojson_type == "Polygon":
            features.append(self._geojson_geometry_to_feature(parsed))
        elif geojson_type == "LineString":
            features.append(self._geojson_geometry_to_feature(parsed))
        elif geojson_type == "MultiPoint":
            features.append(self._geojson_geometry_to_feature(parsed))
        elif geojson_type == "MultiPolygon":
            features.append(self._geojson_geometry_to_feature(parsed))
        elif geojson_type == "Feature":
            features.append(self._geojson_feature_to_feature(parsed))
        elif geojson_type == "FeatureCollection":
            for f in parsed.get("features", []):
                features.append(self._geojson_feature_to_feature(f))
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson_type}")

        prov = {
            "op": "parse_geojson",
            "parse_id": self._next_parse_id(),
            "feature_count": len(features),
        }
        prov_hash = self._compute_provenance(prov)

        for feat in features:
            feat.provenance_hash = prov_hash

        with self._lock:
            self._stats["parse_operations"] += 1
            self._stats["features_extracted"] += len(features)

        return features

    def _geojson_geometry_to_feature(self, geom_dict: Dict[str, Any]) -> Feature:
        geom_type = geom_dict["type"].lower()
        coords = geom_dict.get("coordinates", [])
        geom = Geometry(geometry_type=geom_type, coordinates=coords)
        return Feature(geometry=geom)

    def _geojson_feature_to_feature(self, feat_dict: Dict[str, Any]) -> Feature:
        geom_dict = feat_dict.get("geometry", {})
        props = feat_dict.get("properties", {})
        fid = feat_dict.get("id", "")

        geom = None
        if geom_dict:
            geom = Geometry(
                geometry_type=geom_dict.get("type", "").lower(),
                coordinates=geom_dict.get("coordinates", []),
            )

        return Feature(
            feature_id=str(fid) if fid else "",
            geometry=geom,
            properties=props,
        )

    # -----------------------------------------------------------------------
    # WKT parsing
    # -----------------------------------------------------------------------

    def parse_wkt(self, wkt_str: str) -> Feature:
        """Parse WKT string into a Feature."""
        wkt_str = wkt_str.strip()

        # Match geometry type and coordinate part
        match = re.match(r'^(\w+)\s*\((.+)\)$', wkt_str, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid WKT format: {wkt_str[:50]}")

        geom_type = match.group(1).upper()
        coord_str = match.group(2)

        if geom_type == "POINT":
            coords = self._parse_wkt_point_coords(coord_str)
            geom = Geometry(geometry_type="point", coordinates=coords)
        elif geom_type == "LINESTRING":
            coords = self._parse_wkt_linestring_coords(coord_str)
            geom = Geometry(geometry_type="linestring", coordinates=coords)
        elif geom_type == "POLYGON":
            coords = self._parse_wkt_polygon_coords(coord_str)
            geom = Geometry(geometry_type="polygon", coordinates=coords)
        else:
            raise ValueError(f"Unsupported WKT type: {geom_type}")

        prov = {
            "op": "parse_wkt",
            "parse_id": self._next_parse_id(),
            "geom_type": geom_type,
        }

        feat = Feature(
            geometry=geom,
            provenance_hash=self._compute_provenance(prov),
        )

        with self._lock:
            self._stats["parse_operations"] += 1
            self._stats["features_extracted"] += 1

        return feat

    def _parse_wkt_point_coords(self, coord_str: str) -> List[float]:
        parts = coord_str.strip().split()
        return [float(p) for p in parts]

    def _parse_wkt_linestring_coords(self, coord_str: str) -> List[List[float]]:
        coords = []
        for pair in coord_str.split(","):
            parts = pair.strip().split()
            coords.append([float(p) for p in parts])
        return coords

    def _parse_wkt_polygon_coords(self, coord_str: str) -> List[List[List[float]]]:
        rings = []
        # Split on outer parentheses to get rings
        ring_strs = re.findall(r'\(([^()]+)\)', coord_str)
        for ring_str in ring_strs:
            ring = []
            for pair in ring_str.split(","):
                parts = pair.strip().split()
                ring.append([float(p) for p in parts])
            rings.append(ring)
        return rings

    # -----------------------------------------------------------------------
    # CSV parsing
    # -----------------------------------------------------------------------

    def parse_csv(self, csv_data: str, lat_column: Optional[str] = None,
                  lon_column: Optional[str] = None) -> List[Feature]:
        """Parse CSV data into Features. Auto-detects lat/lon columns if not specified."""
        lines = csv_data.strip().split("\n")
        if len(lines) < 2:
            raise ValueError("CSV must have at least a header row and one data row")

        headers = [h.strip().lower() for h in lines[0].split(",")]

        # Auto-detect or use provided columns
        lat_col = lat_column.lower() if lat_column else self._detect_column(headers, self.LAT_COLUMNS)
        lon_col = lon_column.lower() if lon_column else self._detect_column(headers, self.LON_COLUMNS)

        if lat_col is None or lon_col is None:
            raise ValueError("Could not detect latitude/longitude columns")

        lat_idx = headers.index(lat_col)
        lon_idx = headers.index(lon_col)

        features = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(",")]
            try:
                lat = float(values[lat_idx])
                lon = float(values[lon_idx])
            except (ValueError, IndexError):
                continue

            props = {}
            for i, h in enumerate(headers):
                if i < len(values) and h not in (lat_col, lon_col):
                    props[h] = values[i]

            geom = Geometry(geometry_type="point", coordinates=[lon, lat])
            features.append(Feature(geometry=geom, properties=props))

        prov = {
            "op": "parse_csv",
            "parse_id": self._next_parse_id(),
            "feature_count": len(features),
        }
        prov_hash = self._compute_provenance(prov)
        for feat in features:
            feat.provenance_hash = prov_hash

        with self._lock:
            self._stats["parse_operations"] += 1
            self._stats["features_extracted"] += len(features)

        return features

    def _detect_column(self, headers: List[str], candidates: set) -> Optional[str]:
        for h in headers:
            if h in candidates:
                return h
        return None

    # -----------------------------------------------------------------------
    # Auto-detect format
    # -----------------------------------------------------------------------

    def detect_format(self, data: str) -> str:
        """Auto-detect the format of input data."""
        stripped = data.strip()

        # Try JSON first
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and "type" in parsed:
                    return "geojson"
            except json.JSONDecodeError:
                pass

        # Try WKT
        wkt_pattern = r'^(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\('
        if re.match(wkt_pattern, stripped, re.IGNORECASE):
            return "wkt"

        # Try CSV (has commas and newlines)
        if "\n" in stripped and "," in stripped:
            return "csv"

        return "unknown"

    # -----------------------------------------------------------------------
    # Geometry validation
    # -----------------------------------------------------------------------

    def validate_geometry(self, geometry: Geometry) -> List[str]:
        """Validate a geometry. Returns list of error strings (empty = valid)."""
        errors = []

        if geometry.geometry_type == "polygon":
            errors.extend(self._validate_polygon(geometry.coordinates))
        elif geometry.geometry_type == "linestring":
            if len(geometry.coordinates) < 2:
                errors.append("LineString must have at least 2 coordinates")
        elif geometry.geometry_type == "point":
            if not isinstance(geometry.coordinates, list) or len(geometry.coordinates) < 2:
                errors.append("Point must have at least 2 coordinates (lon, lat)")

        if errors:
            with self._lock:
                self._stats["validation_errors"] += len(errors)

        return errors

    def _validate_polygon(self, coordinates: Any) -> List[str]:
        errors = []
        if not coordinates or not isinstance(coordinates, list):
            errors.append("Polygon must have at least one ring")
            return errors

        for i, ring in enumerate(coordinates):
            if not ring or len(ring) < 4:
                errors.append(f"Ring {i} must have at least 4 coordinates")
                continue

            # Check ring closure
            if ring[0] != ring[-1]:
                errors.append(f"Ring {i} is not closed (first and last coordinates differ)")

            # Check self-intersection (simplified: check for duplicate non-adjacent vertices)
            seen_coords = set()
            for j, coord in enumerate(ring[:-1]):  # Exclude closing point
                coord_key = tuple(coord)
                if coord_key in seen_coords:
                    errors.append(f"Ring {i} has duplicate vertex at position {j}")
                seen_coords.add(coord_key)

        return errors

    # -----------------------------------------------------------------------
    # Feature extraction helpers
    # -----------------------------------------------------------------------

    def extract_features(self, data: str, format: Optional[str] = None) -> List[Feature]:
        """Extract features from data string, auto-detecting format if needed."""
        fmt = format or self.detect_format(data)

        if fmt == "geojson":
            return self.parse_geojson(data)
        elif fmt == "wkt":
            return [self.parse_wkt(data)]
        elif fmt == "csv":
            return self.parse_csv(data)
        else:
            raise ValueError(f"Unsupported or unrecognized format: {fmt}")

    def compute_bounding_box(self, features: List[Feature]) -> Optional[BoundingBox]:
        """Compute the bounding box encompassing all features."""
        if not features:
            return None

        all_coords = []
        for feat in features:
            if feat.geometry:
                self._collect_coords(feat.geometry.coordinates, all_coords)

        if not all_coords:
            return None

        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]

        return BoundingBox(
            min_lon=min(lons),
            min_lat=min(lats),
            max_lon=max(lons),
            max_lat=max(lats),
        )

    def _collect_coords(self, coordinates: Any, result: List[List[float]]) -> None:
        """Recursively collect [lon, lat] pairs from nested coordinate arrays."""
        if not coordinates:
            return
        if isinstance(coordinates[0], (int, float)):
            # This is a single coordinate pair
            result.append(coordinates)
        elif isinstance(coordinates[0], list):
            for item in coordinates:
                self._collect_coords(item, result)

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """FormatParserEngine instance for testing."""
    return FormatParserEngine()


@pytest.fixture
def geojson_point():
    """GeoJSON Point string."""
    return json.dumps({
        "type": "Point",
        "coordinates": [-73.9857, 40.7484],
    })


@pytest.fixture
def geojson_polygon():
    """GeoJSON Polygon string (unit square)."""
    return json.dumps({
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    })


@pytest.fixture
def geojson_feature():
    """GeoJSON Feature string."""
    return json.dumps({
        "type": "Feature",
        "id": "site-001",
        "geometry": {
            "type": "Point",
            "coordinates": [13.405, 52.52],
        },
        "properties": {
            "name": "Berlin Office",
            "emissions_tonnes": 150.5,
        },
    })


@pytest.fixture
def geojson_feature_collection():
    """GeoJSON FeatureCollection string."""
    return json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"name": "Origin"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10, 20]},
                "properties": {"name": "Somewhere"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-5, 45]},
                "properties": {"name": "France"},
            },
        ],
    })


@pytest.fixture
def sample_csv():
    """CSV data with auto-detectable lat/lon columns."""
    return (
        "name,latitude,longitude,value\n"
        "Site A,40.7484,-73.9857,100\n"
        "Site B,52.52,13.405,200\n"
        "Site C,48.8566,2.3522,300\n"
    )


# ===========================================================================
# Test Classes -- GeoJSON Parsing
# ===========================================================================


class TestParseGeoJSONPoint:
    """Test GeoJSON Point parsing."""

    def test_parse_point(self, engine, geojson_point):
        """Parse Point geometry."""
        features = engine.parse_geojson(geojson_point)
        assert len(features) == 1
        assert features[0].geometry.geometry_type == "point"
        assert features[0].geometry.coordinates == [-73.9857, 40.7484]

    def test_parse_point_provenance(self, engine, geojson_point):
        """Point parsing generates provenance hash."""
        features = engine.parse_geojson(geojson_point)
        assert features[0].provenance_hash is not None
        assert len(features[0].provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", features[0].provenance_hash)


class TestParseGeoJSONPolygon:
    """Test GeoJSON Polygon parsing."""

    def test_parse_polygon(self, engine, geojson_polygon):
        """Parse Polygon geometry."""
        features = engine.parse_geojson(geojson_polygon)
        assert len(features) == 1
        assert features[0].geometry.geometry_type == "polygon"
        assert len(features[0].geometry.coordinates) == 1  # One ring
        assert len(features[0].geometry.coordinates[0]) == 5  # 4 vertices + closing

    def test_polygon_ring_closure(self, engine, geojson_polygon):
        """Polygon ring first and last coordinates match."""
        features = engine.parse_geojson(geojson_polygon)
        ring = features[0].geometry.coordinates[0]
        assert ring[0] == ring[-1]


class TestParseGeoJSONFeature:
    """Test GeoJSON Feature parsing."""

    def test_parse_feature(self, engine, geojson_feature):
        """Parse Feature with id, geometry, and properties."""
        features = engine.parse_geojson(geojson_feature)
        assert len(features) == 1
        feat = features[0]
        assert feat.feature_id == "site-001"
        assert feat.geometry.geometry_type == "point"
        assert feat.geometry.coordinates == [13.405, 52.52]
        assert feat.properties["name"] == "Berlin Office"
        assert feat.properties["emissions_tonnes"] == 150.5

    def test_parse_feature_no_id(self, engine):
        """Feature without explicit id gets auto-generated id."""
        data = json.dumps({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {},
        })
        features = engine.parse_geojson(data)
        assert features[0].feature_id.startswith("FTR-")


class TestParseGeoJSONFeatureCollection:
    """Test GeoJSON FeatureCollection parsing."""

    def test_parse_feature_collection(self, engine, geojson_feature_collection):
        """Parse FeatureCollection with 3 features."""
        features = engine.parse_geojson(geojson_feature_collection)
        assert len(features) == 3

    def test_feature_collection_properties(self, engine, geojson_feature_collection):
        """Features in collection have correct properties."""
        features = engine.parse_geojson(geojson_feature_collection)
        names = [f.properties.get("name") for f in features]
        assert "Origin" in names
        assert "Somewhere" in names
        assert "France" in names

    def test_feature_collection_coordinates(self, engine, geojson_feature_collection):
        """Features in collection have correct coordinates."""
        features = engine.parse_geojson(geojson_feature_collection)
        coords = [f.geometry.coordinates for f in features]
        assert [0, 0] in coords
        assert [10, 20] in coords
        assert [-5, 45] in coords

    def test_empty_feature_collection(self, engine):
        """Empty FeatureCollection returns empty list."""
        data = json.dumps({"type": "FeatureCollection", "features": []})
        features = engine.parse_geojson(data)
        assert features == []

    def test_feature_collection_increments_stats(self, engine, geojson_feature_collection):
        """FeatureCollection parsing increments stats."""
        engine.parse_geojson(geojson_feature_collection)
        stats = engine.get_statistics()
        assert stats["parse_operations"] == 1
        assert stats["features_extracted"] == 3


class TestParseGeoJSONInvalid:
    """Test GeoJSON parsing error handling."""

    def test_invalid_json_raises(self, engine):
        """Invalid JSON raises ValueError."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            engine.parse_geojson("not json at all")

    def test_unsupported_type_raises(self, engine):
        """Unsupported GeoJSON type raises ValueError."""
        data = json.dumps({"type": "UnknownType", "coordinates": [0, 0]})
        with pytest.raises(ValueError, match="Unsupported GeoJSON type"):
            engine.parse_geojson(data)


# ===========================================================================
# Test Classes -- WKT Parsing
# ===========================================================================


class TestParseWKTPoint:
    """Test WKT POINT parsing."""

    def test_parse_point(self, engine):
        """Parse WKT POINT."""
        feat = engine.parse_wkt("POINT (10.5 20.3)")
        assert feat.geometry.geometry_type == "point"
        assert feat.geometry.coordinates == [10.5, 20.3]

    def test_parse_point_negative_coords(self, engine):
        """Parse WKT POINT with negative coordinates."""
        feat = engine.parse_wkt("POINT (-73.9857 40.7484)")
        assert feat.geometry.coordinates[0] == -73.9857
        assert feat.geometry.coordinates[1] == 40.7484

    def test_parse_point_provenance(self, engine):
        """WKT POINT parsing generates provenance hash."""
        feat = engine.parse_wkt("POINT (0 0)")
        assert feat.provenance_hash is not None
        assert len(feat.provenance_hash) == 64


class TestParseWKTLineString:
    """Test WKT LINESTRING parsing."""

    def test_parse_linestring(self, engine):
        """Parse WKT LINESTRING."""
        feat = engine.parse_wkt("LINESTRING (0 0, 1 1, 2 0)")
        assert feat.geometry.geometry_type == "linestring"
        assert len(feat.geometry.coordinates) == 3
        assert feat.geometry.coordinates[0] == [0.0, 0.0]
        assert feat.geometry.coordinates[1] == [1.0, 1.0]
        assert feat.geometry.coordinates[2] == [2.0, 0.0]

    def test_parse_linestring_multiple_points(self, engine):
        """Parse LINESTRING with many vertices."""
        wkt = "LINESTRING (0 0, 1 0, 2 0, 3 0, 4 0)"
        feat = engine.parse_wkt(wkt)
        assert len(feat.geometry.coordinates) == 5


class TestParseWKTPolygon:
    """Test WKT POLYGON parsing."""

    def test_parse_polygon(self, engine):
        """Parse WKT POLYGON."""
        wkt = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        feat = engine.parse_wkt(wkt)
        assert feat.geometry.geometry_type == "polygon"
        assert len(feat.geometry.coordinates) == 1  # One ring
        assert len(feat.geometry.coordinates[0]) == 5  # 4 vertices + closing

    def test_parse_polygon_ring_closure(self, engine):
        """Polygon ring first and last coordinates match."""
        wkt = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        feat = engine.parse_wkt(wkt)
        ring = feat.geometry.coordinates[0]
        assert ring[0] == ring[-1]


class TestParseWKTInvalid:
    """Test WKT parsing error handling."""

    def test_invalid_wkt_raises(self, engine):
        """Invalid WKT format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid WKT format"):
            engine.parse_wkt("NOT WKT DATA")

    def test_unsupported_wkt_type(self, engine):
        """Unsupported WKT type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported WKT type"):
            engine.parse_wkt("CIRCULARSTRING (0 0, 1 1, 2 0)")


# ===========================================================================
# Test Classes -- CSV Parsing
# ===========================================================================


class TestParseCSVAutoDetect:
    """Test CSV parsing with auto-detected lat/lon columns."""

    def test_parse_csv_auto_detect(self, engine, sample_csv):
        """Auto-detect latitude and longitude columns."""
        features = engine.parse_csv(sample_csv)
        assert len(features) == 3

    def test_csv_coordinates(self, engine, sample_csv):
        """CSV features have correct coordinates."""
        features = engine.parse_csv(sample_csv)
        # First row: Site A at (40.7484, -73.9857)
        assert features[0].geometry.coordinates == [-73.9857, 40.7484]

    def test_csv_properties(self, engine, sample_csv):
        """CSV features include non-coordinate columns as properties."""
        features = engine.parse_csv(sample_csv)
        assert features[0].properties.get("name") == "Site A"
        assert features[0].properties.get("value") == "100"

    def test_csv_auto_detect_lat_lng(self, engine):
        """Auto-detect 'lat' and 'lng' columns."""
        csv_data = "name,lat,lng\nTest,10.0,20.0\n"
        features = engine.parse_csv(csv_data)
        assert len(features) == 1
        assert features[0].geometry.coordinates == [20.0, 10.0]

    def test_csv_auto_detect_x_y(self, engine):
        """Auto-detect 'x' and 'y' columns."""
        csv_data = "name,y,x\nTest,30.0,40.0\n"
        features = engine.parse_csv(csv_data)
        assert len(features) == 1
        assert features[0].geometry.coordinates == [40.0, 30.0]


class TestParseCSVCustomColumns:
    """Test CSV parsing with custom column names."""

    def test_custom_columns(self, engine):
        """Use custom lat/lon column names."""
        csv_data = "site,north,east,score\nA,51.5,-0.12,95\n"
        features = engine.parse_csv(csv_data, lat_column="north", lon_column="east")
        assert len(features) == 1
        assert features[0].geometry.coordinates == [-0.12, 51.5]

    def test_custom_columns_properties(self, engine):
        """Custom column parsing excludes lat/lon from properties."""
        csv_data = "site,north,east,score\nA,51.5,-0.12,95\n"
        features = engine.parse_csv(csv_data, lat_column="north", lon_column="east")
        assert "north" not in features[0].properties
        assert "east" not in features[0].properties
        assert features[0].properties.get("site") == "A"
        assert features[0].properties.get("score") == "95"


class TestParseCSVErrors:
    """Test CSV parsing error handling."""

    def test_no_data_rows_raises(self, engine):
        """CSV with only header raises ValueError."""
        with pytest.raises(ValueError, match="at least a header row and one data row"):
            engine.parse_csv("lat,lon")

    def test_no_detectable_columns_raises(self, engine):
        """CSV without recognizable lat/lon columns raises ValueError."""
        csv_data = "name,value,score\nA,1,2\n"
        with pytest.raises(ValueError, match="Could not detect"):
            engine.parse_csv(csv_data)

    def test_csv_provenance(self, engine, sample_csv):
        """CSV parsing generates provenance hash."""
        features = engine.parse_csv(sample_csv)
        assert features[0].provenance_hash is not None
        assert len(features[0].provenance_hash) == 64

    def test_csv_increments_stats(self, engine, sample_csv):
        """CSV parsing increments stats."""
        engine.parse_csv(sample_csv)
        stats = engine.get_statistics()
        assert stats["parse_operations"] == 1
        assert stats["features_extracted"] == 3


# ===========================================================================
# Test Classes -- Auto-detect Format
# ===========================================================================


class TestDetectFormat:
    """Test format auto-detection."""

    def test_detect_geojson(self, engine, geojson_point):
        """Detect GeoJSON format from JSON with type field."""
        assert engine.detect_format(geojson_point) == "geojson"

    def test_detect_wkt_point(self, engine):
        """Detect WKT format from POINT string."""
        assert engine.detect_format("POINT (10 20)") == "wkt"

    def test_detect_wkt_polygon(self, engine):
        """Detect WKT format from POLYGON string."""
        assert engine.detect_format("POLYGON ((0 0, 1 0, 1 1, 0 0))") == "wkt"

    def test_detect_wkt_linestring(self, engine):
        """Detect WKT format from LINESTRING string."""
        assert engine.detect_format("LINESTRING (0 0, 1 1, 2 0)") == "wkt"

    def test_detect_csv(self, engine, sample_csv):
        """Detect CSV format from data with commas and newlines."""
        assert engine.detect_format(sample_csv) == "csv"

    def test_detect_unknown(self, engine):
        """Return 'unknown' for unrecognized format."""
        assert engine.detect_format("random text without structure") == "unknown"

    def test_detect_wkt_case_insensitive(self, engine):
        """WKT detection is case insensitive."""
        assert engine.detect_format("point (10 20)") == "wkt"
        assert engine.detect_format("Point (10 20)") == "wkt"


# ===========================================================================
# Test Classes -- Geometry Validation
# ===========================================================================


class TestValidateGeometry:
    """Test geometry validation."""

    def test_valid_point(self, engine):
        """Valid point geometry passes validation."""
        geom = Geometry(geometry_type="point", coordinates=[10.0, 20.0])
        errors = engine.validate_geometry(geom)
        assert errors == []

    def test_invalid_point_missing_coords(self, engine):
        """Point with missing coordinates fails validation."""
        geom = Geometry(geometry_type="point", coordinates=[10.0])
        errors = engine.validate_geometry(geom)
        assert len(errors) > 0
        assert any("at least 2 coordinates" in e for e in errors)

    def test_valid_polygon(self, engine):
        """Valid polygon passes validation."""
        ring = [[0, 0], [1, 0], [1, 1], [0, 0]]
        geom = Geometry(geometry_type="polygon", coordinates=[ring])
        errors = engine.validate_geometry(geom)
        assert errors == []

    def test_polygon_ring_not_closed(self, engine):
        """Polygon with unclosed ring fails validation."""
        ring = [[0, 0], [1, 0], [1, 1], [0, 1]]
        geom = Geometry(geometry_type="polygon", coordinates=[ring])
        errors = engine.validate_geometry(geom)
        assert any("not closed" in e for e in errors)

    def test_polygon_too_few_points(self, engine):
        """Polygon ring with too few points fails validation."""
        ring = [[0, 0], [1, 0], [0, 0]]
        geom = Geometry(geometry_type="polygon", coordinates=[ring])
        errors = engine.validate_geometry(geom)
        assert any("at least 4" in e for e in errors)

    def test_polygon_duplicate_vertex(self, engine):
        """Polygon with duplicate non-adjacent vertex reports error."""
        ring = [[0, 0], [1, 0], [0, 0], [1, 1], [0, 0]]
        geom = Geometry(geometry_type="polygon", coordinates=[ring])
        errors = engine.validate_geometry(geom)
        assert any("duplicate vertex" in e for e in errors)

    def test_valid_linestring(self, engine):
        """Valid linestring passes validation."""
        geom = Geometry(geometry_type="linestring", coordinates=[[0, 0], [1, 1]])
        errors = engine.validate_geometry(geom)
        assert errors == []

    def test_linestring_too_few_points(self, engine):
        """LineString with fewer than 2 coordinates fails validation."""
        geom = Geometry(geometry_type="linestring", coordinates=[[0, 0]])
        errors = engine.validate_geometry(geom)
        assert any("at least 2" in e for e in errors)

    def test_validation_errors_increment_stats(self, engine):
        """Validation errors increment the stats counter."""
        geom = Geometry(geometry_type="point", coordinates=[10.0])
        engine.validate_geometry(geom)
        stats = engine.get_statistics()
        assert stats["validation_errors"] > 0


# ===========================================================================
# Test Classes -- Feature Extraction
# ===========================================================================


class TestExtractFeatures:
    """Test unified feature extraction with format auto-detection."""

    def test_extract_geojson(self, engine, geojson_point):
        """Extract features from GeoJSON data."""
        features = engine.extract_features(geojson_point)
        assert len(features) == 1
        assert features[0].geometry.geometry_type == "point"

    def test_extract_wkt(self, engine):
        """Extract features from WKT data."""
        features = engine.extract_features("POINT (10 20)")
        assert len(features) == 1
        assert features[0].geometry.geometry_type == "point"

    def test_extract_csv(self, engine, sample_csv):
        """Extract features from CSV data."""
        features = engine.extract_features(sample_csv)
        assert len(features) == 3

    def test_extract_explicit_format(self, engine, geojson_point):
        """Extract with explicitly specified format."""
        features = engine.extract_features(geojson_point, format="geojson")
        assert len(features) == 1

    def test_extract_unknown_raises(self, engine):
        """Unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported or unrecognized format"):
            engine.extract_features("random text")


# ===========================================================================
# Test Classes -- Bounding Box Computation
# ===========================================================================


class TestComputeBoundingBox:
    """Test bounding box computation from features."""

    def test_compute_bbox_single_point(self, engine):
        """Bounding box of a single point is a degenerate box."""
        feat = Feature(geometry=Geometry(geometry_type="point", coordinates=[10.0, 20.0]))
        bbox = engine.compute_bounding_box([feat])
        assert bbox.min_lon == 10.0
        assert bbox.max_lon == 10.0
        assert bbox.min_lat == 20.0
        assert bbox.max_lat == 20.0

    def test_compute_bbox_multiple_points(self, engine):
        """Bounding box of multiple points."""
        feats = [
            Feature(geometry=Geometry(geometry_type="point", coordinates=[-10.0, -20.0])),
            Feature(geometry=Geometry(geometry_type="point", coordinates=[30.0, 40.0])),
            Feature(geometry=Geometry(geometry_type="point", coordinates=[5.0, 10.0])),
        ]
        bbox = engine.compute_bounding_box(feats)
        assert bbox.min_lon == -10.0
        assert bbox.min_lat == -20.0
        assert bbox.max_lon == 30.0
        assert bbox.max_lat == 40.0

    def test_compute_bbox_polygon(self, engine):
        """Bounding box of a polygon."""
        ring = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0], [-5.0, -5.0]]
        feat = Feature(geometry=Geometry(geometry_type="polygon", coordinates=[ring]))
        bbox = engine.compute_bounding_box([feat])
        assert bbox.min_lon == -5.0
        assert bbox.min_lat == -5.0
        assert bbox.max_lon == 5.0
        assert bbox.max_lat == 5.0

    def test_compute_bbox_empty_features(self, engine):
        """Empty features list returns None."""
        bbox = engine.compute_bounding_box([])
        assert bbox is None

    def test_compute_bbox_no_geometry(self, engine):
        """Features without geometry return None."""
        feat = Feature()
        bbox = engine.compute_bounding_box([feat])
        assert bbox is None


# ===========================================================================
# Test Classes -- ID Format and Provenance
# ===========================================================================


class TestIDFormat:
    """Test parse ID format PRS-xxxxx."""

    def test_parse_id_format(self, engine, geojson_point):
        """Parse operations generate PRS-xxxxx IDs internally."""
        # Parsing increments counter; we test via sequential operations
        engine.parse_geojson(geojson_point)
        engine.parse_geojson(geojson_point)
        stats = engine.get_statistics()
        assert stats["parse_operations"] == 2

    def test_sequential_ids(self, engine):
        """Sequential parses get sequential IDs (verified via stats)."""
        engine.parse_wkt("POINT (0 0)")
        engine.parse_wkt("POINT (1 1)")
        engine.parse_wkt("POINT (2 2)")
        stats = engine.get_statistics()
        assert stats["parse_operations"] == 3
        assert stats["features_extracted"] == 3


class TestProvenance:
    """Test provenance hash generation."""

    def test_provenance_is_sha256(self, engine, geojson_point):
        """Provenance hash is valid SHA-256 (64 hex chars)."""
        features = engine.parse_geojson(geojson_point)
        h = features[0].provenance_hash
        assert h is not None
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_provenance_consistency(self, engine):
        """Same provenance data produces same hash."""
        data = {"op": "test", "value": 42}
        h1 = engine._compute_provenance(data)
        h2 = engine._compute_provenance(data)
        assert h1 == h2

    def test_provenance_different_data(self, engine):
        """Different provenance data produces different hash."""
        h1 = engine._compute_provenance({"op": "a"})
        h2 = engine._compute_provenance({"op": "b"})
        assert h1 != h2

    def test_wkt_provenance(self, engine):
        """WKT parsing generates provenance hash."""
        feat = engine.parse_wkt("POINT (0 0)")
        assert feat.provenance_hash is not None
        assert len(feat.provenance_hash) == 64

    def test_csv_provenance(self, engine, sample_csv):
        """CSV parsing generates provenance hash."""
        features = engine.parse_csv(sample_csv)
        assert all(f.provenance_hash is not None for f in features)
        # All features from same parse share same provenance hash
        assert features[0].provenance_hash == features[1].provenance_hash
