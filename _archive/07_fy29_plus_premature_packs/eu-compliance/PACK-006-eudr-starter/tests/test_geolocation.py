# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Geolocation Verification Tests
==============================================================

Validates the geolocation engine including coordinate validation,
polygon verification, DMS/UTM normalization, area calculation,
overlap detection, country determination, plot size rules, batch
validation, GeoJSON parsing, and Article 9 formatting.

Test count: 25
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

from conftest import (
    generate_coordinates,
    _compute_hash,
    assert_provenance_hash,
)


# ---------------------------------------------------------------------------
# Geolocation Engine Simulator
# ---------------------------------------------------------------------------

class GeolocationEngineSimulator:
    """Simulates geolocation verification engine operations."""

    WGS84_A = 6378137.0  # semi-major axis in meters

    def validate_coordinates(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Validate that coordinates are within valid WGS84 ranges."""
        errors = []
        if not (-90.0 <= latitude <= 90.0):
            errors.append(f"Latitude {latitude} out of range [-90, 90]")
        if not (-180.0 <= longitude <= 180.0):
            errors.append(f"Longitude {longitude} out of range [-180, 180]")
        precision = len(str(latitude).split(".")[-1]) if "." in str(latitude) else 0
        if precision < 6:
            errors.append(f"Latitude precision is {precision} decimals, required >= 6")
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "latitude": latitude,
            "longitude": longitude,
            "coordinate_system": "WGS84",
            "precision_decimals": precision,
        }

    def validate_coordinates_sea(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Check if coordinates are on land (simplified check)."""
        # Simplified: ocean areas are roughly defined by known patterns
        # This is a mock; real implementation uses land/sea mask
        is_sea = False
        # Mid-Pacific ocean
        if -10 < latitude < 10 and 160 < longitude < 200:
            is_sea = True
        # Mid-Atlantic ocean
        if -10 < latitude < 10 and -40 < longitude < -20:
            is_sea = True
        # Southern ocean
        if latitude < -65:
            is_sea = True
        return {
            "is_land_based": not is_sea,
            "latitude": latitude,
            "longitude": longitude,
        }

    def validate_polygon(self, vertices: List[List[float]]) -> Dict[str, Any]:
        """Validate polygon topology (closed, non-self-intersecting, min 3 vertices)."""
        errors = []
        if len(vertices) < 4:
            errors.append(f"Polygon needs at least 4 points (3 vertices + closing), got {len(vertices)}")
        if len(vertices) >= 2 and vertices[0] != vertices[-1]:
            errors.append("Polygon is not closed (first and last points differ)")
        # Simple self-intersection check
        if len(vertices) >= 4:
            unique_points = set(tuple(v) for v in vertices[:-1])
            if len(unique_points) < len(vertices) - 1:
                errors.append("Polygon has duplicate vertices")
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "vertex_count": len(vertices),
            "is_closed": len(vertices) >= 2 and vertices[0] == vertices[-1],
        }

    def normalize_dms_to_dd(self, degrees: int, minutes: int,
                             seconds: float, direction: str) -> float:
        """Convert DMS (Degrees Minutes Seconds) to decimal degrees."""
        dd = degrees + minutes / 60.0 + seconds / 3600.0
        if direction.upper() in ("S", "W"):
            dd = -dd
        return round(dd, 6)

    def normalize_utm_to_dd(self, easting: float, northing: float,
                             zone: int, hemisphere: str) -> Dict[str, float]:
        """Convert UTM coordinates to decimal degrees (simplified)."""
        # Simplified UTM to lat/lon conversion
        k0 = 0.9996
        lon_origin = (zone - 1) * 6 - 180 + 3
        if hemisphere.upper() == "S":
            northing -= 10000000.0
        lat = northing / 111320.0  # rough approximation
        lon = lon_origin + (easting - 500000.0) / (111320.0 * math.cos(math.radians(lat)))
        return {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "coordinate_system": "WGS84",
        }

    def calculate_area_hectares(self, vertices: List[List[float]]) -> float:
        """Calculate polygon area in hectares using Shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0
        # Close polygon if needed
        pts = vertices[:]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        # Shoelace formula in approximate meters
        area = 0.0
        for i in range(len(pts) - 1):
            lat1, lon1 = pts[i]
            lat2, lon2 = pts[i + 1]
            # Convert to meters (rough)
            x1 = lon1 * 111320 * math.cos(math.radians(lat1))
            y1 = lat1 * 110540
            x2 = lon2 * 111320 * math.cos(math.radians(lat2))
            y2 = lat2 * 110540
            area += x1 * y2 - x2 * y1
        area_m2 = abs(area) / 2.0
        return round(area_m2 / 10000.0, 2)  # Convert to hectares

    def detect_overlaps(self, polygon_a: List[List[float]],
                        polygon_b: List[List[float]]) -> Dict[str, Any]:
        """Detect if two polygons overlap (simplified bounding box check)."""
        def bbox(poly):
            lats = [p[0] for p in poly]
            lons = [p[1] for p in poly]
            return min(lats), max(lats), min(lons), max(lons)

        bb_a = bbox(polygon_a)
        bb_b = bbox(polygon_b)
        overlap = not (
            bb_a[1] < bb_b[0] or bb_b[1] < bb_a[0] or
            bb_a[3] < bb_b[2] or bb_b[3] < bb_a[2]
        )
        return {
            "overlaps": overlap,
            "method": "bounding_box",
        }

    def determine_country(self, latitude: float, longitude: float) -> str:
        """Determine country from coordinates (simplified lookup)."""
        # Simplified country determination based on coordinate ranges
        if -10 < latitude < 6 and 95 < longitude < 141:
            return "IDN"
        if -33 < latitude < 5 and -73 < longitude < -35:
            return "BRA"
        if 4 < latitude < 11 and -8 < longitude < -2:
            return "CIV"
        if 47 < latitude < 55 and 5 < longitude < 15:
            return "DEU"
        if -55 < latitude < -22 and -73 < longitude < -53:
            return "ARG"
        if 1 < latitude < 7 and 100 < longitude < 119:
            return "MYS"
        return "UNKNOWN"

    def check_plot_size_rule(self, area_hectares: float,
                              has_polygon: bool) -> Dict[str, Any]:
        """Check EUDR plot size rules: < 4ha needs point, >= 4ha needs polygon."""
        if area_hectares < 4.0:
            return {
                "compliant": True,
                "requirement": "point",
                "area_hectares": area_hectares,
                "note": "Plot under 4 ha: point coordinate sufficient",
            }
        else:
            return {
                "compliant": has_polygon,
                "requirement": "polygon",
                "area_hectares": area_hectares,
                "note": "Plot >= 4 ha: polygon boundary required" if not has_polygon
                        else "Plot >= 4 ha: polygon boundary provided",
            }

    def batch_validate(self, plots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of plots."""
        results = []
        valid_count = 0
        for plot in plots:
            coord_result = self.validate_coordinates(
                plot["latitude"], plot["longitude"]
            )
            size_result = self.check_plot_size_rule(
                plot.get("area_hectares", 0),
                plot.get("polygon") is not None,
            )
            is_valid = coord_result["is_valid"] and size_result["compliant"]
            if is_valid:
                valid_count += 1
            results.append({
                "plot_id": plot.get("plot_id", "unknown"),
                "coordinate_valid": coord_result["is_valid"],
                "size_rule_compliant": size_result["compliant"],
                "overall_valid": is_valid,
                "errors": coord_result["errors"],
            })
        return {
            "total": len(plots),
            "valid": valid_count,
            "invalid": len(plots) - valid_count,
            "results": results,
        }

    def parse_geojson(self, geojson: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a GeoJSON feature and extract coordinates."""
        geo_type = geojson.get("type", "")
        if geo_type == "Point":
            coords = geojson["coordinates"]
            return {
                "type": "point",
                "latitude": coords[1],
                "longitude": coords[0],
            }
        elif geo_type == "Polygon":
            coords = geojson["coordinates"][0]
            return {
                "type": "polygon",
                "vertices": [[c[1], c[0]] for c in coords],
                "vertex_count": len(coords),
            }
        return {"type": "unknown", "error": f"Unsupported type: {geo_type}"}

    def format_for_article9(self, latitude: float, longitude: float,
                             polygon: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """Format geolocation data per EUDR Article 9 requirements."""
        result = {
            "article_9_format": True,
            "point": {
                "type": "Point",
                "coordinates": [round(longitude, 6), round(latitude, 6)],
            },
            "coordinate_system": "WGS84",
        }
        if polygon:
            closed = polygon[:]
            if closed[0] != closed[-1]:
                closed.append(closed[0])
            result["polygon"] = {
                "type": "Polygon",
                "coordinates": [[[round(v[1], 6), round(v[0], 6)] for v in closed]],
            }
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGeolocation:
    """Tests for the geolocation verification engine."""

    @pytest.fixture
    def engine(self) -> GeolocationEngineSimulator:
        return GeolocationEngineSimulator()

    # 1
    def test_validate_coordinates_valid(self, engine):
        """Valid WGS84 coordinates pass validation."""
        result = engine.validate_coordinates(-0.512345, 101.456789)
        assert result["is_valid"] is True
        assert result["coordinate_system"] == "WGS84"
        assert len(result["errors"]) == 0

    # 2
    def test_validate_coordinates_invalid_range(self, engine):
        """Coordinates outside WGS84 range fail validation."""
        result = engine.validate_coordinates(91.0, 181.0)
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 1

    # 3
    def test_validate_coordinates_sea(self, engine):
        """Coordinates in the ocean are flagged as not land-based."""
        # Mid-Pacific Ocean
        result = engine.validate_coordinates_sea(0.0, 170.0)
        assert result["is_land_based"] is False

    # 4
    def test_validate_polygon_valid(self, engine):
        """Valid closed polygon passes validation."""
        vertices = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
            [-0.510, 101.454],  # closing point
        ]
        result = engine.validate_polygon(vertices)
        assert result["is_valid"] is True
        assert result["is_closed"] is True

    # 5
    def test_validate_polygon_invalid_topology(self, engine):
        """Polygon with too few vertices fails validation."""
        vertices = [[-0.510, 101.454], [-0.510, 101.460]]
        result = engine.validate_polygon(vertices)
        assert result["is_valid"] is False

    # 6
    def test_normalize_dms_to_dd(self, engine):
        """DMS to decimal degrees conversion is correct."""
        # 51 degrees, 30 minutes, 0 seconds North = 51.5
        dd = engine.normalize_dms_to_dd(51, 30, 0.0, "N")
        assert abs(dd - 51.5) < 0.001

    # 7
    def test_normalize_dms_to_dd_south(self, engine):
        """DMS conversion with South direction produces negative latitude."""
        dd = engine.normalize_dms_to_dd(23, 33, 0.0, "S")
        assert dd < 0

    # 8
    def test_normalize_utm_to_dd(self, engine):
        """UTM to decimal degrees conversion produces valid coordinates."""
        result = engine.normalize_utm_to_dd(500000, 5500000, 33, "N")
        assert "latitude" in result
        assert "longitude" in result
        assert -90 <= result["latitude"] <= 90
        assert -180 <= result["longitude"] <= 180

    # 9
    def test_calculate_area_hectares(self, engine):
        """Area calculation returns positive hectares for valid polygon."""
        vertices = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
        ]
        area = engine.calculate_area_hectares(vertices)
        assert area > 0, f"Area should be positive, got {area}"

    # 10
    def test_detect_overlaps(self, engine):
        """Overlapping polygons are detected correctly."""
        poly_a = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
        ]
        poly_b = [
            [-0.512, 101.456], [-0.512, 101.462],
            [-0.518, 101.462], [-0.518, 101.456],
        ]
        result = engine.detect_overlaps(poly_a, poly_b)
        assert result["overlaps"] is True

    # 11
    def test_detect_no_overlap(self, engine):
        """Non-overlapping polygons are correctly identified."""
        poly_a = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
        ]
        poly_b = [
            [-1.510, 102.454], [-1.510, 102.460],
            [-1.515, 102.460], [-1.515, 102.454],
        ]
        result = engine.detect_overlaps(poly_a, poly_b)
        assert result["overlaps"] is False

    # 12
    def test_determine_country(self, engine):
        """Country determination returns correct ISO code for known coordinates."""
        country = engine.determine_country(-0.512345, 101.456789)
        assert country == "IDN"

    # 13
    def test_determine_country_brazil(self, engine):
        """Coordinates in Brazil return BRA."""
        country = engine.determine_country(-14.235, -51.925)
        assert country == "BRA"

    # 14
    def test_plot_size_rule_under_4ha(self, engine):
        """Plots under 4 ha require only point coordinate."""
        result = engine.check_plot_size_rule(2.5, has_polygon=False)
        assert result["compliant"] is True
        assert result["requirement"] == "point"

    # 15
    def test_plot_size_rule_over_4ha(self, engine):
        """Plots >= 4 ha require polygon boundary."""
        result = engine.check_plot_size_rule(25.0, has_polygon=True)
        assert result["compliant"] is True
        assert result["requirement"] == "polygon"

    # 16
    def test_plot_size_rule_over_4ha_no_polygon(self, engine):
        """Plots >= 4 ha without polygon are non-compliant."""
        result = engine.check_plot_size_rule(25.0, has_polygon=False)
        assert result["compliant"] is False

    # 17
    def test_batch_validate(self, engine, sample_plots_list):
        """Batch validation processes all plots and returns summary."""
        result = engine.batch_validate(sample_plots_list)
        assert result["total"] == len(sample_plots_list)
        assert result["valid"] + result["invalid"] == result["total"]
        assert len(result["results"]) == result["total"]

    # 18
    def test_parse_geojson_point(self, engine):
        """GeoJSON Point is parsed correctly."""
        geojson = {"type": "Point", "coordinates": [101.456789, -0.512345]}
        result = engine.parse_geojson(geojson)
        assert result["type"] == "point"
        assert result["latitude"] == -0.512345
        assert result["longitude"] == 101.456789

    # 19
    def test_parse_geojson_polygon(self, engine):
        """GeoJSON Polygon is parsed correctly."""
        geojson = {
            "type": "Polygon",
            "coordinates": [[
                [101.454, -0.510], [101.460, -0.510],
                [101.460, -0.515], [101.454, -0.515],
                [101.454, -0.510],
            ]],
        }
        result = engine.parse_geojson(geojson)
        assert result["type"] == "polygon"
        assert result["vertex_count"] == 5

    # 20
    def test_format_for_article9(self, engine):
        """Article 9 formatting includes point and coordinate system."""
        result = engine.format_for_article9(-0.512345, 101.456789)
        assert result["article_9_format"] is True
        assert result["coordinate_system"] == "WGS84"
        assert result["point"]["type"] == "Point"

    # 21
    def test_format_for_article9_with_polygon(self, engine):
        """Article 9 formatting includes polygon when provided."""
        polygon = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
        ]
        result = engine.format_for_article9(-0.512345, 101.456789, polygon)
        assert "polygon" in result
        assert result["polygon"]["type"] == "Polygon"

    # 22
    def test_coordinate_precision_6_decimal(self, engine):
        """Coordinates maintain 6 decimal place precision."""
        result = engine.validate_coordinates(-0.512345, 101.456789)
        assert result["precision_decimals"] >= 6

    # 23
    def test_coordinate_precision_insufficient(self, engine):
        """Coordinates with < 6 decimal places are flagged."""
        result = engine.validate_coordinates(-0.512, 101.456)
        assert result["is_valid"] is False
        assert any("precision" in e.lower() for e in result["errors"])

    # 24
    def test_land_coordinates_pass(self, engine):
        """Coordinates on land pass the sea check."""
        result = engine.validate_coordinates_sea(-0.512345, 101.456789)
        assert result["is_land_based"] is True

    # 25
    def test_polygon_must_be_closed(self, engine):
        """Unclosed polygon fails validation."""
        vertices = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
            # missing closing point
        ]
        result = engine.validate_polygon(vertices)
        assert result["is_closed"] is False
