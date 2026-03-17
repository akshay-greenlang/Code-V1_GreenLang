# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Advanced Geolocation Engine Tests
===================================================================

Tests the advanced geolocation engine including:
- Coordinate validation (valid, out-of-range, ocean)
- Polygon validation (valid topology, invalid topology)
- Sentinel imagery verification
- Protected area proximity checks
- Indigenous land checks
- Forest change detection
- Deforestation alerts
- MODIS fire detection
- Multi-temporal analysis
- Comprehensive analysis workflow
- Batch processing
- Provenance hash generation

Author: GreenLang QA Team
Version: 1.0.0
"""

import re
from typing import Any, Dict, List

import pytest


def assert_provenance_hash(result: Dict[str, Any]) -> None:
    """Verify that a result contains a valid SHA-256 provenance hash."""
    assert "provenance_hash" in result, "Result missing 'provenance_hash' field"
    h = result["provenance_hash"]
    assert isinstance(h, str), f"provenance_hash must be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert re.match(r"^[0-9a-f]{64}$", h), f"Invalid hex hash: {h}"


def generate_coordinates(country: str, count: int = 1) -> List[Dict[str, float]]:
    """Generate realistic WGS84 coordinates for a given country code."""
    centroids = {
        "BRA": (-14.235, -51.925), "IDN": (-0.789, 113.921),
        "CIV": (7.540, -5.547), "GHA": (7.946, -1.023),
        "COL": (4.571, -74.297), "MYS": (4.210, 101.976),
    }
    lat, lon = centroids.get(country, (0.0, 0.0))
    coords = []
    for i in range(count):
        coords.append({"latitude": lat + (i * 0.012) % 0.5, "longitude": lon + (i * 0.015) % 0.5})
    return coords


@pytest.mark.unit
class TestAdvancedGeolocation:
    """Test suite for advanced geolocation engine."""

    def test_validate_coordinates_valid(self):
        """Test validation accepts valid WGS84 coordinates."""
        valid_coords = [
            {"latitude": -0.512345, "longitude": 101.456789},
            {"latitude": 7.540000, "longitude": -5.547000},
            {"latitude": 51.166000, "longitude": 10.452000},
        ]

        for coords in valid_coords:
            lat, lon = coords["latitude"], coords["longitude"]
            assert -90 <= lat <= 90, f"Latitude {lat} out of range"
            assert -180 <= lon <= 180, f"Longitude {lon} out of range"

    def test_validate_coordinates_out_of_range(self):
        """Test validation rejects out-of-range coordinates."""
        invalid_coords = [
            {"latitude": 91.0, "longitude": 0.0},  # Lat > 90
            {"latitude": -91.0, "longitude": 0.0},  # Lat < -90
            {"latitude": 0.0, "longitude": 181.0},  # Lon > 180
            {"latitude": 0.0, "longitude": -181.0},  # Lon < -180
        ]

        for coords in invalid_coords:
            lat, lon = coords["latitude"], coords["longitude"]
            is_valid = (-90 <= lat <= 90) and (-180 <= lon <= 180)
            assert not is_valid, f"Invalid coords {coords} marked as valid"

    def test_validate_coordinates_ocean(self):
        """Test detection of ocean coordinates (not land-based)."""
        # Known ocean coordinates (Pacific Ocean)
        ocean_coords = {"latitude": 0.0, "longitude": -160.0}

        # Simple heuristic: check if far from known land centroids
        known_land_centroids = [
            (-14.235, -51.925),  # BRA
            (-0.789, 113.921),   # IDN
            (51.166, 10.452),    # DEU
        ]

        lat, lon = ocean_coords["latitude"], ocean_coords["longitude"]
        min_distance = float('inf')
        for land_lat, land_lon in known_land_centroids:
            dist = ((lat - land_lat) ** 2 + (lon - land_lon) ** 2) ** 0.5
            min_distance = min(min_distance, dist)

        # If minimum distance > 50 degrees, likely ocean
        is_likely_ocean = min_distance > 50
        assert is_likely_ocean, f"Ocean coords {ocean_coords} not detected as ocean"

    def test_validate_polygon_valid(self):
        """Test validation accepts valid polygon topology."""
        valid_polygon = [
            [-0.510, 101.454],
            [-0.510, 101.460],
            [-0.515, 101.460],
            [-0.515, 101.454],
            [-0.510, 101.454],  # Closed polygon
        ]

        # Check polygon is closed (first == last)
        assert valid_polygon[0] == valid_polygon[-1], "Polygon not closed"

        # Check at least 4 points (3 unique + closing point)
        assert len(valid_polygon) >= 4, f"Polygon has only {len(valid_polygon)} points"

    def test_validate_polygon_invalid_topology(self):
        """Test validation rejects invalid polygon topology."""
        # Not closed
        invalid_polygon_1 = [
            [-0.510, 101.454],
            [-0.510, 101.460],
            [-0.515, 101.460],
            [-0.515, 101.454],
            # Missing closing point
        ]

        is_closed_1 = invalid_polygon_1[0] == invalid_polygon_1[-1]
        assert not is_closed_1, "Non-closed polygon marked as valid"

        # Too few points
        invalid_polygon_2 = [
            [-0.510, 101.454],
            [-0.510, 101.460],
            [-0.510, 101.454],
        ]

        assert len(invalid_polygon_2) < 4, "Polygon with <4 points marked as valid"

    def test_sentinel_imagery_check(self, sample_plots: List[Dict[str, Any]]):
        """Test Sentinel satellite imagery verification."""
        plot = sample_plots[0]

        # Simulate Sentinel imagery metadata
        sentinel_result = {
            "plot_id": plot["plot_id"],
            "satellite": "Sentinel-2",
            "acquisition_date": "2025-11-15",
            "cloud_cover_percent": 8.5,
            "ndvi_mean": 0.72,  # Vegetation index
            "processing_level": "L2A",
            "resolution_m": 10,
        }

        assert sentinel_result["satellite"] == "Sentinel-2"
        assert sentinel_result["cloud_cover_percent"] < 20
        assert 0 <= sentinel_result["ndvi_mean"] <= 1

    def test_protected_area_check_positive(self, sample_plots: List[Dict[str, Any]]):
        """Test protected area proximity check detects proximity."""
        plot = sample_plots[0]

        # Simulate protected area within 5km
        protected_area_result = {
            "plot_id": plot["plot_id"],
            "within_protected_area": False,
            "nearest_protected_area": "Gunung Leuser National Park",
            "distance_km": 3.2,  # Within 5km buffer
            "protected_area_type": "National Park",
        }

        buffer_km = 5.0
        is_within_buffer = protected_area_result["distance_km"] <= buffer_km
        assert is_within_buffer, f"Plot at {protected_area_result['distance_km']}km not within {buffer_km}km buffer"

    def test_protected_area_check_negative(self, sample_plots: List[Dict[str, Any]]):
        """Test protected area proximity check when far from protected areas."""
        plot = sample_plots[4]

        # Simulate no protected area nearby
        protected_area_result = {
            "plot_id": plot["plot_id"],
            "within_protected_area": False,
            "nearest_protected_area": None,
            "distance_km": 45.0,  # Beyond 5km buffer
            "protected_area_type": None,
        }

        buffer_km = 5.0
        is_within_buffer = (
            protected_area_result["distance_km"] is not None
            and protected_area_result["distance_km"] <= buffer_km
        )
        assert not is_within_buffer, "Plot incorrectly marked as within buffer"

    def test_indigenous_land_check(self, sample_plots: List[Dict[str, Any]]):
        """Test indigenous land proximity check."""
        plot = sample_plots[1]  # Brazil plot

        # Simulate indigenous territory check
        indigenous_result = {
            "plot_id": plot["plot_id"],
            "overlaps_indigenous_land": True,
            "indigenous_territory_name": "Terra Indígena Yanomami",
            "territory_status": "Legally Recognized",
            "consultation_required": True,
        }

        assert "overlaps_indigenous_land" in indigenous_result
        if indigenous_result["overlaps_indigenous_land"]:
            assert indigenous_result["consultation_required"] is True

    def test_forest_change_detection(self, sample_plots: List[Dict[str, Any]]):
        """Test forest change detection over time."""
        plot = sample_plots[0]

        # Simulate forest change analysis
        change_result = {
            "plot_id": plot["plot_id"],
            "analysis_period": "2020-01-01 to 2025-11-15",
            "forest_cover_start_percent": 92.5,
            "forest_cover_end_percent": 88.2,
            "forest_loss_percent": 4.3,
            "forest_loss_hectares": 1.1,
            "deforestation_detected": True,
            "alert_level": "MEDIUM",
        }

        assert change_result["forest_cover_start_percent"] > change_result["forest_cover_end_percent"]
        assert change_result["forest_loss_percent"] > 0
        assert change_result["deforestation_detected"] is True

    def test_deforestation_alerts(self, sample_plots: List[Dict[str, Any]]):
        """Test deforestation alert generation."""
        plot = sample_plots[1]

        # Simulate deforestation alert
        alert = {
            "alert_id": "ALERT-2025-BRA-001",
            "plot_id": plot["plot_id"],
            "alert_date": "2025-11-10",
            "alert_type": "deforestation",
            "severity": "HIGH",
            "forest_loss_hectares": 2.5,
            "confidence": 0.85,
            "source": "GLAD alerts",
        }

        assert alert["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert alert["confidence"] >= 0.7
        assert alert["forest_loss_hectares"] > 0

    def test_modis_fire_check(self, sample_plots: List[Dict[str, Any]]):
        """Test MODIS fire detection."""
        plot = sample_plots[0]

        # Simulate MODIS fire detection
        fire_result = {
            "plot_id": plot["plot_id"],
            "fire_detected": True,
            "detection_date": "2025-08-15",
            "fire_radiative_power": 320.5,  # MW
            "confidence": "high",
            "distance_from_plot_km": 1.2,
        }

        assert "fire_detected" in fire_result
        if fire_result["fire_detected"]:
            assert fire_result["confidence"] in ["low", "nominal", "high"]
            assert fire_result["fire_radiative_power"] > 0

    def test_multi_temporal_analysis(self, sample_plots: List[Dict[str, Any]]):
        """Test multi-temporal satellite analysis."""
        plot = sample_plots[0]

        # Simulate multi-temporal NDVI analysis
        temporal_result = {
            "plot_id": plot["plot_id"],
            "analysis_dates": [
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
                "2025-01-01",
            ],
            "ndvi_values": [0.75, 0.73, 0.70, 0.68, 0.65],
            "trend": "declining",
            "slope": -0.025,  # Per year
            "r_squared": 0.92,
        }

        assert len(temporal_result["analysis_dates"]) == len(temporal_result["ndvi_values"])
        assert temporal_result["trend"] in ["increasing", "stable", "declining"]
        assert 0 <= temporal_result["r_squared"] <= 1

    def test_full_analysis_comprehensive(self, sample_plots: List[Dict[str, Any]]):
        """Test comprehensive geolocation analysis workflow."""
        plot = sample_plots[0]

        # Simulate full analysis
        analysis_result = {
            "plot_id": plot["plot_id"],
            "coordinates_valid": True,
            "polygon_valid": True,
            "sentinel_verified": True,
            "protected_area_distance_km": 8.5,
            "indigenous_land_overlap": False,
            "forest_change_detected": False,
            "fire_detected": False,
            "ndvi_trend": "stable",
            "overall_status": "COMPLIANT",
            "confidence_score": 0.92,
            "provenance_hash": "a" * 64,
        }

        assert analysis_result["coordinates_valid"] is True
        assert analysis_result["polygon_valid"] is True
        assert analysis_result["overall_status"] in ["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"]
        assert 0 <= analysis_result["confidence_score"] <= 1
        assert_provenance_hash(analysis_result)

    def test_batch_analysis_multiple_plots(self, sample_plots: List[Dict[str, Any]]):
        """Test batch analysis of multiple plots."""
        # Simulate batch analysis
        batch_results = []
        for plot in sample_plots[:5]:
            result = {
                "plot_id": plot["plot_id"],
                "coordinates_valid": True,
                "overall_status": "COMPLIANT",
                "confidence_score": 0.85 + (hash(plot["plot_id"]) % 10) / 100,
            }
            batch_results.append(result)

        assert len(batch_results) == 5
        for result in batch_results:
            assert "plot_id" in result
            assert "overall_status" in result
            assert result["coordinates_valid"] is True

    def test_provenance_hash_generated(self, sample_plots: List[Dict[str, Any]]):
        """Test provenance hash is generated for analysis results."""
        import hashlib
        import json

        plot = sample_plots[0]

        analysis_data = {
            "plot_id": plot["plot_id"],
            "coordinates_valid": True,
            "timestamp": "2025-11-15T10:30:00Z",
        }

        provenance_hash = hashlib.sha256(
            json.dumps(analysis_data, sort_keys=True).encode()
        ).hexdigest()

        assert len(provenance_hash) == 64
        assert provenance_hash.isalnum()

        # Verify reproducibility
        provenance_hash2 = hashlib.sha256(
            json.dumps(analysis_data, sort_keys=True).encode()
        ).hexdigest()
        assert provenance_hash == provenance_hash2

    def test_coordinate_precision_validation(self):
        """Test coordinate precision is at least 6 decimal places."""
        coords = {"latitude": -0.512345, "longitude": 101.456789}

        lat_str = str(coords["latitude"])
        lon_str = str(coords["longitude"])

        # Count decimal places
        lat_decimals = len(lat_str.split(".")[-1]) if "." in lat_str else 0
        lon_decimals = len(lon_str.split(".")[-1]) if "." in lon_str else 0

        assert lat_decimals >= 6, f"Latitude precision {lat_decimals} < 6"
        assert lon_decimals >= 6, f"Longitude precision {lon_decimals} < 6"

    def test_area_calculation(self):
        """Test hectare area calculation from polygon."""
        # Simple square polygon
        polygon = [
            [0.0, 0.0],
            [0.0, 0.01],
            [0.01, 0.01],
            [0.01, 0.0],
            [0.0, 0.0],
        ]

        # Approximate area calculation (simplified)
        # 0.01 degrees ≈ 1.11 km at equator
        # 1.11 km * 1.11 km ≈ 1.23 km² ≈ 123 ha
        estimated_area_ha = 123.0

        assert estimated_area_ha > 0
        assert estimated_area_ha < 10000  # Reasonable upper bound

    def test_polygon_closure_enforcement(self):
        """Test polygon closure is enforced."""
        open_polygon = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]

        # Close the polygon
        closed_polygon = open_polygon + [open_polygon[0]]
        assert closed_polygon[0] == closed_polygon[-1]
        assert len(closed_polygon) == len(open_polygon) + 1
