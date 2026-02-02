"""
GL-EUDR-002: Spatial Validation Tests

Test suite for spatial validation services:
- Country boundary validation (GADM)
- Water body detection (OSM)
- Protected area checks (WDPA)
- Urban area detection
- Spatial indexing and queries

Run with: pytest test_spatial.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from .spatial import (
    # Main service
    SpatialValidationService,
    # Individual services
    CountryBoundaryService,
    WaterBodyService,
    ProtectedAreaService,
    UrbanAreaService,
    # Models
    SpatialFeature,
    SpatialQueryResult,
    BoundingBox,
    SpatialIndex,
    # Data loading
    SpatialDataSource,
    BoundaryLevel,
    GeoJSONLoader,
    # Utilities
    normalize_longitude,
    crosses_dateline,
    split_dateline_polygon,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def spatial_service():
    """Create a fresh spatial validation service."""
    return SpatialValidationService()


@pytest.fixture
def country_service():
    """Create a fresh country boundary service."""
    return CountryBoundaryService()


@pytest.fixture
def water_service():
    """Create a fresh water body service."""
    return WaterBodyService()


@pytest.fixture
def protected_service():
    """Create a fresh protected area service."""
    return ProtectedAreaService()


@pytest.fixture
def urban_service():
    """Create a fresh urban area service."""
    return UrbanAreaService()


@pytest.fixture
def sample_geojson_file(tmp_path):
    """Create a sample GeoJSON file for testing."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "test-1",
                    "name": "Test Area 1",
                    "ISO3": "IDN"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [102.0, -5.0],
                        [103.0, -5.0],
                        [103.0, -4.0],
                        [102.0, -4.0],
                        [102.0, -5.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "test-2",
                    "name": "Test Area 2"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [104.0, -5.0],
                        [105.0, -5.0],
                        [105.0, -4.0],
                        [104.0, -4.0],
                        [104.0, -5.0]
                    ]]
                }
            }
        ]
    }

    file_path = tmp_path / "test_features.geojson"
    with open(file_path, 'w') as f:
        json.dump(geojson, f)

    return str(file_path)


@pytest.fixture
def sample_protected_areas_file(tmp_path):
    """Create a sample WDPA-style GeoJSON file."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "wdpa-1",
                    "name": "Bukit Barisan National Park",
                    "IUCN_CAT": "II",
                    "DESIG": "National Park",
                    "STATUS": "Designated",
                    "ISO3": "IDN"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [102.5, -4.5],
                        [102.7, -4.5],
                        [102.7, -4.3],
                        [102.5, -4.3],
                        [102.5, -4.5]
                    ]]
                }
            }
        ]
    }

    file_path = tmp_path / "wdpa_test.geojson"
    with open(file_path, 'w') as f:
        json.dump(geojson, f)

    return str(file_path)


# =============================================================================
# BOUNDING BOX TESTS
# =============================================================================

class TestBoundingBox:
    """Test BoundingBox functionality."""

    def test_contains_point_inside(self):
        """Test point containment check - inside."""
        bbox = BoundingBox(min_lon=100, min_lat=-5, max_lon=105, max_lat=0)

        assert bbox.contains_point(-2.5, 102.5) is True
        assert bbox.contains_point(-5, 100) is True  # Corner
        assert bbox.contains_point(0, 105) is True  # Corner

    def test_contains_point_outside(self):
        """Test point containment check - outside."""
        bbox = BoundingBox(min_lon=100, min_lat=-5, max_lon=105, max_lat=0)

        assert bbox.contains_point(-6, 102.5) is False  # Below
        assert bbox.contains_point(-2.5, 106) is False  # Right
        assert bbox.contains_point(1, 102.5) is False   # Above

    def test_from_coords(self):
        """Test creating bounding box from coordinates."""
        coords = [
            [102.0, -5.0],
            [103.0, -5.0],
            [103.0, -4.0],
            [102.0, -4.0],
        ]

        bbox = BoundingBox.from_coords(coords)

        assert bbox.min_lon == 102.0
        assert bbox.max_lon == 103.0
        assert bbox.min_lat == -5.0
        assert bbox.max_lat == -4.0


# =============================================================================
# SPATIAL INDEX TESTS
# =============================================================================

class TestSpatialIndex:
    """Test SpatialIndex functionality."""

    def test_insert_and_query(self):
        """Test inserting features and querying."""
        index = SpatialIndex(grid_size=1.0)

        feature = SpatialFeature(
            feature_id="test-1",
            name="Test Feature",
            geometry_type="Polygon",
            bounding_box=BoundingBox(102.0, -5.0, 103.0, -4.0),
            properties={},
            coordinates=[[
                [102.0, -5.0],
                [103.0, -5.0],
                [103.0, -4.0],
                [102.0, -4.0],
                [102.0, -5.0]
            ]]
        )

        index.insert(feature)

        # Query point inside
        results = index.query_point(-4.5, 102.5)
        assert len(results) == 1
        assert results[0].feature_id == "test-1"

    def test_query_empty_area(self):
        """Test querying an area with no features."""
        index = SpatialIndex(grid_size=1.0)

        results = index.query_point(0, 0)
        assert len(results) == 0

    def test_multiple_features(self):
        """Test querying with multiple overlapping features."""
        index = SpatialIndex(grid_size=1.0)

        for i in range(3):
            feature = SpatialFeature(
                feature_id=f"test-{i}",
                name=f"Test Feature {i}",
                geometry_type="Polygon",
                bounding_box=BoundingBox(102.0, -5.0, 103.0, -4.0),
                properties={},
                coordinates=[[
                    [102.0, -5.0],
                    [103.0, -5.0],
                    [103.0, -4.0],
                    [102.0, -4.0],
                    [102.0, -5.0]
                ]]
            )
            index.insert(feature)

        results = index.query_point(-4.5, 102.5)
        assert len(results) == 3

    def test_feature_count(self):
        """Test feature counting."""
        index = SpatialIndex()

        assert index.feature_count == 0

        feature = SpatialFeature(
            feature_id="test-1",
            name="Test",
            geometry_type="Point",
            bounding_box=BoundingBox(0, 0, 1, 1),
            properties={},
            coordinates=[0.5, 0.5]
        )
        index.insert(feature)

        assert index.feature_count == 1


# =============================================================================
# SPATIAL FEATURE TESTS
# =============================================================================

class TestSpatialFeature:
    """Test SpatialFeature functionality."""

    def test_polygon_contains_point(self):
        """Test polygon containment."""
        feature = SpatialFeature(
            feature_id="test-1",
            name="Test",
            geometry_type="Polygon",
            bounding_box=BoundingBox(102.0, -5.0, 103.0, -4.0),
            properties={},
            coordinates=[[
                [102.0, -5.0],
                [103.0, -5.0],
                [103.0, -4.0],
                [102.0, -4.0],
                [102.0, -5.0]
            ]]
        )

        # Inside
        assert feature.contains_point(-4.5, 102.5) is True

        # Outside
        assert feature.contains_point(-6, 102.5) is False

    def test_multipolygon_contains_point(self):
        """Test MultiPolygon containment."""
        feature = SpatialFeature(
            feature_id="test-1",
            name="Test",
            geometry_type="MultiPolygon",
            bounding_box=BoundingBox(100.0, -6.0, 106.0, -3.0),
            properties={},
            coordinates=[
                [[  # First polygon
                    [102.0, -5.0],
                    [103.0, -5.0],
                    [103.0, -4.0],
                    [102.0, -4.0],
                    [102.0, -5.0]
                ]],
                [[  # Second polygon
                    [104.0, -5.0],
                    [105.0, -5.0],
                    [105.0, -4.0],
                    [104.0, -4.0],
                    [104.0, -5.0]
                ]]
            ]
        )

        # In first polygon
        assert feature.contains_point(-4.5, 102.5) is True

        # In second polygon
        assert feature.contains_point(-4.5, 104.5) is True

        # Between polygons
        assert feature.contains_point(-4.5, 103.5) is False

    def test_point_feature(self):
        """Test Point feature."""
        feature = SpatialFeature(
            feature_id="test-1",
            name="Test Point",
            geometry_type="Point",
            bounding_box=BoundingBox(102.49, -4.51, 102.51, -4.49),
            properties={},
            coordinates=[102.5, -4.5]
        )

        # Very close
        assert feature.contains_point(-4.5, 102.5) is True

        # Far away
        assert feature.contains_point(-5.0, 103.0) is False


# =============================================================================
# GEOJSON LOADER TESTS
# =============================================================================

class TestGeoJSONLoader:
    """Test GeoJSON loading."""

    def test_load_feature_collection(self, sample_geojson_file):
        """Test loading a FeatureCollection."""
        loader = GeoJSONLoader()
        features = loader.load(sample_geojson_file)

        assert len(features) == 2
        assert features[0].name == "Test Area 1"
        assert features[1].name == "Test Area 2"

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        loader = GeoJSONLoader()
        features = loader.load("/nonexistent/path.geojson")

        assert len(features) == 0

    def test_feature_properties(self, sample_geojson_file):
        """Test that feature properties are preserved."""
        loader = GeoJSONLoader()
        features = loader.load(sample_geojson_file)

        assert features[0].properties.get("ISO3") == "IDN"


# =============================================================================
# COUNTRY BOUNDARY SERVICE TESTS
# =============================================================================

class TestCountryBoundaryService:
    """Test country boundary validation."""

    def test_point_in_country_with_bbox_fallback(self, country_service):
        """Test point-in-country with bounding box fallback."""
        # Indonesia (within bbox)
        assert country_service.is_point_in_country(-4.5, 102.5, "ID") is True

        # Brazil
        assert country_service.is_point_in_country(-15.0, -47.0, "BR") is True

        # Ghana
        assert country_service.is_point_in_country(7.0, -1.0, "GH") is True

    def test_point_outside_country(self, country_service):
        """Test point clearly outside country."""
        # Point in Europe, but claiming to be Indonesia
        # Note: Without full GADM data, this relies on bounding box
        # A point in Germany would be outside Indonesia's bbox
        assert country_service.is_point_in_country(52.0, 13.0, "ID") is False

    def test_unknown_country_code(self, country_service):
        """Test handling of unknown country codes."""
        # Unknown country returns True (permissive without data)
        result = country_service.is_point_in_country(-4.5, 102.5, "XX")
        assert result is True  # No data, so passes

    def test_get_country_at_point(self, country_service):
        """Test country detection at point."""
        # Should detect Indonesia
        country = country_service.get_country_at_point(-4.5, 102.5)
        assert country == "ID"

        # Should detect Brazil
        country = country_service.get_country_at_point(-15.0, -47.0)
        assert country == "BR"


# =============================================================================
# PROTECTED AREA SERVICE TESTS
# =============================================================================

class TestProtectedAreaService:
    """Test protected area detection."""

    def test_check_protected_area_not_loaded(self, protected_service):
        """Test when no data is loaded."""
        result = protected_service.check_protected_area(-4.5, 102.5)

        assert result.found is False

    def test_check_protected_area_with_data(self, tmp_path, sample_protected_areas_file):
        """Test protected area detection with loaded data."""
        service = ProtectedAreaService(str(tmp_path))
        service.load_protected_areas()

        # Point inside protected area
        result = service.check_protected_area(-4.4, 102.6)

        assert result.found is True
        assert "Bukit Barisan" in result.feature_name

    def test_check_polygon_overlap(self, tmp_path, sample_protected_areas_file):
        """Test polygon overlap detection."""
        service = ProtectedAreaService(str(tmp_path))
        service.load_protected_areas()

        # Polygon that overlaps protected area
        polygon_coords = [
            [102.4, -4.6],
            [102.8, -4.6],
            [102.8, -4.2],
            [102.4, -4.2],
            [102.4, -4.6]
        ]

        results = service.check_polygon_overlap(polygon_coords)

        assert len(results) >= 1


# =============================================================================
# SPATIAL VALIDATION SERVICE TESTS
# =============================================================================

class TestSpatialValidationService:
    """Test the main spatial validation service."""

    def test_validate_location(self, spatial_service):
        """Test comprehensive location validation."""
        result = spatial_service.validate_location(-4.5, 102.5, "ID")

        assert "in_expected_country" in result
        assert "in_water" in result
        assert "in_protected_area" in result
        assert "in_urban_area" in result

    def test_validate_polygon(self, spatial_service):
        """Test polygon validation."""
        polygon_coords = [
            [102.0, -5.0],
            [103.0, -5.0],
            [103.0, -4.0],
            [102.0, -4.0],
            [102.0, -5.0]
        ]

        result = spatial_service.validate_polygon(polygon_coords, "ID")

        assert "in_expected_country" in result

    def test_initialize_with_paths(self, tmp_path, sample_geojson_file):
        """Test initialization with data paths."""
        service = SpatialValidationService()
        service.initialize(gadm_path=str(tmp_path))

        # Should not raise
        assert service._initialized is True


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test spatial utility functions."""

    def test_normalize_longitude(self):
        """Test longitude normalization."""
        assert normalize_longitude(180) == 180
        assert normalize_longitude(-180) == -180
        assert normalize_longitude(181) == -179
        assert normalize_longitude(-181) == 179
        assert normalize_longitude(360) == 0
        assert normalize_longitude(540) == 180

    def test_crosses_dateline_false(self):
        """Test polygon that doesn't cross dateline."""
        coords = [
            [102.0, -5.0],
            [103.0, -5.0],
            [103.0, -4.0],
            [102.0, -4.0],
            [102.0, -5.0]
        ]

        assert crosses_dateline(coords) is False

    def test_crosses_dateline_true(self):
        """Test polygon that crosses dateline."""
        coords = [
            [179.0, 0.0],
            [-179.0, 0.0],  # Jump across dateline
            [-179.0, 1.0],
            [179.0, 1.0],
            [179.0, 0.0]
        ]

        assert crosses_dateline(coords) is True

    def test_split_dateline_polygon(self):
        """Test splitting a dateline-crossing polygon."""
        coords = [
            [179.0, 0.0],
            [-179.0, 0.0],
            [-179.0, 1.0],
            [179.0, 1.0],
            [179.0, 0.0]
        ]

        result = split_dateline_polygon(coords)

        assert len(result) == 2
        # Both parts should have the same number of points
        assert len(result[0]) == len(result[1])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSpatialIntegration:
    """Integration tests for spatial services."""

    def test_full_validation_workflow(self, tmp_path, sample_protected_areas_file):
        """Test complete spatial validation workflow."""
        # Setup
        service = SpatialValidationService()
        service.initialize(wdpa_path=str(tmp_path))

        # Validate a location
        result = service.validate_location(-4.4, 102.6, "ID")

        # Check all fields are present
        assert "in_expected_country" in result
        assert "in_water" in result
        assert "in_protected_area" in result
        assert "in_urban_area" in result

        # This point is in the protected area
        assert result["in_protected_area"] is True

    def test_country_bbox_coverage(self, country_service):
        """Test that EUDR-relevant countries have bounding boxes."""
        eudr_countries = ["BR", "ID", "MY", "PE", "CO", "GH", "CI"]

        for country in eudr_countries:
            # Should have bbox data
            bbox = country_service.COUNTRY_BBOX.get(country)
            assert bbox is not None, f"Missing bbox for {country}"

    def test_spatial_index_performance(self):
        """Test spatial index handles many features."""
        index = SpatialIndex(grid_size=1.0)

        # Insert 100 features
        for i in range(100):
            lat = -5.0 + (i * 0.1)
            lon = 102.0 + (i * 0.1)
            feature = SpatialFeature(
                feature_id=f"test-{i}",
                name=f"Feature {i}",
                geometry_type="Polygon",
                bounding_box=BoundingBox(lon, lat, lon + 0.5, lat + 0.5),
                properties={},
                coordinates=[[
                    [lon, lat],
                    [lon + 0.5, lat],
                    [lon + 0.5, lat + 0.5],
                    [lon, lat + 0.5],
                    [lon, lat]
                ]]
            )
            index.insert(feature)

        assert index.feature_count == 100

        # Query should be fast
        import time
        start = time.time()
        for _ in range(100):
            index.query_point(-4.0, 103.0)
        elapsed = time.time() - start

        # 100 queries should be fast
        assert elapsed < 0.5
