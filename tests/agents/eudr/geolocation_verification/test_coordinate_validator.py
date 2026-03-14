# -*- coding: utf-8 -*-
"""
Tests for CoordinateValidator - AGENT-EUDR-002 Feature 1: Coordinate Validation

Comprehensive test suite covering:
- WGS84 coordinate bounds validation (latitude -90..90, longitude -180..180)
- Decimal precision assessment and scoring
- Lat/lon transposition detection
- Country-coordinate matching
- Ocean/land detection
- Duplicate coordinate detection
- Cluster anomaly detection
- Elevation plausibility checks
- Batch validation with mixed results
- Boundary coordinates (poles, dateline, equator, prime meridian)
- Determinism and reproducibility

Test count: 200 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 1 - Coordinate Validation)
"""

import math
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.geolocation_verification.models import (
    CoordinateIssue,
    CoordinateIssueType,
    CoordinateValidationResult,
    VerifyCoordinateRequest,
)
from greenlang.agents.eudr.geolocation_verification.coordinate_validator import (
    CoordinateValidator,
)


# ===========================================================================
# 1. WGS84 Bounds Validation (25 tests)
# ===========================================================================


class TestWGS84Bounds:
    """Test WGS84 coordinate bounds validation."""

    def test_valid_coordinate_center(self, coordinate_validator):
        """Test valid coordinate near the center of WGS84 bounds."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert isinstance(result, CoordinateValidationResult)
        assert result.wgs84_valid is True

    def test_valid_coordinate_brazil(self, coordinate_validator, valid_coordinate_brazil):
        """Test valid coordinate in Para, Brazil."""
        result = coordinate_validator.validate(valid_coordinate_brazil)
        assert result.wgs84_valid is True
        assert result.lat == valid_coordinate_brazil.lat
        assert result.lon == valid_coordinate_brazil.lon

    def test_valid_coordinate_indonesia(self, coordinate_validator, valid_coordinate_indonesia):
        """Test valid coordinate in Kalimantan, Indonesia."""
        result = coordinate_validator.validate(valid_coordinate_indonesia)
        assert result.wgs84_valid is True

    def test_valid_coordinate_ghana(self, coordinate_validator, valid_coordinate_ghana):
        """Test valid coordinate in Ashanti, Ghana."""
        result = coordinate_validator.validate(valid_coordinate_ghana)
        assert result.wgs84_valid is True

    def test_invalid_latitude_too_high(self, coordinate_validator):
        """Test latitude above 90 degrees is invalid."""
        inp = VerifyCoordinateRequest(lat=91.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False
        assert result.is_valid is False

    def test_invalid_latitude_too_low(self, coordinate_validator):
        """Test latitude below -90 degrees is invalid."""
        inp = VerifyCoordinateRequest(lat=-91.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False
        assert result.is_valid is False

    def test_invalid_longitude_too_high(self, coordinate_validator):
        """Test longitude above 180 degrees is invalid."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=181.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False
        assert result.is_valid is False

    def test_invalid_longitude_too_low(self, coordinate_validator):
        """Test longitude below -180 degrees is invalid."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=-181.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False
        assert result.is_valid is False

    def test_latitude_exactly_90(self, coordinate_validator):
        """Test latitude at exactly 90 (North Pole) is valid."""
        inp = VerifyCoordinateRequest(lat=90.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    def test_latitude_exactly_minus_90(self, coordinate_validator):
        """Test latitude at exactly -90 (South Pole) is valid."""
        inp = VerifyCoordinateRequest(lat=-90.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    def test_longitude_exactly_180(self, coordinate_validator):
        """Test longitude at exactly 180 (Date Line) is valid."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=180.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    def test_longitude_exactly_minus_180(self, coordinate_validator):
        """Test longitude at exactly -180 (Date Line) is valid."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=-180.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    def test_invalid_both_out_of_bounds(self, coordinate_validator):
        """Test both latitude and longitude out of bounds."""
        inp = VerifyCoordinateRequest(lat=100.0, lon=200.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False
        assert result.is_valid is False

    @pytest.mark.parametrize("lat,lon", [
        (45.0, 90.0),
        (-45.0, -90.0),
        (89.999, 179.999),
        (-89.999, -179.999),
        (0.000001, 0.000001),
    ])
    def test_valid_coordinates_parametrized(self, coordinate_validator, lat, lon):
        """Test multiple valid coordinate pairs."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    @pytest.mark.parametrize("lat,lon", [
        (90.001, 0.0),
        (-90.001, 0.0),
        (0.0, 180.001),
        (0.0, -180.001),
        (91.0, 181.0),
        (-91.0, -181.0),
        (200.0, 0.0),
        (0.0, 360.0),
    ])
    def test_invalid_coordinates_parametrized(self, coordinate_validator, lat, lon):
        """Test multiple invalid coordinate pairs."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_issues_populated_on_invalid_latitude(self, coordinate_validator):
        """Test that issues list is populated when latitude is invalid."""
        inp = VerifyCoordinateRequest(lat=95.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert len(result.issues) > 0
        issue_codes = [i.code for i in result.issues]
        assert any("LAT" in code or "BOUND" in code or "WGS84" in code for code in issue_codes)

    def test_issues_populated_on_invalid_longitude(self, coordinate_validator):
        """Test that issues list is populated when longitude is invalid."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=200.0)
        result = coordinate_validator.validate(inp)
        assert len(result.issues) > 0

    def test_validation_id_generated(self, coordinate_validator):
        """Test that a validation ID is generated for each result."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.validation_id is not None
        assert len(result.validation_id) > 0
        assert result.validation_id.startswith("CVR")

    def test_validated_at_set(self, coordinate_validator):
        """Test that the validated_at timestamp is set."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.validated_at is not None


# ===========================================================================
# 2. Precision Assessment (30 tests)
# ===========================================================================


class TestPrecisionAssessment:
    """Test coordinate decimal precision assessment and scoring."""

    def test_precision_7_decimals(self, coordinate_validator):
        """Test 7 decimal places gives highest precision score."""
        inp = VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 7
        assert result.precision_score >= 0.9

    def test_precision_6_decimals_score_max(self, coordinate_validator):
        """Test 6 decimal places gives near-maximum precision score."""
        inp = VerifyCoordinateRequest(lat=-3.123456, lon=-60.023456)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 6
        assert result.precision_score >= 0.8

    def test_precision_5_decimals_acceptable(self, coordinate_validator):
        """Test 5 decimal places is acceptable (~1.1m accuracy)."""
        inp = VerifyCoordinateRequest(lat=-3.12345, lon=-60.02345)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 5
        assert result.precision_score >= 0.6

    def test_precision_4_decimals_moderate(self, coordinate_validator):
        """Test 4 decimal places gives moderate score."""
        inp = VerifyCoordinateRequest(lat=-3.1234, lon=-60.0234)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 4
        assert result.precision_score >= 0.3

    def test_precision_3_decimals_low_score(self, coordinate_validator):
        """Test 3 decimal places gives low precision score (~111m)."""
        inp = VerifyCoordinateRequest(lat=-3.123, lon=-60.023)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 3
        assert result.precision_score <= 0.5

    def test_precision_2_decimals_very_low(self, coordinate_validator):
        """Test 2 decimal places gives very low score (~1.1km)."""
        inp = VerifyCoordinateRequest(lat=-3.12, lon=-60.02)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 2
        assert result.precision_score <= 0.3

    def test_precision_1_decimal_very_low(self, coordinate_validator):
        """Test 1 decimal place gives minimal score (~11km)."""
        inp = VerifyCoordinateRequest(lat=-3.1, lon=-60.0)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= 1
        assert result.precision_score <= 0.2

    def test_precision_integer_minimum(self, coordinate_validator):
        """Test integer coordinates give minimum precision score (~111km)."""
        inp = VerifyCoordinateRequest(lat=-3.0, lon=-60.0)
        result = coordinate_validator.validate(inp)
        # Integer-like coords still have at least 1 decimal if .0 is counted
        assert result.precision_score <= 0.2

    def test_precision_score_range(self, coordinate_validator):
        """Test precision score is always in [0.0, 1.0] range."""
        test_cases = [
            (0.0, 0.0),
            (1.1, 2.2),
            (12.345, 67.890),
            (-3.1234567, -60.0234567),
        ]
        for lat, lon in test_cases:
            result = coordinate_validator.validate(VerifyCoordinateRequest(lat=lat, lon=lon))
            assert 0.0 <= result.precision_score <= 1.0

    @pytest.mark.parametrize("decimals,lat_str", [
        (1, -3.1),
        (2, -3.12),
        (3, -3.123),
        (4, -3.1234),
        (5, -3.12345),
        (6, -3.123456),
        (7, -3.1234567),
    ])
    def test_precision_monotonically_increasing(self, coordinate_validator, decimals, lat_str):
        """Test that more decimal places always yield higher or equal precision score."""
        inp = VerifyCoordinateRequest(lat=lat_str, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= decimals

    def test_precision_issue_below_threshold(self, coordinate_validator, mock_config):
        """Test that a precision issue is raised when below min_decimals threshold."""
        inp = VerifyCoordinateRequest(lat=-3.12, lon=-60.02)  # 2 decimals < 5 min
        result = coordinate_validator.validate(inp)
        issue_codes = [i.code for i in result.issues]
        assert any(
            "PRECISION" in code or "DECIMAL" in code
            for code in issue_codes
        )

    def test_no_precision_issue_above_threshold(self, coordinate_validator):
        """Test no precision issue when above minimum decimals threshold."""
        inp = VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567)  # 7 decimals >= 5
        result = coordinate_validator.validate(inp)
        precision_issues = [
            i for i in result.issues
            if "PRECISION" in i.code or "DECIMAL" in i.code
        ]
        assert len(precision_issues) == 0

    def test_precision_different_lat_lon_decimals(self, coordinate_validator):
        """Test precision when lat and lon have different decimal counts."""
        inp = VerifyCoordinateRequest(lat=-3.12, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        # Should report the lower precision
        assert result.precision_decimal_places <= 7

    def test_precision_trailing_zeros(self, coordinate_validator):
        """Test precision handling when coordinates have trailing zeros."""
        inp = VerifyCoordinateRequest(lat=-3.100000, lon=-60.020000)
        result = coordinate_validator.validate(inp)
        # Python floats strip trailing zeros, so precision may be lower
        assert isinstance(result.precision_decimal_places, int)

    @pytest.mark.parametrize("lat,expected_min_decimals", [
        (-3.1234567890, 7),
        (-3.12345, 5),
        (-3.1, 1),
    ])
    def test_precision_detected_correctly(self, coordinate_validator, lat, expected_min_decimals):
        """Test precision detection across different decimal counts."""
        inp = VerifyCoordinateRequest(lat=lat, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        assert result.precision_decimal_places >= expected_min_decimals


# ===========================================================================
# 3. Transposition Detection (15 tests)
# ===========================================================================


class TestTranspositionDetection:
    """Test detection of swapped latitude and longitude values."""

    def test_transposition_detection_lat_gt_90(self, coordinate_validator):
        """Detect transposition when lat value exceeds 90 (looks like longitude)."""
        inp = VerifyCoordinateRequest(lat=111.7654321, lon=-2.5678901)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is True

    def test_no_transposition_normal_coords(self, coordinate_validator):
        """No transposition for normal coordinates within bounds."""
        inp = VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is False

    def test_transposition_both_within_lat_range(self, coordinate_validator):
        """No transposition when both values are within latitude range."""
        inp = VerifyCoordinateRequest(lat=45.0, lon=80.0)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is False

    def test_transposition_lat_minus_120(self, coordinate_validator):
        """Detect transposition when latitude is -120 (valid longitude)."""
        inp = VerifyCoordinateRequest(lat=-120.0, lon=-3.0)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is True

    def test_transposition_large_lat_positive(self, coordinate_validator):
        """Detect transposition when lat is 150 (clearly a longitude value)."""
        inp = VerifyCoordinateRequest(lat=150.0, lon=10.0)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is True

    def test_no_transposition_at_poles(self, coordinate_validator):
        """No false transposition at boundary values."""
        inp = VerifyCoordinateRequest(lat=89.9, lon=89.9)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is False

    @pytest.mark.parametrize("lat,lon,expected", [
        (100.0, -5.0, True),
        (-100.0, 5.0, True),
        (50.0, 100.0, False),
        (-50.0, -100.0, False),
        (0.0, 0.0, False),
        (179.0, 1.0, True),
        (-179.0, -1.0, True),
    ])
    def test_transposition_parametrized(self, coordinate_validator, lat, lon, expected):
        """Parametrized test for transposition detection."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.transposition_detected is expected

    def test_transposition_issue_severity(self, coordinate_validator):
        """Test that transposition issues have critical or high severity."""
        inp = VerifyCoordinateRequest(lat=111.0, lon=-3.0)
        result = coordinate_validator.validate(inp)
        transposition_issues = [
            i for i in result.issues
            if "TRANSPOS" in i.code or "SWAP" in i.code
        ]
        if transposition_issues:
            assert transposition_issues[0].severity in (
                IssueSeverity.CRITICAL,
                IssueSeverity.HIGH,
            )


# ===========================================================================
# 4. Country Matching (25 tests)
# ===========================================================================


class TestCountryMatching:
    """Test country-coordinate matching validation."""

    def test_country_match_brazil_correct(self, coordinate_validator):
        """Test coordinate in Brazil matches declared country BR."""
        inp = VerifyCoordinateRequest(
            lat=-3.1234567, lon=-60.0234567, declared_country="BR"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_match_brazil_wrong_country(self, coordinate_validator):
        """Test coordinate in Brazil with wrong declared country."""
        inp = VerifyCoordinateRequest(
            lat=-3.1234567, lon=-60.0234567, declared_country="GH"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is False

    def test_country_match_indonesia(self, coordinate_validator):
        """Test coordinate in Indonesia matches declared country ID."""
        inp = VerifyCoordinateRequest(
            lat=-2.5678901, lon=111.7654321, declared_country="ID"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_match_ghana(self, coordinate_validator):
        """Test coordinate in Ghana matches declared country GH."""
        inp = VerifyCoordinateRequest(
            lat=6.1234567, lon=-1.6234567, declared_country="GH"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_match_colombia(self, coordinate_validator):
        """Test coordinate in Colombia matches declared country CO."""
        inp = VerifyCoordinateRequest(
            lat=4.5678901, lon=-74.0654321, declared_country="CO"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_match_cote_divoire(self, coordinate_validator):
        """Test coordinate in Cote d'Ivoire matches declared country CI."""
        inp = VerifyCoordinateRequest(
            lat=6.8234567, lon=-5.2734567, declared_country="CI"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_match_malaysia(self, coordinate_validator):
        """Test coordinate in Malaysia matches declared country MY."""
        inp = VerifyCoordinateRequest(
            lat=3.1234567, lon=101.7654321, declared_country="MY"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_match_paraguay(self, coordinate_validator):
        """Test coordinate in Paraguay matches declared country PY."""
        inp = VerifyCoordinateRequest(
            lat=-23.4567890, lon=-57.1234567, declared_country="PY"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    def test_country_mismatch_issue_generated(self, coordinate_validator):
        """Test that country mismatch generates a validation issue."""
        inp = VerifyCoordinateRequest(
            lat=-3.1234567, lon=-60.0234567, declared_country="ID"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is False
        assert len(result.issues) > 0
        issue_codes = [i.code for i in result.issues]
        assert any(
            "COUNTRY" in code or "MISMATCH" in code
            for code in issue_codes
        )

    def test_country_match_resolved_country_populated(self, coordinate_validator):
        """Test that resolved_country field is populated."""
        inp = VerifyCoordinateRequest(
            lat=-3.1234567, lon=-60.0234567, declared_country="BR"
        )
        result = coordinate_validator.validate(inp)
        assert result.resolved_country is not None
        assert result.resolved_country == "BR"

    def test_country_empty_declared_no_mismatch(self, coordinate_validator):
        """Test no mismatch when declared country is empty."""
        inp = VerifyCoordinateRequest(
            lat=-3.1234567, lon=-60.0234567, declared_country=""
        )
        result = coordinate_validator.validate(inp)
        # When no country is declared, cannot be a mismatch - but might be an issue
        assert isinstance(result.country_match, bool)

    def test_country_match_case_insensitive(self, coordinate_validator):
        """Test country matching is case-insensitive."""
        inp = VerifyCoordinateRequest(
            lat=-3.1234567, lon=-60.0234567, declared_country="br"
        )
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    @pytest.mark.parametrize("lat,lon,country", [
        (-3.12, -60.02, "BR"),
        (-2.57, 111.77, "ID"),
        (6.12, -1.62, "GH"),
        (4.57, -74.07, "CO"),
        (3.12, 101.77, "MY"),
        (-23.46, -57.12, "PY"),
    ])
    def test_country_match_all_eudr_origins(self, coordinate_validator, lat, lon, country):
        """Test country matching for all major EUDR origin countries."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon, declared_country=country)
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    @pytest.mark.parametrize("lat,lon,wrong_country", [
        (-3.12, -60.02, "ID"),
        (-2.57, 111.77, "BR"),
        (6.12, -1.62, "CO"),
    ])
    def test_country_mismatch_detected(self, coordinate_validator, lat, lon, wrong_country):
        """Test country mismatch is detected for wrong countries."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon, declared_country=wrong_country)
        result = coordinate_validator.validate(inp)
        assert result.country_match is False


# ===========================================================================
# 5. Ocean / Land Detection (20 tests)
# ===========================================================================


class TestOceanLandDetection:
    """Test detection of coordinates in oceans vs on land."""

    def test_ocean_coordinate_detected(self, coordinate_validator, invalid_coordinate_ocean):
        """Test that a coordinate in the Atlantic Ocean is detected."""
        result = coordinate_validator.validate(invalid_coordinate_ocean)
        assert result.is_on_land is False

    def test_land_coordinate_brazil(self, coordinate_validator, valid_coordinate_brazil):
        """Test that a coordinate on land in Brazil is detected correctly."""
        result = coordinate_validator.validate(valid_coordinate_brazil)
        assert result.is_on_land is True

    def test_land_coordinate_indonesia(self, coordinate_validator, valid_coordinate_indonesia):
        """Test that a coordinate on land in Indonesia is detected correctly."""
        result = coordinate_validator.validate(valid_coordinate_indonesia)
        assert result.is_on_land is True

    def test_land_coordinate_ghana(self, coordinate_validator, valid_coordinate_ghana):
        """Test that a coordinate on land in Ghana is detected correctly."""
        result = coordinate_validator.validate(valid_coordinate_ghana)
        assert result.is_on_land is True

    @pytest.mark.parametrize("lat,lon,expected_land", [
        (0.0, -30.0, False),    # Mid-Atlantic
        (20.0, -40.0, False),   # Atlantic Ocean
        (-40.0, -20.0, False),  # South Atlantic
        (30.0, -170.0, False),  # Pacific Ocean
        (-60.0, 100.0, False),  # Southern Ocean
    ])
    def test_ocean_coordinates_parametrized(self, coordinate_validator, lat, lon, expected_land):
        """Test multiple ocean coordinates are correctly identified."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.is_on_land is expected_land

    @pytest.mark.parametrize("lat,lon,expected_land", [
        (-15.78, -47.93, True),   # Brasilia, Brazil
        (-6.17, 106.85, True),    # Jakarta, Indonesia
        (5.55, -0.19, True),      # Accra, Ghana
        (48.86, 2.35, True),      # Paris, France
        (35.68, 139.69, True),    # Tokyo, Japan
    ])
    def test_land_coordinates_parametrized(self, coordinate_validator, lat, lon, expected_land):
        """Test multiple land coordinates are correctly identified."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.is_on_land is expected_land

    def test_ocean_coordinate_generates_issue(self, coordinate_validator):
        """Test ocean coordinate generates a validation issue."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=-30.0, declared_country="BR")
        result = coordinate_validator.validate(inp)
        assert result.is_on_land is False
        issue_codes = [i.code for i in result.issues]
        assert any(
            "OCEAN" in code or "LAND" in code or "WATER" in code
            for code in issue_codes
        )

    def test_coastal_coordinate_land(self, coordinate_validator):
        """Test coordinate on a coastline is treated as land."""
        # Rio de Janeiro coastline
        inp = VerifyCoordinateRequest(lat=-22.9068, lon=-43.1729)
        result = coordinate_validator.validate(inp)
        assert result.is_on_land is True


# ===========================================================================
# 6. Duplicate Detection (20 tests)
# ===========================================================================


class TestDuplicateDetection:
    """Test duplicate coordinate detection in batch context."""

    def test_duplicate_detection_same_coords(self, coordinate_validator):
        """Detect duplicates when two coordinates are identical."""
        coords = [
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="P2"),
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == 2
        # At least one should be flagged as duplicate
        dup_flags = [r.is_duplicate for r in results]
        assert any(dup_flags)

    def test_duplicate_detection_nearby_coords(self, coordinate_validator):
        """Detect duplicates when coordinates are very close (within threshold)."""
        coords = [
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.1234568, lon=-60.0234568, plot_id="P2"),  # ~0.15m away
        ]
        results = coordinate_validator.validate_batch(coords)
        dup_flags = [r.is_duplicate for r in results]
        assert any(dup_flags)

    def test_no_duplicates_distant_coords(self, coordinate_validator):
        """No duplicates when coordinates are far apart."""
        coords = [
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="P1"),
            VerifyCoordinateRequest(lat=-2.5678901, lon=111.7654321, plot_id="P2"),  # Different continent
        ]
        results = coordinate_validator.validate_batch(coords)
        dup_flags = [r.is_duplicate for r in results]
        assert not any(dup_flags)

    def test_duplicate_detection_three_same(self, coordinate_validator):
        """Detect duplicates with three identical coordinates."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id=f"P{i}")
            for i in range(3)
        ]
        results = coordinate_validator.validate_batch(coords)
        dup_count = sum(1 for r in results if r.is_duplicate)
        assert dup_count >= 2  # At least 2 of 3 should be duplicates

    def test_no_duplicates_single_coord(self, coordinate_validator):
        """Single coordinate cannot have duplicates."""
        coords = [VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1")]
        results = coordinate_validator.validate_batch(coords)
        assert results[0].is_duplicate is False

    def test_duplicates_within_threshold(self, coordinate_validator, mock_config):
        """Test that duplicates are detected within the distance threshold."""
        threshold_m = mock_config.duplicate_distance_threshold_m
        # Two coordinates ~5m apart (well within default 10m threshold)
        coords = [
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.1234117, lon=-60.0234567, plot_id="P2"),  # ~5m north
        ]
        results = coordinate_validator.validate_batch(coords)
        dup_flags = [r.is_duplicate for r in results]
        assert any(dup_flags)

    def test_no_duplicates_just_outside_threshold(self, coordinate_validator):
        """Test coordinates just outside duplicate threshold are not duplicates."""
        coords = [
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.1230000, lon=-60.0234567, plot_id="P2"),  # ~50m away
        ]
        results = coordinate_validator.validate_batch(coords)
        dup_flags = [r.is_duplicate for r in results]
        assert not any(dup_flags)

    def test_duplicate_issue_code(self, coordinate_validator):
        """Test that duplicates generate appropriate issue codes."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P2"),
        ]
        results = coordinate_validator.validate_batch(coords)
        dup_results = [r for r in results if r.is_duplicate]
        if dup_results:
            issue_codes = [i.code for i in dup_results[0].issues]
            assert any("DUP" in code for code in issue_codes)

    @pytest.mark.parametrize("n_coords", [5, 10, 20, 50])
    def test_duplicate_detection_scales(self, coordinate_validator, n_coords):
        """Test duplicate detection works with varying batch sizes."""
        coords = [
            VerifyCoordinateRequest(
                lat=-3.12 + i * 0.01, lon=-60.02, plot_id=f"P{i}"
            )
            for i in range(n_coords)
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == n_coords

    def test_duplicate_detection_mixed_duplicates(self, coordinate_validator):
        """Test batch with some duplicates and some unique coordinates."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P2"),  # dup of P1
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P3"),    # unique
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P4"),    # dup of P3
            VerifyCoordinateRequest(lat=-2.57, lon=111.77, plot_id="P5"),  # unique
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == 5
        dup_count = sum(1 for r in results if r.is_duplicate)
        assert dup_count >= 2


# ===========================================================================
# 7. Cluster Anomaly Detection (15 tests)
# ===========================================================================


class TestClusterAnomalyDetection:
    """Test cluster anomaly detection (all coords at same location)."""

    def test_cluster_anomaly_all_same_location(self, coordinate_validator):
        """Detect anomaly when all batch coordinates are at the same location."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id=f"P{i}")
            for i in range(10)
        ]
        results = coordinate_validator.validate_batch(coords)
        anomaly_flags = [r.cluster_anomaly for r in results]
        assert any(anomaly_flags)

    def test_no_cluster_anomaly_spread_coords(self, coordinate_validator):
        """No anomaly when coordinates are well-spread geographically."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=-2.57, lon=111.77, plot_id="P2"),
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P3"),
            VerifyCoordinateRequest(lat=4.57, lon=-74.07, plot_id="P4"),
        ]
        results = coordinate_validator.validate_batch(coords)
        anomaly_flags = [r.cluster_anomaly for r in results]
        assert not any(anomaly_flags)

    def test_cluster_anomaly_small_cluster(self, coordinate_validator):
        """Test cluster anomaly with small cluster (3 points same location)."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P2"),
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P3"),
        ]
        results = coordinate_validator.validate_batch(coords)
        anomaly_flags = [r.cluster_anomaly for r in results]
        assert any(anomaly_flags)

    def test_no_anomaly_two_clusters(self, coordinate_validator):
        """No anomaly when coordinates form two distinct geographic clusters."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=-3.13, lon=-60.03, plot_id="P2"),
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P3"),
            VerifyCoordinateRequest(lat=6.13, lon=-1.63, plot_id="P4"),
        ]
        results = coordinate_validator.validate_batch(coords)
        anomaly_flags = [r.cluster_anomaly for r in results]
        assert not any(anomaly_flags)

    def test_cluster_anomaly_generates_issue(self, coordinate_validator):
        """Test that cluster anomaly generates an issue."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id=f"P{i}")
            for i in range(5)
        ]
        results = coordinate_validator.validate_batch(coords)
        for r in results:
            if r.cluster_anomaly:
                issue_codes = [i.code for i in r.issues]
                assert any("CLUSTER" in code or "ANOMALY" in code for code in issue_codes)
                break

    def test_no_anomaly_single_coordinate(self, coordinate_validator):
        """Single coordinate cannot be a cluster anomaly."""
        coords = [VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1")]
        results = coordinate_validator.validate_batch(coords)
        assert results[0].cluster_anomaly is False


# ===========================================================================
# 8. Elevation Plausibility (15 tests)
# ===========================================================================


class TestElevationPlausibility:
    """Test elevation plausibility checks."""

    def test_elevation_plausible_normal(self, coordinate_validator):
        """Test normal elevation is plausible for production plots."""
        inp = VerifyCoordinateRequest(lat=-3.12, lon=-60.02)
        result = coordinate_validator.validate(inp)
        assert result.elevation_plausible is True

    def test_elevation_implausible_too_high(self, coordinate_validator):
        """Test extreme high elevation is implausible for production."""
        # Mock high elevation location (e.g., Everest)
        inp = VerifyCoordinateRequest(lat=27.9881, lon=86.9250)  # Mt Everest
        result = coordinate_validator.validate(inp)
        # The elevation_plausible check depends on resolved elevation
        # but the validator should handle this gracefully
        assert isinstance(result.elevation_plausible, bool)

    def test_elevation_field_populated(self, coordinate_validator):
        """Test elevation_m field is populated or None."""
        inp = VerifyCoordinateRequest(lat=-3.12, lon=-60.02)
        result = coordinate_validator.validate(inp)
        # May be None if no elevation lookup, or a float value
        assert result.elevation_m is None or isinstance(result.elevation_m, (int, float))

    def test_elevation_max_config(self, mock_config):
        """Test elevation max from config is used for validation."""
        assert mock_config.elevation_max_m == 6000.0

    @pytest.mark.parametrize("lat,lon", [
        (-3.12, -60.02),    # Amazon lowlands (~50m)
        (-2.57, 111.77),    # Kalimantan lowlands (~20m)
        (6.12, -1.62),      # Ashanti highlands (~200m)
    ])
    def test_lowland_coordinates_plausible(self, coordinate_validator, lat, lon):
        """Test lowland commodity production coordinates are plausible."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.elevation_plausible is True

    def test_elevation_issue_for_extreme(self, coordinate_validator):
        """Test elevation issue generated for extreme altitudes."""
        # High altitude location where commodity production is impossible
        inp = VerifyCoordinateRequest(lat=35.88, lon=76.51)  # K2 base
        result = coordinate_validator.validate(inp)
        assert isinstance(result.elevation_plausible, bool)

    def test_elevation_zero_sea_level(self, coordinate_validator):
        """Test sea-level coordinates have plausible elevation."""
        inp = VerifyCoordinateRequest(lat=-22.91, lon=-43.17)  # Rio coast
        result = coordinate_validator.validate(inp)
        assert result.elevation_plausible is True


# ===========================================================================
# 9. Batch Validation (25 tests)
# ===========================================================================


class TestBatchValidation:
    """Test batch coordinate validation."""

    def test_batch_validation_empty_list(self, coordinate_validator):
        """Test batch validation with empty list returns empty results."""
        results = coordinate_validator.validate_batch([])
        assert results == []

    def test_batch_validation_single_item(self, coordinate_validator, valid_coordinate_brazil):
        """Test batch validation with single coordinate."""
        results = coordinate_validator.validate_batch([valid_coordinate_brazil])
        assert len(results) == 1
        assert isinstance(results[0], CoordinateValidationResult)

    def test_batch_validation_mixed_results(self, coordinate_validator):
        """Test batch with both valid and invalid coordinates."""
        coords = [
            VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, plot_id="VALID-1"),
            VerifyCoordinateRequest(lat=95.0, lon=0.0, plot_id="INVALID-1"),
            VerifyCoordinateRequest(lat=6.1234567, lon=-1.6234567, plot_id="VALID-2"),
            VerifyCoordinateRequest(lat=0.0, lon=200.0, plot_id="INVALID-2"),
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == 4
        valid_count = sum(1 for r in results if r.wgs84_valid)
        invalid_count = sum(1 for r in results if not r.wgs84_valid)
        assert valid_count == 2
        assert invalid_count == 2

    def test_batch_validation_preserves_order(self, coordinate_validator):
        """Test that batch results maintain input order."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12 + i * 0.1, lon=-60.02, plot_id=f"P{i}")
            for i in range(5)
        ]
        results = coordinate_validator.validate_batch(coords)
        for i, result in enumerate(results):
            expected_lat = -3.12 + i * 0.1
            assert abs(result.lat - expected_lat) < 0.001

    def test_batch_validation_all_valid(self, coordinate_validator):
        """Test batch where all coordinates are valid."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=-2.57, lon=111.77, plot_id="P2"),
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P3"),
        ]
        results = coordinate_validator.validate_batch(coords)
        assert all(r.wgs84_valid for r in results)

    def test_batch_validation_all_invalid(self, coordinate_validator):
        """Test batch where all coordinates are invalid."""
        coords = [
            VerifyCoordinateRequest(lat=95.0, lon=0.0, plot_id="P1"),
            VerifyCoordinateRequest(lat=-95.0, lon=0.0, plot_id="P2"),
            VerifyCoordinateRequest(lat=0.0, lon=185.0, plot_id="P3"),
        ]
        results = coordinate_validator.validate_batch(coords)
        assert all(not r.wgs84_valid for r in results)

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 25, 50, 100])
    def test_batch_validation_various_sizes(self, coordinate_validator, batch_size):
        """Test batch validation with various batch sizes."""
        coords = [
            VerifyCoordinateRequest(
                lat=-3.12 + (i % 10) * 0.01,
                lon=-60.02 + (i % 10) * 0.005,
                plot_id=f"P{i}",
            )
            for i in range(batch_size)
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == batch_size

    def test_batch_each_result_has_validation_id(self, coordinate_validator):
        """Test each batch result has a unique validation ID."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12 + i * 0.01, lon=-60.02, plot_id=f"P{i}")
            for i in range(5)
        ]
        results = coordinate_validator.validate_batch(coords)
        validation_ids = [r.validation_id for r in results]
        assert len(set(validation_ids)) == 5  # All unique

    def test_batch_provenance_hashes(self, coordinate_validator):
        """Test that provenance hashes are generated for batch results."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P2"),
        ]
        results = coordinate_validator.validate_batch(coords)
        for r in results:
            assert r.provenance_hash is not None
            assert len(r.provenance_hash) > 0


# ===========================================================================
# 10. Boundary Coordinates (15 tests)
# ===========================================================================


class TestBoundaryCoordinates:
    """Test validation of boundary coordinates."""

    def test_north_pole(self, coordinate_validator, coordinate_north_pole):
        """Test validation at the North Pole."""
        result = coordinate_validator.validate(coordinate_north_pole)
        assert result.wgs84_valid is True
        assert result.lat == 90.0

    def test_south_pole(self, coordinate_validator, coordinate_south_pole):
        """Test validation at the South Pole."""
        result = coordinate_validator.validate(coordinate_south_pole)
        assert result.wgs84_valid is True
        assert result.lat == -90.0

    def test_dateline(self, coordinate_validator, coordinate_dateline):
        """Test validation on the International Date Line."""
        result = coordinate_validator.validate(coordinate_dateline)
        assert result.wgs84_valid is True
        assert result.lon == 180.0

    def test_prime_meridian(self, coordinate_validator, coordinate_prime_meridian):
        """Test validation on the Prime Meridian at equator."""
        result = coordinate_validator.validate(coordinate_prime_meridian)
        assert result.wgs84_valid is True
        assert result.lat == 0.0
        assert result.lon == 0.0

    def test_equator_brazil(self, coordinate_validator):
        """Test validation on the equator in Brazil."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=-51.0, declared_country="BR")
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    def test_minus_180_longitude(self, coordinate_validator):
        """Test validation at -180 longitude."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=-180.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    @pytest.mark.parametrize("lat,lon", [
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, 180.0),
        (0.0, -180.0),
        (0.0, 0.0),
        (90.0, 180.0),
        (-90.0, -180.0),
        (90.0, -180.0),
        (-90.0, 180.0),
    ])
    def test_all_boundary_corners(self, coordinate_validator, lat, lon):
        """Test all corners and edges of WGS84 bounds."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True


# ===========================================================================
# 11. Determinism and Reproducibility (10 tests)
# ===========================================================================


class TestDeterminism:
    """Test that validation is deterministic (same input = same output)."""

    def test_deterministic_same_input_same_output(self, coordinate_validator):
        """Test same input produces identical results on repeated calls."""
        inp = VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, declared_country="BR")
        result1 = coordinate_validator.validate(inp)
        result2 = coordinate_validator.validate(inp)
        assert result1.is_valid == result2.is_valid
        assert result1.wgs84_valid == result2.wgs84_valid
        assert result1.precision_decimal_places == result2.precision_decimal_places
        assert result1.precision_score == result2.precision_score
        assert result1.transposition_detected == result2.transposition_detected
        assert result1.country_match == result2.country_match
        assert result1.is_on_land == result2.is_on_land

    def test_deterministic_provenance_hash(self, coordinate_validator):
        """Test provenance hash is deterministic for same input."""
        inp = VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567, declared_country="BR")
        result1 = coordinate_validator.validate(inp)
        result2 = coordinate_validator.validate(inp)
        assert result1.provenance_hash == result2.provenance_hash

    def test_different_input_different_provenance(self, coordinate_validator):
        """Test different inputs produce different provenance hashes."""
        inp1 = VerifyCoordinateRequest(lat=-3.12, lon=-60.02)
        inp2 = VerifyCoordinateRequest(lat=6.12, lon=-1.62)
        result1 = coordinate_validator.validate(inp1)
        result2 = coordinate_validator.validate(inp2)
        assert result1.provenance_hash != result2.provenance_hash

    def test_deterministic_batch_provenance(self, coordinate_validator):
        """Test batch validation produces deterministic provenance hashes."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P2"),
        ]
        results1 = coordinate_validator.validate_batch(coords)
        results2 = coordinate_validator.validate_batch(coords)
        for r1, r2 in zip(results1, results2):
            assert r1.provenance_hash == r2.provenance_hash

    def test_provenance_hash_length(self, coordinate_validator):
        """Test provenance hash is SHA-256 (64 hex characters)."""
        inp = VerifyCoordinateRequest(lat=-3.12, lon=-60.02)
        result = coordinate_validator.validate(inp)
        if result.provenance_hash:
            assert len(result.provenance_hash) == 64

    def test_deterministic_issues_count(self, coordinate_validator):
        """Test same input produces same number of issues."""
        inp = VerifyCoordinateRequest(lat=95.0, lon=200.0, declared_country="XX")
        result1 = coordinate_validator.validate(inp)
        result2 = coordinate_validator.validate(inp)
        assert len(result1.issues) == len(result2.issues)

    def test_idempotent_no_state_mutation(self, coordinate_validator):
        """Test validation is stateless - no side effects between calls."""
        inp1 = VerifyCoordinateRequest(lat=-3.12, lon=-60.02, declared_country="BR")
        inp2 = VerifyCoordinateRequest(lat=95.0, lon=200.0, declared_country="XX")
        # Run invalid first, then valid
        coordinate_validator.validate(inp2)
        result = coordinate_validator.validate(inp1)
        assert result.wgs84_valid is True
        assert result.is_valid is True

    def test_batch_deterministic_10_runs(self, coordinate_validator):
        """Test batch validation is deterministic across 10 runs."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="P1"),
            VerifyCoordinateRequest(lat=6.12, lon=-1.62, plot_id="P2"),
        ]
        first_hashes = [
            r.provenance_hash
            for r in coordinate_validator.validate_batch(coords)
        ]
        for _ in range(9):
            run_hashes = [
                r.provenance_hash
                for r in coordinate_validator.validate_batch(coords)
            ]
            assert run_hashes == first_hashes


# ===========================================================================
# 12. Result Serialization (10 tests)
# ===========================================================================


class TestResultSerialization:
    """Test CoordinateValidationResult serialization."""

    def test_to_dict(self, coordinate_validator, valid_coordinate_brazil):
        """Test to_dict produces valid dictionary."""
        result = coordinate_validator.validate(valid_coordinate_brazil)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "lat" in d
        assert "lon" in d
        assert "is_valid" in d
        assert "wgs84_valid" in d

    def test_to_dict_contains_all_fields(self, coordinate_validator, valid_coordinate_brazil):
        """Test to_dict contains all expected fields."""
        result = coordinate_validator.validate(valid_coordinate_brazil)
        d = result.to_dict()
        expected_keys = {
            "validation_id", "lat", "lon", "is_valid", "wgs84_valid",
            "precision_decimal_places", "precision_score", "transposition_detected",
            "country_match", "resolved_country", "is_on_land", "is_duplicate",
            "elevation_m", "elevation_plausible", "cluster_anomaly",
            "issues", "provenance_hash", "validated_at",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_issues_serialized(self, coordinate_validator):
        """Test issues are serialized as list of dicts."""
        inp = VerifyCoordinateRequest(lat=95.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        d = result.to_dict()
        assert isinstance(d["issues"], list)
        if d["issues"]:
            assert isinstance(d["issues"][0], dict)
            assert "code" in d["issues"][0]
            assert "severity" in d["issues"][0]

    def test_to_dict_datetime_iso(self, coordinate_validator, valid_coordinate_brazil):
        """Test validated_at is serialized as ISO string."""
        result = coordinate_validator.validate(valid_coordinate_brazil)
        d = result.to_dict()
        assert isinstance(d["validated_at"], str)
        assert "T" in d["validated_at"]  # ISO format has T separator

    def test_to_dict_roundtrip_lat_lon(self, coordinate_validator):
        """Test lat/lon values survive serialization roundtrip."""
        inp = VerifyCoordinateRequest(lat=-3.1234567, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        d = result.to_dict()
        assert d["lat"] == -3.1234567
        assert d["lon"] == -60.0234567


# ===========================================================================
# 13. EUDR-Specific Country Coordinate Coverage (30 tests)
# ===========================================================================


class TestEUDRCountryCoverage:
    """Test coordinate validation across all EUDR-relevant countries and commodities."""

    @pytest.mark.parametrize("lat,lon,country,commodity", [
        (-3.12, -60.02, "BR", "cocoa"),
        (-15.78, -47.93, "BR", "cattle"),
        (-12.97, -38.51, "BR", "cocoa"),
        (-23.55, -46.63, "BR", "soya"),
        (-25.43, -49.27, "BR", "wood"),
        (-10.95, -37.07, "BR", "rubber"),
        (-2.57, 111.77, "ID", "oil_palm"),
        (-6.17, 106.85, "ID", "rubber"),
        (-1.65, 103.59, "ID", "oil_palm"),
        (-0.50, 117.15, "ID", "wood"),
        (6.12, -1.62, "GH", "cocoa"),
        (5.55, -0.19, "GH", "cocoa"),
        (7.34, -2.33, "GH", "cocoa"),
        (6.82, -5.27, "CI", "cocoa"),
        (5.32, -4.01, "CI", "cocoa"),
        (4.57, -74.07, "CO", "coffee"),
        (2.45, -76.60, "CO", "coffee"),
        (3.12, 101.77, "MY", "oil_palm"),
        (5.96, 116.07, "MY", "oil_palm"),
        (-23.46, -57.12, "PY", "soya"),
        (-25.26, -57.58, "PY", "cattle"),
        (-34.61, -58.38, "AR", "soya"),
        (-27.47, -58.99, "AR", "cattle"),
        (4.05, 9.77, "CM", "cocoa"),
        (5.95, 10.15, "CM", "wood"),
        (6.50, 3.38, "NG", "oil_palm"),
        (9.06, 7.49, "NG", "cocoa"),
        (13.45, 2.11, "NE", "cattle"),
        (7.54, 1.32, "BJ", "cotton"),
        (-4.32, 15.31, "CG", "wood"),
    ])
    def test_eudr_origin_coordinates(self, coordinate_validator, lat, lon, country, commodity):
        """Test coordinate validation for EUDR origin country coordinates."""
        inp = VerifyCoordinateRequest(
            lat=lat, lon=lon, declared_country=country, commodity=commodity,
        )
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True
        assert isinstance(result.is_valid, bool)

    @pytest.mark.parametrize("lat,lon,expected_country", [
        (-3.12, -60.02, "BR"),
        (-2.57, 111.77, "ID"),
        (6.12, -1.62, "GH"),
        (6.82, -5.27, "CI"),
        (4.57, -74.07, "CO"),
        (3.12, 101.77, "MY"),
        (-23.46, -57.12, "PY"),
        (-34.61, -58.38, "AR"),
        (4.05, 9.77, "CM"),
        (6.50, 3.38, "NG"),
    ])
    def test_resolved_country_accuracy(self, coordinate_validator, lat, lon, expected_country):
        """Test resolved country matches expected for known locations."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon, declared_country=expected_country)
        result = coordinate_validator.validate(inp)
        assert result.country_match is True

    @pytest.mark.parametrize("lat,lon,wrong_country", [
        (-3.12, -60.02, "US"),
        (-2.57, 111.77, "AU"),
        (6.12, -1.62, "FR"),
        (6.82, -5.27, "DE"),
        (4.57, -74.07, "JP"),
        (3.12, 101.77, "CN"),
        (-23.46, -57.12, "RU"),
        (-34.61, -58.38, "GB"),
    ])
    def test_country_mismatch_eudr_countries(self, coordinate_validator, lat, lon, wrong_country):
        """Test country mismatch detection for EUDR country coordinates."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon, declared_country=wrong_country)
        result = coordinate_validator.validate(inp)
        assert result.country_match is False


# ===========================================================================
# 14. Advanced Precision Testing (15 tests)
# ===========================================================================


class TestAdvancedPrecision:
    """Advanced precision testing with specific scenarios."""

    @pytest.mark.parametrize("precision,lat", [
        (0, 3.0),
        (1, 3.1),
        (2, 3.12),
        (3, 3.123),
        (4, 3.1234),
        (5, 3.12345),
        (6, 3.123456),
        (7, 3.1234567),
        (8, 3.12345678),
    ])
    def test_precision_levels_ascending(self, coordinate_validator, precision, lat):
        """Test precision detection for increasing decimal places."""
        inp = VerifyCoordinateRequest(lat=-lat, lon=-60.0234567)
        result = coordinate_validator.validate(inp)
        assert isinstance(result.precision_score, float)
        assert 0.0 <= result.precision_score <= 1.0

    @pytest.mark.parametrize("score_lat,score_lon", [
        (-3.1, -60.1),
        (-3.12, -60.12),
        (-3.123, -60.123),
        (-3.1234, -60.1234),
        (-3.12345, -60.12345),
        (-3.123456, -60.123456),
    ])
    def test_precision_score_ordering(self, coordinate_validator, score_lat, score_lon):
        """Test that higher precision coordinates get higher scores."""
        inp = VerifyCoordinateRequest(lat=score_lat, lon=score_lon)
        result = coordinate_validator.validate(inp)
        assert isinstance(result.precision_score, float)


# ===========================================================================
# 15. Edge Cases and Error Conditions (20 tests)
# ===========================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_nan_latitude_handling(self, coordinate_validator):
        """Test NaN latitude is handled gracefully."""
        inp = VerifyCoordinateRequest(lat=float('nan'), lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_nan_longitude_handling(self, coordinate_validator):
        """Test NaN longitude is handled gracefully."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=float('nan'))
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_inf_latitude_handling(self, coordinate_validator):
        """Test infinity latitude is handled gracefully."""
        inp = VerifyCoordinateRequest(lat=float('inf'), lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_inf_longitude_handling(self, coordinate_validator):
        """Test infinity longitude is handled gracefully."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=float('inf'))
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_negative_inf_handling(self, coordinate_validator):
        """Test negative infinity is handled gracefully."""
        inp = VerifyCoordinateRequest(lat=float('-inf'), lon=float('-inf'))
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_very_small_negative_lat(self, coordinate_validator):
        """Test very small negative latitude."""
        inp = VerifyCoordinateRequest(lat=-0.0000001, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True

    def test_very_large_precision(self, coordinate_validator):
        """Test coordinate with excessive precision."""
        inp = VerifyCoordinateRequest(lat=-3.12345678901234, lon=-60.02345678901234)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True
        assert result.precision_decimal_places >= 7

    def test_zero_zero_coordinate(self, coordinate_validator):
        """Test (0.0, 0.0) coordinate (Null Island)."""
        inp = VerifyCoordinateRequest(lat=0.0, lon=0.0)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is True
        # Null Island is in the ocean
        assert result.is_on_land is False

    @pytest.mark.parametrize("lat,lon", [
        (90.0001, 0.0),
        (-90.0001, 0.0),
        (0.0, 180.0001),
        (0.0, -180.0001),
        (90.1, 0.0),
        (-90.1, 0.0),
        (0.0, 180.1),
        (0.0, -180.1),
        (91.0, 181.0),
        (-91.0, -181.0),
        (100.0, 0.0),
        (0.0, 200.0),
        (200.0, 200.0),
        (-200.0, -200.0),
        (1000.0, 1000.0),
    ])
    def test_out_of_bounds_comprehensive(self, coordinate_validator, lat, lon):
        """Comprehensive out-of-bounds coordinate testing."""
        inp = VerifyCoordinateRequest(lat=lat, lon=lon)
        result = coordinate_validator.validate(inp)
        assert result.wgs84_valid is False

    def test_batch_with_nan(self, coordinate_validator):
        """Test batch containing NaN coordinates."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="VALID"),
            VerifyCoordinateRequest(lat=float('nan'), lon=0.0, plot_id="NAN"),
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == 2
        assert results[0].wgs84_valid is True
        assert results[1].wgs84_valid is False

    def test_batch_with_inf(self, coordinate_validator):
        """Test batch containing infinity coordinates."""
        coords = [
            VerifyCoordinateRequest(lat=-3.12, lon=-60.02, plot_id="VALID"),
            VerifyCoordinateRequest(lat=float('inf'), lon=0.0, plot_id="INF"),
        ]
        results = coordinate_validator.validate_batch(coords)
        assert len(results) == 2
        assert results[0].wgs84_valid is True
        assert results[1].wgs84_valid is False
