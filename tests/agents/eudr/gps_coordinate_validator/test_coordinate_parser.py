# -*- coding: utf-8 -*-
"""
Tests for CoordinateParser - AGENT-EUDR-007 Engine 1: Multi-Format Parsing

Comprehensive test suite covering:
- Decimal degrees (DD) parsing with positive and negative values
- Degrees, minutes, seconds (DMS) parsing with NSEW suffixes
- DMS with Unicode degree/minute/second symbols
- DMS with various separator styles (d, deg, degree symbol)
- Degrees and decimal minutes (DDM) parsing
- Universal Transverse Mercator (UTM) parsing
- Military Grid Reference System (MGRS) parsing
- Signed decimal degrees parsing
- Decimal degrees with N/S/E/W suffix parsing
- Auto-detection of coordinate format with confidence scoring
- Batch parsing of multiple formats
- Whitespace and separator handling
- Invalid / malformed input handling
- Parametrized tests across all 8 supported formats

Test count: 60+ tests
Coverage target: >= 85% of CoordinateParser module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import math

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    CoordinateFormat,
    ParsedCoordinate,
    RawCoordinate,
)
from tests.agents.eudr.gps_coordinate_validator.conftest import (
    COCOA_FARM_GHANA,
    PALM_PLANTATION_INDONESIA,
    COFFEE_FARM_COLOMBIA,
    SOYA_FIELD_BRAZIL,
    RUBBER_FARM_THAILAND,
    CATTLE_RANCH_BRAZIL,
    TIMBER_FOREST_CONGO,
    COFFEE_FARM_ETHIOPIA,
    HIGH_PRECISION,
    LOW_PRECISION,
    TRUNCATED,
    NULL_ISLAND,
    BOUNDARY_LATITUDE,
    BOUNDARY_LONGITUDE,
    SOUTH_POLE,
    ANTIMERIDIAN_EAST,
    DMS_GHANA,
    DMS_BRAZIL,
    DMS_INDONESIA,
    DDM_GHANA,
    DDM_COLOMBIA,
    UTM_GHANA,
    UTM_BRAZIL,
    UTM_INDONESIA,
    MGRS_GHANA,
    DD_SUFFIX_GHANA,
    DD_COMMA_GHANA,
    DD_SPACE_GHANA,
    DD_SEMICOLON_GHANA,
    DMS_WITH_DEG_SYMBOL,
    DMS_WITH_D_SEPARATOR,
    DMS_WITH_DEG_TEXT,
    DMS_UNICODE_SYMBOLS,
    DMS_COMPACT,
    DD_COMMA_DECIMAL_EUROPEAN,
    UTM_LOWERCASE_GHANA,
    DD_EXCESSIVE_WHITESPACE,
    DD_TRAILING_ZEROS,
    DD_PLUS_SIGN,
    GARBAGE_INPUT,
    EMPTY_INPUT,
    PARTIAL_INPUT,
    SHA256_HEX_LENGTH,
    make_raw,
    assert_close,
    dms_to_decimal,
)


# ===========================================================================
# 1. Decimal Degrees (DD) Parsing
# ===========================================================================


class TestParseDecimalDegrees:
    """Test parsing of decimal degrees format coordinates."""

    def test_parse_decimal_degrees(self, coordinate_parser):
        """Standard DD format with comma separator."""
        raw = make_raw("5.603716, -0.186964")
        result = coordinate_parser.parse(raw)
        assert isinstance(result, ParsedCoordinate)
        assert result.parse_successful is True
        assert_close(result.latitude, 5.603716, tolerance=0.0001)
        assert_close(result.longitude, -0.186964, tolerance=0.0001)

    def test_parse_negative_dd_southern_hemisphere(self, coordinate_parser):
        """Negative latitude for southern hemisphere (Brazil)."""
        raw = make_raw("-12.970000, -55.320000", country="BR", commodity="soya")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude < 0
        assert_close(result.latitude, -12.97, tolerance=0.001)

    def test_parse_negative_dd_western_hemisphere(self, coordinate_parser):
        """Negative longitude for western hemisphere (Colombia)."""
        raw = make_raw("4.570868, -75.678000", country="CO", commodity="coffee")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.longitude < 0
        assert_close(result.longitude, -75.678, tolerance=0.001)

    def test_parse_dd_positive_both(self, coordinate_parser):
        """Both positive lat and lon (Indonesia)."""
        raw = make_raw("-2.524000, 111.876000", country="ID", commodity="oil_palm")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, -2.524, tolerance=0.001)
        assert_close(result.longitude, 111.876, tolerance=0.001)

    def test_parse_dd_zero_values(self, coordinate_parser):
        """Parsing coordinates at origin (0, 0)."""
        raw = make_raw("0.0, 0.0")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, 0.0, tolerance=0.0001)
        assert_close(result.longitude, 0.0, tolerance=0.0001)

    def test_parse_dd_boundary_latitude_90(self, coordinate_parser):
        """Parsing latitude at +90 (North Pole)."""
        raw = make_raw("90.0, 0.0")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, 90.0, tolerance=0.0001)

    def test_parse_dd_boundary_latitude_neg90(self, coordinate_parser):
        """Parsing latitude at -90 (South Pole)."""
        raw = make_raw("-90.0, 0.0")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, -90.0, tolerance=0.0001)

    def test_parse_dd_boundary_longitude_180(self, coordinate_parser):
        """Parsing longitude at 180 (antimeridian)."""
        raw = make_raw("0.0, 180.0")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.longitude, 180.0, tolerance=0.0001)

    def test_parse_dd_boundary_longitude_neg180(self, coordinate_parser):
        """Parsing longitude at -180 (antimeridian)."""
        raw = make_raw("0.0, -180.0")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.longitude, -180.0, tolerance=0.0001)


# ===========================================================================
# 2. DMS Parsing
# ===========================================================================


class TestParseDMS:
    """Test parsing of Degrees, Minutes, Seconds (DMS) format."""

    def test_parse_dms_north_west(self, coordinate_parser):
        """DMS with North latitude and West longitude (Ghana)."""
        raw = make_raw(DMS_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.01)
        assert_close(result.longitude, COCOA_FARM_GHANA[1], tolerance=0.01)

    def test_parse_dms_south_west(self, coordinate_parser):
        """DMS with South latitude and West longitude (Brazil)."""
        raw = make_raw(DMS_BRAZIL, country="BR", commodity="soya")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude < 0
        assert result.longitude < 0

    def test_parse_dms_south_east(self, coordinate_parser):
        """DMS with South latitude and East longitude (Indonesia)."""
        raw = make_raw(DMS_INDONESIA, country="ID", commodity="oil_palm")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude < 0
        assert result.longitude > 0

    def test_parse_dms_unicode_symbols(self, coordinate_parser):
        """DMS with Unicode prime/double-prime symbols."""
        raw = make_raw(DMS_UNICODE_SYMBOLS)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.01)

    def test_parse_dms_d_separator(self, coordinate_parser):
        """DMS with 'd' as degree separator."""
        raw = make_raw(DMS_WITH_D_SEPARATOR)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.01)

    def test_parse_dms_deg_text_separator(self, coordinate_parser):
        """DMS with 'deg' as degree separator."""
        raw = make_raw(DMS_WITH_DEG_TEXT)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.01)

    def test_parse_dms_degree_symbol(self, coordinate_parser):
        """DMS with standard degree symbol."""
        raw = make_raw(DMS_WITH_DEG_SYMBOL)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True


# ===========================================================================
# 3. DDM Parsing
# ===========================================================================


class TestParseDDM:
    """Test parsing of Degrees and Decimal Minutes (DDM) format."""

    def test_parse_ddm_ghana(self, coordinate_parser):
        """DDM format for Ghana cocoa farm."""
        raw = make_raw(DDM_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.01)
        assert_close(result.longitude, COCOA_FARM_GHANA[1], tolerance=0.01)

    def test_parse_ddm_colombia(self, coordinate_parser):
        """DDM format for Colombia coffee farm."""
        raw = make_raw(DDM_COLOMBIA, country="CO", commodity="coffee")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COFFEE_FARM_COLOMBIA[0], tolerance=0.01)


# ===========================================================================
# 4. UTM Parsing
# ===========================================================================


class TestParseUTM:
    """Test parsing of Universal Transverse Mercator (UTM) format."""

    def test_parse_utm_ghana(self, coordinate_parser):
        """UTM coordinate for Ghana."""
        raw = make_raw(UTM_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        # UTM conversion is approximate; allow wider tolerance
        assert -90 <= result.latitude <= 90
        assert -180 <= result.longitude <= 180

    def test_parse_utm_brazil(self, coordinate_parser):
        """UTM coordinate for Brazil."""
        raw = make_raw(UTM_BRAZIL, country="BR", commodity="soya")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude < 0  # Southern hemisphere

    def test_parse_utm_indonesia(self, coordinate_parser):
        """UTM coordinate for Indonesia."""
        raw = make_raw(UTM_INDONESIA, country="ID", commodity="oil_palm")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.longitude > 90  # Eastern hemisphere


# ===========================================================================
# 5. MGRS Parsing
# ===========================================================================


class TestParseMGRS:
    """Test parsing of Military Grid Reference System (MGRS) format."""

    def test_parse_mgrs_ghana(self, coordinate_parser):
        """MGRS grid reference for Ghana."""
        raw = make_raw(MGRS_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert -90 <= result.latitude <= 90
        assert -180 <= result.longitude <= 180


# ===========================================================================
# 6. Signed DD and DD with Suffix
# ===========================================================================


class TestParseSignedAndSuffix:
    """Test signed DD and DD with N/S/E/W suffix formats."""

    def test_parse_signed_dd(self, coordinate_parser):
        """Signed decimal degrees (negative = S/W)."""
        raw = make_raw("-12.97, -55.32")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, -12.97, tolerance=0.001)
        assert_close(result.longitude, -55.32, tolerance=0.001)

    def test_parse_dd_suffix_north_west(self, coordinate_parser):
        """DD with N/W suffix."""
        raw = make_raw(DD_SUFFIX_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.001)
        assert_close(result.longitude, COCOA_FARM_GHANA[1], tolerance=0.001)


# ===========================================================================
# 7. Format Auto-Detection
# ===========================================================================


class TestFormatDetection:
    """Test automatic coordinate format detection."""

    def test_format_detection_dd(self, coordinate_parser):
        """Auto-detect decimal degrees format."""
        raw = make_raw(DD_COMMA_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.detected_format in (
            CoordinateFormat.DECIMAL_DEGREES,
            CoordinateFormat.SIGNED_DD,
        )

    def test_format_detection_dms(self, coordinate_parser):
        """Auto-detect DMS format."""
        raw = make_raw(DMS_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.detected_format == CoordinateFormat.DMS

    def test_format_detection_utm(self, coordinate_parser):
        """Auto-detect UTM format."""
        raw = make_raw(UTM_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.detected_format == CoordinateFormat.UTM

    def test_format_detection_confidence_dd(self, coordinate_parser):
        """High confidence for unambiguous DD format."""
        raw = make_raw(DD_COMMA_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.format_confidence >= 0.7

    def test_format_detection_confidence_dms(self, coordinate_parser):
        """High confidence for unambiguous DMS format."""
        raw = make_raw(DMS_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.format_confidence >= 0.7


# ===========================================================================
# 8. Batch Parsing
# ===========================================================================


class TestBatchParse:
    """Test batch parsing of multiple coordinate formats."""

    def test_batch_parse_multiple_formats(self, coordinate_parser):
        """Parse a batch of coordinates in different formats."""
        raw_inputs = [
            make_raw(DD_COMMA_GHANA),
            make_raw(DMS_GHANA),
            make_raw(UTM_GHANA),
        ]
        results = coordinate_parser.parse_batch(raw_inputs)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ParsedCoordinate)

    def test_batch_parse_empty_list(self, coordinate_parser):
        """Batch parse with empty input list returns empty."""
        results = coordinate_parser.parse_batch([])
        assert results == []

    def test_batch_parse_single_item(self, coordinate_parser):
        """Batch parse with a single item."""
        raw_inputs = [make_raw(DD_COMMA_GHANA)]
        results = coordinate_parser.parse_batch(raw_inputs)
        assert len(results) == 1
        assert results[0].parse_successful is True


# ===========================================================================
# 9. Whitespace and Separator Handling
# ===========================================================================


class TestWhitespaceHandling:
    """Test whitespace and separator handling in coordinate parsing."""

    def test_parse_whitespace_leading_trailing(self, coordinate_parser):
        """Leading and trailing whitespace is trimmed."""
        raw = make_raw("  5.603716, -0.186964  ")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, 5.603716, tolerance=0.001)

    def test_parse_whitespace_tabs(self, coordinate_parser):
        """Tab characters between components are handled."""
        raw = make_raw("5.603716\t-0.186964")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True

    def test_parse_comma_separator(self, coordinate_parser):
        """Comma separator between lat and lon."""
        raw = make_raw(DD_COMMA_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True

    def test_parse_space_separator(self, coordinate_parser):
        """Space separator between lat and lon."""
        raw = make_raw(DD_SPACE_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True

    def test_parse_semicolon_separator(self, coordinate_parser):
        """Semicolon separator between lat and lon."""
        raw = make_raw(DD_SEMICOLON_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True


# ===========================================================================
# 10. Invalid Input Handling
# ===========================================================================


class TestInvalidInput:
    """Test handling of invalid or malformed coordinate input."""

    def test_parse_garbage_input(self, coordinate_parser):
        """Garbage text returns parse_successful=False."""
        raw = make_raw(GARBAGE_INPUT)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is False

    def test_parse_partial_input(self, coordinate_parser):
        """Single number without a pair returns parse_successful=False."""
        raw = make_raw(PARTIAL_INPUT)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is False

    def test_parse_result_has_provenance_hash(self, coordinate_parser):
        """Parse result includes a provenance hash."""
        raw = make_raw(DD_COMMA_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256


# ===========================================================================
# 11. Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    "input_str,expected_lat,expected_lon,tolerance",
    [
        ("5.603716, -0.186964", 5.603716, -0.186964, 0.001),
        ("-12.970000, -55.320000", -12.97, -55.32, 0.001),
        ("4.570868, -75.678000", 4.570868, -75.678, 0.001),
        ("-2.524000, 111.876000", -2.524, 111.876, 0.001),
        ("7.880400, 98.392300", 7.8804, 98.3923, 0.001),
        ("-15.780000, -47.930000", -15.78, -47.93, 0.001),
        ("-4.322000, 15.313000", -4.322, 15.313, 0.001),
        ("0.0, 0.0", 0.0, 0.0, 0.0001),
        ("90.0, 180.0", 90.0, 180.0, 0.0001),
        ("-90.0, -180.0", -90.0, -180.0, 0.0001),
    ],
    ids=[
        "ghana", "brazil", "colombia", "indonesia", "thailand",
        "brazil_cattle", "congo", "origin", "ne_corner", "sw_corner",
    ],
)
def test_parametrized_dd_parsing(
    coordinate_parser, input_str, expected_lat, expected_lon, tolerance
):
    """Parametrized test: DD format parsing across 10 sample coordinates."""
    raw = make_raw(input_str)
    result = coordinate_parser.parse(raw)
    assert result.parse_successful is True
    assert_close(result.latitude, expected_lat, tolerance=tolerance)
    assert_close(result.longitude, expected_lon, tolerance=tolerance)


@pytest.mark.parametrize(
    "fmt_str,expected_format",
    [
        ("5.603716, -0.186964", CoordinateFormat.DECIMAL_DEGREES),
        (DMS_GHANA, CoordinateFormat.DMS),
        (DDM_GHANA, CoordinateFormat.DDM),
        (UTM_GHANA, CoordinateFormat.UTM),
        (DD_SUFFIX_GHANA, CoordinateFormat.DD_SUFFIX),
    ],
    ids=["dd", "dms", "ddm", "utm", "dd_suffix"],
)
def test_parametrized_format_detection(coordinate_parser, fmt_str, expected_format):
    """Parametrized test: format auto-detection across 5 formats."""
    raw = make_raw(fmt_str)
    result = coordinate_parser.parse(raw)
    assert result.detected_format == expected_format


# ===========================================================================
# 12. Additional Edge Cases - DMS Conversion Accuracy
# ===========================================================================


class TestDMSConversionAccuracy:
    """Test DMS-to-DD conversion accuracy for various representations."""

    def test_dms_conversion_ghana_accuracy(self, coordinate_parser):
        """DMS Ghana conversion matches expected decimal within 0.01 degrees."""
        expected_lat = dms_to_decimal(5, 36, 13.4, "N")
        expected_lon = dms_to_decimal(0, 11, 13.1, "W")
        raw = make_raw(DMS_GHANA)
        result = coordinate_parser.parse(raw)
        assert_close(result.latitude, expected_lat, tolerance=0.01)
        assert_close(result.longitude, expected_lon, tolerance=0.01)

    def test_dms_conversion_brazil_accuracy(self, coordinate_parser):
        """DMS Brazil conversion matches expected decimal."""
        expected_lat = dms_to_decimal(12, 58, 12.0, "S")
        expected_lon = dms_to_decimal(55, 19, 12.0, "W")
        raw = make_raw(DMS_BRAZIL, country="BR", commodity="soya")
        result = coordinate_parser.parse(raw)
        assert_close(result.latitude, expected_lat, tolerance=0.01)
        assert_close(result.longitude, expected_lon, tolerance=0.01)

    def test_dms_conversion_indonesia_accuracy(self, coordinate_parser):
        """DMS Indonesia conversion matches expected decimal."""
        expected_lat = dms_to_decimal(2, 31, 26.4, "S")
        expected_lon = dms_to_decimal(111, 52, 33.6, "E")
        raw = make_raw(DMS_INDONESIA, country="ID", commodity="oil_palm")
        result = coordinate_parser.parse(raw)
        assert_close(result.latitude, expected_lat, tolerance=0.01)
        assert_close(result.longitude, expected_lon, tolerance=0.01)

    def test_dms_compact_notation(self, coordinate_parser):
        """Compact DMS (no space between lat/lon) is parsed correctly."""
        raw = make_raw(DMS_COMPACT)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=0.01)


# ===========================================================================
# 13. Additional Edge Cases - Unusual Input Formats
# ===========================================================================


class TestUnusualInputFormats:
    """Test parsing of unusual but valid input formats."""

    def test_trailing_zeros_preserved(self, coordinate_parser):
        """Trailing zeros in DD string are parsed correctly."""
        raw = make_raw(DD_TRAILING_ZEROS)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, 5.603716, tolerance=0.0001)
        assert_close(result.longitude, -0.186964, tolerance=0.0001)

    def test_plus_sign_positive(self, coordinate_parser):
        """Leading plus sign on latitude is handled."""
        raw = make_raw(DD_PLUS_SIGN)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude > 0

    def test_excessive_whitespace(self, coordinate_parser):
        """Excessive whitespace around coordinates is trimmed."""
        raw = make_raw(DD_EXCESSIVE_WHITESPACE)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, 5.603716, tolerance=0.001)

    def test_utm_lowercase_letter(self, coordinate_parser):
        """UTM with lowercase zone letter is handled."""
        raw = make_raw(UTM_LOWERCASE_GHANA)
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True

    def test_very_high_precision_dd(self, coordinate_parser):
        """DD with 10+ decimal places parsed correctly."""
        raw = make_raw("5.6037158900, -0.1869642300")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, 5.60371589, tolerance=0.00001)

    def test_parse_negative_zero(self, coordinate_parser):
        """Negative zero (-0.0) is treated as zero."""
        raw = make_raw("-0.0, 0.0")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        # Both -0.0 and 0.0 should be treated equivalently
        assert abs(result.latitude) < 0.001


# ===========================================================================
# 14. Provenance and Determinism
# ===========================================================================


class TestParserProvenance:
    """Test provenance tracking and deterministic behaviour."""

    def test_parse_provenance_hash_present(self, coordinate_parser):
        """Parsed result includes a SHA-256 provenance hash."""
        raw = make_raw(DD_COMMA_GHANA)
        result = coordinate_parser.parse(raw)
        assert hasattr(result, "provenance_hash")
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_parse_deterministic_same_input(self, coordinate_parser):
        """Same input always produces the same parsed result."""
        raw1 = make_raw("5.603716, -0.186964")
        raw2 = make_raw("5.603716, -0.186964")
        r1 = coordinate_parser.parse(raw1)
        r2 = coordinate_parser.parse(raw2)
        assert r1.latitude == r2.latitude
        assert r1.longitude == r2.longitude
        assert r1.detected_format == r2.detected_format
        assert r1.provenance_hash == r2.provenance_hash

    def test_parse_different_input_different_hash(self, coordinate_parser):
        """Different inputs produce different provenance hashes."""
        raw1 = make_raw("5.603716, -0.186964")
        raw2 = make_raw("-12.97, -55.32")
        r1 = coordinate_parser.parse(raw1)
        r2 = coordinate_parser.parse(raw2)
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# 15. All EUDR Production Regions
# ===========================================================================


class TestAllEUDRRegions:
    """Test parsing for all 10 sample EUDR production region coordinates."""

    def test_parse_rubber_thailand(self, coordinate_parser):
        """Parse rubber farm Thailand coordinate."""
        raw = make_raw(
            f"{RUBBER_FARM_THAILAND[0]}, {RUBBER_FARM_THAILAND[1]}",
            country="TH",
            commodity="rubber",
        )
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, RUBBER_FARM_THAILAND[0], tolerance=0.001)

    def test_parse_cattle_brazil(self, coordinate_parser):
        """Parse cattle ranch Brazil coordinate."""
        raw = make_raw(
            f"{CATTLE_RANCH_BRAZIL[0]}, {CATTLE_RANCH_BRAZIL[1]}",
            country="BR",
            commodity="cattle",
        )
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, CATTLE_RANCH_BRAZIL[0], tolerance=0.001)

    def test_parse_timber_congo(self, coordinate_parser):
        """Parse timber forest Congo coordinate."""
        raw = make_raw(
            f"{TIMBER_FOREST_CONGO[0]}, {TIMBER_FOREST_CONGO[1]}",
            country="CD",
            commodity="wood",
        )
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude < 0

    def test_parse_coffee_ethiopia(self, coordinate_parser):
        """Parse coffee farm Ethiopia coordinate."""
        raw = make_raw(
            f"{COFFEE_FARM_ETHIOPIA[0]}, {COFFEE_FARM_ETHIOPIA[1]}",
            country="ET",
            commodity="coffee",
        )
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert result.latitude > 0
        assert result.longitude > 0

    def test_parse_high_precision_coord(self, coordinate_parser):
        """Parse 8-decimal-place high-precision coordinate."""
        raw = make_raw(f"{HIGH_PRECISION[0]}, {HIGH_PRECISION[1]}")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, HIGH_PRECISION[0], tolerance=0.0001)

    def test_parse_low_precision_coord(self, coordinate_parser):
        """Parse 1-decimal-place low-precision coordinate."""
        raw = make_raw(f"{LOW_PRECISION[0]}, {LOW_PRECISION[1]}")
        result = coordinate_parser.parse(raw)
        assert result.parse_successful is True
        assert_close(result.latitude, LOW_PRECISION[0], tolerance=0.1)


# ===========================================================================
# 16. Batch Parsing - Extended
# ===========================================================================


class TestBatchParseExtended:
    """Extended batch parsing tests."""

    def test_batch_parse_10_coordinates(self, coordinate_parser):
        """Parse a batch of 10 EUDR coordinates."""
        coords = [
            COCOA_FARM_GHANA, PALM_PLANTATION_INDONESIA, COFFEE_FARM_COLOMBIA,
            SOYA_FIELD_BRAZIL, RUBBER_FARM_THAILAND, CATTLE_RANCH_BRAZIL,
            TIMBER_FOREST_CONGO, COFFEE_FARM_ETHIOPIA,
            NULL_ISLAND, HIGH_PRECISION,
        ]
        raw_inputs = [make_raw(f"{c[0]}, {c[1]}") for c in coords]
        results = coordinate_parser.parse_batch(raw_inputs)
        assert len(results) == 10
        # All should parse successfully since they are valid DD strings
        for r in results:
            assert r.parse_successful is True

    def test_batch_parse_with_invalid_mixed(self, coordinate_parser):
        """Batch parse with mix of valid and invalid inputs."""
        raw_inputs = [
            make_raw("5.603716, -0.186964"),
            make_raw(GARBAGE_INPUT),
            make_raw("-12.97, -55.32"),
            make_raw(PARTIAL_INPUT),
        ]
        results = coordinate_parser.parse_batch(raw_inputs)
        assert len(results) == 4
        assert results[0].parse_successful is True
        assert results[1].parse_successful is False
        assert results[2].parse_successful is True
        assert results[3].parse_successful is False

    def test_batch_parse_preserves_order(self, coordinate_parser):
        """Batch parse results maintain input order."""
        lats = [5.603716, -12.97, 7.8804]
        raw_inputs = [make_raw(f"{lat}, 0.0") for lat in lats]
        results = coordinate_parser.parse_batch(raw_inputs)
        for i, r in enumerate(results):
            assert_close(r.latitude, lats[i], tolerance=0.001)


# ===========================================================================
# 17. Parametrized All EUDR Coordinate Parsing
# ===========================================================================


@pytest.mark.parametrize(
    "coord,name",
    [
        (COCOA_FARM_GHANA, "ghana_cocoa"),
        (PALM_PLANTATION_INDONESIA, "indonesia_palm"),
        (COFFEE_FARM_COLOMBIA, "colombia_coffee"),
        (SOYA_FIELD_BRAZIL, "brazil_soya"),
        (RUBBER_FARM_THAILAND, "thailand_rubber"),
        (CATTLE_RANCH_BRAZIL, "brazil_cattle"),
        (TIMBER_FOREST_CONGO, "congo_timber"),
        (COFFEE_FARM_ETHIOPIA, "ethiopia_coffee"),
        (BOUNDARY_LATITUDE, "north_pole"),
        (SOUTH_POLE, "south_pole"),
        (ANTIMERIDIAN_EAST, "antimeridian"),
    ],
    ids=[
        "ghana_cocoa", "indonesia_palm", "colombia_coffee", "brazil_soya",
        "thailand_rubber", "brazil_cattle", "congo_timber", "ethiopia_coffee",
        "north_pole", "south_pole", "antimeridian",
    ],
)
def test_parametrized_all_eudr_coords(coordinate_parser, coord, name):
    """Parametrized: all EUDR + boundary coordinates parse successfully."""
    raw = make_raw(f"{coord[0]}, {coord[1]}")
    result = coordinate_parser.parse(raw)
    assert result.parse_successful is True, f"Failed to parse {name}: {coord}"
    assert_close(result.latitude, coord[0], tolerance=0.001)
    assert_close(result.longitude, coord[1], tolerance=0.001)
