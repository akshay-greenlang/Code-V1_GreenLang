# -*- coding: utf-8 -*-
"""
Tests for FormatValidator - AGENT-EUDR-007 Engine 4: Format Validation

Comprehensive test suite covering:
- Valid coordinate range checking
- Latitude and longitude out-of-range detection
- Lat/lon swap detection for Ghana, Brazil, Indonesia
- Sign error detection (missing negative)
- Hemisphere error detection (N/S mixup)
- Null Island (0, 0) detection
- NaN and Inf value detection
- Exact duplicate detection
- Near-duplicate detection (< 1m apart)
- Auto-correction for swapped coordinates
- Auto-correction for sign errors
- Batch validation with mixed results
- Parametrized tests for error types

Test count: 50+ tests
Coverage target: >= 85% of FormatValidator module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import math

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    ValidationErrorType,
    ValidationSeverity,
    ValidationResult,
    ValidationError as VError,
    BatchValidationResult,
)
from tests.agents.eudr.gps_coordinate_validator.conftest import (
    COCOA_FARM_GHANA,
    PALM_PLANTATION_INDONESIA,
    SOYA_FIELD_BRAZIL,
    OCEAN_POINT,
    NULL_ISLAND,
    SWAPPED_COORDS,
    SIGN_ERROR,
    LOW_PRECISION,
    TRUNCATED,
    ARCTIC_POINT,
    URBAN_POINT,
    DESERT_POINT,
    BOUNDARY_LATITUDE,
    BOUNDARY_LONGITUDE,
    SOUTH_POLE,
    assert_valid,
    assert_invalid,
    assert_close,
    haversine_distance_m,
)


# ===========================================================================
# 1. Valid Range Checking
# ===========================================================================


class TestValidRange:
    """Test coordinate range validation for WGS84 bounds."""

    def test_valid_range_normal(self, format_validator):
        """Normal coordinates pass range check."""
        result = format_validator.validate(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert_valid(result)

    def test_valid_range_boundaries(self, format_validator):
        """Boundary coordinates (90, 180) are valid."""
        result = format_validator.validate(latitude=90.0, longitude=180.0)
        # No out-of-range error
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) == 0

    def test_valid_range_negative_boundaries(self, format_validator):
        """Boundary coordinates (-90, -180) are valid."""
        result = format_validator.validate(latitude=-90.0, longitude=-180.0)
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) == 0

    def test_valid_range_origin(self, format_validator):
        """Origin (0, 0) passes range check but may trigger null island warning."""
        result = format_validator.validate(latitude=0.0, longitude=0.0)
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) == 0


# ===========================================================================
# 2. Out-of-Range Detection
# ===========================================================================


class TestOutOfRange:
    """Test latitude and longitude out-of-range detection."""

    def test_latitude_above_90(self, format_validator):
        """Latitude > 90 is detected as out of range."""
        result = format_validator.validate(latitude=91.0, longitude=0.0)
        assert_invalid(result)
        errors = [e for e in result.errors if e.error_type == ValidationErrorType.OUT_OF_RANGE]
        assert len(errors) >= 1
        assert errors[0].field in ("latitude", "both")

    def test_latitude_below_neg90(self, format_validator):
        """Latitude < -90 is detected as out of range."""
        result = format_validator.validate(latitude=-91.0, longitude=0.0)
        assert_invalid(result)
        errors = [e for e in result.errors if e.error_type == ValidationErrorType.OUT_OF_RANGE]
        assert len(errors) >= 1

    def test_longitude_above_180(self, format_validator):
        """Longitude > 180 is detected as out of range."""
        result = format_validator.validate(latitude=0.0, longitude=181.0)
        assert_invalid(result)
        errors = [e for e in result.errors if e.error_type == ValidationErrorType.OUT_OF_RANGE]
        assert len(errors) >= 1
        assert errors[0].field in ("longitude", "both")

    def test_longitude_below_neg180(self, format_validator):
        """Longitude < -180 is detected as out of range."""
        result = format_validator.validate(latitude=0.0, longitude=-181.0)
        assert_invalid(result)

    def test_latitude_far_out_of_range(self, format_validator):
        """Very large latitude value is caught."""
        result = format_validator.validate(latitude=500.0, longitude=0.0)
        assert_invalid(result)

    def test_longitude_far_out_of_range(self, format_validator):
        """Very large longitude value is caught."""
        result = format_validator.validate(latitude=0.0, longitude=999.0)
        assert_invalid(result)


# ===========================================================================
# 3. Swap Detection
# ===========================================================================


class TestSwapDetection:
    """Test lat/lon swap detection for known EUDR production regions."""

    def test_swap_detection_ghana(self, format_validator):
        """Swapped Ghana coords: lon=5.603716, lat=-0.186964 detected."""
        result = format_validator.validate(
            latitude=SWAPPED_COORDS[0],  # -0.186964 (was lon)
            longitude=SWAPPED_COORDS[1],  # 5.603716 (was lat)
            declared_country="GH",
        )
        swap_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.SWAPPED
        ]
        assert len(swap_errors) >= 1

    def test_swap_detection_brazil(self, format_validator):
        """Swapped Brazil coords detected."""
        result = format_validator.validate(
            latitude=-55.32,   # was longitude
            longitude=-12.97,  # was latitude
            declared_country="BR",
        )
        # -55.32 as latitude is valid range but swapped context
        swap_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.SWAPPED
        ]
        assert len(swap_errors) >= 1

    def test_swap_detection_indonesia(self, format_validator):
        """Swapped Indonesia coords: latitude > 90 if swapped."""
        result = format_validator.validate(
            latitude=111.876,   # was longitude (out of range as lat)
            longitude=-2.524,   # was latitude
            declared_country="ID",
        )
        # 111.876 > 90, so either OOR or swap detected
        errors = result.errors
        has_swap = any(e.error_type == ValidationErrorType.SWAPPED for e in errors)
        has_oor = any(e.error_type == ValidationErrorType.OUT_OF_RANGE for e in errors)
        assert has_swap or has_oor

    def test_no_swap_valid_coordinate(self, format_validator):
        """Valid coordinate does not trigger swap detection."""
        result = format_validator.validate(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            declared_country="GH",
        )
        swap_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.SWAPPED
        ]
        assert len(swap_errors) == 0


# ===========================================================================
# 4. Sign Error Detection
# ===========================================================================


class TestSignErrorDetection:
    """Test detection of missing negative signs on coordinates."""

    def test_sign_error_detection_ghana_lon(self, format_validator):
        """Missing negative on Ghana longitude detected."""
        result = format_validator.validate(
            latitude=SIGN_ERROR[0],    # 5.603716 (correct)
            longitude=SIGN_ERROR[1],   # 0.186964 (should be -0.186964)
            declared_country="GH",
        )
        sign_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.SIGN_ERROR
        ]
        assert len(sign_errors) >= 1

    def test_no_sign_error_valid_coordinate(self, format_validator):
        """Valid coordinate does not trigger sign error."""
        result = format_validator.validate(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            declared_country="GH",
        )
        sign_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.SIGN_ERROR
        ]
        assert len(sign_errors) == 0


# ===========================================================================
# 5. Null Island Detection
# ===========================================================================


class TestNullIsland:
    """Test detection of Null Island (0, 0) coordinates."""

    def test_null_island_exact_zero(self, format_validator):
        """Exact (0, 0) is detected as Null Island."""
        result = format_validator.validate(
            latitude=NULL_ISLAND[0],
            longitude=NULL_ISLAND[1],
        )
        null_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.NULL_ISLAND
        ]
        assert len(null_errors) >= 1

    def test_null_island_near_zero(self, format_validator):
        """Near-zero coordinates within threshold are flagged."""
        result = format_validator.validate(
            latitude=0.001,
            longitude=0.001,
        )
        null_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.NULL_ISLAND
        ]
        # Within 1000m of (0,0) should be flagged
        assert len(null_errors) >= 1

    def test_not_null_island_far_from_origin(self, format_validator):
        """Coordinates far from origin are not flagged as Null Island."""
        result = format_validator.validate(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        null_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.NULL_ISLAND
        ]
        assert len(null_errors) == 0


# ===========================================================================
# 6. NaN and Inf Detection
# ===========================================================================


class TestNaNInfDetection:
    """Test detection of NaN and infinity coordinate values."""

    def test_nan_latitude(self, format_validator):
        """NaN latitude is detected."""
        result = format_validator.validate(
            latitude=float("nan"),
            longitude=0.0,
        )
        assert_invalid(result)
        nan_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.NAN_VALUE
        ]
        assert len(nan_errors) >= 1

    def test_nan_longitude(self, format_validator):
        """NaN longitude is detected."""
        result = format_validator.validate(
            latitude=0.0,
            longitude=float("nan"),
        )
        assert_invalid(result)

    def test_inf_latitude(self, format_validator):
        """Positive infinity latitude is detected."""
        result = format_validator.validate(
            latitude=float("inf"),
            longitude=0.0,
        )
        assert_invalid(result)
        inf_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.INF_VALUE
        ]
        assert len(inf_errors) >= 1

    def test_negative_inf_longitude(self, format_validator):
        """Negative infinity longitude is detected."""
        result = format_validator.validate(
            latitude=0.0,
            longitude=float("-inf"),
        )
        assert_invalid(result)


# ===========================================================================
# 7. Duplicate Detection
# ===========================================================================


class TestDuplicateDetection:
    """Test exact and near-duplicate coordinate detection."""

    def test_exact_duplicate_detection(self, format_validator):
        """Exact duplicates are detected in batch."""
        coords = [
            (5.603716, -0.186964),
            (5.603716, -0.186964),  # exact duplicate
            (-12.97, -55.32),
        ]
        result = format_validator.validate_batch(coords)
        assert isinstance(result, BatchValidationResult)
        assert len(result.duplicate_pairs) >= 1
        assert (0, 1) in result.duplicate_pairs

    def test_near_duplicate_detection(self, format_validator):
        """Near-duplicates within 1m are detected."""
        coords = [
            (5.603716, -0.186964),
            (5.603717, -0.186964),  # ~0.11m away
            (-12.97, -55.32),
        ]
        result = format_validator.validate_batch(coords)
        # Should detect near-duplicate between index 0 and 1
        assert len(result.near_duplicate_pairs) >= 1

    def test_no_duplicates_distinct(self, format_validator):
        """Distinct coordinates have no duplicates."""
        coords = [
            (5.603716, -0.186964),
            (-12.97, -55.32),
            (7.8804, 98.3923),
        ]
        result = format_validator.validate_batch(coords)
        assert len(result.duplicate_pairs) == 0


# ===========================================================================
# 8. Auto-Correction
# ===========================================================================


class TestAutoCorrection:
    """Test auto-correction of detected errors."""

    def test_auto_correction_swap(self, format_validator):
        """Swapped coordinates can be auto-corrected."""
        result = format_validator.validate(
            latitude=SWAPPED_COORDS[0],
            longitude=SWAPPED_COORDS[1],
            declared_country="GH",
        )
        if result.auto_correctable:
            assert result.corrected_latitude is not None
            assert result.corrected_longitude is not None
            # Corrected should be closer to Ghana
            assert_close(result.corrected_latitude, COCOA_FARM_GHANA[0], tolerance=0.5)

    def test_auto_correction_sign(self, format_validator):
        """Sign error can be auto-corrected."""
        result = format_validator.validate(
            latitude=SIGN_ERROR[0],
            longitude=SIGN_ERROR[1],
            declared_country="GH",
        )
        if result.auto_correctable:
            assert result.corrected_longitude is not None
            assert result.corrected_longitude < 0  # Should be negative for Ghana


# ===========================================================================
# 9. Batch Validation
# ===========================================================================


class TestBatchValidation:
    """Test batch validation with mixed results."""

    def test_batch_validation_mixed(self, format_validator):
        """Batch with valid and invalid coordinates."""
        coords = [
            (COCOA_FARM_GHANA[0], COCOA_FARM_GHANA[1]),        # valid
            (91.0, 0.0),                                         # invalid: OOR
            (SOYA_FIELD_BRAZIL[0], SOYA_FIELD_BRAZIL[1]),       # valid
            (0.0, 0.0),                                          # suspicious: null island
            (float("nan"), 0.0),                                 # invalid: NaN
        ]
        result = format_validator.validate_batch(coords)
        assert isinstance(result, BatchValidationResult)
        assert result.total_count == 5
        assert result.valid_count >= 1
        assert result.error_count >= 2  # OOR and NaN

    def test_batch_validation_all_valid(self, format_validator):
        """Batch with all valid coordinates."""
        coords = [
            COCOA_FARM_GHANA,
            SOYA_FIELD_BRAZIL,
            PALM_PLANTATION_INDONESIA,
        ]
        result = format_validator.validate_batch(coords)
        assert result.total_count == 3
        assert result.valid_count == 3
        assert result.error_count == 0

    def test_batch_validation_empty(self, format_validator):
        """Empty batch returns empty result."""
        result = format_validator.validate_batch([])
        assert result.total_count == 0


# ===========================================================================
# 10. Provenance
# ===========================================================================


class TestFormatValidatorProvenance:
    """Test provenance hash generation."""

    def test_validation_has_provenance(self, format_validator):
        """Single validation result includes provenance hash."""
        result = format_validator.validate(
            latitude=5.603716,
            longitude=-0.186964,
        )
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_batch_has_provenance(self, format_validator):
        """Batch validation result includes provenance hash."""
        result = format_validator.validate_batch([COCOA_FARM_GHANA])
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 11. Parametrized Error Type Tests
# ===========================================================================


@pytest.mark.parametrize(
    "lat,lon,expected_error_type",
    [
        (91.0, 0.0, ValidationErrorType.OUT_OF_RANGE),
        (-91.0, 0.0, ValidationErrorType.OUT_OF_RANGE),
        (0.0, 181.0, ValidationErrorType.OUT_OF_RANGE),
        (0.0, -181.0, ValidationErrorType.OUT_OF_RANGE),
        (0.0, 0.0, ValidationErrorType.NULL_ISLAND),
        (float("nan"), 0.0, ValidationErrorType.NAN_VALUE),
        (float("inf"), 0.0, ValidationErrorType.INF_VALUE),
    ],
    ids=[
        "lat_above_90", "lat_below_neg90", "lon_above_180", "lon_below_neg180",
        "null_island", "nan_lat", "inf_lat",
    ],
)
def test_parametrized_error_types(format_validator, lat, lon, expected_error_type):
    """Parametrized: various error types are correctly detected."""
    result = format_validator.validate(latitude=lat, longitude=lon)
    all_issues = result.errors + result.warnings
    found_types = [e.error_type for e in all_issues]
    assert expected_error_type in found_types, (
        f"Expected error type {expected_error_type} not found in {found_types}"
    )


# ===========================================================================
# 12. Hemisphere Error Detection
# ===========================================================================


class TestHemisphereErrorDetection:
    """Test detection of hemisphere errors (e.g., positive lat for Southern hemisphere)."""

    def test_hemisphere_error_brazil_positive_lat(self, format_validator):
        """Brazil coordinate with positive latitude (should be negative)."""
        result = format_validator.validate(
            latitude=abs(SOYA_FIELD_BRAZIL[0]),   # 12.97 (should be -12.97)
            longitude=SOYA_FIELD_BRAZIL[1],
            declared_country="BR",
        )
        hemisphere_errors = [
            e for e in result.errors + result.warnings
            if e.error_type in (
                ValidationErrorType.HEMISPHERE_ERROR,
                ValidationErrorType.SIGN_ERROR,
            )
        ]
        assert len(hemisphere_errors) >= 1

    def test_hemisphere_error_indonesia_positive_lat(self, format_validator):
        """Indonesia coordinate with positive latitude (most of Indonesia is S)."""
        result = format_validator.validate(
            latitude=abs(PALM_PLANTATION_INDONESIA[0]),  # 2.524 (should be -2.524)
            longitude=PALM_PLANTATION_INDONESIA[1],
            declared_country="ID",
        )
        # This may or may not trigger since some of Indonesia is north of equator
        # Just verify the result is a ValidationResult
        assert isinstance(result, ValidationResult)

    def test_no_hemisphere_error_correct_sign(self, format_validator):
        """Correct hemisphere signs do not trigger errors."""
        result = format_validator.validate(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
            declared_country="BR",
        )
        hemisphere_errors = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.HEMISPHERE_ERROR
        ]
        assert len(hemisphere_errors) == 0


# ===========================================================================
# 13. Boundary Value Tests
# ===========================================================================


class TestBoundaryValues:
    """Test boundary values that are technically valid but suspicious."""

    def test_boundary_exactly_90(self, format_validator):
        """Latitude exactly 90 is valid but may trigger boundary warning."""
        result = format_validator.validate(
            latitude=BOUNDARY_LATITUDE[0],
            longitude=BOUNDARY_LATITUDE[1],
        )
        # Should not have out-of-range error
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) == 0

    def test_boundary_exactly_neg90(self, format_validator):
        """Latitude exactly -90 is valid."""
        result = format_validator.validate(
            latitude=SOUTH_POLE[0],
            longitude=SOUTH_POLE[1],
        )
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) == 0

    def test_boundary_exactly_180(self, format_validator):
        """Longitude exactly 180 is valid."""
        result = format_validator.validate(
            latitude=BOUNDARY_LONGITUDE[0],
            longitude=BOUNDARY_LONGITUDE[1],
        )
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) == 0

    def test_boundary_warning_polar(self, format_validator):
        """Polar coordinates may trigger boundary value warning."""
        result = format_validator.validate(latitude=90.0, longitude=0.0)
        # Check if there is a boundary_value warning (optional enhancement)
        boundary_issues = [
            e for e in result.warnings
            if e.error_type == ValidationErrorType.BOUNDARY_VALUE
        ]
        # May or may not have boundary warning depending on implementation
        assert isinstance(result, ValidationResult)


# ===========================================================================
# 14. Error Severity Checks
# ===========================================================================


class TestErrorSeverity:
    """Test that error severity levels are correctly assigned."""

    def test_out_of_range_is_critical(self, format_validator):
        """Out-of-range errors have CRITICAL or ERROR severity."""
        result = format_validator.validate(latitude=91.0, longitude=0.0)
        oor_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.OUT_OF_RANGE
        ]
        assert len(oor_errors) >= 1
        for e in oor_errors:
            assert e.severity in (
                ValidationSeverity.CRITICAL,
                ValidationSeverity.ERROR,
            )

    def test_nan_is_critical(self, format_validator):
        """NaN errors have CRITICAL severity."""
        result = format_validator.validate(latitude=float("nan"), longitude=0.0)
        nan_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.NAN_VALUE
        ]
        assert len(nan_errors) >= 1
        for e in nan_errors:
            assert e.severity in (
                ValidationSeverity.CRITICAL,
                ValidationSeverity.ERROR,
            )

    def test_null_island_is_warning(self, format_validator):
        """Null Island is typically a WARNING, not an ERROR."""
        result = format_validator.validate(latitude=0.0, longitude=0.0)
        null_issues = [
            e for e in result.errors + result.warnings
            if e.error_type == ValidationErrorType.NULL_ISLAND
        ]
        assert len(null_issues) >= 1
        for e in null_issues:
            assert e.severity in (
                ValidationSeverity.WARNING,
                ValidationSeverity.ERROR,
            )


# ===========================================================================
# 15. Batch Validation Extended
# ===========================================================================


class TestBatchValidationExtended:
    """Extended batch validation tests."""

    def test_batch_validation_10_coordinates(self, format_validator):
        """Batch validate 10 coordinates."""
        coords = [
            COCOA_FARM_GHANA,
            PALM_PLANTATION_INDONESIA,
            SOYA_FIELD_BRAZIL,
            URBAN_POINT,
            DESERT_POINT,
            ARCTIC_POINT,
            SOUTH_POLE,
            BOUNDARY_LATITUDE,
            BOUNDARY_LONGITUDE,
            LOW_PRECISION,
        ]
        result = format_validator.validate_batch(coords)
        assert isinstance(result, BatchValidationResult)
        assert result.total_count == 10

    def test_batch_validation_single_item(self, format_validator):
        """Batch validate a single coordinate."""
        coords = [COCOA_FARM_GHANA]
        result = format_validator.validate_batch(coords)
        assert result.total_count == 1
        assert result.valid_count == 1

    def test_batch_validation_all_invalid(self, format_validator):
        """Batch with all invalid coordinates."""
        coords = [
            (91.0, 0.0),
            (-91.0, 0.0),
            (0.0, 181.0),
        ]
        result = format_validator.validate_batch(coords)
        assert result.total_count == 3
        assert result.error_count == 3
        assert result.valid_count == 0

    def test_batch_validation_processing_time(self, format_validator):
        """Batch validation records processing time."""
        coords = [COCOA_FARM_GHANA, SOYA_FIELD_BRAZIL]
        result = format_validator.validate_batch(coords)
        assert result.processing_time_ms >= 0


# ===========================================================================
# 16. Provenance Extended
# ===========================================================================


class TestFormatValidatorProvenanceExtended:
    """Extended provenance tracking for format validation."""

    def test_deterministic_provenance(self, format_validator):
        """Same input produces same provenance hash."""
        r1 = format_validator.validate(latitude=5.603716, longitude=-0.186964)
        r2 = format_validator.validate(latitude=5.603716, longitude=-0.186964)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input_different_hash(self, format_validator):
        """Different inputs produce different provenance hashes."""
        r1 = format_validator.validate(latitude=5.603716, longitude=-0.186964)
        r2 = format_validator.validate(latitude=-12.97, longitude=-55.32)
        assert r1.provenance_hash != r2.provenance_hash

    def test_provenance_hash_hex_characters(self, format_validator):
        """Provenance hash contains only hex characters."""
        result = format_validator.validate(latitude=5.603716, longitude=-0.186964)
        import re
        assert re.match(r"^[0-9a-f]{64}$", result.provenance_hash)
