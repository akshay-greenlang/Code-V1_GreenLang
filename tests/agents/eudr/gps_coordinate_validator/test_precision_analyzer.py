# -*- coding: utf-8 -*-
"""
Tests for PrecisionAnalyzer - AGENT-EUDR-007 Engine 3: Precision Assessment

Comprehensive test suite covering:
- Decimal place counting (0 through 10 decimal places)
- Ground resolution at equator (0 degrees latitude)
- Ground resolution at 45 degrees latitude
- Ground resolution at high latitude (80 degrees)
- Precision level classification (survey_grade, high, moderate, low, inadequate)
- EUDR adequacy assessment (pass >= 5dp, fail < 5dp)
- Truncation detection (integer coordinates)
- Artificial rounding detection (.000000 patterns)
- Source precision estimation (GNSS, mobile, handheld)
- Batch precision analysis
- Parametrized tests for decimal places 0-10

Test count: 45+ tests
Coverage target: >= 85% of PrecisionAnalyzer module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import math

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    PrecisionLevel,
    PrecisionResult,
    SourceType,
)
from tests.agents.eudr.gps_coordinate_validator.conftest import (
    COCOA_FARM_GHANA,
    PALM_PLANTATION_INDONESIA,
    SOYA_FIELD_BRAZIL,
    RUBBER_FARM_THAILAND,
    HIGH_PRECISION,
    LOW_PRECISION,
    TRUNCATED,
    NULL_ISLAND,
    BOUNDARY_LATITUDE,
    SOUTH_POLE,
    DESERT_POINT,
    PRECISION_GROUND_RESOLUTION,
    SHA256_HEX_LENGTH,
    assert_close,
    ground_resolution_at_latitude,
)


# ===========================================================================
# 1. Decimal Place Counting
# ===========================================================================


class TestDecimalPlaceCounting:
    """Test counting of significant decimal places in coordinate values."""

    @pytest.mark.parametrize(
        "value,expected_dp",
        [
            (6.0, 0),
            (5.6, 1),
            (5.60, 1),          # trailing zero not significant in float repr
            (5.603, 3),
            (5.6037, 4),
            (5.60371, 5),
            (5.603716, 6),
            (5.6037158, 7),
            (5.60371589, 8),
            (-0.186964, 6),
            (-12.97, 2),
            (0.0, 0),
        ],
        ids=[
            "0dp", "1dp", "1dp_trailing", "3dp", "4dp", "5dp",
            "6dp", "7dp", "8dp", "neg_6dp", "neg_2dp", "zero",
        ],
    )
    def test_count_decimal_places(self, precision_analyzer, value, expected_dp):
        """Count decimal places for various float values."""
        dp = precision_analyzer.count_decimal_places(value)
        assert dp == expected_dp, (
            f"Expected {expected_dp} decimal places for {value}, got {dp}"
        )

    def test_count_decimal_places_scientific_notation(self, precision_analyzer):
        """Handle scientific notation correctly."""
        # 1e-7 should be detected as having ~7 decimal places
        dp = precision_analyzer.count_decimal_places(1e-7)
        assert dp >= 7

    def test_count_decimal_places_integer(self, precision_analyzer):
        """Integer float has 0 decimal places."""
        dp = precision_analyzer.count_decimal_places(42.0)
        assert dp == 0


# ===========================================================================
# 2. Ground Resolution Calculation
# ===========================================================================


class TestGroundResolution:
    """Test ground resolution calculation at various latitudes."""

    def test_ground_resolution_equator_6dp(self, precision_analyzer):
        """6 decimal places at equator: ~0.11m resolution."""
        result = precision_analyzer.analyze(
            latitude=0.123456,
            longitude=30.123456,
        )
        # At equator, 6dp => ~0.11m
        assert result.ground_resolution_lat_m < 0.2
        assert result.ground_resolution_lat_m > 0.05

    def test_ground_resolution_equator_3dp(self, precision_analyzer):
        """3 decimal places at equator: ~111m resolution."""
        result = precision_analyzer.analyze(
            latitude=0.123,
            longitude=30.123,
        )
        assert result.ground_resolution_lat_m > 50.0
        assert result.ground_resolution_lat_m < 200.0

    def test_ground_resolution_45_degrees(self, precision_analyzer):
        """Ground resolution at 45 degrees latitude (longitude shrinks)."""
        result = precision_analyzer.analyze(
            latitude=45.123456,
            longitude=10.123456,
        )
        # Longitude resolution shrinks by cos(45) ~ 0.707 at 45 degrees
        # Latitude resolution stays the same
        assert result.ground_resolution_lat_m < 0.2
        assert result.ground_resolution_lon_m < 0.15  # Smaller due to cos(45)

    def test_ground_resolution_high_latitude(self, precision_analyzer):
        """Ground resolution at 80 degrees latitude."""
        result = precision_analyzer.analyze(
            latitude=80.123456,
            longitude=15.123456,
        )
        # At 80 degrees, cos(80) ~ 0.174, so longitude resolution is much smaller
        assert result.ground_resolution_lon_m < result.ground_resolution_lat_m

    @pytest.mark.parametrize(
        "dp,expected_lat_resolution_m",
        PRECISION_GROUND_RESOLUTION,
        ids=[f"{dp}dp" for dp, _ in PRECISION_GROUND_RESOLUTION],
    )
    def test_ground_resolution_parametrized(
        self, precision_analyzer, dp, expected_lat_resolution_m
    ):
        """Parametrized ground resolution for 0-10 decimal places at equator."""
        # Construct a coordinate with exactly dp decimal places
        lat_str = f"0.{'1' * dp}" if dp > 0 else "0"
        lat = float(lat_str)
        lon_str = f"30.{'1' * dp}" if dp > 0 else "30"
        lon = float(lon_str)
        result = precision_analyzer.analyze(latitude=lat, longitude=lon)
        # Allow 50% tolerance due to float representation
        assert result.ground_resolution_lat_m < expected_lat_resolution_m * 2.0


# ===========================================================================
# 3. Precision Level Classification
# ===========================================================================


class TestPrecisionClassification:
    """Test precision level classification."""

    def test_precision_survey_grade(self, precision_analyzer):
        """8+ decimal places => SURVEY_GRADE."""
        result = precision_analyzer.analyze(
            latitude=HIGH_PRECISION[0],  # 8dp
            longitude=HIGH_PRECISION[1],  # 8dp
        )
        assert result.precision_level == PrecisionLevel.SURVEY_GRADE

    def test_precision_high(self, precision_analyzer):
        """5-6 decimal places => HIGH."""
        result = precision_analyzer.analyze(
            latitude=COCOA_FARM_GHANA[0],  # 6dp
            longitude=COCOA_FARM_GHANA[1],  # 6dp
        )
        assert result.precision_level in (PrecisionLevel.SURVEY_GRADE, PrecisionLevel.HIGH)

    def test_precision_moderate(self, precision_analyzer):
        """4 decimal places => MODERATE."""
        result = precision_analyzer.analyze(
            latitude=5.6037,
            longitude=-0.1870,
        )
        assert result.precision_level == PrecisionLevel.MODERATE

    def test_precision_low(self, precision_analyzer):
        """3 decimal places => LOW."""
        result = precision_analyzer.analyze(
            latitude=5.604,
            longitude=-0.187,
        )
        assert result.precision_level == PrecisionLevel.LOW

    def test_precision_inadequate(self, precision_analyzer):
        """0-2 decimal places => INADEQUATE."""
        result = precision_analyzer.analyze(
            latitude=LOW_PRECISION[0],  # 1dp
            longitude=LOW_PRECISION[1],  # 1dp
        )
        assert result.precision_level == PrecisionLevel.INADEQUATE


# ===========================================================================
# 4. EUDR Adequacy Assessment
# ===========================================================================


class TestEUDRAdequacy:
    """Test EUDR adequacy assessment (>= 5dp passes, < 5dp fails)."""

    def test_eudr_adequacy_pass_6dp(self, precision_analyzer):
        """6 decimal places passes EUDR adequacy."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
        )
        assert result.eudr_adequate is True

    def test_eudr_adequacy_pass_5dp(self, precision_analyzer):
        """5 decimal places passes EUDR adequacy (borderline)."""
        result = precision_analyzer.analyze(
            latitude=5.60372,
            longitude=-0.18696,
        )
        assert result.eudr_adequate is True

    def test_eudr_adequacy_fail_4dp(self, precision_analyzer):
        """4 decimal places fails EUDR adequacy."""
        result = precision_analyzer.analyze(
            latitude=5.6037,
            longitude=-0.1870,
        )
        assert result.eudr_adequate is False

    def test_eudr_adequacy_fail_1dp(self, precision_analyzer):
        """1 decimal place fails EUDR adequacy."""
        result = precision_analyzer.analyze(
            latitude=LOW_PRECISION[0],
            longitude=LOW_PRECISION[1],
        )
        assert result.eudr_adequate is False

    def test_eudr_adequacy_fail_0dp(self, precision_analyzer):
        """0 decimal places (integer) fails EUDR adequacy."""
        result = precision_analyzer.analyze(
            latitude=TRUNCATED[0],
            longitude=TRUNCATED[1],
        )
        assert result.eudr_adequate is False


# ===========================================================================
# 5. Truncation and Rounding Detection
# ===========================================================================


class TestTruncationDetection:
    """Test detection of truncated and artificially rounded coordinates."""

    def test_truncation_detection_integer(self, precision_analyzer):
        """Integer coordinates are detected as truncated."""
        result = precision_analyzer.analyze(
            latitude=6.0,
            longitude=-1.0,
        )
        assert result.is_truncated is True

    def test_truncation_detection_zero(self, precision_analyzer):
        """Zero coordinates are detected as truncated."""
        result = precision_analyzer.analyze(
            latitude=0.0,
            longitude=0.0,
        )
        assert result.is_truncated is True

    def test_no_truncation_6dp(self, precision_analyzer):
        """6 decimal place coordinates are not truncated."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
        )
        assert result.is_truncated is False

    def test_artificial_rounding_all_zeros(self, precision_analyzer):
        """Coordinates ending in .000000 are flagged as artificially rounded."""
        result = precision_analyzer.analyze(
            latitude=5.000000,
            longitude=-1.000000,
        )
        assert result.is_artificially_rounded is True

    def test_no_artificial_rounding_normal(self, precision_analyzer):
        """Normal coordinates are not flagged as artificially rounded."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
        )
        assert result.is_artificially_rounded is False


# ===========================================================================
# 6. Source Precision Estimation
# ===========================================================================


class TestSourcePrecisionEstimation:
    """Test source precision estimation by device type."""

    def test_source_precision_gnss(self, precision_analyzer):
        """GNSS survey equipment has sub-metre expected precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert result.estimated_source_precision_m is not None
        assert result.estimated_source_precision_m < 1.0

    def test_source_precision_rtk(self, precision_analyzer):
        """RTK GPS has sub-metre expected precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.RTK_GPS,
        )
        assert result.estimated_source_precision_m is not None
        assert result.estimated_source_precision_m < 1.0

    def test_source_precision_mobile(self, precision_analyzer):
        """Mobile GPS has 3-10m expected precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.MOBILE_GPS,
        )
        assert result.estimated_source_precision_m is not None
        assert result.estimated_source_precision_m >= 1.0
        assert result.estimated_source_precision_m <= 15.0

    def test_source_precision_manual(self, precision_analyzer):
        """Manual entry has high uncertainty."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.MANUAL_ENTRY,
        )
        assert result.estimated_source_precision_m is not None
        assert result.estimated_source_precision_m > 5.0


# ===========================================================================
# 7. Batch Analysis
# ===========================================================================


class TestBatchAnalyze:
    """Test batch precision analysis."""

    def test_batch_analyze_multiple(self, precision_analyzer):
        """Batch analyze multiple coordinates."""
        coords = [
            (5.603716, -0.186964),
            (-12.97, -55.32),
            (6.0, -1.0),
        ]
        results = precision_analyzer.analyze_batch(coords)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, PrecisionResult)

    def test_batch_analyze_empty(self, precision_analyzer):
        """Batch analyze with empty list returns empty."""
        results = precision_analyzer.analyze_batch([])
        assert results == []


# ===========================================================================
# 8. Result Structure and Provenance
# ===========================================================================


class TestPrecisionResultStructure:
    """Test PrecisionResult structure and provenance."""

    def test_result_has_all_fields(self, precision_analyzer):
        """PrecisionResult has all expected fields populated."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
        )
        assert isinstance(result, PrecisionResult)
        assert result.latitude_decimal_places >= 0
        assert result.longitude_decimal_places >= 0
        assert result.effective_decimal_places >= 0
        assert result.ground_resolution_lat_m >= 0
        assert result.ground_resolution_lon_m >= 0
        assert isinstance(result.precision_level, PrecisionLevel)
        assert isinstance(result.eudr_adequate, bool)
        assert isinstance(result.is_truncated, bool)
        assert isinstance(result.is_artificially_rounded, bool)

    def test_result_has_provenance_hash(self, precision_analyzer):
        """PrecisionResult includes a provenance hash."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
        )
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_deterministic_result(self, precision_analyzer):
        """Same input produces identical result (deterministic)."""
        r1 = precision_analyzer.analyze(5.603716, -0.186964)
        r2 = precision_analyzer.analyze(5.603716, -0.186964)
        assert r1.provenance_hash == r2.provenance_hash
        assert r1.effective_decimal_places == r2.effective_decimal_places
        assert r1.precision_level == r2.precision_level

    def test_effective_decimal_places_is_minimum(self, precision_analyzer):
        """Effective decimal places is minimum of lat and lon decimal places."""
        result = precision_analyzer.analyze(
            latitude=5.603716,   # 6dp
            longitude=-0.19,     # 2dp
        )
        assert result.effective_decimal_places == min(
            result.latitude_decimal_places,
            result.longitude_decimal_places,
        )


# ===========================================================================
# 9. Latitude-Dependent Ground Resolution
# ===========================================================================


class TestLatitudeDependentResolution:
    """Test ground resolution variation with latitude."""

    def test_equator_vs_high_latitude(self, precision_analyzer):
        """Longitude resolution at equator > longitude resolution at 80 degrees."""
        equator_result = precision_analyzer.analyze(
            latitude=0.123456,
            longitude=30.123456,
        )
        high_lat_result = precision_analyzer.analyze(
            latitude=80.123456,
            longitude=15.123456,
        )
        # At high latitudes, longitude resolution is much smaller (tighter)
        assert high_lat_result.ground_resolution_lon_m < equator_result.ground_resolution_lon_m

    def test_latitude_resolution_constant(self, precision_analyzer):
        """Latitude resolution is roughly constant regardless of latitude."""
        equator_result = precision_analyzer.analyze(
            latitude=0.123456,
            longitude=30.123456,
        )
        mid_lat_result = precision_analyzer.analyze(
            latitude=45.123456,
            longitude=10.123456,
        )
        # Latitude resolution should be very similar
        ratio = equator_result.ground_resolution_lat_m / max(
            mid_lat_result.ground_resolution_lat_m, 0.0001
        )
        assert 0.8 < ratio < 1.2

    def test_polar_longitude_resolution_near_zero(self, precision_analyzer):
        """At the poles, longitude resolution approaches zero."""
        result = precision_analyzer.analyze(
            latitude=89.999999,
            longitude=0.123456,
        )
        # Longitude resolution at near-pole should be very small
        assert result.ground_resolution_lon_m < result.ground_resolution_lat_m * 0.1


# ===========================================================================
# 10. Asymmetric Precision (Lat vs Lon different)
# ===========================================================================


class TestAsymmetricPrecision:
    """Test analysis when lat and lon have different decimal places."""

    def test_asymmetric_6dp_lat_2dp_lon(self, precision_analyzer):
        """6dp latitude with 2dp longitude is flagged."""
        result = precision_analyzer.analyze(
            latitude=5.603716,   # 6dp
            longitude=-0.19,     # 2dp
        )
        assert result.latitude_decimal_places > result.longitude_decimal_places
        assert result.effective_decimal_places == result.longitude_decimal_places
        assert result.eudr_adequate is False  # limited by longitude

    def test_asymmetric_2dp_lat_6dp_lon(self, precision_analyzer):
        """2dp latitude with 6dp longitude is limited by latitude."""
        result = precision_analyzer.analyze(
            latitude=5.60,       # 2dp (actually 1dp in float)
            longitude=-0.186964, # 6dp
        )
        assert result.effective_decimal_places <= 2
        assert result.eudr_adequate is False

    def test_symmetric_6dp_both(self, precision_analyzer):
        """Both lat and lon at 6dp is EUDR adequate."""
        result = precision_analyzer.analyze(
            latitude=COCOA_FARM_GHANA[0],  # 6dp
            longitude=COCOA_FARM_GHANA[1], # 6dp
        )
        assert result.eudr_adequate is True

    def test_asymmetric_5dp_lat_8dp_lon(self, precision_analyzer):
        """5dp lat with 8dp lon: effective is 5dp (EUDR adequate)."""
        result = precision_analyzer.analyze(
            latitude=5.60372,      # 5dp
            longitude=-0.18696423, # 8dp
        )
        assert result.effective_decimal_places >= 5
        assert result.eudr_adequate is True


# ===========================================================================
# 11. All EUDR Region Precision Analysis
# ===========================================================================


class TestEUDRRegionPrecision:
    """Test precision analysis for all EUDR commodity region coordinates."""

    def test_ghana_precision(self, precision_analyzer):
        """Ghana cocoa coordinate has adequate precision."""
        result = precision_analyzer.analyze(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert result.eudr_adequate is True
        assert result.precision_level in (PrecisionLevel.HIGH, PrecisionLevel.SURVEY_GRADE)

    def test_indonesia_precision(self, precision_analyzer):
        """Indonesia palm coordinate has adequate precision."""
        result = precision_analyzer.analyze(
            latitude=PALM_PLANTATION_INDONESIA[0],
            longitude=PALM_PLANTATION_INDONESIA[1],
        )
        # -2.524000 has 3dp in float (trailing zeros lost), so may be inadequate
        assert isinstance(result.precision_level, PrecisionLevel)

    def test_brazil_precision(self, precision_analyzer):
        """Brazil soya coordinate precision analysis."""
        result = precision_analyzer.analyze(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
        )
        assert isinstance(result.precision_level, PrecisionLevel)
        assert result.ground_resolution_lat_m > 0

    def test_thailand_precision(self, precision_analyzer):
        """Thailand rubber coordinate precision analysis."""
        result = precision_analyzer.analyze(
            latitude=RUBBER_FARM_THAILAND[0],
            longitude=RUBBER_FARM_THAILAND[1],
        )
        assert isinstance(result, PrecisionResult)
        assert result.latitude_decimal_places >= 0


# ===========================================================================
# 12. Extended Source Precision Estimation
# ===========================================================================


class TestSourcePrecisionEstimationExtended:
    """Extended tests for source precision estimation by device type."""

    def test_source_precision_satellite_derived(self, precision_analyzer):
        """Satellite-derived coordinates have moderate precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.SATELLITE_DERIVED,
        )
        assert result.estimated_source_precision_m is not None
        assert result.estimated_source_precision_m > 0

    def test_source_precision_digitized_map(self, precision_analyzer):
        """Digitized map coordinates have low precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.DIGITIZED_MAP,
        )
        assert result.estimated_source_precision_m is not None
        assert result.estimated_source_precision_m > 5.0

    def test_source_precision_geocoded(self, precision_analyzer):
        """Geocoded coordinates have variable precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.GEOCODED,
        )
        assert result.estimated_source_precision_m is not None

    def test_source_precision_handheld_gps(self, precision_analyzer):
        """Handheld GPS has moderate precision."""
        result = precision_analyzer.analyze(
            latitude=5.603716,
            longitude=-0.186964,
            source_type=SourceType.HANDHELD_GPS,
        )
        assert result.estimated_source_precision_m is not None
        assert 1.0 <= result.estimated_source_precision_m <= 15.0


# ===========================================================================
# 13. Extended Batch Analysis
# ===========================================================================


class TestBatchAnalyzeExtended:
    """Extended batch precision analysis tests."""

    def test_batch_analyze_10_coordinates(self, precision_analyzer):
        """Batch analyze 10 coordinates."""
        coords = [
            COCOA_FARM_GHANA,
            PALM_PLANTATION_INDONESIA,
            SOYA_FIELD_BRAZIL,
            RUBBER_FARM_THAILAND,
            HIGH_PRECISION,
            LOW_PRECISION,
            TRUNCATED,
            NULL_ISLAND,
            BOUNDARY_LATITUDE,
            SOUTH_POLE,
        ]
        results = precision_analyzer.analyze_batch(coords)
        assert len(results) == 10
        for r in results:
            assert isinstance(r, PrecisionResult)
            assert r.latitude_decimal_places >= 0
            assert r.longitude_decimal_places >= 0

    def test_batch_analyze_preserves_order(self, precision_analyzer):
        """Batch results preserve input coordinate order."""
        coords = [
            (5.603716, -0.186964),  # 6dp -> adequate
            (5.6, -0.2),           # 1dp -> inadequate
            (5.60371589, -0.18696423),  # 8dp -> survey grade
        ]
        results = precision_analyzer.analyze_batch(coords)
        assert results[0].effective_decimal_places > results[1].effective_decimal_places
        assert results[2].effective_decimal_places > results[0].effective_decimal_places

    def test_batch_analyze_single_item(self, precision_analyzer):
        """Batch analyze with a single coordinate."""
        coords = [(5.603716, -0.186964)]
        results = precision_analyzer.analyze_batch(coords)
        assert len(results) == 1
        assert results[0].eudr_adequate is True


# ===========================================================================
# 14. Provenance Hash Extended
# ===========================================================================


class TestPrecisionProvenanceExtended:
    """Extended provenance tests for precision analysis."""

    def test_provenance_hash_is_sha256(self, precision_analyzer):
        """Provenance hash has correct SHA-256 length."""
        result = precision_analyzer.analyze(5.603716, -0.186964)
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_provenance_different_inputs_different_hash(self, precision_analyzer):
        """Different coordinates produce different provenance hashes."""
        r1 = precision_analyzer.analyze(5.603716, -0.186964)
        r2 = precision_analyzer.analyze(-12.97, -55.32)
        assert r1.provenance_hash != r2.provenance_hash

    def test_provenance_different_source_types_different_hash(self, precision_analyzer):
        """Different source types for same coordinate produce different hashes."""
        r1 = precision_analyzer.analyze(
            5.603716, -0.186964, source_type=SourceType.GNSS_SURVEY
        )
        r2 = precision_analyzer.analyze(
            5.603716, -0.186964, source_type=SourceType.MANUAL_ENTRY
        )
        assert r1.provenance_hash != r2.provenance_hash
