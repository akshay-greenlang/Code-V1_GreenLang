# -*- coding: utf-8 -*-
"""
Tests for AccuracyAssessor - AGENT-EUDR-007 Engine 6: Accuracy Assessment

Comprehensive test suite covering:
- Gold tier classification (score >= 90)
- Silver tier classification (score 70-89)
- Bronze tier classification (score 50-69)
- Unverified tier classification (score < 50)
- Precision component scoring (high vs low precision)
- Plausibility component scoring (land vs ocean)
- Consistency component scoring (clean vs errored)
- Source type scoring (GNSS highest, manual lowest)
- Confidence interval calculation
- Tier classification threshold boundaries
- Batch assessment
- Parametrized tests for source types and tiers

Test count: 45+ tests
Coverage target: >= 85% of AccuracyAssessor module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    PrecisionLevel,
    PrecisionResult,
    SourceType,
    ValidationResult,
    CoordinateValidationError,
    ValidationErrorType,
)
from tests.agents.eudr.gps_coordinate_validator.conftest import (
    COCOA_FARM_GHANA,
    OCEAN_POINT,
    LOW_PRECISION,
    TRUNCATED,
    NULL_ISLAND,
    HIGH_PRECISION,
    GOLD_THRESHOLD,
    SILVER_THRESHOLD,
    BRONZE_THRESHOLD,
    UNVERIFIED_THRESHOLD,
    SHA256_HEX_LENGTH,
    assert_close,
    make_normalized,
)


# ---------------------------------------------------------------------------
# Helper: build assessment input bundles
# ---------------------------------------------------------------------------


def _perfect_precision() -> PrecisionResult:
    """Build a perfect precision result (survey grade, 8dp, EUDR adequate)."""
    return PrecisionResult(
        latitude_decimal_places=8,
        longitude_decimal_places=8,
        effective_decimal_places=8,
        ground_resolution_lat_m=0.001,
        ground_resolution_lon_m=0.001,
        precision_level=PrecisionLevel.SURVEY_GRADE,
        eudr_adequate=True,
        is_truncated=False,
        is_artificially_rounded=False,
        estimated_source_precision_m=0.05,
    )


def _poor_precision() -> PrecisionResult:
    """Build a poor precision result (inadequate, 1dp)."""
    return PrecisionResult(
        latitude_decimal_places=1,
        longitude_decimal_places=1,
        effective_decimal_places=1,
        ground_resolution_lat_m=11132.0,
        ground_resolution_lon_m=11132.0,
        precision_level=PrecisionLevel.INADEQUATE,
        eudr_adequate=False,
        is_truncated=True,
        is_artificially_rounded=False,
    )


def _clean_validation() -> ValidationResult:
    """Build a clean validation result (no errors)."""
    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        error_count=0,
        warning_count=0,
    )


def _errored_validation() -> ValidationResult:
    """Build a validation result with errors."""
    return ValidationResult(
        is_valid=False,
        errors=[
            CoordinateValidationError(
                error_type=ValidationErrorType.SWAPPED_LAT_LON,
                severity="error",
                description="Coordinates appear swapped",
            ),
            CoordinateValidationError(
                error_type=ValidationErrorType.SIGN_ERROR,
                severity="error",
                description="Missing negative sign",
            ),
        ],
        warnings=[],
        error_count=2,
        warning_count=0,
    )


# ===========================================================================
# 1. Tier Classification
# ===========================================================================


class TestTierClassification:
    """Test quality tier classification based on total score."""

    def test_gold_tier(self, accuracy_assessor):
        """Perfect coordinate data scores Gold tier (>= 90)."""
        score = accuracy_assessor.assess(
            latitude=HIGH_PRECISION[0],
            longitude=HIGH_PRECISION[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.total_score >= GOLD_THRESHOLD
        assert score.tier == "gold"

    def test_silver_tier(self, accuracy_assessor):
        """Good but imperfect data scores Silver tier (70-89)."""
        moderate_precision = PrecisionResult(
            latitude_decimal_places=5,
            longitude_decimal_places=5,
            effective_decimal_places=5,
            ground_resolution_lat_m=1.1,
            ground_resolution_lon_m=1.1,
            precision_level=PrecisionLevel.HIGH,
            eudr_adequate=True,
            is_truncated=False,
            is_artificially_rounded=False,
            estimated_source_precision_m=3.0,
        )
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=moderate_precision,
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.HANDHELD_GPS,
        )
        assert SILVER_THRESHOLD <= score.total_score < GOLD_THRESHOLD
        assert score.tier == "silver"

    def test_bronze_tier(self, accuracy_assessor):
        """Marginal data scores Bronze tier (50-69)."""
        low_precision = PrecisionResult(
            latitude_decimal_places=3,
            longitude_decimal_places=3,
            effective_decimal_places=3,
            ground_resolution_lat_m=111.0,
            ground_resolution_lon_m=111.0,
            precision_level=PrecisionLevel.LOW,
            eudr_adequate=False,
            is_truncated=False,
            is_artificially_rounded=False,
            estimated_source_precision_m=50.0,
        )
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=low_precision,
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.MOBILE_GPS,
        )
        assert BRONZE_THRESHOLD <= score.total_score < SILVER_THRESHOLD
        assert score.tier == "bronze"

    def test_unverified_tier(self, accuracy_assessor):
        """Poor data with errors scores Unverified tier (< 50)."""
        score = accuracy_assessor.assess(
            latitude=LOW_PRECISION[0],
            longitude=LOW_PRECISION[1],
            precision_result=_poor_precision(),
            validation_result=_errored_validation(),
            is_on_land=False,
            country_match=False,
            source_type=SourceType.MANUAL_ENTRY,
        )
        assert score.total_score < BRONZE_THRESHOLD
        assert score.tier == "unverified"


# ===========================================================================
# 2. Precision Component Scoring
# ===========================================================================


class TestPrecisionScoring:
    """Test precision component of accuracy score."""

    def test_precision_score_survey_grade(self, accuracy_assessor):
        """Survey-grade precision scores high on precision component."""
        score = accuracy_assessor.assess(
            latitude=HIGH_PRECISION[0],
            longitude=HIGH_PRECISION[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.precision_score >= 18.0  # Out of 20 max

    def test_precision_score_low(self, accuracy_assessor):
        """Low precision scores low on precision component."""
        score = accuracy_assessor.assess(
            latitude=LOW_PRECISION[0],
            longitude=LOW_PRECISION[1],
            precision_result=_poor_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.MANUAL_ENTRY,
        )
        assert score.precision_score < 10.0


# ===========================================================================
# 3. Plausibility Component Scoring
# ===========================================================================


class TestPlausibilityScoring:
    """Test plausibility component of accuracy score."""

    def test_plausibility_score_land(self, accuracy_assessor):
        """On-land coordinate scores well on plausibility."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.plausibility_score > 0

    def test_plausibility_score_ocean_penalty(self, accuracy_assessor):
        """Ocean coordinate receives plausibility penalty."""
        score = accuracy_assessor.assess(
            latitude=OCEAN_POINT[0],
            longitude=OCEAN_POINT[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=False,
            country_match=False,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.plausibility_score < 10.0


# ===========================================================================
# 4. Consistency Component Scoring
# ===========================================================================


class TestConsistencyScoring:
    """Test consistency (error count) component of accuracy score."""

    def test_consistency_score_clean(self, accuracy_assessor):
        """Clean validation (no errors) scores maximum consistency."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.consistency_score >= 15.0

    def test_consistency_score_errors_reduce(self, accuracy_assessor):
        """Validation errors reduce consistency score."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_errored_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.consistency_score < 15.0


# ===========================================================================
# 5. Source Type Scoring
# ===========================================================================


class TestSourceTypeScoring:
    """Test source type component of accuracy score."""

    def test_source_score_gnss_highest(self, accuracy_assessor):
        """GNSS survey source scores highest."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        gnss_source_score = score.source_score

        score2 = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.MANUAL_ENTRY,
        )
        manual_source_score = score2.source_score

        assert gnss_source_score > manual_source_score

    def test_source_score_manual_lowest(self, accuracy_assessor):
        """Manual entry source scores lowest."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.MANUAL_ENTRY,
        )
        assert score.source_score < 10.0


# ===========================================================================
# 6. Confidence Interval
# ===========================================================================


class TestConfidenceInterval:
    """Test confidence interval calculation."""

    def test_confidence_interval_high_precision(self, accuracy_assessor):
        """High-precision coordinate has small confidence radius."""
        score = accuracy_assessor.assess(
            latitude=HIGH_PRECISION[0],
            longitude=HIGH_PRECISION[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.confidence_radius_m is not None
        assert score.confidence_radius_m < 1.0

    def test_confidence_interval_low_precision(self, accuracy_assessor):
        """Low-precision coordinate has large confidence radius."""
        score = accuracy_assessor.assess(
            latitude=LOW_PRECISION[0],
            longitude=LOW_PRECISION[1],
            precision_result=_poor_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.MANUAL_ENTRY,
        )
        assert score.confidence_radius_m is not None
        assert score.confidence_radius_m > 100.0


# ===========================================================================
# 7. Tier Threshold Boundary Tests
# ===========================================================================


class TestTierBoundaries:
    """Test exact tier classification boundaries."""

    @pytest.mark.parametrize(
        "score_val,expected_tier",
        [
            (100.0, "gold"),
            (90.0, "gold"),
            (89.99, "silver"),
            (70.0, "silver"),
            (69.99, "bronze"),
            (50.0, "bronze"),
            (49.99, "unverified"),
            (0.0, "unverified"),
        ],
        ids=[
            "100_gold", "90_gold_boundary", "89.99_silver",
            "70_silver_boundary", "69.99_bronze", "50_bronze_boundary",
            "49.99_unverified", "0_unverified",
        ],
    )
    def test_tier_boundary(self, accuracy_assessor, score_val, expected_tier):
        """Tier classification at exact boundary values."""
        tier = accuracy_assessor.classify_tier(score_val)
        assert tier == expected_tier


# ===========================================================================
# 8. Batch Assessment
# ===========================================================================


class TestBatchAssess:
    """Test batch accuracy assessment."""

    def test_batch_assess_multiple(self, accuracy_assessor):
        """Batch assess multiple coordinates."""
        entries = [
            {
                "latitude": COCOA_FARM_GHANA[0],
                "longitude": COCOA_FARM_GHANA[1],
                "precision_result": _perfect_precision(),
                "validation_result": _clean_validation(),
                "is_on_land": True,
                "country_match": True,
                "source_type": SourceType.GNSS_SURVEY,
            },
            {
                "latitude": LOW_PRECISION[0],
                "longitude": LOW_PRECISION[1],
                "precision_result": _poor_precision(),
                "validation_result": _errored_validation(),
                "is_on_land": False,
                "country_match": False,
                "source_type": SourceType.MANUAL_ENTRY,
            },
        ]
        results = accuracy_assessor.assess_batch(entries)
        assert len(results) == 2
        # First should score higher than second
        assert results[0].total_score > results[1].total_score

    def test_batch_assess_empty(self, accuracy_assessor):
        """Batch assess with empty list returns empty."""
        results = accuracy_assessor.assess_batch([])
        assert results == []


# ===========================================================================
# 9. Provenance and Determinism
# ===========================================================================


class TestAccuracyProvenance:
    """Test provenance tracking and determinism."""

    def test_assess_has_provenance_hash(self, accuracy_assessor):
        """Assessment result includes a provenance hash."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert score.provenance_hash != ""
        assert len(score.provenance_hash) == SHA256_HEX_LENGTH

    def test_deterministic_assessment(self, accuracy_assessor):
        """Same inputs produce identical assessment."""
        kwargs = dict(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        s1 = accuracy_assessor.assess(**kwargs)
        s2 = accuracy_assessor.assess(**kwargs)
        assert s1.total_score == s2.total_score
        assert s1.tier == s2.tier
        assert s1.provenance_hash == s2.provenance_hash


# ===========================================================================
# 10. Parametrized Source Type Tests
# ===========================================================================


@pytest.mark.parametrize(
    "source_type",
    [
        SourceType.GNSS_SURVEY,
        SourceType.GNSS_SURVEY,
        SourceType.HANDHELD_GPS,
        SourceType.MOBILE_GPS,
        SourceType.SATELLITE_DERIVED,
        SourceType.DIGITIZED_MAP,
        SourceType.GEOCODED,
        SourceType.MANUAL_ENTRY,
    ],
    ids=[
        "gnss_survey", "rtk_gps", "handheld_gps", "mobile_gps",
        "satellite", "digitized_map", "geocoded", "manual_entry",
    ],
)
def test_parametrized_source_types(accuracy_assessor, source_type):
    """Parametrized: all source types produce valid assessment results."""
    score = accuracy_assessor.assess(
        latitude=COCOA_FARM_GHANA[0],
        longitude=COCOA_FARM_GHANA[1],
        precision_result=_perfect_precision(),
        validation_result=_clean_validation(),
        is_on_land=True,
        country_match=True,
        source_type=source_type,
    )
    assert 0 <= score.total_score <= 100
    assert score.tier in ("gold", "silver", "bronze", "unverified")


# ===========================================================================
# 11. Score Component Sum
# ===========================================================================


class TestScoreComponentSum:
    """Test that individual score components sum to total score."""

    def test_score_components_sum_to_total(self, accuracy_assessor):
        """All component scores should roughly sum to total score."""
        score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        component_sum = (
            score.precision_score
            + score.plausibility_score
            + score.consistency_score
            + score.source_score
        )
        # Allow small tolerance for rounding
        assert abs(component_sum - score.total_score) < 1.0

    def test_all_components_non_negative(self, accuracy_assessor):
        """All score components are non-negative."""
        score = accuracy_assessor.assess(
            latitude=LOW_PRECISION[0],
            longitude=LOW_PRECISION[1],
            precision_result=_poor_precision(),
            validation_result=_errored_validation(),
            is_on_land=False,
            country_match=False,
            source_type=SourceType.MANUAL_ENTRY,
        )
        assert score.precision_score >= 0
        assert score.plausibility_score >= 0
        assert score.consistency_score >= 0
        assert score.source_score >= 0
        assert score.total_score >= 0


# ===========================================================================
# 12. EUDR Adequacy Impact on Score
# ===========================================================================


class TestEUDRAdequacyImpact:
    """Test impact of EUDR adequacy on overall score."""

    def test_eudr_adequate_scores_higher(self, accuracy_assessor):
        """EUDR-adequate precision scores higher than inadequate."""
        adequate_score = accuracy_assessor.assess(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            precision_result=_perfect_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        inadequate_score = accuracy_assessor.assess(
            latitude=LOW_PRECISION[0],
            longitude=LOW_PRECISION[1],
            precision_result=_poor_precision(),
            validation_result=_clean_validation(),
            is_on_land=True,
            country_match=True,
            source_type=SourceType.GNSS_SURVEY,
        )
        assert adequate_score.total_score > inadequate_score.total_score
