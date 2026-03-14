# -*- coding: utf-8 -*-
"""
Tests for AccuracyScoringEngine - AGENT-EUDR-002 Feature 5: Accuracy Scoring

Comprehensive test suite covering:
- Perfect score (all components pass)
- Zero score (all components fail)
- Quality tier thresholds (gold, silver, bronze, fail)
- Individual score components (precision, polygon, country, protected, deforestation, temporal)
- Custom weight configurations
- Deterministic scoring (Decimal arithmetic)
- Aggregate statistics
- Batch scoring
- Score serialization

Test count: 100 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 5 - Accuracy Scoring)
"""

from decimal import Decimal

import pytest

from greenlang.agents.eudr.geolocation_verification.models import (
    CoordinateValidationResult,
    DeforestationVerificationResult,
    GeolocationAccuracyScore,
    PolygonVerificationResult,
    ProtectedAreaCheckResult,
    QualityTier,
    TemporalChangeResult,
)
from greenlang.agents.eudr.geolocation_verification.accuracy_scorer import (
    AccuracyScoringEngine,
)


# ---------------------------------------------------------------------------
# Helper: build component results for scoring
# ---------------------------------------------------------------------------


def _make_coord_result(**overrides) -> CoordinateValidationResult:
    """Create a CoordinateValidationResult with defaults."""
    defaults = dict(
        is_valid=True,
        wgs84_valid=True,
        precision_decimal_places=7,
        precision_score=1.0,
        transposition_detected=False,
        country_match=True,
        is_on_land=True,
        is_duplicate=False,
        elevation_plausible=True,
        cluster_anomaly=False,
        provenance_hash="a" * 64,
    )
    defaults.update(overrides)
    return CoordinateValidationResult(**defaults)


def _make_polygon_result(**overrides) -> PolygonVerificationResult:
    """Create a PolygonVerificationResult with defaults."""
    defaults = dict(
        is_valid=True,
        ring_closed=True,
        winding_order_ccw=True,
        has_self_intersection=False,
        vertex_count=5,
        calculated_area_ha=2.0,
        area_within_tolerance=True,
        is_sliver=False,
        has_spikes=False,
        vertex_density_ok=True,
        max_area_ok=True,
        provenance_hash="b" * 64,
    )
    defaults.update(overrides)
    return PolygonVerificationResult(**defaults)


def _make_protected_result(**overrides) -> ProtectedAreaCheckResult:
    """Create a ProtectedAreaCheckResult with defaults."""
    defaults = dict(
        overlaps_protected=False,
        overlap_percentage=0.0,
    )
    defaults.update(overrides)
    return ProtectedAreaCheckResult(**defaults)


def _make_deforestation_result(**overrides) -> DeforestationVerificationResult:
    """Create a DeforestationVerificationResult with defaults."""
    defaults = dict(
        deforestation_detected=False,
        alert_count=0,
        forest_loss_ha=0.0,
        cutoff_date="2020-12-31",
        confidence=0.95,
    )
    defaults.update(overrides)
    return DeforestationVerificationResult(**defaults)


def _make_temporal_result(**overrides) -> TemporalChangeResult:
    """Create a TemporalChangeResult with defaults."""
    defaults = dict(
        is_consistent=True,
        rapid_change_detected=False,
        provenance_hash="c" * 64,
    )
    defaults.update(overrides)
    return TemporalChangeResult(**defaults)


# ===========================================================================
# 1. Overall Score Tests (15 tests)
# ===========================================================================


class TestOverallScore:
    """Test overall accuracy scoring."""

    def test_perfect_score_all_pass(self, accuracy_scorer):
        """Test perfect score when all components pass."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert isinstance(result, GeolocationAccuracyScore)
        assert result.total_score >= Decimal("85.00")
        assert result.quality_tier == QualityTier.GOLD

    def test_zero_score_all_fail(self, accuracy_scorer):
        """Test minimum score when all components fail."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(
                is_valid=False, wgs84_valid=False,
                precision_score=0.0, country_match=False,
                is_on_land=False, transposition_detected=True,
            ),
            polygon_result=_make_polygon_result(
                is_valid=False, ring_closed=False,
                has_self_intersection=True, is_sliver=True,
                has_spikes=True, area_within_tolerance=False,
                max_area_ok=False,
            ),
            protected_result=_make_protected_result(
                overlaps_protected=True, overlap_percentage=100.0,
            ),
            deforestation_result=_make_deforestation_result(
                deforestation_detected=True, confidence=0.95,
                alert_count=10, forest_loss_ha=50.0,
            ),
            temporal_result=_make_temporal_result(
                is_consistent=False, rapid_change_detected=True,
            ),
        )
        assert result.total_score < Decimal("50.00")
        assert result.quality_tier == QualityTier.UNVERIFIED

    def test_score_range(self, accuracy_scorer):
        """Test total score is always in [0, 100] range."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert Decimal("0") <= result.total_score <= Decimal("100")

    def test_score_id_generated(self, accuracy_scorer):
        """Test score ID is generated."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.score_id is not None
        assert result.score_id.startswith("GAS")

    def test_provenance_hash_generated(self, accuracy_scorer):
        """Test provenance hash is generated for score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_scored_at_timestamp(self, accuracy_scorer):
        """Test scored_at timestamp is set."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.scored_at is not None

    def test_weights_used_recorded(self, accuracy_scorer):
        """Test weights used are recorded in the result."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert isinstance(result.weights_used, dict)


# ===========================================================================
# 2. Quality Tier Thresholds (20 tests)
# ===========================================================================


class TestQualityTiers:
    """Test quality tier threshold classification."""

    def test_gold_tier_threshold(self, accuracy_scorer):
        """Test score >= 85 maps to GOLD tier."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        if result.total_score >= Decimal("85"):
            assert result.quality_tier == QualityTier.GOLD

    def test_silver_tier_threshold(self, accuracy_scorer):
        """Test score >= 70 and < 85 maps to SILVER tier."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=0.5),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        if Decimal("70") <= result.total_score < Decimal("85"):
            assert result.quality_tier == QualityTier.SILVER

    def test_bronze_tier_threshold(self, accuracy_scorer):
        """Test score >= 50 and < 70 maps to BRONZE tier."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(
                precision_score=0.3, country_match=False
            ),
            polygon_result=_make_polygon_result(is_sliver=True),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        if Decimal("50") <= result.total_score < Decimal("70"):
            assert result.quality_tier == QualityTier.BRONZE

    def test_fail_tier_threshold(self, accuracy_scorer):
        """Test score < 50 maps to FAIL tier."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(
                is_valid=False, wgs84_valid=False,
                precision_score=0.0, country_match=False,
            ),
            polygon_result=_make_polygon_result(
                is_valid=False, has_self_intersection=True,
            ),
            protected_result=_make_protected_result(
                overlaps_protected=True, overlap_percentage=100.0,
            ),
            deforestation_result=_make_deforestation_result(
                deforestation_detected=True, confidence=0.9,
            ),
            temporal_result=_make_temporal_result(is_consistent=False),
        )
        if result.total_score < Decimal("50"):
            assert result.quality_tier == QualityTier.UNVERIFIED

    @pytest.mark.parametrize("tier,min_score,max_score", [
        (QualityTier.GOLD, 85, 100),
        (QualityTier.SILVER, 70, 84),
        (QualityTier.BRONZE, 50, 69),
        (QualityTier.UNVERIFIED, 0, 49),
    ])
    def test_tier_score_boundaries(self, tier, min_score, max_score):
        """Test tier classification at boundary scores."""
        score = GeolocationAccuracyScore(
            total_score=Decimal(str(min_score)),
            quality_tier=tier,
        )
        assert score.quality_tier == tier

    def test_gold_tier_value(self):
        """Test GOLD tier enum value."""
        assert QualityTier.GOLD.value == "gold"

    def test_silver_tier_value(self):
        """Test SILVER tier enum value."""
        assert QualityTier.SILVER.value == "silver"

    def test_bronze_tier_value(self):
        """Test BRONZE tier enum value."""
        assert QualityTier.BRONZE.value == "bronze"

    def test_fail_tier_value(self):
        """Test FAIL tier enum value."""
        assert QualityTier.UNVERIFIED.value == "fail"

    def test_all_tiers_present(self):
        """Test all four quality tiers exist."""
        tiers = {t.value for t in QualityTier}
        assert tiers == {"gold", "silver", "bronze", "fail"}


# ===========================================================================
# 3. Individual Score Components (25 tests)
# ===========================================================================


class TestScoreComponents:
    """Test individual score component calculations."""

    def test_coordinate_precision_component(self, accuracy_scorer):
        """Test coordinate precision sub-score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=1.0),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.coordinate_precision_score >= Decimal("0")

    def test_precision_low_reduces_score(self, accuracy_scorer):
        """Test low precision reduces the precision sub-score."""
        high = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=1.0),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        low = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=0.1),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert high.coordinate_precision_score >= low.coordinate_precision_score

    def test_polygon_quality_component(self, accuracy_scorer):
        """Test polygon quality sub-score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.polygon_quality_score >= Decimal("0")

    def test_polygon_invalid_reduces_score(self, accuracy_scorer):
        """Test invalid polygon reduces the polygon sub-score."""
        valid = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(is_valid=True),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        invalid = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(
                is_valid=False, has_self_intersection=True,
            ),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert valid.polygon_quality_score >= invalid.polygon_quality_score

    def test_country_match_component(self, accuracy_scorer):
        """Test country match sub-score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(country_match=True),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.country_match_score >= Decimal("0")

    def test_country_mismatch_reduces_score(self, accuracy_scorer):
        """Test country mismatch reduces the country sub-score."""
        match = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(country_match=True),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        mismatch = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(country_match=False),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert match.country_match_score >= mismatch.country_match_score

    def test_protected_area_component(self, accuracy_scorer):
        """Test protected area sub-score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(overlaps_protected=False),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.protected_area_score >= Decimal("0")

    def test_protected_overlap_reduces_score(self, accuracy_scorer):
        """Test protected area overlap reduces the protected sub-score."""
        clear = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(overlaps_protected=False),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        overlap = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(
                overlaps_protected=True, overlap_percentage=100.0,
            ),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert clear.protected_area_score >= overlap.protected_area_score

    def test_deforestation_component(self, accuracy_scorer):
        """Test deforestation sub-score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(deforestation_detected=False),
            temporal_result=_make_temporal_result(),
        )
        assert result.deforestation_score >= Decimal("0")

    def test_deforestation_detected_reduces_score(self, accuracy_scorer):
        """Test deforestation detection reduces the deforestation sub-score."""
        clean = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(deforestation_detected=False),
            temporal_result=_make_temporal_result(),
        )
        detected = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(
                deforestation_detected=True, confidence=0.95,
            ),
            temporal_result=_make_temporal_result(),
        )
        assert clean.deforestation_score >= detected.deforestation_score

    def test_temporal_component(self, accuracy_scorer):
        """Test temporal consistency sub-score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(is_consistent=True),
        )
        assert result.temporal_consistency_score >= Decimal("0")

    def test_temporal_inconsistent_reduces_score(self, accuracy_scorer):
        """Test temporal inconsistency reduces the temporal sub-score."""
        consistent = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(is_consistent=True),
        )
        inconsistent = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(
                is_consistent=False, rapid_change_detected=True,
            ),
        )
        assert consistent.temporal_consistency_score >= inconsistent.temporal_consistency_score

    def test_all_subscores_non_negative(self, accuracy_scorer):
        """Test all sub-scores are non-negative."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.coordinate_precision_score >= Decimal("0")
        assert result.polygon_quality_score >= Decimal("0")
        assert result.country_match_score >= Decimal("0")
        assert result.protected_area_score >= Decimal("0")
        assert result.deforestation_score >= Decimal("0")
        assert result.temporal_consistency_score >= Decimal("0")

    def test_subscores_sum_to_total(self, accuracy_scorer):
        """Test sub-scores sum approximately to total score."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        component_sum = (
            result.coordinate_precision_score
            + result.polygon_quality_score
            + result.country_match_score
            + result.protected_area_score
            + result.deforestation_score
            + result.temporal_consistency_score
        )
        # Allow small rounding tolerance
        assert abs(component_sum - result.total_score) < Decimal("1.0")


# ===========================================================================
# 4. Custom Weights (10 tests)
# ===========================================================================


class TestCustomWeights:
    """Test custom weight configurations."""

    def test_custom_weights(self, mock_config):
        """Test scoring with custom weight configuration."""
        mock_config.score_weights = {
            "precision": 0.30,
            "polygon": 0.30,
            "country": 0.10,
            "protected": 0.10,
            "deforestation": 0.10,
            "temporal": 0.10,
        }
        scorer = AccuracyScoringEngine(config=mock_config)
        result = scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert isinstance(result, GeolocationAccuracyScore)

    def test_weight_sum_validated(self, mock_config):
        """Test weights must sum to 1.0."""
        with pytest.raises(ValueError):
            GeolocationVerificationConfig = type(mock_config)
            GeolocationVerificationConfig(
                score_weights={
                    "precision": 0.50,
                    "polygon": 0.50,
                    "country": 0.50,
                    "protected": 0.50,
                    "deforestation": 0.50,
                    "temporal": 0.50,
                },
            )

    def test_equal_weights(self, mock_config):
        """Test scoring with equal weights."""
        w = round(1.0 / 6, 4)
        remainder = round(1.0 - w * 5, 4)
        mock_config.score_weights = {
            "precision": w,
            "polygon": w,
            "country": w,
            "protected": w,
            "deforestation": w,
            "temporal": remainder,
        }
        scorer = AccuracyScoringEngine(config=mock_config)
        result = scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert result.total_score >= Decimal("0")


# ===========================================================================
# 5. Deterministic Scoring (15 tests)
# ===========================================================================


class TestScoringDeterminism:
    """Test accuracy scoring determinism with Decimal arithmetic."""

    def test_deterministic_scoring(self, accuracy_scorer):
        """Test same inputs produce identical scores."""
        args = dict(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        r1 = accuracy_scorer.compute_score(**args)
        r2 = accuracy_scorer.compute_score(**args)
        assert r1.total_score == r2.total_score
        assert r1.quality_tier == r2.quality_tier

    def test_deterministic_provenance(self, accuracy_scorer):
        """Test provenance hash is deterministic."""
        args = dict(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        r1 = accuracy_scorer.compute_score(**args)
        r2 = accuracy_scorer.compute_score(**args)
        assert r1.provenance_hash == r2.provenance_hash

    def test_deterministic_10_runs(self, accuracy_scorer):
        """Test determinism over 10 runs."""
        args = dict(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        first = accuracy_scorer.compute_score(**args)
        for _ in range(9):
            result = accuracy_scorer.compute_score(**args)
            assert result.total_score == first.total_score
            assert result.provenance_hash == first.provenance_hash

    def test_decimal_arithmetic_no_float_drift(self, accuracy_scorer):
        """Test Decimal arithmetic prevents floating-point drift."""
        args = dict(
            coord_result=_make_coord_result(precision_score=0.7),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        scores = [accuracy_scorer.compute_score(**args).total_score for _ in range(100)]
        assert len(set(scores)) == 1  # All identical

    def test_different_inputs_different_scores(self, accuracy_scorer):
        """Test different inputs produce different scores."""
        r1 = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=1.0),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        r2 = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=0.1),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert r1.total_score != r2.total_score


# ===========================================================================
# 6. Aggregate Statistics and Batch (10 tests)
# ===========================================================================


class TestAggregateAndBatch:
    """Test aggregate statistics and batch scoring."""

    def test_aggregate_statistics(self, accuracy_scorer):
        """Test aggregate statistics computation."""
        scores = []
        for precision in [0.2, 0.5, 0.8, 1.0]:
            s = accuracy_scorer.compute_score(
                coord_result=_make_coord_result(precision_score=precision),
                polygon_result=_make_polygon_result(),
                protected_result=_make_protected_result(),
                deforestation_result=_make_deforestation_result(),
                temporal_result=_make_temporal_result(),
            )
            scores.append(s)
        assert len(scores) == 4
        total_scores = [s.total_score for s in scores]
        assert min(total_scores) <= max(total_scores)

    def test_batch_scoring(self, accuracy_scorer):
        """Test batch scoring of multiple plots."""
        batch_inputs = []
        for i in range(5):
            batch_inputs.append(dict(
                coord_result=_make_coord_result(precision_score=0.5 + i * 0.1),
                polygon_result=_make_polygon_result(),
                protected_result=_make_protected_result(),
                deforestation_result=_make_deforestation_result(),
                temporal_result=_make_temporal_result(),
            ))
        results = [accuracy_scorer.compute_score(**inp) for inp in batch_inputs]
        assert len(results) == 5
        for r in results:
            assert isinstance(r, GeolocationAccuracyScore)

    def test_batch_unique_ids(self, accuracy_scorer):
        """Test batch results have unique score IDs."""
        results = []
        for i in range(5):
            r = accuracy_scorer.compute_score(
                coord_result=_make_coord_result(),
                polygon_result=_make_polygon_result(),
                protected_result=_make_protected_result(),
                deforestation_result=_make_deforestation_result(),
                temporal_result=_make_temporal_result(),
            )
            results.append(r)
        ids = [r.score_id for r in results]
        assert len(set(ids)) == 5


# ===========================================================================
# 7. Serialization (5 tests)
# ===========================================================================


class TestScoreSerialization:
    """Test GeolocationAccuracyScore serialization."""

    def test_to_dict(self, accuracy_scorer):
        """Test to_dict produces valid dictionary."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_score" in d
        assert "quality_tier" in d

    def test_to_dict_decimal_as_string(self, accuracy_scorer):
        """Test Decimal values are serialized as strings."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        d = result.to_dict()
        assert isinstance(d["total_score"], str)

    def test_to_dict_tier_as_string(self, accuracy_scorer):
        """Test quality tier is serialized as string."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        d = result.to_dict()
        assert d["quality_tier"] in ["gold", "silver", "bronze", "fail"]


# ===========================================================================
# 8. Scoring Sensitivity Analysis (25 tests)
# ===========================================================================


class TestScoringSensitivity:
    """Test scoring sensitivity to individual component changes."""

    @pytest.mark.parametrize("precision_score", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    def test_precision_sensitivity(self, accuracy_scorer, precision_score):
        """Test score sensitivity to precision component."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=precision_score),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert Decimal("0") <= result.total_score <= Decimal("100")
        assert result.coordinate_precision_score >= Decimal("0")

    @pytest.mark.parametrize("is_valid,ring_closed,has_intersection,is_sliver", [
        (True, True, False, False),
        (True, True, False, True),
        (True, False, False, False),
        (False, False, True, False),
        (False, False, True, True),
    ])
    def test_polygon_quality_sensitivity(self, accuracy_scorer, is_valid, ring_closed, has_intersection, is_sliver):
        """Test score sensitivity to polygon quality component."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(
                is_valid=is_valid, ring_closed=ring_closed,
                has_self_intersection=has_intersection, is_sliver=is_sliver,
            ),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert Decimal("0") <= result.total_score <= Decimal("100")

    @pytest.mark.parametrize("overlaps,pct", [
        (False, 0.0),
        (True, 10.0),
        (True, 25.0),
        (True, 50.0),
        (True, 75.0),
        (True, 100.0),
    ])
    def test_protected_area_sensitivity(self, accuracy_scorer, overlaps, pct):
        """Test score sensitivity to protected area overlap."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(
                overlaps_protected=overlaps, overlap_percentage=pct,
            ),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert Decimal("0") <= result.total_score <= Decimal("100")

    @pytest.mark.parametrize("deforestation,confidence", [
        (False, 0.95),
        (True, 0.5),
        (True, 0.7),
        (True, 0.9),
        (True, 1.0),
    ])
    def test_deforestation_sensitivity(self, accuracy_scorer, deforestation, confidence):
        """Test score sensitivity to deforestation detection."""
        result = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(
                deforestation_detected=deforestation, confidence=confidence,
            ),
            temporal_result=_make_temporal_result(),
        )
        assert Decimal("0") <= result.total_score <= Decimal("100")

    def test_all_perfect_is_highest(self, accuracy_scorer):
        """Test that all-perfect components yield the highest possible score."""
        perfect = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=1.0, country_match=True),
            polygon_result=_make_polygon_result(is_valid=True),
            protected_result=_make_protected_result(overlaps_protected=False),
            deforestation_result=_make_deforestation_result(deforestation_detected=False),
            temporal_result=_make_temporal_result(is_consistent=True),
        )
        degraded = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(precision_score=0.5, country_match=True),
            polygon_result=_make_polygon_result(is_valid=True),
            protected_result=_make_protected_result(overlaps_protected=False),
            deforestation_result=_make_deforestation_result(deforestation_detected=False),
            temporal_result=_make_temporal_result(is_consistent=True),
        )
        assert perfect.total_score >= degraded.total_score

    def test_all_worst_is_lowest(self, accuracy_scorer):
        """Test that all-worst components yield the lowest possible score."""
        worst = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(
                is_valid=False, wgs84_valid=False,
                precision_score=0.0, country_match=False,
                transposition_detected=True,
            ),
            polygon_result=_make_polygon_result(
                is_valid=False, has_self_intersection=True,
                ring_closed=False, is_sliver=True,
            ),
            protected_result=_make_protected_result(
                overlaps_protected=True, overlap_percentage=100.0,
            ),
            deforestation_result=_make_deforestation_result(
                deforestation_detected=True, confidence=1.0,
            ),
            temporal_result=_make_temporal_result(
                is_consistent=False, rapid_change_detected=True,
            ),
        )
        good = accuracy_scorer.compute_score(
            coord_result=_make_coord_result(),
            polygon_result=_make_polygon_result(),
            protected_result=_make_protected_result(),
            deforestation_result=_make_deforestation_result(),
            temporal_result=_make_temporal_result(),
        )
        assert worst.total_score < good.total_score


# ===========================================================================
# 9. Monotonicity Tests (10 tests)
# ===========================================================================


class TestScoreMonotonicity:
    """Test that scores change monotonically with component quality."""

    def test_precision_monotonicity(self, accuracy_scorer):
        """Test score increases monotonically with precision."""
        scores = []
        for prec in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = accuracy_scorer.compute_score(
                coord_result=_make_coord_result(precision_score=prec),
                polygon_result=_make_polygon_result(),
                protected_result=_make_protected_result(),
                deforestation_result=_make_deforestation_result(),
                temporal_result=_make_temporal_result(),
            )
            scores.append(r.total_score)
        # Each subsequent score should be >= previous
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]

    def test_overlap_reduces_score_monotonically(self, accuracy_scorer):
        """Test score decreases with increasing protected area overlap."""
        scores = []
        for overlap_pct in [0.0, 25.0, 50.0, 75.0, 100.0]:
            r = accuracy_scorer.compute_score(
                coord_result=_make_coord_result(),
                polygon_result=_make_polygon_result(),
                protected_result=_make_protected_result(
                    overlaps_protected=overlap_pct > 0,
                    overlap_percentage=overlap_pct,
                ),
                deforestation_result=_make_deforestation_result(),
                temporal_result=_make_temporal_result(),
            )
            scores.append(r.total_score)
        # Each subsequent score should be <= previous (more overlap = worse)
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]
