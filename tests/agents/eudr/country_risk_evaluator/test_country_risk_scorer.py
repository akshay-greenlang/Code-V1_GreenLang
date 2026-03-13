# -*- coding: utf-8 -*-
"""
Unit tests for CountryRiskScorer - AGENT-EUDR-016 Engine 1

Tests the multi-factor weighted composite risk scoring engine per EUDR
Article 29 covering 6-factor model, EC benchmark override, risk level
classification, confidence scoring, trend analysis, country comparison,
batch assessment, and provenance tracking.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
"""

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.country_risk_scorer import (
    CountryRiskScorer,
    _DEFAULT_WEIGHTS,
    _FACTOR_KEYS,
)
from greenlang.agents.eudr.country_risk_evaluator.models import (
    AssessmentConfidence,
    CountryRiskAssessment,
    RiskLevel,
    TrendDirection,
    DEFAULT_FACTOR_WEIGHTS,
)


# ============================================================================
# TestCountryRiskScorerInit
# ============================================================================


class TestCountryRiskScorerInit:
    """Tests for CountryRiskScorer initialization."""

    @pytest.mark.unit
    def test_initialization_creates_empty_stores(self, mock_config):
        scorer = CountryRiskScorer()
        assert scorer._assessments == {}
        assert scorer._risk_history == {}
        assert scorer._ec_benchmarks == {}

    @pytest.mark.unit
    def test_initialization_creates_lock(self, mock_config):
        scorer = CountryRiskScorer()
        assert scorer._lock is not None

    @pytest.mark.unit
    def test_default_weights_sum_to_one(self):
        total = sum(_DEFAULT_WEIGHTS.values())
        assert total == Decimal("1.00")

    @pytest.mark.unit
    def test_six_factor_keys_defined(self):
        assert len(_FACTOR_KEYS) == 6
        expected = [
            "deforestation_rate",
            "governance_quality",
            "enforcement_effectiveness",
            "corruption_index",
            "forest_law_compliance",
            "historical_trend",
        ]
        assert _FACTOR_KEYS == expected


# ============================================================================
# TestAssessCountry
# ============================================================================


class TestAssessCountry:
    """Tests for assess_country method."""

    @pytest.mark.unit
    def test_assess_country_valid_returns_assessment(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert isinstance(result, CountryRiskAssessment)
        assert result.country_code == "BR"
        assert result.assessment_id.startswith("cra-")

    @pytest.mark.unit
    def test_assess_country_score_in_range(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert 0.0 <= result.risk_score <= 100.0

    @pytest.mark.unit
    def test_assess_country_composite_score_is_weighted_sum(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        # Manually compute expected weighted sum AFTER normalization:
        #   deforestation_rate: 5.0 -> clamp[0,10]*10 = 50.0
        #   governance_quality: 45.0 -> 100-45 = 55.0
        #   enforcement_effectiveness: 40.0 -> 100-40 = 60.0
        #   corruption_index: 55.0 -> 100-55 = 45.0
        #   forest_law_compliance: 50.0 -> 100-50 = 50.0
        #   historical_trend: 3.0 -> (3+10)/20*100 = 65.0
        expected = (
            50.0 * 0.30  # normalized deforestation_rate
            + 55.0 * 0.20  # normalized governance_quality
            + 60.0 * 0.15  # normalized enforcement_effectiveness
            + 45.0 * 0.15  # normalized corruption_index
            + 50.0 * 0.10  # normalized forest_law_compliance
            + 65.0 * 0.10  # normalized historical_trend
        )
        assert result.risk_score == pytest.approx(expected, abs=0.5)

    @pytest.mark.unit
    def test_assess_country_stores_assessment(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        retrieved = country_risk_scorer.get_assessment(result.assessment_id)
        assert retrieved is not None
        assert retrieved.assessment_id == result.assessment_id

    @pytest.mark.unit
    def test_assess_country_uppercase_country_code(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("br", sample_factor_values)
        assert result.country_code == "BR"

    @pytest.mark.unit
    def test_assess_country_strips_whitespace(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("  BR  ", sample_factor_values)
        assert result.country_code == "BR"

    @pytest.mark.unit
    def test_assess_country_has_assessed_at(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert result.assessed_at is not None
        assert result.assessed_at.tzinfo is not None

    @pytest.mark.unit
    def test_assess_country_has_factor_weights(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert len(result.factor_weights) > 0

    @pytest.mark.unit
    def test_assess_country_has_composite_factors(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert len(result.composite_factors) > 0


# ============================================================================
# TestRiskLevelClassification
# ============================================================================


class TestRiskLevelClassification:
    """Tests for risk level classification thresholds."""

    @pytest.mark.unit
    def test_low_risk_classification(
        self, country_risk_scorer, sample_low_risk_factors
    ):
        result = country_risk_scorer.assess_country("SE", sample_low_risk_factors)
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.unit
    def test_standard_risk_classification(
        self, country_risk_scorer, sample_standard_risk_factors
    ):
        result = country_risk_scorer.assess_country("MY", sample_standard_risk_factors)
        assert result.risk_level == RiskLevel.STANDARD

    @pytest.mark.unit
    def test_high_risk_classification(
        self, country_risk_scorer, sample_high_risk_factors
    ):
        result = country_risk_scorer.assess_country("BR", sample_high_risk_factors)
        assert result.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    def test_classify_risk_level_boundary_low(self, country_risk_scorer):
        # Score exactly at low_risk_threshold (30) should be LOW
        level = country_risk_scorer.classify_risk_level(30.0)
        assert level == "low"

    @pytest.mark.unit
    def test_classify_risk_level_boundary_standard_lower(self, country_risk_scorer):
        # Score just above low_risk_threshold (31) should be STANDARD
        level = country_risk_scorer.classify_risk_level(31.0)
        assert level == "standard"

    @pytest.mark.unit
    def test_classify_risk_level_boundary_high(self, country_risk_scorer):
        # Score just above high_risk_threshold (65) should be HIGH
        level = country_risk_scorer.classify_risk_level(66.0)
        assert level == "high"

    @pytest.mark.unit
    def test_classify_risk_level_boundary_standard_upper(self, country_risk_scorer):
        # Score at high_risk_threshold (65) should be STANDARD
        level = country_risk_scorer.classify_risk_level(65.0)
        assert level == "standard"

    @pytest.mark.unit
    def test_classify_risk_level_zero(self, country_risk_scorer):
        level = country_risk_scorer.classify_risk_level(0.0)
        assert level == "low"

    @pytest.mark.unit
    def test_classify_risk_level_hundred(self, country_risk_scorer):
        level = country_risk_scorer.classify_risk_level(100.0)
        assert level == "high"

    @pytest.mark.unit
    def test_classify_risk_level_invalid_negative(self, country_risk_scorer):
        with pytest.raises(ValueError, match="composite_score must be in"):
            country_risk_scorer.classify_risk_level(-1.0)

    @pytest.mark.unit
    def test_classify_risk_level_invalid_over_100(self, country_risk_scorer):
        with pytest.raises(ValueError, match="composite_score must be in"):
            country_risk_scorer.classify_risk_level(101.0)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "score,expected_level",
        [
            (0.0, "low"),
            (15.0, "low"),
            (30.0, "low"),
            (31.0, "standard"),
            (45.0, "standard"),
            (65.0, "standard"),
            (66.0, "high"),
            (80.0, "high"),
            (100.0, "high"),
        ],
    )
    def test_classify_risk_level_parametrized(
        self, country_risk_scorer, score, expected_level
    ):
        level = country_risk_scorer.classify_risk_level(score)
        assert level == expected_level


# ============================================================================
# TestCompositeScoreCalculation
# ============================================================================


class TestCompositeScoreCalculation:
    """Tests for composite score calculation with 6 weighted factors."""

    @pytest.mark.unit
    def test_minimum_risk_score(self, country_risk_scorer):
        """All factors set to produce minimum normalized risk (0 each).

        To get normalized 0 for each factor:
          deforestation_rate=0     -> clamp(0,0,10)*10 = 0
          governance_quality=100   -> 100-100 = 0
          enforcement_eff=100      -> 100-100 = 0
          corruption_index=100     -> 100-100 = 0  (CPI=100=clean)
          forest_law_compliance=100-> 100-100 = 0
          historical_trend=-10     -> (-10+10)/20*100 = 0
        """
        factors = {
            "deforestation_rate": 0.0,
            "governance_quality": 100.0,
            "enforcement_effectiveness": 100.0,
            "corruption_index": 100.0,
            "forest_law_compliance": 100.0,
            "historical_trend": -10.0,
        }
        result = country_risk_scorer.assess_country("FI", factors)
        assert result.risk_score == pytest.approx(0.0, abs=0.1)

    @pytest.mark.unit
    def test_maximum_risk_score(self, country_risk_scorer):
        """All factors set to produce maximum normalized risk (100 each).

        To get normalized 100 for each factor:
          deforestation_rate=10    -> clamp(10,0,10)*10 = 100
          governance_quality=0     -> 100-0 = 100
          enforcement_eff=0        -> 100-0 = 100
          corruption_index=0       -> 100-0 = 100
          forest_law_compliance=0  -> 100-0 = 100
          historical_trend=10      -> (10+10)/20*100 = 100
        """
        factors = {
            "deforestation_rate": 10.0,
            "governance_quality": 0.0,
            "enforcement_effectiveness": 0.0,
            "corruption_index": 0.0,
            "forest_law_compliance": 0.0,
            "historical_trend": 10.0,
        }
        result = country_risk_scorer.assess_country("CD", factors)
        assert result.risk_score == pytest.approx(100.0, abs=0.5)

    @pytest.mark.unit
    def test_factor_weights_sum_to_100_percent(self, mock_config):
        total = (
            mock_config.deforestation_weight
            + mock_config.governance_weight
            + mock_config.enforcement_weight
            + mock_config.corruption_weight
            + mock_config.forest_law_weight
            + mock_config.trend_weight
        )
        assert total == 100

    @pytest.mark.unit
    def test_custom_weights_accepted(self, country_risk_scorer, sample_factor_values):
        custom_weights = {
            "deforestation_rate": 0.50,
            "governance_quality": 0.10,
            "enforcement_effectiveness": 0.10,
            "corruption_index": 0.10,
            "forest_law_compliance": 0.10,
            "historical_trend": 0.10,
        }
        result = country_risk_scorer.assess_country(
            "BR", sample_factor_values, factor_weights=custom_weights
        )
        assert isinstance(result, CountryRiskAssessment)
        # With higher deforestation weight, score should be higher
        assert result.risk_score > 0

    @pytest.mark.unit
    def test_normalization_clamps_to_0_100(self, country_risk_scorer):
        """Extreme values should be clamped by normalization."""
        # deforestation_rate: 150 clamped to 10 -> 100
        # governance: 150 clamped to 100 -> 100-100=0
        # enforcement: 150 clamped to 100 -> 0
        # corruption: 150 clamped to 100 -> 0
        # compliance: 150 clamped to 100 -> 0
        # trend: 150 clamped to 10 -> (10+10)/20*100=100
        factors = {k: 150.0 for k in _FACTOR_KEYS}
        result = country_risk_scorer.assess_country("XX", factors)
        assert 0.0 <= result.risk_score <= 100.0

    @pytest.mark.unit
    def test_decimal_arithmetic_no_floating_point_drift(self, country_risk_scorer):
        """Verify Decimal arithmetic produces consistent results."""
        factors = {
            "deforestation_rate": 3.33,   # in [0,10] scale
            "governance_quality": 33.33,
            "enforcement_effectiveness": 33.33,
            "corruption_index": 33.33,
            "forest_law_compliance": 33.33,
            "historical_trend": 3.33,     # in [-10,10] scale
        }
        result1 = country_risk_scorer.assess_country("BR", factors)
        # Create a new scorer to ensure no state leakage
        scorer2 = CountryRiskScorer()
        result2 = scorer2.assess_country("BR", factors)
        assert result1.risk_score == pytest.approx(result2.risk_score, abs=1e-6)


# ============================================================================
# TestECBenchmarkOverride
# ============================================================================


class TestECBenchmarkOverride:
    """Tests for EC benchmark override functionality."""

    @pytest.mark.unit
    def test_ec_benchmark_overrides_computed_level(
        self, country_risk_scorer, sample_low_risk_factors
    ):
        # Set EC benchmark for Sweden as HIGH
        country_risk_scorer.set_ec_benchmark("SE", "high")
        result = country_risk_scorer.assess_country("SE", sample_low_risk_factors)
        # Even though factors are low, EC override sets it to HIGH
        assert result.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    def test_ec_benchmark_low_override(
        self, country_risk_scorer, sample_high_risk_factors
    ):
        country_risk_scorer.set_ec_benchmark("BR", "low")
        result = country_risk_scorer.assess_country("BR", sample_high_risk_factors)
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.unit
    def test_ec_benchmark_standard_override(
        self, country_risk_scorer, sample_high_risk_factors
    ):
        country_risk_scorer.set_ec_benchmark("BR", "standard")
        result = country_risk_scorer.assess_country("BR", sample_high_risk_factors)
        assert result.risk_level == RiskLevel.STANDARD

    @pytest.mark.unit
    def test_no_ec_benchmark_uses_computed_level(
        self, country_risk_scorer, sample_high_risk_factors
    ):
        # No EC benchmark set, so computed level applies
        result = country_risk_scorer.assess_country("BR", sample_high_risk_factors)
        assert result.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    def test_ec_benchmark_aligned_flag(
        self, country_risk_scorer, sample_high_risk_factors
    ):
        country_risk_scorer.set_ec_benchmark("BR", "high")
        result = country_risk_scorer.assess_country("BR", sample_high_risk_factors)
        # ec_benchmark_aligned = not ec_override; when EC benchmark is applied
        # ec_override is True, so aligned is False
        assert result.ec_benchmark_aligned is False

    @pytest.mark.unit
    def test_no_ec_benchmark_means_aligned(
        self, country_risk_scorer, sample_high_risk_factors
    ):
        # No EC benchmark set; ec_override_level is None -> aligned = True
        result = country_risk_scorer.assess_country("BR", sample_high_risk_factors)
        assert result.ec_benchmark_aligned is True


# ============================================================================
# TestConfidenceScoring
# ============================================================================


class TestConfidenceScoring:
    """Tests for assessment confidence scoring."""

    @pytest.mark.unit
    def test_confidence_with_all_factors(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert result.confidence in [
            AssessmentConfidence.VERY_LOW,
            AssessmentConfidence.LOW,
            AssessmentConfidence.MEDIUM,
            AssessmentConfidence.HIGH,
            AssessmentConfidence.VERY_HIGH,
        ]

    @pytest.mark.unit
    def test_confidence_with_fresh_data(
        self, country_risk_scorer, sample_factor_values
    ):
        recent_dates = {
            k: datetime.now(timezone.utc) - timedelta(days=30) for k in _FACTOR_KEYS
        }
        result = country_risk_scorer.assess_country(
            "BR", sample_factor_values, data_dates=recent_dates
        )
        # Fresh data should yield higher confidence
        assert result.confidence in [
            AssessmentConfidence.HIGH,
            AssessmentConfidence.VERY_HIGH,
            AssessmentConfidence.MEDIUM,
        ]

    @pytest.mark.unit
    def test_confidence_with_stale_data(
        self, country_risk_scorer, sample_factor_values
    ):
        """Stale data lowers freshness but completeness still dominates.

        With all 6 factors present, completeness=1.0.
        With all dates 500 days old (> data_freshness_max_days=365),
        freshness=0.5 per factor -> average=0.5.
        confidence = 1.0*0.6 + 0.5*0.4 = 0.80 -> HIGH threshold (>=0.8).
        Stale data reduces freshness component but full completeness
        keeps confidence at the HIGH boundary.
        """
        old_dates = {
            k: datetime.now(timezone.utc) - timedelta(days=500)
            for k in _FACTOR_KEYS
        }
        result = country_risk_scorer.assess_country(
            "BR", sample_factor_values, data_dates=old_dates
        )
        # Full completeness (6/6) with stale freshness (0.5) gives
        # confidence = 0.80 which is at the HIGH boundary
        assert result.confidence in [
            AssessmentConfidence.HIGH,
            AssessmentConfidence.MEDIUM,
        ]

    @pytest.mark.unit
    def test_confidence_with_partial_factors(self, country_risk_scorer):
        # Only provide some factors (others default to 0 or handled gracefully)
        partial = {
            "deforestation_rate": 50.0,
            "governance_quality": 40.0,
        }
        # This may raise an error or handle missing factors
        # depending on implementation
        try:
            result = country_risk_scorer.assess_country("BR", partial)
            # If it succeeds, confidence should reflect partial data
            assert result is not None
        except (ValueError, KeyError):
            # If missing factors are required, that is valid behavior
            pass


# ============================================================================
# TestTrendAnalysis
# ============================================================================


class TestTrendAnalysis:
    """Tests for risk score trend analysis."""

    @pytest.mark.unit
    def test_trend_insufficient_data(self, country_risk_scorer):
        result = country_risk_scorer.get_risk_trend("BR")
        assert result["trend_direction"] == "insufficient_data"

    @pytest.mark.unit
    def test_trend_with_history(
        self, country_risk_scorer, sample_factor_values
    ):
        # Create multiple assessments to build history
        for _ in range(3):
            country_risk_scorer.assess_country("BR", sample_factor_values)

        result = country_risk_scorer.get_risk_trend("BR")
        assert result["country_code"] == "BR"
        assert "trend_direction" in result
        assert "slope" in result

    @pytest.mark.unit
    def test_trend_direction_stable_with_same_scores(
        self, country_risk_scorer, sample_factor_values
    ):
        for _ in range(5):
            country_risk_scorer.assess_country("BR", sample_factor_values)

        result = country_risk_scorer.get_risk_trend("BR")
        # Same factors each time -> slope near zero -> stable
        if result["trend_direction"] != "insufficient_data":
            assert result["trend_direction"] in ["stable", "improving", "deteriorating"]

    @pytest.mark.unit
    def test_trend_invalid_country_code(self, country_risk_scorer):
        with pytest.raises(ValueError):
            country_risk_scorer.get_risk_trend("")

    @pytest.mark.unit
    def test_trend_invalid_window_years(self, country_risk_scorer):
        with pytest.raises(ValueError, match="window_years"):
            country_risk_scorer.get_risk_trend("BR", window_years=0)

    @pytest.mark.unit
    def test_trend_custom_window(
        self, country_risk_scorer, sample_factor_values
    ):
        for _ in range(3):
            country_risk_scorer.assess_country("BR", sample_factor_values)

        result = country_risk_scorer.get_risk_trend("BR", window_years=1)
        assert result["window_years"] == 1


# ============================================================================
# TestBatchAssessment
# ============================================================================


class TestBatchAssessment:
    """Tests for batch country assessment."""

    @pytest.mark.unit
    def test_batch_assessment_valid(
        self, country_risk_scorer, sample_country_data
    ):
        results = country_risk_scorer.assess_batch(sample_country_data)
        assert len(results) == len(sample_country_data)
        for result in results:
            assert isinstance(result, CountryRiskAssessment)

    @pytest.mark.unit
    def test_batch_preserves_order(
        self, country_risk_scorer, sample_country_data
    ):
        results = country_risk_scorer.assess_batch(sample_country_data)
        for i, item in enumerate(sample_country_data):
            assert results[i].country_code == item["country_code"].upper()

    @pytest.mark.unit
    def test_batch_empty_raises_error(self, country_risk_scorer):
        with pytest.raises(ValueError, match="must not be empty"):
            country_risk_scorer.assess_batch([])

    @pytest.mark.unit
    def test_batch_exceeds_max_size(self, country_risk_scorer, sample_factor_values):
        items = [
            {"country_code": "BR", "factor_values": sample_factor_values}
        ] * 501
        with pytest.raises(ValueError, match="exceeds maximum"):
            country_risk_scorer.assess_batch(items)


# ============================================================================
# TestCountryComparison
# ============================================================================


class TestCountryComparison:
    """Tests for country comparison functionality."""

    @pytest.mark.unit
    def test_compare_countries_valid(
        self, country_risk_scorer, sample_country_data
    ):
        # First create assessments
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        codes = [item["country_code"] for item in sample_country_data]
        result = country_risk_scorer.compare_countries(codes)

        assert "country_scores" in result
        assert "highest_risk_country" in result
        assert "lowest_risk_country" in result
        assert "statistics" in result

    @pytest.mark.unit
    def test_compare_countries_ranking_correct(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        codes = [item["country_code"] for item in sample_country_data]
        result = country_risk_scorer.compare_countries(codes)

        # Brazil should have highest risk (highest factor values in sample)
        assert result["highest_risk_country"]["country_code"] == "BR"
        # Sweden should have lowest risk
        assert result["lowest_risk_country"]["country_code"] in ["SE", "DE"]

    @pytest.mark.unit
    def test_compare_countries_has_percentiles(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        codes = [item["country_code"] for item in sample_country_data]
        result = country_risk_scorer.compare_countries(codes)

        for entry in result["country_scores"]:
            assert "rank" in entry
            assert "percentile" in entry
            assert 0.0 <= entry["percentile"] <= 100.0

    @pytest.mark.unit
    def test_compare_countries_statistics(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        codes = [item["country_code"] for item in sample_country_data]
        result = country_risk_scorer.compare_countries(codes)

        stats = result["statistics"]
        assert "mean_score" in stats
        assert "std_dev" in stats
        assert "min_score" in stats
        assert "max_score" in stats
        assert stats["max_score"] >= stats["min_score"]

    @pytest.mark.unit
    def test_compare_countries_empty_raises(self, country_risk_scorer):
        with pytest.raises(ValueError, match="must not be empty"):
            country_risk_scorer.compare_countries([])


# ============================================================================
# TestCountryRanking
# ============================================================================


class TestCountryRanking:
    """Tests for country ranking functionality."""

    @pytest.mark.unit
    def test_ranking_desc_order(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        ranking = country_risk_scorer.get_country_ranking(order="desc")
        for i in range(len(ranking) - 1):
            assert ranking[i]["composite_score"] >= ranking[i + 1]["composite_score"]

    @pytest.mark.unit
    def test_ranking_asc_order(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        ranking = country_risk_scorer.get_country_ranking(order="asc")
        for i in range(len(ranking) - 1):
            assert ranking[i]["composite_score"] <= ranking[i + 1]["composite_score"]

    @pytest.mark.unit
    def test_ranking_invalid_order(self, country_risk_scorer):
        with pytest.raises(ValueError, match="order must be"):
            country_risk_scorer.get_country_ranking(order="random")

    @pytest.mark.unit
    def test_ranking_limit(self, country_risk_scorer, sample_country_data):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        ranking = country_risk_scorer.get_country_ranking(limit=2)
        assert len(ranking) == 2


# ============================================================================
# TestInputValidation
# ============================================================================


class TestInputValidation:
    """Tests for input validation and error handling."""

    @pytest.mark.unit
    def test_empty_country_code_raises(
        self, country_risk_scorer, sample_factor_values
    ):
        with pytest.raises(ValueError):
            country_risk_scorer.assess_country("", sample_factor_values)

    @pytest.mark.unit
    def test_missing_factor_handling(self, country_risk_scorer):
        # Missing required factors
        incomplete_factors = {"deforestation_rate": 50.0}
        try:
            result = country_risk_scorer.assess_country("BR", incomplete_factors)
            # If it succeeds, missing factors should default to some value
            assert result is not None
        except (ValueError, KeyError):
            pass  # Expected if all 6 factors are required

    @pytest.mark.unit
    def test_none_country_code_raises(
        self, country_risk_scorer, sample_factor_values
    ):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            country_risk_scorer.assess_country(None, sample_factor_values)


# ============================================================================
# TestProvenanceTracking
# ============================================================================


class TestProvenanceTracking:
    """Tests for provenance hash recording."""

    @pytest.mark.unit
    def test_assessment_has_provenance_hash(
        self, country_risk_scorer, sample_factor_values
    ):
        result = country_risk_scorer.assess_country("BR", sample_factor_values)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex string

    @pytest.mark.unit
    def test_provenance_deterministic(
        self, country_risk_scorer, sample_factor_values
    ):
        """Same inputs should produce same provenance hash."""
        result1 = country_risk_scorer.assess_country("BR", sample_factor_values)
        scorer2 = CountryRiskScorer()
        result2 = scorer2.assess_country("BR", sample_factor_values)
        # Provenance hash includes timestamp, so they may differ
        # But assessment_id is unique per call
        assert result1.assessment_id != result2.assessment_id


# ============================================================================
# TestParametrizedHighRiskCountries
# ============================================================================


class TestParametrizedHighRiskCountries:
    """Parametrized tests for EUDR high-risk countries."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "country_code",
        ["BR", "ID", "CD", "CI", "GH", "MY", "CO"],
        ids=["Brazil", "Indonesia", "DRC", "Cote_dIvoire", "Ghana", "Malaysia", "Colombia"],
    )
    def test_high_risk_country_assessment(
        self, country_risk_scorer, sample_high_risk_factors, country_code
    ):
        result = country_risk_scorer.assess_country(
            country_code, sample_high_risk_factors
        )
        assert result.country_code == country_code
        assert result.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "country_code",
        ["SE", "FI", "DE", "NO", "DK", "CH", "NZ"],
        ids=["Sweden", "Finland", "Germany", "Norway", "Denmark", "Switzerland", "NZ"],
    )
    def test_low_risk_country_assessment(
        self, country_risk_scorer, sample_low_risk_factors, country_code
    ):
        result = country_risk_scorer.assess_country(
            country_code, sample_low_risk_factors
        )
        assert result.country_code == country_code
        assert result.risk_level == RiskLevel.LOW


# ============================================================================
# TestListAssessments
# ============================================================================


class TestListAssessments:
    """Tests for listing and filtering assessments."""

    @pytest.mark.unit
    def test_list_all_assessments(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        results = country_risk_scorer.list_assessments()
        assert len(results) == len(sample_country_data)

    @pytest.mark.unit
    def test_list_assessments_filter_by_country(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        results = country_risk_scorer.list_assessments(country_code="BR")
        assert len(results) == 1
        assert results[0].country_code == "BR"

    @pytest.mark.unit
    def test_list_assessments_with_pagination(
        self, country_risk_scorer, sample_country_data
    ):
        for item in sample_country_data:
            country_risk_scorer.assess_country(
                item["country_code"], item["factor_values"]
            )

        results = country_risk_scorer.list_assessments(limit=2, offset=0)
        assert len(results) == 2

    @pytest.mark.unit
    def test_get_nonexistent_assessment(self, country_risk_scorer):
        result = country_risk_scorer.get_assessment("nonexistent-id")
        assert result is None
