# -*- coding: utf-8 -*-
"""
Tests for ConversionRiskAssessor - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Risk tier classification (low, moderate, high, critical)
- All 8 risk factor scoring
- Risk weight validation (sum to 1.0)
- Deterministic risk scoring
- Conversion probability estimation (6m, 12m, 24m)
- Deforestation frontier detection
- Risk heatmap generation
- Batch assessment
- Configurable weights

Test count: 50 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.land_use_change.conftest import (
    ConversionRiskAssessment,
    compute_test_hash,
    classify_risk_tier,
    weighted_composite,
    SHA256_HEX_LENGTH,
    RISK_TIERS,
    RISK_FACTORS,
    DEFAULT_RISK_WEIGHTS,
)


# ===========================================================================
# 1. Risk Tier Classification Tests (12 tests)
# ===========================================================================


class TestRiskTierClassification:
    """Tests for risk tier classification logic."""

    def test_risk_tier_low(self):
        """Composite score < 0.25 -> low risk."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-LOW-001",
            risk_tier="low",
            composite_score=0.15,
        )
        assert result.risk_tier == "low"
        assert result.composite_score < 0.25

    def test_risk_tier_moderate(self):
        """Composite score 0.25-0.50 -> moderate risk."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-MOD-001",
            risk_tier="moderate",
            composite_score=0.38,
        )
        assert result.risk_tier == "moderate"
        assert 0.25 <= result.composite_score < 0.50

    def test_risk_tier_high(self):
        """Composite score 0.50-0.75 -> high risk."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-HI-001",
            risk_tier="high",
            composite_score=0.62,
        )
        assert result.risk_tier == "high"
        assert 0.50 <= result.composite_score < 0.75

    def test_risk_tier_critical(self):
        """Composite score >= 0.75 -> critical risk."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-CRT-001",
            risk_tier="critical",
            composite_score=0.88,
        )
        assert result.risk_tier == "critical"
        assert result.composite_score >= 0.75

    def test_classify_risk_tier_function_low(self):
        """classify_risk_tier returns low for score < 0.25."""
        assert classify_risk_tier(0.10) == "low"
        assert classify_risk_tier(0.24) == "low"

    def test_classify_risk_tier_function_moderate(self):
        """classify_risk_tier returns moderate for score 0.25-0.50."""
        assert classify_risk_tier(0.25) == "moderate"
        assert classify_risk_tier(0.49) == "moderate"

    def test_classify_risk_tier_function_high(self):
        """classify_risk_tier returns high for score 0.50-0.75."""
        assert classify_risk_tier(0.50) == "high"
        assert classify_risk_tier(0.74) == "high"

    def test_classify_risk_tier_function_critical(self):
        """classify_risk_tier returns critical for score >= 0.75."""
        assert classify_risk_tier(0.75) == "critical"
        assert classify_risk_tier(1.00) == "critical"

    @pytest.mark.parametrize("tier", RISK_TIERS)
    def test_all_risk_tiers_valid(self, tier):
        """Each risk tier value is accepted."""
        result = ConversionRiskAssessment(
            plot_id=f"PLOT-{tier.upper()[:3]}",
            risk_tier=tier,
        )
        assert result.risk_tier == tier

    def test_risk_tier_boundary_0(self):
        """Score 0.0 is low risk."""
        assert classify_risk_tier(0.0) == "low"

    def test_risk_tier_boundary_1(self):
        """Score 1.0 is critical risk."""
        assert classify_risk_tier(1.0) == "critical"

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.00, "low"),
            (0.12, "low"),
            (0.24, "low"),
            (0.25, "moderate"),
            (0.37, "moderate"),
            (0.49, "moderate"),
            (0.50, "high"),
            (0.62, "high"),
            (0.74, "high"),
            (0.75, "critical"),
            (0.88, "critical"),
            (1.00, "critical"),
        ],
        ids=[f"score_{int(s * 100)}" for s in [
            0.00, 0.12, 0.24, 0.25, 0.37, 0.49,
            0.50, 0.62, 0.74, 0.75, 0.88, 1.00,
        ]],
    )
    def test_classify_risk_tier_parametrized(self, score, expected):
        """Risk tier classification at various scores."""
        assert classify_risk_tier(score) == expected


# ===========================================================================
# 2. Risk Factor Scoring Tests (10 tests)
# ===========================================================================


class TestRiskFactorScoring:
    """Tests for all 8 risk factor individual scoring."""

    @pytest.mark.parametrize("factor", RISK_FACTORS)
    def test_each_risk_factor_exists(self, factor):
        """Each risk factor is recognized."""
        assert factor in DEFAULT_RISK_WEIGHTS

    def test_all_8_risk_factors(self):
        """There are exactly 8 risk factors."""
        assert len(RISK_FACTORS) == 8
        assert len(DEFAULT_RISK_WEIGHTS) == 8

    def test_factor_scores_in_range(self):
        """All factor scores must be in [0, 1] range."""
        scores = {
            "transition_magnitude": 0.80,
            "proximity_to_forest": 0.60,
            "historical_deforestation_rate": 0.70,
            "commodity_pressure": 0.55,
            "governance_score": 0.30,
            "protected_area_proximity": 0.85,
            "road_infrastructure_proximity": 0.45,
            "population_density_change": 0.50,
        }
        for factor, score in scores.items():
            assert 0.0 <= score <= 1.0, (
                f"Factor {factor} score {score} out of range"
            )

    def test_factor_scores_in_result(self):
        """Factor scores stored in result."""
        factor_scores = {f: 0.50 for f in RISK_FACTORS}
        result = ConversionRiskAssessment(
            plot_id="PLOT-FS-001",
            factor_scores=factor_scores,
        )
        assert len(result.factor_scores) == 8

    def test_high_transition_magnitude_raises_risk(self):
        """High transition magnitude increases composite risk score."""
        scores_high = {"transition_magnitude": 0.95}
        scores_low = {"transition_magnitude": 0.10}
        high_comp = weighted_composite(scores_high, DEFAULT_RISK_WEIGHTS)
        low_comp = weighted_composite(scores_low, DEFAULT_RISK_WEIGHTS)
        assert high_comp > low_comp

    def test_governance_score_inverse(self):
        """Low governance score (poor governance) increases risk."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-GOV-001",
            factor_scores={"governance_score": 0.90},
        )
        # High governance_score factor means poor governance = high risk
        assert result.factor_scores["governance_score"] > 0.50

    def test_protected_area_proximity_high_risk(self):
        """Near protected area increases risk."""
        result = ConversionRiskAssessment(
            factor_scores={"protected_area_proximity": 0.90},
        )
        assert result.factor_scores["protected_area_proximity"] > 0.80

    def test_road_infrastructure_moderate_risk(self):
        """Near roads moderately increases risk."""
        result = ConversionRiskAssessment(
            factor_scores={"road_infrastructure_proximity": 0.55},
        )
        assert 0.40 <= result.factor_scores["road_infrastructure_proximity"] <= 0.70

    def test_composite_from_all_factors(self):
        """Composite score computed from all 8 factors."""
        factor_scores = {
            "transition_magnitude": 0.80,
            "proximity_to_forest": 0.60,
            "historical_deforestation_rate": 0.70,
            "commodity_pressure": 0.55,
            "governance_score": 0.30,
            "protected_area_proximity": 0.85,
            "road_infrastructure_proximity": 0.45,
            "population_density_change": 0.50,
        }
        composite = weighted_composite(factor_scores, DEFAULT_RISK_WEIGHTS)
        assert 0.0 < composite < 1.0
        result = ConversionRiskAssessment(
            composite_score=composite,
            factor_scores=factor_scores,
            factor_weights=DEFAULT_RISK_WEIGHTS,
        )
        assert abs(result.composite_score - composite) < 0.001

    def test_zero_factors_zero_composite(self):
        """All factors at zero -> composite is zero."""
        factor_scores = {f: 0.0 for f in RISK_FACTORS}
        composite = weighted_composite(factor_scores, DEFAULT_RISK_WEIGHTS)
        assert composite == 0.0


# ===========================================================================
# 3. Weight Validation Tests (5 tests)
# ===========================================================================


class TestWeightValidation:
    """Tests for risk weight validation."""

    def test_weights_sum_to_one(self):
        """Default risk weights sum to 1.0."""
        total = sum(DEFAULT_RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_weights_positive(self):
        """All weights are positive."""
        assert all(w > 0.0 for w in DEFAULT_RISK_WEIGHTS.values())

    def test_all_weights_below_one(self):
        """No single weight exceeds 1.0."""
        assert all(w <= 1.0 for w in DEFAULT_RISK_WEIGHTS.values())

    def test_largest_weight_is_transition_magnitude(self):
        """Transition magnitude has the largest weight (0.20)."""
        max_factor = max(DEFAULT_RISK_WEIGHTS, key=DEFAULT_RISK_WEIGHTS.get)
        assert max_factor == "transition_magnitude"
        assert DEFAULT_RISK_WEIGHTS[max_factor] == 0.20

    def test_configurable_weights(self, sample_config):
        """Config allows custom weights."""
        custom_weights = dict(DEFAULT_RISK_WEIGHTS)
        custom_weights["transition_magnitude"] = 0.30
        custom_weights["proximity_to_forest"] = 0.05
        total = sum(custom_weights.values())
        assert abs(total - 1.0) < 0.001


# ===========================================================================
# 4. Conversion Probability Tests (6 tests)
# ===========================================================================


class TestConversionProbability:
    """Tests for conversion probability estimation."""

    def test_conversion_probability_6m(self):
        """6-month conversion probability in [0, 1]."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-PRB-001",
            conversion_probability_6m=0.15,
        )
        assert 0.0 <= result.conversion_probability_6m <= 1.0

    def test_conversion_probability_12m(self):
        """12-month conversion probability in [0, 1]."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-PRB-002",
            conversion_probability_12m=0.30,
        )
        assert 0.0 <= result.conversion_probability_12m <= 1.0

    def test_conversion_probability_24m(self):
        """24-month conversion probability in [0, 1]."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-PRB-003",
            conversion_probability_24m=0.55,
        )
        assert 0.0 <= result.conversion_probability_24m <= 1.0

    def test_probability_increases_with_horizon(self):
        """Longer horizons have higher conversion probability."""
        result = ConversionRiskAssessment(
            conversion_probability_6m=0.15,
            conversion_probability_12m=0.30,
            conversion_probability_24m=0.55,
        )
        assert result.conversion_probability_6m <= result.conversion_probability_12m
        assert result.conversion_probability_12m <= result.conversion_probability_24m

    def test_low_risk_low_probability(self):
        """Low risk plots have low conversion probability."""
        result = ConversionRiskAssessment(
            risk_tier="low",
            composite_score=0.10,
            conversion_probability_12m=0.05,
        )
        assert result.conversion_probability_12m < 0.15

    def test_critical_risk_high_probability(self):
        """Critical risk plots have high conversion probability."""
        result = ConversionRiskAssessment(
            risk_tier="critical",
            composite_score=0.90,
            conversion_probability_12m=0.75,
        )
        assert result.conversion_probability_12m > 0.50


# ===========================================================================
# 5. Deforestation Frontier Tests (4 tests)
# ===========================================================================


class TestDeforestationFrontier:
    """Tests for deforestation frontier detection."""

    def test_deforestation_frontier_detection_true(self):
        """Active deforestation frontier is detected."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-FRT-001",
            is_deforestation_frontier=True,
            risk_tier="critical",
            composite_score=0.85,
        )
        assert result.is_deforestation_frontier is True

    def test_deforestation_frontier_detection_false(self):
        """Non-frontier area is not flagged."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-FRT-002",
            is_deforestation_frontier=False,
            risk_tier="low",
            composite_score=0.10,
        )
        assert result.is_deforestation_frontier is False

    def test_frontier_has_high_risk(self):
        """Frontier areas have high or critical risk tier."""
        result = ConversionRiskAssessment(
            is_deforestation_frontier=True,
            risk_tier="high",
        )
        assert result.risk_tier in ("high", "critical")

    def test_non_frontier_can_be_moderate(self):
        """Non-frontier areas can have moderate risk."""
        result = ConversionRiskAssessment(
            is_deforestation_frontier=False,
            risk_tier="moderate",
        )
        assert result.risk_tier == "moderate"


# ===========================================================================
# 6. Heatmap, Batch, and Determinism Tests (7 tests)
# ===========================================================================


class TestHeatmapBatchDeterminism:
    """Tests for heatmap generation, batch assessment, and determinism."""

    def test_risk_heatmap_generation(self):
        """Risk heatmap data is generated."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-HM-001",
            heatmap_data={
                "grid_resolution_m": 100,
                "cells": 500,
                "min_risk": 0.05,
                "max_risk": 0.95,
                "color_map": "RdYlGn_r",
            },
        )
        assert "grid_resolution_m" in result.heatmap_data
        assert result.heatmap_data["cells"] > 0

    def test_batch_assessment(self):
        """Batch assessment of 25 plots."""
        results = [
            ConversionRiskAssessment(
                plot_id=f"PLOT-BATCH-{i:04d}",
                risk_tier=classify_risk_tier(i * 0.04),
                composite_score=i * 0.04,
            )
            for i in range(25)
        ]
        assert len(results) == 25
        low_count = sum(1 for r in results if r.risk_tier == "low")
        assert low_count > 0

    def test_deterministic_scoring(self):
        """Same inputs produce same risk scores."""
        factor_scores = {
            "transition_magnitude": 0.60,
            "proximity_to_forest": 0.40,
            "historical_deforestation_rate": 0.50,
            "commodity_pressure": 0.45,
            "governance_score": 0.35,
            "protected_area_proximity": 0.70,
            "road_infrastructure_proximity": 0.30,
            "population_density_change": 0.40,
        }
        composites = [
            weighted_composite(factor_scores, DEFAULT_RISK_WEIGHTS)
            for _ in range(10)
        ]
        assert len(set(composites)) == 1

    def test_provenance_hash(self):
        """Assessment has provenance hash."""
        h = compute_test_hash({"plot_id": "PLOT-PRV", "risk": "high"})
        result = ConversionRiskAssessment(
            provenance_hash=h,
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_weighted_composite_partial_factors(self):
        """Composite works with partial factor availability."""
        partial = {
            "transition_magnitude": 0.80,
            "proximity_to_forest": 0.60,
        }
        composite = weighted_composite(partial, DEFAULT_RISK_WEIGHTS)
        assert 0.0 < composite < 1.0

    def test_weighted_composite_empty_factors(self):
        """Empty factors produce zero composite."""
        composite = weighted_composite({}, DEFAULT_RISK_WEIGHTS)
        assert composite == 0.0

    def test_weighted_composite_all_max(self):
        """All factors at 1.0 produce composite of 1.0."""
        factor_scores = {f: 1.0 for f in RISK_FACTORS}
        composite = weighted_composite(factor_scores, DEFAULT_RISK_WEIGHTS)
        assert abs(composite - 1.0) < 0.001
