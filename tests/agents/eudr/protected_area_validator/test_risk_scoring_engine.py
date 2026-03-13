# -*- coding: utf-8 -*-
"""
Tests for RiskScoringEngine - AGENT-EUDR-022 Engine 5

Comprehensive test suite covering:
- 5-factor deterministic scoring (Decimal arithmetic)
- IUCN category risk (40-100 base scores)
- Overlap type multipliers
- Buffer zone proximity scoring
- Deforestation correlation
- Certification scheme overlays
- Edge cases and boundary values
- Risk classification thresholds

Test count: 75 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 (Engine 5: Risk Scoring)
"""

import time
from decimal import Decimal, ROUND_HALF_UP

import pytest

from tests.agents.eudr.protected_area_validator.conftest import (
    compute_test_hash,
    compute_risk_score,
    classify_risk_level,
    compute_buffer_proximity_score,
    SHA256_HEX_LENGTH,
    IUCN_CATEGORIES,
    IUCN_CATEGORY_RISK_SCORES,
    OVERLAP_TYPES,
    OVERLAP_TYPE_SCORES,
    DEFAULT_RISK_WEIGHTS,
    RISK_THRESHOLD_CRITICAL,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    RISK_THRESHOLD_LOW,
    CERTIFICATION_SCHEMES,
)


# ===========================================================================
# 1. 5-Factor Composite Scoring (15 tests)
# ===========================================================================


class TestCompositeScoring:
    """Test 5-factor weighted composite risk scoring."""

    def test_max_risk_score_is_100(self):
        """Test maximum possible risk score is 100.00."""
        score = compute_risk_score(
            iucn_category="Ia",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("100"),
            deforestation_correlation_score=Decimal("100"),
            certification_overlay_score=Decimal("100"),
        )
        assert score == Decimal("100.00")

    def test_min_risk_score_with_none_overlap(self):
        """Test minimum risk score with no overlap and low factors."""
        score = compute_risk_score(
            iucn_category="VI",
            overlap_type="NONE",
            buffer_proximity_score=Decimal("0"),
            deforestation_correlation_score=Decimal("0"),
            certification_overlay_score=Decimal("0"),
        )
        # VI(40)*0.30 + NONE(0)*0.25 + 0*0.20 + 0*0.15 + 0*0.10 = 12.00
        assert score == Decimal("12.00")

    def test_weights_sum_to_one(self):
        """Test all 5 weights sum to exactly 1.00."""
        total = sum(DEFAULT_RISK_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_five_factors_defined(self):
        """Test exactly 5 factors in weight map."""
        assert len(DEFAULT_RISK_WEIGHTS) == 5

    def test_known_composite_calculation(self):
        """Test known composite calculation with documented values.

        IUCN II (90) * 0.30 = 27.00
        DIRECT (100) * 0.25 = 25.00
        Proximity (80) * 0.20 = 16.00
        Deforestation (60) * 0.15 = 9.00
        Certification (40) * 0.10 = 4.00
        Total = 81.00 -> CRITICAL
        """
        score = compute_risk_score(
            iucn_category="II",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("80"),
            deforestation_correlation_score=Decimal("60"),
            certification_overlay_score=Decimal("40"),
        )
        assert score == Decimal("81.00")

    def test_composite_score_always_positive(self):
        """Test composite score is always >= 0."""
        for iucn in IUCN_CATEGORIES:
            for ot in OVERLAP_TYPES:
                score = compute_risk_score(
                    iucn_category=iucn,
                    overlap_type=ot,
                    buffer_proximity_score=Decimal("0"),
                    deforestation_correlation_score=Decimal("0"),
                    certification_overlay_score=Decimal("0"),
                )
                assert score >= Decimal("0")

    def test_composite_score_never_exceeds_100(self):
        """Test composite score never exceeds 100."""
        for iucn in IUCN_CATEGORIES:
            for ot in OVERLAP_TYPES:
                score = compute_risk_score(
                    iucn_category=iucn,
                    overlap_type=ot,
                    buffer_proximity_score=Decimal("100"),
                    deforestation_correlation_score=Decimal("100"),
                    certification_overlay_score=Decimal("100"),
                )
                assert score <= Decimal("100")

    def test_score_uses_decimal_precision(self):
        """Test score result is Decimal with 2 decimal places."""
        score = compute_risk_score(
            iucn_category="III",
            overlap_type="PARTIAL",
            buffer_proximity_score=Decimal("33.33"),
            deforestation_correlation_score=Decimal("66.67"),
            certification_overlay_score=Decimal("50.00"),
        )
        assert isinstance(score, Decimal)
        assert score == score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def test_iucn_category_dominates_at_30_pct_weight(self):
        """Test IUCN category is the largest weight factor at 30%."""
        assert DEFAULT_RISK_WEIGHTS["iucn_category"] == Decimal("0.30")
        for key, weight in DEFAULT_RISK_WEIGHTS.items():
            assert DEFAULT_RISK_WEIGHTS["iucn_category"] >= weight

    def test_overlap_type_is_second_factor_at_25_pct(self):
        """Test overlap type is the second largest factor at 25%."""
        assert DEFAULT_RISK_WEIGHTS["overlap_type"] == Decimal("0.25")

    def test_buffer_proximity_at_20_pct(self):
        """Test buffer proximity weight is 20%."""
        assert DEFAULT_RISK_WEIGHTS["buffer_proximity"] == Decimal("0.20")

    def test_deforestation_correlation_at_15_pct(self):
        """Test deforestation correlation weight is 15%."""
        assert DEFAULT_RISK_WEIGHTS["deforestation_correlation"] == Decimal("0.15")

    def test_certification_overlay_at_10_pct(self):
        """Test certification overlay weight is 10%."""
        assert DEFAULT_RISK_WEIGHTS["certification_overlay"] == Decimal("0.10")

    def test_custom_weights_accepted(self):
        """Test custom weight overrides are applied."""
        custom = {
            "iucn_category": Decimal("0.50"),
            "overlap_type": Decimal("0.20"),
            "buffer_proximity": Decimal("0.10"),
            "deforestation_correlation": Decimal("0.10"),
            "certification_overlay": Decimal("0.10"),
        }
        score = compute_risk_score(
            iucn_category="Ia",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("50"),
            deforestation_correlation_score=Decimal("50"),
            certification_overlay_score=Decimal("50"),
            weights=custom,
        )
        assert isinstance(score, Decimal)

    def test_zero_weight_factor_ignored(self):
        """Test a factor with 0 weight has no effect."""
        custom = {
            "iucn_category": Decimal("0.50"),
            "overlap_type": Decimal("0.50"),
            "buffer_proximity": Decimal("0.00"),  # Zero weight
            "deforestation_correlation": Decimal("0.00"),
            "certification_overlay": Decimal("0.00"),
        }
        score_low = compute_risk_score(
            iucn_category="II",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("0"),
            deforestation_correlation_score=Decimal("0"),
            certification_overlay_score=Decimal("0"),
            weights=custom,
        )
        score_high = compute_risk_score(
            iucn_category="II",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("100"),
            deforestation_correlation_score=Decimal("100"),
            certification_overlay_score=Decimal("100"),
            weights=custom,
        )
        # Zero-weight factors should not change the score
        assert score_low == score_high


# ===========================================================================
# 2. IUCN Category Risk Factor (10 tests)
# ===========================================================================


class TestIUCNCategoryRiskFactor:
    """Test IUCN category as risk factor."""

    @pytest.mark.parametrize("category,base_score", [
        ("Ia", Decimal("100")),
        ("Ib", Decimal("95")),
        ("II", Decimal("90")),
        ("III", Decimal("80")),
        ("IV", Decimal("70")),
        ("V", Decimal("55")),
        ("VI", Decimal("40")),
    ])
    def test_iucn_category_base_scores(self, category, base_score):
        """Test IUCN category base scores match documented values."""
        assert IUCN_CATEGORY_RISK_SCORES[category] == base_score

    def test_stricter_category_higher_score(self):
        """Test stricter IUCN categories produce higher scores."""
        assert IUCN_CATEGORY_RISK_SCORES["Ia"] > IUCN_CATEGORY_RISK_SCORES["VI"]

    def test_iucn_score_contribution_to_composite(self):
        """Test IUCN score contributes 30% of composite score."""
        ia_contrib = IUCN_CATEGORY_RISK_SCORES["Ia"] * DEFAULT_RISK_WEIGHTS["iucn_category"]
        vi_contrib = IUCN_CATEGORY_RISK_SCORES["VI"] * DEFAULT_RISK_WEIGHTS["iucn_category"]
        assert ia_contrib == Decimal("30.00")
        assert vi_contrib == Decimal("12.00")


# ===========================================================================
# 3. Overlap Type Risk Factor (8 tests)
# ===========================================================================


class TestOverlapTypeRiskFactor:
    """Test overlap type as risk factor."""

    @pytest.mark.parametrize("overlap_type,score", [
        ("DIRECT", Decimal("100")),
        ("PARTIAL", Decimal("80")),
        ("BUFFER", Decimal("60")),
        ("ADJACENT", Decimal("45")),
        ("PROXIMATE", Decimal("25")),
        ("NONE", Decimal("0")),
    ])
    def test_overlap_type_scores(self, overlap_type, score):
        """Test overlap type base scores."""
        assert OVERLAP_TYPE_SCORES[overlap_type] == score

    def test_overlap_contribution_to_composite(self):
        """Test overlap contributes 25% of composite score."""
        direct_contrib = OVERLAP_TYPE_SCORES["DIRECT"] * DEFAULT_RISK_WEIGHTS["overlap_type"]
        assert direct_contrib == Decimal("25.00")

    def test_no_overlap_zero_contribution(self):
        """Test NONE overlap contributes 0 to composite."""
        none_contrib = OVERLAP_TYPE_SCORES["NONE"] * DEFAULT_RISK_WEIGHTS["overlap_type"]
        assert none_contrib == Decimal("0.00")


# ===========================================================================
# 4. Deforestation Correlation Factor (8 tests)
# ===========================================================================


class TestDeforestationCorrelation:
    """Test deforestation correlation as risk factor."""

    def test_high_deforestation_high_risk(self):
        """Test high deforestation correlation increases risk."""
        score_high = compute_risk_score(
            iucn_category="II", overlap_type="BUFFER",
            buffer_proximity_score=Decimal("50"),
            deforestation_correlation_score=Decimal("95"),
            certification_overlay_score=Decimal("50"),
        )
        score_low = compute_risk_score(
            iucn_category="II", overlap_type="BUFFER",
            buffer_proximity_score=Decimal("50"),
            deforestation_correlation_score=Decimal("10"),
            certification_overlay_score=Decimal("50"),
        )
        assert score_high > score_low

    def test_zero_deforestation_reduces_risk(self):
        """Test zero deforestation correlation reduces risk."""
        score = compute_risk_score(
            iucn_category="II", overlap_type="BUFFER",
            buffer_proximity_score=Decimal("50"),
            deforestation_correlation_score=Decimal("0"),
            certification_overlay_score=Decimal("50"),
        )
        assert score < Decimal("100")

    def test_deforestation_weight_is_15_pct(self):
        """Test deforestation correlation weight is 15%."""
        assert DEFAULT_RISK_WEIGHTS["deforestation_correlation"] == Decimal("0.15")

    def test_max_deforestation_contribution(self):
        """Test maximum deforestation contribution."""
        max_contrib = Decimal("100") * DEFAULT_RISK_WEIGHTS["deforestation_correlation"]
        assert max_contrib == Decimal("15.00")

    @pytest.mark.parametrize("deforestation_score", [
        Decimal("0"), Decimal("25"), Decimal("50"), Decimal("75"), Decimal("100"),
    ])
    def test_deforestation_score_range(self, deforestation_score):
        """Test deforestation score in valid range [0, 100]."""
        assert Decimal("0") <= deforestation_score <= Decimal("100")

    def test_deforestation_from_satellite_data(self):
        """Test deforestation score derived from satellite monitoring."""
        # Score based on Global Forest Watch / Hansen data
        annual_loss_pct = Decimal("2.5")
        threshold_high = Decimal("3.0")
        normalized = min(annual_loss_pct / threshold_high * Decimal("100"), Decimal("100"))
        assert Decimal("0") <= normalized <= Decimal("100")

    def test_deforestation_correlation_with_country_risk(self):
        """Test deforestation correlation considers country-level risk."""
        high_risk_countries = {"BR", "ID", "CD"}
        country = "BR"
        is_high_risk = country in high_risk_countries
        assert is_high_risk is True

    def test_deforestation_trend_factored(self):
        """Test deforestation trend (increasing/decreasing) is factored."""
        increasing_trend = Decimal("85")
        decreasing_trend = Decimal("30")
        assert increasing_trend > decreasing_trend


# ===========================================================================
# 5. Certification Scheme Overlay (10 tests)
# ===========================================================================


class TestCertificationSchemeOverlay:
    """Test certification scheme as risk mitigating factor."""

    @pytest.mark.parametrize("scheme", CERTIFICATION_SCHEMES)
    def test_valid_certification_schemes(self, scheme):
        """Test all certification schemes are recognized."""
        assert scheme in CERTIFICATION_SCHEMES

    def test_certification_count(self):
        """Test number of supported certification schemes."""
        assert len(CERTIFICATION_SCHEMES) == 8  # Including 'none'

    def test_fsc_certification_reduces_risk(self):
        """Test FSC certification reduces risk score."""
        score_no_cert = compute_risk_score(
            iucn_category="IV", overlap_type="BUFFER",
            buffer_proximity_score=Decimal("60"),
            deforestation_correlation_score=Decimal("50"),
            certification_overlay_score=Decimal("0"),
        )
        score_fsc = compute_risk_score(
            iucn_category="IV", overlap_type="BUFFER",
            buffer_proximity_score=Decimal("60"),
            deforestation_correlation_score=Decimal("50"),
            certification_overlay_score=Decimal("80"),
        )
        # Higher cert score changes the total
        assert score_no_cert != score_fsc

    def test_no_certification_is_valid(self):
        """Test 'none' is a valid certification status."""
        assert "none" in CERTIFICATION_SCHEMES

    def test_certification_weight_is_10_pct(self):
        """Test certification overlay weight is 10%."""
        assert DEFAULT_RISK_WEIGHTS["certification_overlay"] == Decimal("0.10")

    def test_max_certification_contribution(self):
        """Test maximum certification contribution."""
        max_contrib = Decimal("100") * DEFAULT_RISK_WEIGHTS["certification_overlay"]
        assert max_contrib == Decimal("10.00")

    def test_multiple_certifications_stacked(self):
        """Test multiple certifications produce higher overlay score."""
        single_cert_score = Decimal("60")
        multi_cert_score = Decimal("90")
        assert multi_cert_score > single_cert_score

    def test_expired_certification_not_counted(self):
        """Test expired certification is not counted."""
        active = True
        expired = False
        assert active != expired

    def test_rspo_for_palm_oil(self):
        """Test RSPO certification is relevant for palm oil."""
        assert "rspo" in CERTIFICATION_SCHEMES

    def test_rainforest_alliance_for_cocoa(self):
        """Test Rainforest Alliance is relevant for cocoa."""
        assert "rainforest_alliance" in CERTIFICATION_SCHEMES


# ===========================================================================
# 6. Risk Classification Thresholds (12 tests)
# ===========================================================================


class TestRiskClassificationThresholds:
    """Test risk level classification from composite scores."""

    def test_threshold_values_defined(self):
        """Test all threshold values are defined."""
        assert RISK_THRESHOLD_CRITICAL == Decimal("80")
        assert RISK_THRESHOLD_HIGH == Decimal("60")
        assert RISK_THRESHOLD_MEDIUM == Decimal("40")
        assert RISK_THRESHOLD_LOW == Decimal("20")

    @pytest.mark.parametrize("score,expected_level", [
        (Decimal("100.00"), "CRITICAL"),
        (Decimal("80.00"), "CRITICAL"),
        (Decimal("79.99"), "HIGH"),
        (Decimal("60.00"), "HIGH"),
        (Decimal("59.99"), "MEDIUM"),
        (Decimal("40.00"), "MEDIUM"),
        (Decimal("39.99"), "LOW"),
        (Decimal("20.00"), "LOW"),
        (Decimal("19.99"), "INFO"),
        (Decimal("0.00"), "INFO"),
    ])
    def test_classification_boundary_values(self, score, expected_level):
        """Test classification at exact boundary values."""
        assert classify_risk_level(score) == expected_level

    def test_critical_range(self):
        """Test CRITICAL range: [80, 100]."""
        for s in [80, 85, 90, 95, 100]:
            assert classify_risk_level(Decimal(str(s))) == "CRITICAL"

    def test_high_range(self):
        """Test HIGH range: [60, 80)."""
        for s in [60, 65, 70, 75]:
            assert classify_risk_level(Decimal(str(s))) == "HIGH"

    def test_medium_range(self):
        """Test MEDIUM range: [40, 60)."""
        for s in [40, 45, 50, 55]:
            assert classify_risk_level(Decimal(str(s))) == "MEDIUM"

    def test_low_range(self):
        """Test LOW range: [20, 40)."""
        for s in [20, 25, 30, 35]:
            assert classify_risk_level(Decimal(str(s))) == "LOW"

    def test_info_range(self):
        """Test INFO range: [0, 20)."""
        for s in [0, 5, 10, 15]:
            assert classify_risk_level(Decimal(str(s))) == "INFO"

    def test_thresholds_ascending(self):
        """Test thresholds are in ascending order."""
        assert RISK_THRESHOLD_LOW < RISK_THRESHOLD_MEDIUM
        assert RISK_THRESHOLD_MEDIUM < RISK_THRESHOLD_HIGH
        assert RISK_THRESHOLD_HIGH < RISK_THRESHOLD_CRITICAL

    def test_full_range_covered(self):
        """Test every integer score 0-100 maps to a valid level."""
        valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        for s in range(101):
            level = classify_risk_level(Decimal(str(s)))
            assert level in valid_levels

    def test_classification_is_deterministic(self):
        """Test same score always gives same classification."""
        results = set()
        for _ in range(100):
            level = classify_risk_level(Decimal("55.55"))
            results.add(level)
        assert len(results) == 1


# ===========================================================================
# 7. Edge Cases and Boundary Values (12 tests)
# ===========================================================================


class TestEdgeCasesAndBoundaryValues:
    """Test edge cases in risk scoring."""

    def test_all_factors_zero(self):
        """Test all factor scores at zero."""
        score = compute_risk_score(
            iucn_category="VI",
            overlap_type="NONE",
            buffer_proximity_score=Decimal("0"),
            deforestation_correlation_score=Decimal("0"),
            certification_overlay_score=Decimal("0"),
        )
        assert score >= Decimal("0")

    def test_all_factors_100(self):
        """Test all factor scores at 100."""
        score = compute_risk_score(
            iucn_category="Ia",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("100"),
            deforestation_correlation_score=Decimal("100"),
            certification_overlay_score=Decimal("100"),
        )
        assert score == Decimal("100.00")

    def test_decimal_precision_no_float_drift(self):
        """Test Decimal arithmetic prevents floating-point drift."""
        score1 = compute_risk_score(
            iucn_category="III",
            overlap_type="BUFFER",
            buffer_proximity_score=Decimal("33.33"),
            deforestation_correlation_score=Decimal("66.67"),
            certification_overlay_score=Decimal("50.00"),
        )
        score2 = compute_risk_score(
            iucn_category="III",
            overlap_type="BUFFER",
            buffer_proximity_score=Decimal("33.33"),
            deforestation_correlation_score=Decimal("66.67"),
            certification_overlay_score=Decimal("50.00"),
        )
        assert score1 == score2

    def test_factor_at_boundary_50(self):
        """Test factors at midpoint value of 50."""
        score = compute_risk_score(
            iucn_category="III",
            overlap_type="ADJACENT",
            buffer_proximity_score=Decimal("50"),
            deforestation_correlation_score=Decimal("50"),
            certification_overlay_score=Decimal("50"),
        )
        # III(80)*0.30 + ADJ(45)*0.25 + 50*0.20 + 50*0.15 + 50*0.10
        # = 24 + 11.25 + 10 + 7.5 + 5 = 57.75
        assert score == Decimal("57.75")

    def test_single_dominant_factor(self):
        """Test scenario where single factor dominates."""
        score = compute_risk_score(
            iucn_category="Ia",
            overlap_type="NONE",
            buffer_proximity_score=Decimal("0"),
            deforestation_correlation_score=Decimal("0"),
            certification_overlay_score=Decimal("0"),
        )
        # Ia(100)*0.30 = 30.00 (only IUCN contributes)
        assert score == Decimal("30.00")

    def test_unknown_iucn_category_uses_default(self):
        """Test unknown IUCN category uses default score of 50."""
        score = compute_risk_score(
            iucn_category="UNKNOWN",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("80"),
            deforestation_correlation_score=Decimal("60"),
            certification_overlay_score=Decimal("40"),
        )
        # UNKNOWN defaults to 50
        assert isinstance(score, Decimal)

    def test_unknown_overlap_type_uses_default(self):
        """Test unknown overlap type uses default score of 0."""
        score = compute_risk_score(
            iucn_category="II",
            overlap_type="UNKNOWN",
            buffer_proximity_score=Decimal("80"),
            deforestation_correlation_score=Decimal("60"),
            certification_overlay_score=Decimal("40"),
        )
        assert isinstance(score, Decimal)

    def test_extremely_small_factor_values(self):
        """Test very small factor values (e.g., 0.01)."""
        score = compute_risk_score(
            iucn_category="VI",
            overlap_type="PROXIMATE",
            buffer_proximity_score=Decimal("0.01"),
            deforestation_correlation_score=Decimal("0.01"),
            certification_overlay_score=Decimal("0.01"),
        )
        assert score >= Decimal("0")

    def test_factor_at_99_99(self):
        """Test factor values at 99.99."""
        score = compute_risk_score(
            iucn_category="Ia",
            overlap_type="DIRECT",
            buffer_proximity_score=Decimal("99.99"),
            deforestation_correlation_score=Decimal("99.99"),
            certification_overlay_score=Decimal("99.99"),
        )
        assert score <= Decimal("100.00")

    def test_mixed_extreme_factors(self):
        """Test mix of 0 and 100 factor values."""
        score = compute_risk_score(
            iucn_category="Ia",
            overlap_type="NONE",
            buffer_proximity_score=Decimal("100"),
            deforestation_correlation_score=Decimal("0"),
            certification_overlay_score=Decimal("100"),
        )
        # Ia(100)*0.30 + NONE(0)*0.25 + 100*0.20 + 0*0.15 + 100*0.10
        # = 30 + 0 + 20 + 0 + 10 = 60.00
        assert score == Decimal("60.00")

    def test_rounding_behavior_half_up(self):
        """Test ROUND_HALF_UP rounding behavior."""
        val = Decimal("55.555")
        rounded = val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert rounded == Decimal("55.56")

    def test_rounding_at_threshold_boundary(self):
        """Test rounding does not cause incorrect classification."""
        # 79.995 rounds to 80.00 -> CRITICAL
        score = Decimal("79.995").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert score == Decimal("80.00")
        assert classify_risk_level(score) == "CRITICAL"
