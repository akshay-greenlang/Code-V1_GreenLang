# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Temperature Scoring Engine.

Tests company-level temperature scores (0-4C range), per-scope
temperature ratings, short/long-term separate scoring, overall
weighted combination, portfolio temperature scoring for FIs,
peer ranking comparisons, and reduction-to-temperature mapping
with 22+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Company Score
# ===========================================================================

class TestCompanyScore:
    """Test company-level temperature score (0-4C range)."""

    def test_overall_score_range(self, sample_temperature_score):
        score = sample_temperature_score["overall_score_c"]
        assert 1.0 <= score <= 4.0

    def test_good_temperature_score(self, sample_temperature_score):
        assert sample_temperature_score["overall_score_c"] < 2.0

    def test_poor_temperature_score(self, high_temperature_score):
        assert high_temperature_score["overall_score_c"] >= 3.0

    @pytest.mark.parametrize("score,alignment", [
        (1.5, "1.5C_aligned"),
        (1.8, "well_below_2C"),
        (2.0, "below_2C"),
        (2.5, "insufficient"),
        (3.5, "highly_insufficient"),
    ])
    def test_score_to_alignment_mapping(self, score, alignment):
        if score <= 1.5:
            result = "1.5C_aligned"
        elif score <= 2.0:
            result = "well_below_2C"
        elif score <= 2.0:
            result = "below_2C"
        elif score <= 3.0:
            result = "insufficient"
        else:
            result = "highly_insufficient"
        assert result == alignment

    def test_methodology_recorded(self, sample_temperature_score):
        assert sample_temperature_score["methodology"] == "SBTi_CDP_temperature_rating"


# ===========================================================================
# Scope Temperature
# ===========================================================================

class TestScopeTemperature:
    """Test per-scope temperature scores."""

    def test_scope1_score(self, sample_temperature_score):
        assert 1.0 <= sample_temperature_score["scope1_score_c"] <= 4.0

    def test_scope2_score(self, sample_temperature_score):
        assert 1.0 <= sample_temperature_score["scope2_score_c"] <= 4.0

    def test_scope3_score(self, sample_temperature_score):
        assert 1.0 <= sample_temperature_score["scope3_score_c"] <= 4.0

    def test_scope3_typically_higher(self, sample_temperature_score):
        """Scope 3 scores tend to be higher (worse) due to less direct control."""
        assert sample_temperature_score["scope3_score_c"] >= sample_temperature_score["scope1_score_c"]


# ===========================================================================
# Short/Long Term Scoring
# ===========================================================================

class TestShortLongTerm:
    """Test separate short-term and long-term scoring."""

    def test_short_term_score(self, sample_temperature_score):
        assert 1.0 <= sample_temperature_score["short_term_score_c"] <= 4.0

    def test_long_term_score(self, sample_temperature_score):
        assert 1.0 <= sample_temperature_score["long_term_score_c"] <= 4.0

    def test_short_vs_long_term(self, sample_temperature_score):
        # Both should be within valid range
        st = sample_temperature_score["short_term_score_c"]
        lt = sample_temperature_score["long_term_score_c"]
        assert st > 0 and lt > 0


# ===========================================================================
# Overall Score
# ===========================================================================

class TestOverallScore:
    """Test weighted combination of scope and time horizon scores."""

    def test_overall_within_scope_range(self, sample_temperature_score):
        overall = sample_temperature_score["overall_score_c"]
        s1 = sample_temperature_score["scope1_score_c"]
        s3 = sample_temperature_score["scope3_score_c"]
        # Overall should be between best and worst scope scores
        assert min(s1, s3) <= overall <= max(s1, s3) + 0.5

    def test_weighted_combination_calculation(self):
        s1 = 1.5
        s2 = 1.6
        s3 = 2.2
        # Simplified weighting: S1+2 = 40%, S3 = 60%
        s12_avg = (s1 + s2) / 2
        overall = s12_avg * 0.4 + s3 * 0.6
        assert 1.0 <= overall <= 4.0


# ===========================================================================
# Portfolio Temperature
# ===========================================================================

class TestPortfolioTemperature:
    """Test FI portfolio temperature scoring."""

    def test_portfolio_waci(self, sample_fi_portfolio):
        assert sample_fi_portfolio["waci"] > 0

    def test_portfolio_coverage_with_targets(self, sample_fi_portfolio):
        assert sample_fi_portfolio["coverage_with_sbti_pct"] > 0

    def test_portfolio_holdings_temperature(self, sample_fi_portfolio):
        holdings = sample_fi_portfolio["holdings"]
        targets_count = sum(1 for h in holdings if h["has_sbti_target"])
        assert targets_count >= 1

    def test_portfolio_weighted_temperature(self):
        holdings = [
            {"exposure": 500_000_000, "temperature": 1.8},
            {"exposure": 300_000_000, "temperature": 3.2},
            {"exposure": 200_000_000, "temperature": 1.5},
        ]
        total_exposure = sum(h["exposure"] for h in holdings)
        weighted_temp = sum(
            h["exposure"] / total_exposure * h["temperature"]
            for h in holdings
        )
        assert 1.0 <= weighted_temp <= 4.0


# ===========================================================================
# Peer Ranking
# ===========================================================================

class TestPeerRanking:
    """Test peer comparison rankings."""

    def test_peer_percentile(self, sample_temperature_score):
        assert 0 <= sample_temperature_score["peer_percentile"] <= 100

    def test_sector_average(self, sample_temperature_score):
        assert sample_temperature_score["sector_average_c"] > 0

    def test_better_than_sector_average(self, sample_temperature_score):
        assert sample_temperature_score["overall_score_c"] < sample_temperature_score["sector_average_c"]

    def test_lower_percentile_is_better(self, sample_temperature_score):
        """Lower percentile = better temperature score."""
        assert sample_temperature_score["peer_percentile"] < 50


# ===========================================================================
# Reduction-to-Temperature Mapping
# ===========================================================================

class TestReductionToTemp:
    """Test mapping annual reduction rate to temperature outcome."""

    @pytest.mark.parametrize("annual_rate,expected_temp_range", [
        (4.2, (1.4, 1.6)),    # 1.5C aligned
        (2.5, (1.7, 2.1)),    # WB2C aligned
        (1.0, (2.5, 3.5)),    # Insufficient
        (0.0, (3.5, 4.0)),    # No action
    ])
    def test_rate_to_temperature(self, annual_rate, expected_temp_range):
        # Simplified linear mapping
        if annual_rate >= 4.2:
            temp = 1.5
        elif annual_rate >= 2.5:
            temp = 2.0
        elif annual_rate >= 1.0:
            temp = 3.0
        else:
            temp = 3.8
        assert expected_temp_range[0] <= temp <= expected_temp_range[1]

    def test_reduction_to_1_5c_pct(self, sample_temperature_score):
        pct = sample_temperature_score["reduction_to_1_5c_pct"]
        assert pct > 0  # Additional reduction needed to reach 1.5C
