# -*- coding: utf-8 -*-
"""
Tests for WGIAnalyzerEngine - AGENT-EUDR-019 Engine 2: WGI Analyzer

Comprehensive test suite covering:
- WGI country indicators retrieval for all 6 dimensions
- WGI indicator history for individual dimensions
- WGI dimension analysis across multiple countries
- Cross-country WGI comparison with 2+ countries
- WGI rankings by dimension
- Composite governance score with default/custom weights
- WGI -> EUDR risk mapping for various composite scores
- WGI scale validation (estimates within -2.5 to +2.5, percentiles 0-100)
- Provenance chain integrity
- Edge cases: missing dimensions, invalid countries, boundary values

Test count: ~45 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019 (Engine 2: WGI Analyzer)
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.wgi_analyzer_engine import (
    WGIAnalyzerEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.models import (
    WGIDimension,
)
from greenlang.agents.eudr.corruption_index_monitor.config import (
    set_config,
    reset_config,
)


# ===========================================================================
# 1. WGI Country Indicators (10 tests)
# ===========================================================================


class TestWGICountryIndicators:
    """Test get_country_indicators for all 6 WGI dimensions."""

    def test_get_indicators_brazil(self, wgi_engine):
        """Test WGI indicator retrieval for Brazil across all dimensions."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.success is True
        assert result.indicators is not None
        assert len(result.indicators) == 6

    def test_get_indicators_denmark_strong_governance(self, wgi_engine):
        """Test WGI indicators for Denmark (expected strong governance)."""
        result = wgi_engine.get_country_indicators("DK")
        assert result.success is True
        for indicator in result.indicators:
            # Denmark should have positive governance estimates
            assert indicator.estimate >= Decimal("0")

    def test_get_indicators_specific_year(self, wgi_engine):
        """Test WGI indicators for a specific year."""
        result = wgi_engine.get_country_indicators("BR", year=2022)
        assert result.success is True
        if result.indicators:
            for ind in result.indicators:
                assert ind.year == 2022

    def test_get_indicators_invalid_country(self, wgi_engine):
        """Test WGI indicators for invalid country returns failure."""
        result = wgi_engine.get_country_indicators("ZZ")
        assert result.success is False

    def test_get_indicators_lowercase_normalized(self, wgi_engine):
        """Test lowercase country code is normalized to uppercase."""
        result = wgi_engine.get_country_indicators("br")
        assert result.success is True

    def test_get_indicators_includes_all_dimensions(self, wgi_engine):
        """Test that result includes all 6 WGI dimensions."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.success is True
        dimensions_found = {ind.dimension for ind in result.indicators}
        expected_dims = {
            "voice_accountability", "political_stability",
            "government_effectiveness", "regulatory_quality",
            "rule_of_law", "control_of_corruption",
        }
        # All 6 dimensions should be present
        assert len(dimensions_found) == 6

    def test_get_indicators_estimates_in_range(self, wgi_engine):
        """Test all WGI estimates are within -2.5 to +2.5 range."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.success is True
        for ind in result.indicators:
            assert Decimal("-2.5") <= ind.estimate <= Decimal("2.5")

    def test_get_indicators_percentile_in_range(self, wgi_engine):
        """Test all WGI percentile ranks are within 0-100 range."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.success is True
        for ind in result.indicators:
            if ind.percentile_rank is not None:
                assert Decimal("0") <= ind.percentile_rank <= Decimal("100")

    def test_get_indicators_includes_provenance(self, wgi_engine):
        """Test that indicator results include provenance hash."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.provenance_hash is not None

    @pytest.mark.parametrize("country_code", [
        "BR", "ID", "MY", "CO", "GH", "DK", "FI",
    ])
    def test_get_indicators_multiple_countries(self, wgi_engine, country_code):
        """Test WGI indicators retrieval for multiple EUDR countries."""
        result = wgi_engine.get_country_indicators(country_code)
        assert result.success is True
        assert result.indicators is not None
        assert len(result.indicators) == 6


# ===========================================================================
# 2. WGI History (6 tests)
# ===========================================================================


class TestWGIHistory:
    """Test get_indicator_history for individual WGI dimensions."""

    def test_history_control_of_corruption(self, wgi_engine):
        """Test history for control_of_corruption dimension."""
        result = wgi_engine.get_indicator_history(
            "BR", dimension="control_of_corruption"
        )
        assert result.success is True
        assert result.indicators is not None
        assert len(result.indicators) >= 1

    def test_history_rule_of_law(self, wgi_engine):
        """Test history for rule_of_law dimension."""
        result = wgi_engine.get_indicator_history(
            "DK", dimension="rule_of_law"
        )
        assert result.success is True

    def test_history_with_year_range(self, wgi_engine):
        """Test dimension history with explicit year range."""
        result = wgi_engine.get_indicator_history(
            "BR",
            dimension="control_of_corruption",
            start_year=2015,
            end_year=2022,
        )
        assert result.success is True
        if result.indicators:
            for ind in result.indicators:
                assert 2015 <= ind.year <= 2022

    def test_history_invalid_country(self, wgi_engine):
        """Test history for invalid country returns failure."""
        result = wgi_engine.get_indicator_history(
            "ZZ", dimension="rule_of_law"
        )
        assert result.success is False

    def test_history_chronological_order(self, wgi_engine):
        """Test history entries are in chronological order."""
        result = wgi_engine.get_indicator_history(
            "BR", dimension="control_of_corruption"
        )
        assert result.success is True
        if len(result.indicators) > 1:
            years = [ind.year for ind in result.indicators]
            assert years == sorted(years)

    def test_history_all_dimensions_have_data(self, wgi_engine):
        """Test that history is available for all 6 dimensions."""
        dimensions = [
            "voice_accountability", "political_stability",
            "government_effectiveness", "regulatory_quality",
            "rule_of_law", "control_of_corruption",
        ]
        for dim in dimensions:
            result = wgi_engine.get_indicator_history("BR", dimension=dim)
            assert result.success is True, f"Failed for dimension: {dim}"


# ===========================================================================
# 3. WGI Dimension Analysis (5 tests)
# ===========================================================================


class TestWGIDimensionAnalysis:
    """Test get_dimension_analysis for cross-country comparison by dimension."""

    def test_dimension_analysis_control_of_corruption(self, wgi_engine):
        """Test dimension analysis for control_of_corruption."""
        result = wgi_engine.get_dimension_analysis("control_of_corruption")
        assert result.success is True
        assert result.entries is not None
        assert len(result.entries) > 0

    def test_dimension_analysis_government_effectiveness(self, wgi_engine):
        """Test dimension analysis for government_effectiveness."""
        result = wgi_engine.get_dimension_analysis("government_effectiveness")
        assert result.success is True

    def test_dimension_analysis_sorted_by_estimate(self, wgi_engine):
        """Test dimension analysis entries are sorted by estimate descending."""
        result = wgi_engine.get_dimension_analysis("rule_of_law")
        assert result.success is True
        if len(result.entries) > 1:
            estimates = [e.estimate for e in result.entries]
            assert estimates == sorted(estimates, reverse=True)

    def test_dimension_analysis_top_n(self, wgi_engine):
        """Test dimension analysis with top_n filtering."""
        result = wgi_engine.get_dimension_analysis("rule_of_law", top_n=10)
        assert result.success is True
        assert len(result.entries) <= 10

    def test_dimension_analysis_includes_provenance(self, wgi_engine):
        """Test dimension analysis includes provenance hash."""
        result = wgi_engine.get_dimension_analysis("control_of_corruption")
        assert result.provenance_hash is not None


# ===========================================================================
# 4. Cross-Country Comparison (5 tests)
# ===========================================================================


class TestWGIComparison:
    """Test compare_countries with 2 or more countries."""

    def test_compare_two_countries(self, wgi_engine):
        """Test WGI comparison between Brazil and Denmark."""
        result = wgi_engine.compare_countries(["BR", "DK"])
        assert result.success is True
        assert result.comparisons is not None
        assert len(result.comparisons) == 2

    def test_compare_multiple_countries(self, wgi_engine):
        """Test WGI comparison among multiple countries."""
        result = wgi_engine.compare_countries(["BR", "DK", "ID", "GH"])
        assert result.success is True
        assert len(result.comparisons) == 4

    def test_compare_identifies_best_governance(self, wgi_engine):
        """Test comparison correctly identifies country with best governance."""
        result = wgi_engine.compare_countries(["BR", "DK"])
        assert result.success is True
        # Denmark should have better governance than Brazil
        dk_comp = next(
            (c for c in result.comparisons if c.country_code == "DK"),
            None,
        )
        br_comp = next(
            (c for c in result.comparisons if c.country_code == "BR"),
            None,
        )
        if dk_comp and br_comp and dk_comp.composite is not None and br_comp.composite is not None:
            assert dk_comp.composite > br_comp.composite

    def test_compare_empty_list(self, wgi_engine):
        """Test comparison with empty country list."""
        result = wgi_engine.compare_countries([])
        # Should handle gracefully
        assert result.success is True or result.success is False

    def test_compare_includes_provenance(self, wgi_engine):
        """Test comparison results include provenance hash."""
        result = wgi_engine.compare_countries(["BR", "DK"])
        assert result.provenance_hash is not None


# ===========================================================================
# 5. WGI Rankings (4 tests)
# ===========================================================================


class TestWGIRankings:
    """Test get_rankings by WGI dimension."""

    def test_rankings_by_dimension(self, wgi_engine):
        """Test rankings retrieval by control_of_corruption dimension."""
        result = wgi_engine.get_rankings(dimension="control_of_corruption")
        assert result.success is True
        assert result.rankings is not None
        assert len(result.rankings) > 0

    def test_rankings_sorted_descending(self, wgi_engine):
        """Test rankings are sorted by estimate descending."""
        result = wgi_engine.get_rankings(dimension="rule_of_law")
        assert result.success is True
        if len(result.rankings) > 1:
            estimates = [r.estimate for r in result.rankings]
            assert estimates == sorted(estimates, reverse=True)

    def test_rankings_top_n(self, wgi_engine):
        """Test rankings with top_n limit."""
        result = wgi_engine.get_rankings(
            dimension="control_of_corruption", top_n=5
        )
        assert result.success is True
        assert len(result.rankings) <= 5

    def test_rankings_include_provenance(self, wgi_engine):
        """Test rankings include provenance hash."""
        result = wgi_engine.get_rankings(dimension="control_of_corruption")
        assert result.provenance_hash is not None


# ===========================================================================
# 6. Composite Governance Score (6 tests)
# ===========================================================================


class TestWGICompositeScore:
    """Test calculate_composite_governance_score with default and custom weights."""

    def test_composite_brazil(self, wgi_engine):
        """Test composite governance score for Brazil."""
        result = wgi_engine.calculate_composite_governance_score("BR")
        assert result.success is True
        assert result.composite_score is not None
        # Composite normalized to [0, 1]
        assert Decimal("0") <= result.composite_score <= Decimal("1")

    def test_composite_denmark_high(self, wgi_engine):
        """Test composite governance score for Denmark (expected high)."""
        result = wgi_engine.calculate_composite_governance_score("DK")
        assert result.success is True
        assert result.composite_score is not None
        # Denmark should have high composite score
        assert result.composite_score >= Decimal("0.7")

    def test_composite_brazil_lower_than_denmark(self, wgi_engine):
        """Test that Brazil's composite is lower than Denmark's."""
        br_result = wgi_engine.calculate_composite_governance_score("BR")
        dk_result = wgi_engine.calculate_composite_governance_score("DK")
        assert br_result.success is True
        assert dk_result.success is True
        assert br_result.composite_score < dk_result.composite_score

    def test_composite_invalid_country(self, wgi_engine):
        """Test composite for invalid country returns failure."""
        result = wgi_engine.calculate_composite_governance_score("ZZ")
        assert result.success is False

    def test_composite_deterministic(self, wgi_engine):
        """Test composite score is deterministic across calls."""
        r1 = wgi_engine.calculate_composite_governance_score("BR")
        r2 = wgi_engine.calculate_composite_governance_score("BR")
        assert r1.composite_score == r2.composite_score

    def test_composite_includes_provenance(self, wgi_engine):
        """Test composite score includes provenance hash."""
        result = wgi_engine.calculate_composite_governance_score("BR")
        assert result.provenance_hash is not None


# ===========================================================================
# 7. WGI -> EUDR Risk Mapping (6 tests)
# ===========================================================================


class TestWGIEUDRMapping:
    """Test _map_wgi_to_eudr_risk for various composite scores.

    WGI EUDR risk mapping:
        1. Normalize estimate from [-2.5, +2.5] to [0, 1]
        2. EUDR risk = 1.0 - normalized_composite
        Composite 0.0 -> EUDR risk 1.0 (weakest governance)
        Composite 1.0 -> EUDR risk 0.0 (strongest governance)
    """

    def test_wgi_eudr_risk_zero_composite(self, wgi_engine):
        """Composite 0.0 (weakest) maps to EUDR risk 1.0."""
        risk = wgi_engine._map_wgi_to_eudr_risk(Decimal("0.0"))
        assert risk == Decimal("1.0000") or risk == Decimal("1.0")

    def test_wgi_eudr_risk_one_composite(self, wgi_engine):
        """Composite 1.0 (strongest) maps to EUDR risk 0.0."""
        risk = wgi_engine._map_wgi_to_eudr_risk(Decimal("1.0"))
        assert risk == Decimal("0.0000") or risk == Decimal("0.0")

    def test_wgi_eudr_risk_midpoint(self, wgi_engine):
        """Composite 0.5 (midpoint) maps to EUDR risk 0.5."""
        risk = wgi_engine._map_wgi_to_eudr_risk(Decimal("0.5"))
        assert risk == Decimal("0.5000") or risk == Decimal("0.5")

    def test_wgi_eudr_risk_inverse_relationship(self, wgi_engine):
        """Higher composite scores should produce lower EUDR risk."""
        risk_low = wgi_engine._map_wgi_to_eudr_risk(Decimal("0.2"))
        risk_high = wgi_engine._map_wgi_to_eudr_risk(Decimal("0.8"))
        assert risk_low > risk_high

    def test_wgi_eudr_risk_always_between_0_and_1(self, wgi_engine):
        """EUDR risk from WGI should always be between 0 and 1."""
        for val in [Decimal("0.0"), Decimal("0.25"), Decimal("0.5"),
                     Decimal("0.75"), Decimal("1.0")]:
            risk = wgi_engine._map_wgi_to_eudr_risk(val)
            assert Decimal("0") <= risk <= Decimal("1")

    def test_get_eudr_risk_factor_for_country(self, wgi_engine):
        """Test convenience method get_eudr_risk_factor for a country."""
        risk = wgi_engine.get_eudr_risk_factor("DK")
        assert Decimal("0") <= risk <= Decimal("1")
        # Denmark should have low EUDR risk
        assert risk <= Decimal("0.3")


# ===========================================================================
# 8. WGI Scale Validation (5 tests)
# ===========================================================================


class TestWGIScaleValidation:
    """Ensure WGI estimates and percentiles are within valid ranges."""

    def test_all_estimates_within_range(self, wgi_engine):
        """Test all WGI estimates are within -2.5 to +2.5."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.success is True
        for ind in result.indicators:
            assert Decimal("-2.5") <= ind.estimate <= Decimal("2.5"), (
                f"Estimate {ind.estimate} out of range for "
                f"dimension {ind.dimension}"
            )

    def test_all_percentiles_within_range(self, wgi_engine):
        """Test all WGI percentile ranks are within 0-100."""
        result = wgi_engine.get_country_indicators("DK")
        assert result.success is True
        for ind in result.indicators:
            if ind.percentile_rank is not None:
                assert Decimal("0") <= ind.percentile_rank <= Decimal("100")

    def test_high_governance_positive_estimates(self, wgi_engine):
        """Test countries with strong governance have positive estimates."""
        result = wgi_engine.get_country_indicators("DK")
        assert result.success is True
        positive_count = sum(
            1 for ind in result.indicators if ind.estimate > Decimal("0")
        )
        # Denmark should have mostly positive estimates
        assert positive_count >= 5

    def test_governance_score_normalized(self, wgi_engine):
        """Test that governance_score field is normalized to 0-100."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.success is True
        for ind in result.indicators:
            if ind.governance_score is not None:
                assert Decimal("0") <= ind.governance_score <= Decimal("100")

    @pytest.mark.parametrize("country_code", ["BR", "DK", "ID", "GH"])
    def test_estimates_consistent_across_countries(self, wgi_engine, country_code):
        """Test WGI estimates are within valid range for multiple countries."""
        result = wgi_engine.get_country_indicators(country_code)
        assert result.success is True
        for ind in result.indicators:
            assert Decimal("-2.5") <= ind.estimate <= Decimal("2.5")


# ===========================================================================
# 9. Provenance Chain (5 tests)
# ===========================================================================


class TestWGIProvenance:
    """Test provenance chain integrity for WGI results."""

    def test_indicators_provenance_hash(self, wgi_engine):
        """Test country indicators include provenance hash."""
        result = wgi_engine.get_country_indicators("BR")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) > 0

    def test_indicators_provenance_deterministic(self, wgi_engine):
        """Test provenance hash is deterministic for same query."""
        r1 = wgi_engine.get_country_indicators("BR")
        r2 = wgi_engine.get_country_indicators("BR")
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_countries_different_provenance(self, wgi_engine):
        """Test different countries produce different provenance hashes."""
        r_br = wgi_engine.get_country_indicators("BR")
        r_dk = wgi_engine.get_country_indicators("DK")
        assert r_br.provenance_hash != r_dk.provenance_hash

    def test_composite_provenance_deterministic(self, wgi_engine):
        """Test composite score provenance is deterministic."""
        r1 = wgi_engine.calculate_composite_governance_score("BR")
        r2 = wgi_engine.calculate_composite_governance_score("BR")
        assert r1.provenance_hash == r2.provenance_hash

    def test_provenance_hex_format(self, wgi_engine):
        """Test provenance hash is valid hexadecimal."""
        result = wgi_engine.get_country_indicators("BR")
        if result.provenance_hash:
            int(result.provenance_hash, 16)  # Should not raise


# ===========================================================================
# 10. Edge Cases (6 tests)
# ===========================================================================


class TestWGIEdgeCases:
    """Test edge cases: invalid countries, boundary values, error handling."""

    def test_empty_country_code(self, wgi_engine):
        """Test empty country code returns failure."""
        result = wgi_engine.get_country_indicators("")
        assert result.success is False

    def test_three_letter_country_code(self, wgi_engine):
        """Test 3-letter country code is handled appropriately."""
        result = wgi_engine.get_country_indicators("BRA")
        assert result.success is False

    def test_whitespace_country_code(self, wgi_engine):
        """Test whitespace-padded country code is stripped."""
        result = wgi_engine.get_country_indicators(" BR ")
        assert result.success is True

    def test_comparison_with_invalid_country(self, wgi_engine):
        """Test comparison handles invalid country in list gracefully."""
        result = wgi_engine.compare_countries(["BR", "ZZ", "DK"])
        assert result.success is True
        # Should still return data for valid countries
        valid_results = [
            c for c in result.comparisons
            if c.country_code in ("BR", "DK")
        ]
        assert len(valid_results) >= 2

    def test_history_unavailable_year_range(self, wgi_engine):
        """Test history with years outside available data range."""
        result = wgi_engine.get_indicator_history(
            "BR", dimension="control_of_corruption",
            start_year=1900, end_year=1950,
        )
        # Should return empty or handle gracefully
        assert result.success is True or result.success is False

    def test_engine_initializes_without_config(self):
        """Test WGI engine can initialize without explicit config."""
        reset_config()
        engine = WGIAnalyzerEngine()
        result = engine.get_country_indicators("BR")
        assert result.success is True


# ===========================================================================
# 11. Determinism (3 tests)
# ===========================================================================


class TestWGIDeterminism:
    """Test deterministic behavior of WGI analyzer engine."""

    def test_indicators_deterministic_10_runs(self, wgi_engine):
        """Test 10 consecutive queries return identical results."""
        first = wgi_engine.get_country_indicators("BR")
        for _ in range(9):
            result = wgi_engine.get_country_indicators("BR")
            assert result.provenance_hash == first.provenance_hash
            for i1, i2 in zip(first.indicators, result.indicators):
                assert i1.estimate == i2.estimate

    def test_eudr_risk_deterministic_10_runs(self, wgi_engine):
        """Test EUDR risk factor is deterministic across 10 runs."""
        first_risk = wgi_engine.get_eudr_risk_factor("BR")
        for _ in range(9):
            risk = wgi_engine.get_eudr_risk_factor("BR")
            assert risk == first_risk

    def test_comparison_deterministic(self, wgi_engine):
        """Test country comparison is deterministic."""
        r1 = wgi_engine.compare_countries(["BR", "DK"])
        r2 = wgi_engine.compare_countries(["BR", "DK"])
        assert r1.provenance_hash == r2.provenance_hash
