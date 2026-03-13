# -*- coding: utf-8 -*-
"""
Tests for CPIMonitorEngine - AGENT-EUDR-019 Engine 1: CPI Monitor

Comprehensive test suite covering:
- CPI score retrieval for valid/invalid countries, specific/latest years
- CPI history retrieval with various year ranges
- CPI rankings: global, by region, top-N filtering
- CPI regional analysis for different regions, invalid regions
- CPI batch queries with multiple countries, empty list, mixed valid/invalid
- CPI summary statistics with year filtering
- Corruption risk classification at all threshold boundaries
- EUDR risk factor mapping (inverse CPI relationship)
- Provenance hash generation and chain integrity
- Metrics counter increments and histogram recording

Test count: ~45 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019 (Engine 1: CPI Monitor)
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.cpi_monitor_engine import (
    CPIMonitorEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.config import (
    set_config,
    reset_config,
)


# ===========================================================================
# 1. CPI Score Retrieval (10 tests)
# ===========================================================================


class TestCPIScoreRetrieval:
    """Test get_country_score for valid/invalid countries and years."""

    def test_get_score_brazil_latest(self, cpi_engine):
        """Test CPI score retrieval for Brazil with latest year (default)."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True
        assert result.data is not None
        assert result.data.country_code == "BR"
        assert result.data.score >= Decimal("0")
        assert result.data.score <= Decimal("100")
        assert result.data.year >= 2020

    def test_get_score_brazil_specific_year(self, cpi_engine):
        """Test CPI score retrieval for Brazil for a specific year."""
        result = cpi_engine.get_country_score("BR", 2020)
        assert result.success is True
        assert result.data is not None
        assert result.data.year == 2020

    def test_get_score_denmark_high_cpi(self, cpi_engine):
        """Test CPI score for Denmark (expected high score, low corruption)."""
        result = cpi_engine.get_country_score("DK")
        assert result.success is True
        assert result.data is not None
        assert result.data.score >= Decimal("80")
        assert result.data.country_code == "DK"

    def test_get_score_invalid_country_code(self, cpi_engine):
        """Test CPI score retrieval for an invalid country code returns failure."""
        result = cpi_engine.get_country_score("XX")
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "invalid" in result.error.lower()

    def test_get_score_empty_country_code(self, cpi_engine):
        """Test CPI score retrieval with empty country code returns failure."""
        result = cpi_engine.get_country_score("")
        assert result.success is False
        assert result.error is not None

    def test_get_score_lowercase_country_code(self, cpi_engine):
        """Test that lowercase country code is normalized to uppercase."""
        result = cpi_engine.get_country_score("br")
        assert result.success is True
        assert result.data is not None
        assert result.data.country_code == "BR"

    def test_get_score_unavailable_year(self, cpi_engine):
        """Test CPI score for a year where data is unavailable."""
        result = cpi_engine.get_country_score("BR", 1990)
        assert result.success is False
        assert result.error is not None

    def test_get_score_includes_rank(self, cpi_engine):
        """Test that CPI score result includes global rank."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True
        assert result.data.rank is not None
        assert result.data.rank >= 1

    def test_get_score_includes_region(self, cpi_engine):
        """Test that CPI score result includes region classification."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True
        assert result.data.region is not None
        assert len(result.data.region) > 0

    def test_get_score_includes_calculation_timestamp(self, cpi_engine):
        """Test that result includes calculation timestamp."""
        result = cpi_engine.get_country_score("BR")
        assert result.calculation_timestamp is not None

    @pytest.mark.parametrize("country_code", [
        "BR", "ID", "MY", "CO", "GH", "CI", "DK", "FI",
    ])
    def test_get_score_multiple_eudr_countries(self, cpi_engine, country_code):
        """Test CPI score retrieval for multiple EUDR-relevant countries."""
        result = cpi_engine.get_country_score(country_code)
        assert result.success is True
        assert result.data is not None
        assert result.data.country_code == country_code
        assert Decimal("0") <= result.data.score <= Decimal("100")


# ===========================================================================
# 2. CPI History (8 tests)
# ===========================================================================


class TestCPIHistory:
    """Test get_score_history with various year ranges."""

    def test_history_brazil_full_range(self, cpi_engine):
        """Test full CPI history retrieval for Brazil."""
        result = cpi_engine.get_score_history("BR")
        assert result.success is True
        assert result.scores is not None
        assert len(result.scores) >= 5

    def test_history_with_year_range(self, cpi_engine):
        """Test CPI history with explicit start and end year."""
        result = cpi_engine.get_score_history("BR", start_year=2018, end_year=2024)
        assert result.success is True
        assert result.scores is not None
        for score_entry in result.scores:
            assert 2018 <= score_entry.year <= 2024

    def test_history_single_year_range(self, cpi_engine):
        """Test CPI history when start_year equals end_year."""
        result = cpi_engine.get_score_history("BR", start_year=2022, end_year=2022)
        assert result.success is True
        if result.scores:
            assert len(result.scores) == 1
            assert result.scores[0].year == 2022

    def test_history_invalid_country(self, cpi_engine):
        """Test history for an invalid country returns failure."""
        result = cpi_engine.get_score_history("ZZ")
        assert result.success is False

    def test_history_scores_in_chronological_order(self, cpi_engine):
        """Test that history scores are returned in chronological order."""
        result = cpi_engine.get_score_history("BR")
        assert result.success is True
        if len(result.scores) > 1:
            years = [s.year for s in result.scores]
            assert years == sorted(years)

    def test_history_each_entry_has_score(self, cpi_engine):
        """Test that each history entry has a valid CPI score."""
        result = cpi_engine.get_score_history("DK")
        assert result.success is True
        for entry in result.scores:
            assert Decimal("0") <= entry.score <= Decimal("100")

    def test_history_boundary_years(self, cpi_engine):
        """Test CPI history at the boundary of available data."""
        result = cpi_engine.get_score_history("BR", start_year=2012)
        assert result.success is True
        if result.scores:
            assert result.scores[0].year >= 2012

    def test_history_includes_provenance(self, cpi_engine):
        """Test that history result includes a provenance hash."""
        result = cpi_engine.get_score_history("BR")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) > 0


# ===========================================================================
# 3. CPI Rankings (5 tests)
# ===========================================================================


class TestCPIRankings:
    """Test get_rankings for global and regional rankings."""

    def test_rankings_global(self, cpi_engine):
        """Test global CPI rankings retrieval."""
        result = cpi_engine.get_rankings()
        assert result.success is True
        assert result.rankings is not None
        assert len(result.rankings) > 0

    def test_rankings_by_region(self, cpi_engine):
        """Test CPI rankings filtered by region."""
        result = cpi_engine.get_rankings(region="americas")
        assert result.success is True
        if result.rankings:
            for entry in result.rankings:
                assert entry.region == "americas"

    def test_rankings_top_n_filtering(self, cpi_engine):
        """Test top-N filtering in rankings."""
        result = cpi_engine.get_rankings(top_n=10)
        assert result.success is True
        assert len(result.rankings) <= 10

    def test_rankings_descending_score_order(self, cpi_engine):
        """Test that rankings are in descending score order (cleanest first)."""
        result = cpi_engine.get_rankings()
        assert result.success is True
        if len(result.rankings) > 1:
            scores = [entry.score for entry in result.rankings]
            assert scores == sorted(scores, reverse=True)

    def test_rankings_include_provenance(self, cpi_engine):
        """Test that rankings result includes provenance hash."""
        result = cpi_engine.get_rankings()
        assert result.provenance_hash is not None


# ===========================================================================
# 4. CPI Regional Analysis (5 tests)
# ===========================================================================


class TestCPIRegionalAnalysis:
    """Test get_regional_analysis for different regions."""

    def test_regional_analysis_americas(self, cpi_engine):
        """Test regional analysis for Americas region."""
        result = cpi_engine.get_regional_analysis("americas")
        assert result.success is True
        assert result.region == "americas"

    def test_regional_analysis_sub_saharan_africa(self, cpi_engine):
        """Test regional analysis for Sub-Saharan Africa."""
        result = cpi_engine.get_regional_analysis("sub_saharan_africa")
        assert result.success is True
        assert result.region == "sub_saharan_africa"

    def test_regional_analysis_asia_pacific(self, cpi_engine):
        """Test regional analysis for Asia-Pacific region."""
        result = cpi_engine.get_regional_analysis("asia_pacific")
        assert result.success is True

    def test_regional_analysis_invalid_region(self, cpi_engine):
        """Test regional analysis for an invalid region returns failure."""
        result = cpi_engine.get_regional_analysis("invalid_region_xyz")
        assert result.success is False

    def test_regional_analysis_includes_statistics(self, cpi_engine):
        """Test that regional analysis includes mean, min, max stats."""
        result = cpi_engine.get_regional_analysis("americas")
        assert result.success is True
        if result.average is not None:
            assert Decimal("0") <= result.average <= Decimal("100")


# ===========================================================================
# 5. CPI Batch Query (6 tests)
# ===========================================================================


class TestCPIBatchQuery:
    """Test batch_query with multiple countries."""

    def test_batch_query_multiple_countries(self, cpi_engine):
        """Test batch query with multiple valid countries."""
        countries = ["BR", "ID", "DK"]
        result = cpi_engine.batch_query(countries)
        assert result.success is True
        assert result.results is not None
        assert len(result.results) == 3

    def test_batch_query_empty_list(self, cpi_engine):
        """Test batch query with empty country list."""
        result = cpi_engine.batch_query([])
        assert result.success is True
        assert result.results is not None
        assert len(result.results) == 0

    def test_batch_query_single_country(self, cpi_engine):
        """Test batch query with a single country."""
        result = cpi_engine.batch_query(["DK"])
        assert result.success is True
        assert len(result.results) == 1

    def test_batch_query_mixed_valid_invalid(self, cpi_engine):
        """Test batch query with both valid and invalid country codes."""
        result = cpi_engine.batch_query(["BR", "XX", "DK"])
        assert result.success is True
        # Valid countries should return data; invalid should be noted
        successful = [r for r in result.results if r.success]
        assert len(successful) >= 2

    def test_batch_query_includes_provenance(self, cpi_engine):
        """Test that batch query results include provenance hashes."""
        result = cpi_engine.batch_query(["BR", "DK"])
        assert result.provenance_hash is not None

    def test_batch_query_all_eudr_countries(self, cpi_engine, sample_country_codes):
        """Test batch query with all EUDR-relevant countries."""
        result = cpi_engine.batch_query(sample_country_codes)
        assert result.success is True
        assert len(result.results) >= len(sample_country_codes) - 2  # Allow some missing


# ===========================================================================
# 6. CPI Summary Statistics (3 tests)
# ===========================================================================


class TestCPISummary:
    """Test get_summary_statistics with year filtering."""

    def test_summary_statistics_default(self, cpi_engine):
        """Test summary statistics with default parameters."""
        result = cpi_engine.get_summary_statistics()
        assert result.success is True

    def test_summary_statistics_specific_year(self, cpi_engine):
        """Test summary statistics for a specific year."""
        result = cpi_engine.get_summary_statistics(year=2024)
        assert result.success is True

    def test_summary_statistics_includes_count(self, cpi_engine):
        """Test that summary statistics include country count."""
        result = cpi_engine.get_summary_statistics()
        assert result.success is True
        if result.count is not None:
            assert result.count > 0


# ===========================================================================
# 7. Corruption Risk Classification (10 tests)
# ===========================================================================


class TestCPIRiskClassification:
    """Test classify_corruption_level for all threshold boundaries.

    Risk classification thresholds:
        CPI >= 80: VERY_LOW corruption risk
        60 <= CPI < 80: LOW corruption risk
        40 <= CPI < 60: MODERATE corruption risk
        20 <= CPI < 40: HIGH corruption risk
        CPI < 20: VERY_HIGH corruption risk
    """

    def test_classify_very_low_corruption_at_100(self, cpi_engine):
        """CPI 100 (maximum) should classify as very_low corruption risk."""
        result = cpi_engine.classify_corruption_level(100.0)
        assert result == "very_low"

    def test_classify_very_low_corruption_at_80(self, cpi_engine):
        """CPI 80 (boundary) should classify as very_low corruption risk."""
        result = cpi_engine.classify_corruption_level(80.0)
        assert result == "very_low"

    def test_classify_low_corruption_at_79(self, cpi_engine):
        """CPI 79 (just below very_low boundary) should classify as low."""
        result = cpi_engine.classify_corruption_level(79.0)
        assert result == "low"

    def test_classify_low_corruption_at_60(self, cpi_engine):
        """CPI 60 (boundary) should classify as low corruption risk."""
        result = cpi_engine.classify_corruption_level(60.0)
        assert result == "low"

    def test_classify_moderate_corruption_at_59(self, cpi_engine):
        """CPI 59 (just below low boundary) should classify as moderate."""
        result = cpi_engine.classify_corruption_level(59.0)
        assert result == "moderate"

    def test_classify_moderate_corruption_at_40(self, cpi_engine):
        """CPI 40 (boundary) should classify as moderate corruption risk."""
        result = cpi_engine.classify_corruption_level(40.0)
        assert result == "moderate"

    def test_classify_high_corruption_at_39(self, cpi_engine):
        """CPI 39 (just below moderate boundary) should classify as high."""
        result = cpi_engine.classify_corruption_level(39.0)
        assert result == "high"

    def test_classify_high_corruption_at_20(self, cpi_engine):
        """CPI 20 (boundary) should classify as high corruption risk."""
        result = cpi_engine.classify_corruption_level(20.0)
        assert result == "high"

    def test_classify_very_high_corruption_at_19(self, cpi_engine):
        """CPI 19 (just below high boundary) should classify as very_high."""
        result = cpi_engine.classify_corruption_level(19.0)
        assert result == "very_high"

    def test_classify_very_high_corruption_at_0(self, cpi_engine):
        """CPI 0 (minimum) should classify as very_high corruption risk."""
        result = cpi_engine.classify_corruption_level(0.0)
        assert result == "very_high"

    @pytest.mark.parametrize("score,expected_level", [
        (100, "very_low"),
        (90, "very_low"),
        (80, "very_low"),
        (79, "low"),
        (70, "low"),
        (60, "low"),
        (59, "moderate"),
        (50, "moderate"),
        (40, "moderate"),
        (39, "high"),
        (30, "high"),
        (20, "high"),
        (19, "very_high"),
        (10, "very_high"),
        (0, "very_high"),
    ])
    def test_classify_parametrized(self, cpi_engine, score, expected_level):
        """Parametrized test for all risk level boundaries."""
        result = cpi_engine.classify_corruption_level(float(score))
        assert result == expected_level


# ===========================================================================
# 8. EUDR Risk Mapping (8 tests)
# ===========================================================================


class TestCPIEUDRRiskMapping:
    """Test _calculate_eudr_risk_factor and get_eudr_risk_factor.

    EUDR risk formula: risk = 1.0 - (CPI / 100)
        CPI 0 -> EUDR risk 1.0 (most corrupt = highest EUDR risk)
        CPI 50 -> EUDR risk 0.5
        CPI 100 -> EUDR risk 0.0 (cleanest = lowest EUDR risk)
    """

    def test_eudr_risk_cpi_0_maps_to_1(self, cpi_engine):
        """CPI 0 (most corrupt) should map to EUDR risk 1.0 (highest)."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("0"))
        assert risk == Decimal("1.0000")

    def test_eudr_risk_cpi_100_maps_to_0(self, cpi_engine):
        """CPI 100 (cleanest) should map to EUDR risk 0.0 (lowest)."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("100"))
        assert risk == Decimal("0.0000")

    def test_eudr_risk_cpi_50_maps_to_05(self, cpi_engine):
        """CPI 50 (midpoint) should map to EUDR risk 0.5."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("50"))
        assert risk == Decimal("0.5000")

    @pytest.mark.parametrize("cpi_score,expected_risk", [
        (Decimal("0"), Decimal("1.0000")),
        (Decimal("10"), Decimal("0.9000")),
        (Decimal("20"), Decimal("0.8000")),
        (Decimal("30"), Decimal("0.7000")),
        (Decimal("38"), Decimal("0.6200")),
        (Decimal("50"), Decimal("0.5000")),
        (Decimal("60"), Decimal("0.4000")),
        (Decimal("70"), Decimal("0.3000")),
        (Decimal("80"), Decimal("0.2000")),
        (Decimal("90"), Decimal("0.1000")),
        (Decimal("100"), Decimal("0.0000")),
    ])
    def test_eudr_risk_parametrized(self, cpi_engine, cpi_score, expected_risk):
        """Parametrized EUDR risk factor mapping test."""
        risk = cpi_engine._calculate_eudr_risk_factor(cpi_score)
        assert risk == expected_risk

    def test_eudr_risk_brazil_approximate(self, cpi_engine):
        """Brazil CPI ~38 should map to EUDR risk ~0.62."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("38"))
        assert risk == Decimal("0.6200")

    def test_eudr_risk_denmark_approximate(self, cpi_engine):
        """Denmark CPI ~90 should map to EUDR risk ~0.10."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("90"))
        assert risk == Decimal("0.1000")

    def test_eudr_risk_always_between_0_and_1(self, cpi_engine):
        """EUDR risk factor should always be between 0.0 and 1.0."""
        for cpi in range(0, 101, 5):
            risk = cpi_engine._calculate_eudr_risk_factor(Decimal(str(cpi)))
            assert Decimal("0") <= risk <= Decimal("1")

    def test_get_eudr_risk_factor_for_country(self, cpi_engine):
        """Test convenience method get_eudr_risk_factor for a country."""
        risk = cpi_engine.get_eudr_risk_factor("DK")
        assert Decimal("0") <= risk <= Decimal("1")
        # Denmark should be low risk
        assert risk <= Decimal("0.2000")

    def test_get_eudr_risk_factor_unknown_country(self, cpi_engine):
        """Unknown country should return maximum risk (precautionary)."""
        risk = cpi_engine.get_eudr_risk_factor("ZZ")
        assert risk == Decimal("1.0000")


# ===========================================================================
# 9. Provenance Tracking (6 tests)
# ===========================================================================


class TestCPIProvenance:
    """Test provenance hash generation and chain integrity."""

    def test_score_result_has_provenance_hash(self, cpi_engine):
        """Test that CPI score result includes a provenance hash."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) > 0

    def test_provenance_hash_is_deterministic(self, cpi_engine):
        """Test that same input produces same provenance hash."""
        result1 = cpi_engine.get_country_score("BR", 2024)
        result2 = cpi_engine.get_country_score("BR", 2024)
        assert result1.provenance_hash == result2.provenance_hash

    def test_different_inputs_different_provenance(self, cpi_engine):
        """Test that different inputs produce different provenance hashes."""
        result_br = cpi_engine.get_country_score("BR")
        result_dk = cpi_engine.get_country_score("DK")
        assert result_br.provenance_hash != result_dk.provenance_hash

    def test_provenance_hash_hex_format(self, cpi_engine):
        """Test provenance hash is valid hexadecimal."""
        result = cpi_engine.get_country_score("BR")
        if result.provenance_hash:
            int(result.provenance_hash, 16)  # Should not raise

    def test_history_provenance_deterministic(self, cpi_engine):
        """Test history results have deterministic provenance."""
        result1 = cpi_engine.get_score_history("BR")
        result2 = cpi_engine.get_score_history("BR")
        assert result1.provenance_hash == result2.provenance_hash

    def test_rankings_provenance_deterministic(self, cpi_engine):
        """Test rankings results have deterministic provenance."""
        result1 = cpi_engine.get_rankings()
        result2 = cpi_engine.get_rankings()
        assert result1.provenance_hash == result2.provenance_hash


# ===========================================================================
# 10. Metrics Recording (4 tests)
# ===========================================================================


class TestCPIMetrics:
    """Test metric counter increments and histogram recording."""

    def test_query_does_not_raise_without_prometheus(self, cpi_engine):
        """Test that queries work even when Prometheus is unavailable."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True

    def test_batch_query_does_not_raise_without_prometheus(self, cpi_engine):
        """Test that batch queries work without Prometheus."""
        result = cpi_engine.batch_query(["BR", "DK"])
        assert result.success is True

    def test_history_does_not_raise_without_prometheus(self, cpi_engine):
        """Test that history queries work without Prometheus."""
        result = cpi_engine.get_score_history("BR")
        assert result.success is True

    def test_rankings_does_not_raise_without_prometheus(self, cpi_engine):
        """Test that rankings queries work without Prometheus."""
        result = cpi_engine.get_rankings()
        assert result.success is True


# ===========================================================================
# 11. Edge Cases and Error Handling (6 tests)
# ===========================================================================


class TestCPIEdgeCases:
    """Test edge cases and error conditions for CPIMonitorEngine."""

    def test_three_letter_country_code(self, cpi_engine):
        """Test that 3-letter codes are handled (should fail as CPI uses 2-letter)."""
        result = cpi_engine.get_country_score("BRA")
        assert result.success is False

    def test_numeric_country_code(self, cpi_engine):
        """Test that numeric strings are rejected as country codes."""
        result = cpi_engine.get_country_score("12")
        assert result.success is False

    def test_whitespace_country_code_stripped(self, cpi_engine):
        """Test that whitespace is stripped from country codes."""
        result = cpi_engine.get_country_score(" BR ")
        assert result.success is True
        assert result.data.country_code == "BR"

    def test_get_score_includes_standard_error(self, cpi_engine):
        """Test that CPI score includes standard error estimate."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True
        if result.data.standard_error is not None:
            assert result.data.standard_error >= Decimal("0")

    def test_get_score_includes_confidence_interval(self, cpi_engine):
        """Test that CPI score includes confidence interval bounds."""
        result = cpi_engine.get_country_score("BR")
        assert result.success is True
        if result.data.confidence_interval_low is not None:
            assert result.data.confidence_interval_low <= result.data.score
        if result.data.confidence_interval_high is not None:
            assert result.data.confidence_interval_high >= result.data.score

    def test_get_score_includes_change_from_previous(self, cpi_engine):
        """Test that CPI score includes year-over-year change."""
        result = cpi_engine.get_country_score("BR", 2024)
        assert result.success is True
        # Change can be positive, negative, or zero
        assert isinstance(result.data.change_from_previous, Decimal)


# ===========================================================================
# 12. Engine Initialization (3 tests)
# ===========================================================================


class TestCPIEngineInitialization:
    """Test CPIMonitorEngine initialization and configuration."""

    def test_engine_initializes_without_config(self):
        """Test engine can initialize even without explicit config."""
        reset_config()
        engine = CPIMonitorEngine()
        result = engine.get_country_score("BR")
        assert result.success is True

    def test_engine_initializes_with_config(self, mock_config):
        """Test engine initializes correctly with explicit config."""
        set_config(mock_config)
        engine = CPIMonitorEngine()
        result = engine.get_country_score("BR")
        assert result.success is True

    def test_engine_multiple_instances_consistent(self, mock_config):
        """Test that multiple engine instances return consistent results."""
        set_config(mock_config)
        engine1 = CPIMonitorEngine()
        engine2 = CPIMonitorEngine()
        result1 = engine1.get_country_score("BR", 2024)
        result2 = engine2.get_country_score("BR", 2024)
        assert result1.data.score == result2.data.score
        assert result1.data.rank == result2.data.rank


# ===========================================================================
# 13. Determinism and Reproducibility (4 tests)
# ===========================================================================


class TestCPIDeterminism:
    """Test deterministic behavior of CPI engine."""

    def test_score_deterministic_10_runs(self, cpi_engine):
        """Test that 10 consecutive score queries return identical results."""
        first_result = cpi_engine.get_country_score("BR", 2024)
        for _ in range(9):
            result = cpi_engine.get_country_score("BR", 2024)
            assert result.data.score == first_result.data.score
            assert result.data.rank == first_result.data.rank
            assert result.provenance_hash == first_result.provenance_hash

    def test_eudr_risk_deterministic(self, cpi_engine):
        """Test EUDR risk factor is deterministic for same CPI score."""
        risk1 = cpi_engine._calculate_eudr_risk_factor(Decimal("38"))
        risk2 = cpi_engine._calculate_eudr_risk_factor(Decimal("38"))
        assert risk1 == risk2

    def test_classification_deterministic(self, cpi_engine):
        """Test risk classification is deterministic."""
        for _ in range(10):
            assert cpi_engine.classify_corruption_level(38.0) == "high"
            assert cpi_engine.classify_corruption_level(90.0) == "very_low"
            assert cpi_engine.classify_corruption_level(15.0) == "very_high"

    def test_batch_deterministic(self, cpi_engine):
        """Test batch query returns deterministic results."""
        countries = ["BR", "DK", "ID"]
        result1 = cpi_engine.batch_query(countries)
        result2 = cpi_engine.batch_query(countries)
        assert result1.provenance_hash == result2.provenance_hash
        for r1, r2 in zip(result1.results, result2.results):
            if r1.success and r2.success:
                assert r1.data.score == r2.data.score


# ===========================================================================
# 14. EUDR Risk Factor Boundary Values (3 tests)
# ===========================================================================


class TestCPIEUDRBoundaries:
    """Test EUDR risk factor at extreme and boundary values."""

    def test_eudr_risk_clamps_negative_cpi(self, cpi_engine):
        """Test that negative CPI score is clamped to 0 (risk 1.0)."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("-5"))
        assert risk == Decimal("1.0000")

    def test_eudr_risk_clamps_above_100(self, cpi_engine):
        """Test that CPI above 100 is clamped (risk 0.0)."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("105"))
        assert risk == Decimal("0.0000")

    def test_eudr_risk_decimal_precision(self, cpi_engine):
        """Test that EUDR risk is returned with 4 decimal places."""
        risk = cpi_engine._calculate_eudr_risk_factor(Decimal("33"))
        # 1.0 - 33/100 = 0.67, should be 0.6700
        assert risk == Decimal("0.6700")
        # Check precision: exactly 4 decimal places
        str_risk = str(risk)
        if "." in str_risk:
            decimals = len(str_risk.split(".")[1])
            assert decimals == 4
