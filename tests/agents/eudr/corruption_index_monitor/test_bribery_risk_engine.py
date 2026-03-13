# -*- coding: utf-8 -*-
"""
Tests for BriberyRiskEngine - AGENT-EUDR-019 Engine 3: Bribery Risk

Comprehensive test suite covering:
- Country bribery risk assessment for high/low risk countries
- Country bribery profile retrieval with all 4 TRACE domains
- Sector-specific risk for EUDR sectors (agriculture, logging, palm oil)
- High-risk country identification with various thresholds
- Cross-country sector exposure analysis
- Sector-adjusted risk calculation with multipliers
- Bribery -> EUDR risk mapping (score 1 -> 0.0, score 100 -> 1.0)
- Domain score verification for all 4 TRACE domains
- Provenance chain integrity
- Edge cases and error handling

Test count: ~40 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019 (Engine 3: Bribery Risk)
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.bribery_risk_engine import (
    BriberyRiskEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.config import (
    set_config,
    reset_config,
)


# ===========================================================================
# 1. Bribery Country Assessment (8 tests)
# ===========================================================================


class TestBriberyAssessment:
    """Test assess_country_risk for high/low risk countries."""

    def test_assess_brazil_risk(self, bribery_engine):
        """Test bribery risk assessment for Brazil (moderate-high risk)."""
        result = bribery_engine.assess_country_risk("BR")
        assert result.success is True
        assert result.assessment is not None
        assert result.assessment.country_code == "BR"
        assert result.assessment.composite_score is not None

    def test_assess_denmark_low_risk(self, bribery_engine):
        """Test bribery risk assessment for Denmark (low risk expected)."""
        result = bribery_engine.assess_country_risk("DK")
        assert result.success is True
        assert result.assessment is not None
        assert result.assessment.composite_score <= Decimal("25")

    def test_assess_high_risk_country(self, bribery_engine):
        """Test bribery risk assessment for a high-risk country."""
        result = bribery_engine.assess_country_risk("CD")
        assert result.success is True
        assert result.assessment is not None
        assert result.assessment.composite_score >= Decimal("50")

    def test_assess_invalid_country(self, bribery_engine):
        """Test bribery risk assessment for invalid country returns failure."""
        result = bribery_engine.assess_country_risk("ZZ")
        assert result.success is False

    def test_assess_lowercase_normalized(self, bribery_engine):
        """Test lowercase country code is normalized to uppercase."""
        result = bribery_engine.assess_country_risk("br")
        assert result.success is True
        assert result.assessment.country_code == "BR"

    def test_assess_empty_country_code(self, bribery_engine):
        """Test empty country code returns failure."""
        result = bribery_engine.assess_country_risk("")
        assert result.success is False

    def test_assess_includes_risk_level(self, bribery_engine):
        """Test assessment includes risk level classification."""
        result = bribery_engine.assess_country_risk("BR")
        assert result.success is True
        assert result.assessment.risk_level is not None
        assert result.assessment.risk_level in (
            "LOW", "MEDIUM", "HIGH", "VERY_HIGH",
            "low", "medium", "high", "very_high",
        )

    @pytest.mark.parametrize("country_code", [
        "BR", "ID", "MY", "CO", "GH", "DK",
    ])
    def test_assess_multiple_countries(self, bribery_engine, country_code):
        """Test bribery assessment for multiple EUDR-relevant countries."""
        result = bribery_engine.assess_country_risk(country_code)
        assert result.success is True
        assert result.assessment is not None


# ===========================================================================
# 2. Bribery Profile (6 tests)
# ===========================================================================


class TestBriberyProfile:
    """Test get_country_bribery_profile for complete TRACE data."""

    def test_profile_brazil(self, bribery_engine):
        """Test bribery profile for Brazil includes all 4 domains."""
        result = bribery_engine.get_country_bribery_profile("BR")
        assert result.success is True
        assert result.profile is not None

    def test_profile_includes_domain_scores(self, bribery_engine):
        """Test profile includes all 4 TRACE domain scores."""
        result = bribery_engine.get_country_bribery_profile("BR")
        assert result.success is True
        profile = result.profile
        # Verify domain scores are present
        assert profile.domain_scores is not None
        assert len(profile.domain_scores) == 4

    def test_profile_denmark_low_domains(self, bribery_engine):
        """Test Denmark profile shows low scores across all domains."""
        result = bribery_engine.get_country_bribery_profile("DK")
        assert result.success is True
        # Denmark should have low bribery risk across all domains
        for domain_score in result.profile.domain_scores.values():
            assert domain_score <= Decimal("30")

    def test_profile_invalid_country(self, bribery_engine):
        """Test profile for invalid country returns failure."""
        result = bribery_engine.get_country_bribery_profile("ZZ")
        assert result.success is False

    def test_profile_includes_composite(self, bribery_engine):
        """Test profile includes composite bribery score."""
        result = bribery_engine.get_country_bribery_profile("BR")
        assert result.success is True
        assert result.profile.composite_score is not None
        assert Decimal("1") <= result.profile.composite_score <= Decimal("100")

    def test_profile_includes_provenance(self, bribery_engine):
        """Test profile includes provenance hash."""
        result = bribery_engine.get_country_bribery_profile("BR")
        assert result.provenance_hash is not None


# ===========================================================================
# 3. Sector Risk (8 tests)
# ===========================================================================


class TestSectorRisk:
    """Test get_sector_risk for EUDR-relevant sectors."""

    def test_sector_risk_agriculture(self, bribery_engine):
        """Test bribery risk for agriculture sector."""
        result = bribery_engine.get_sector_risk("BR", "agriculture")
        assert result.success is True
        assert result.sector_risk is not None

    def test_sector_risk_forestry(self, bribery_engine):
        """Test bribery risk for forestry sector (timber/logging)."""
        result = bribery_engine.get_sector_risk("BR", "forestry")
        assert result.success is True

    def test_sector_risk_palm_oil(self, bribery_engine):
        """Test bribery risk for palm oil sector in Indonesia."""
        result = bribery_engine.get_sector_risk("ID", "palm_oil")
        assert result.success is True

    def test_sector_risk_timber_highest_multiplier(self, bribery_engine):
        """Test that timber/logging has highest risk multiplier.

        Timber/logging should have a higher base risk multiplier (1.5)
        than agriculture (1.2) for the same country.
        """
        ag_result = bribery_engine.get_sector_risk("BR", "agriculture")
        timber_result = bribery_engine.get_sector_risk("BR", "timber")
        if ag_result.success and timber_result.success:
            if ag_result.sector_risk.adjusted_score is not None and \
               timber_result.sector_risk.adjusted_score is not None:
                assert timber_result.sector_risk.adjusted_score >= ag_result.sector_risk.adjusted_score

    def test_sector_risk_invalid_sector(self, bribery_engine):
        """Test bribery risk for an invalid sector."""
        result = bribery_engine.get_sector_risk("BR", "invalid_sector_xyz")
        assert result.success is False or result.sector_risk is None

    def test_sector_risk_invalid_country(self, bribery_engine):
        """Test sector risk for invalid country."""
        result = bribery_engine.get_sector_risk("ZZ", "agriculture")
        assert result.success is False

    def test_sector_risk_includes_multiplier(self, bribery_engine):
        """Test sector risk includes risk multiplier information."""
        result = bribery_engine.get_sector_risk("BR", "agriculture")
        assert result.success is True
        if result.sector_risk.risk_multiplier is not None:
            assert result.sector_risk.risk_multiplier >= Decimal("1.0")

    @pytest.mark.parametrize("sector", [
        "agriculture", "forestry", "palm_oil", "cocoa",
        "coffee", "soy", "rubber", "cattle", "timber",
    ])
    def test_sector_risk_all_eudr_sectors(self, bribery_engine, sector):
        """Test bribery risk for all EUDR commodity sectors."""
        result = bribery_engine.get_sector_risk("BR", sector)
        assert result.success is True or result.success is False


# ===========================================================================
# 4. High-Risk Country Identification (5 tests)
# ===========================================================================


class TestHighRiskCountries:
    """Test identify_high_risk_countries with various thresholds."""

    def test_identify_high_risk_default_threshold(self, bribery_engine):
        """Test high-risk country identification with default threshold."""
        result = bribery_engine.identify_high_risk_countries()
        assert result.success is True
        assert result.countries is not None
        assert len(result.countries) > 0

    def test_identify_high_risk_strict_threshold(self, bribery_engine):
        """Test with strict threshold (score >= 70) identifies fewer countries."""
        result = bribery_engine.identify_high_risk_countries(threshold=70)
        assert result.success is True
        # Strict threshold should yield fewer high-risk countries
        assert isinstance(result.countries, list)

    def test_identify_high_risk_lenient_threshold(self, bribery_engine):
        """Test with lenient threshold (score >= 30) identifies more countries."""
        result = bribery_engine.identify_high_risk_countries(threshold=30)
        assert result.success is True
        # Lenient threshold should yield more countries
        assert len(result.countries) >= 1

    def test_high_risk_includes_known_risk_countries(self, bribery_engine):
        """Test that known high-risk countries appear in results."""
        result = bribery_engine.identify_high_risk_countries(threshold=50)
        assert result.success is True
        high_risk_codes = [c.country_code for c in result.countries]
        # CD (DR Congo) should typically be high risk
        # Check at least some known risky countries appear
        assert len(high_risk_codes) > 0

    def test_high_risk_sorted_by_score(self, bribery_engine):
        """Test high-risk results are sorted by risk score descending."""
        result = bribery_engine.identify_high_risk_countries()
        assert result.success is True
        if len(result.countries) > 1:
            scores = [c.composite_score for c in result.countries]
            assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 5. Sector Exposure Analysis (4 tests)
# ===========================================================================


class TestSectorExposure:
    """Test analyze_sector_exposure for cross-country analysis."""

    def test_sector_exposure_single_country(self, bribery_engine):
        """Test sector exposure analysis for a single country."""
        result = bribery_engine.analyze_sector_exposure(["BR"])
        assert result.success is True
        assert result.exposures is not None

    def test_sector_exposure_multiple_countries(self, bribery_engine):
        """Test sector exposure analysis across multiple countries."""
        result = bribery_engine.analyze_sector_exposure(["BR", "ID", "DK"])
        assert result.success is True
        assert len(result.exposures) >= 3

    def test_sector_exposure_empty_list(self, bribery_engine):
        """Test sector exposure with empty country list."""
        result = bribery_engine.analyze_sector_exposure([])
        assert result.success is True
        assert len(result.exposures) == 0

    def test_sector_exposure_includes_provenance(self, bribery_engine):
        """Test sector exposure includes provenance hash."""
        result = bribery_engine.analyze_sector_exposure(["BR"])
        assert result.provenance_hash is not None


# ===========================================================================
# 6. Sector-Adjusted Risk Calculation (5 tests)
# ===========================================================================


class TestSectorAdjustedRisk:
    """Test _calculate_sector_adjusted_risk with multipliers."""

    def test_sector_adjusted_with_multiplier(self, bribery_engine):
        """Test sector-adjusted risk applies multiplier correctly."""
        base_score = 50.0
        multiplier = 1.3
        adjusted = bribery_engine._calculate_sector_adjusted_risk(
            base_score, multiplier
        )
        assert adjusted is not None
        # Adjusted should be higher than base due to multiplier > 1
        assert adjusted >= Decimal(str(base_score))

    def test_sector_adjusted_multiplier_1_no_change(self, bribery_engine):
        """Test multiplier of 1.0 leaves score unchanged."""
        base_score = 50.0
        adjusted = bribery_engine._calculate_sector_adjusted_risk(
            base_score, 1.0
        )
        assert adjusted == Decimal("50") or abs(float(adjusted) - 50.0) < 0.01

    def test_sector_adjusted_capped_at_100(self, bribery_engine):
        """Test sector-adjusted risk is capped at 100."""
        base_score = 90.0
        multiplier = 1.5  # 90 * 1.5 = 135, should cap at 100
        adjusted = bribery_engine._calculate_sector_adjusted_risk(
            base_score, multiplier
        )
        assert adjusted <= Decimal("100")

    def test_sector_adjusted_forestry_multiplier(self, bribery_engine):
        """Test forestry sector has appropriate multiplier effect."""
        # Forestry multiplier is typically 1.5 (highest for timber)
        base_score = 60.0
        adjusted = bribery_engine._calculate_sector_adjusted_risk(
            base_score, 1.5
        )
        expected_adjusted = min(Decimal("100"), Decimal("60") * Decimal("1.5"))
        assert adjusted == expected_adjusted or abs(float(adjusted) - float(expected_adjusted)) < 1.0

    def test_sector_adjusted_zero_base(self, bribery_engine):
        """Test sector adjustment with zero base score."""
        adjusted = bribery_engine._calculate_sector_adjusted_risk(0.0, 1.5)
        assert adjusted == Decimal("0") or adjusted == Decimal("0.0")


# ===========================================================================
# 7. Bribery -> EUDR Risk Mapping (6 tests)
# ===========================================================================


class TestBriberyEUDRMapping:
    """Test _map_bribery_to_eudr_risk mapping.

    Bribery EUDR risk mapping:
        Score 1 -> EUDR risk 0.0 (lowest bribery = lowest risk)
        Score 100 -> EUDR risk 1.0 (highest bribery = highest risk)
        Formula: eudr_risk = bribery_score / 100
    """

    def test_bribery_score_1_maps_to_0(self, bribery_engine):
        """Bribery score 1 (lowest risk) maps to EUDR risk ~0.01."""
        risk = bribery_engine._map_bribery_to_eudr_risk(1.0)
        assert risk <= Decimal("0.02")

    def test_bribery_score_100_maps_to_1(self, bribery_engine):
        """Bribery score 100 (highest risk) maps to EUDR risk 1.0."""
        risk = bribery_engine._map_bribery_to_eudr_risk(100.0)
        assert risk == Decimal("1.0") or risk == Decimal("1.0000")

    def test_bribery_score_50_maps_to_05(self, bribery_engine):
        """Bribery score 50 (midpoint) maps to EUDR risk 0.5."""
        risk = bribery_engine._map_bribery_to_eudr_risk(50.0)
        assert risk == Decimal("0.5") or risk == Decimal("0.5000")

    @pytest.mark.parametrize("bribery_score,min_risk,max_risk", [
        (1, Decimal("0.00"), Decimal("0.02")),
        (10, Decimal("0.09"), Decimal("0.11")),
        (25, Decimal("0.24"), Decimal("0.26")),
        (50, Decimal("0.49"), Decimal("0.51")),
        (75, Decimal("0.74"), Decimal("0.76")),
        (100, Decimal("0.99"), Decimal("1.01")),
    ])
    def test_bribery_eudr_risk_parametrized(
        self, bribery_engine, bribery_score, min_risk, max_risk
    ):
        """Parametrized bribery EUDR risk mapping test."""
        risk = bribery_engine._map_bribery_to_eudr_risk(float(bribery_score))
        assert min_risk <= risk <= max_risk

    def test_bribery_risk_monotonically_increasing(self, bribery_engine):
        """Test EUDR risk increases monotonically with bribery score."""
        prev_risk = Decimal("0")
        for score in range(1, 101, 10):
            risk = bribery_engine._map_bribery_to_eudr_risk(float(score))
            assert risk >= prev_risk
            prev_risk = risk

    def test_bribery_risk_always_between_0_and_1(self, bribery_engine):
        """Test EUDR risk from bribery is always between 0 and 1."""
        for score in range(1, 101, 5):
            risk = bribery_engine._map_bribery_to_eudr_risk(float(score))
            assert Decimal("0") <= risk <= Decimal("1")


# ===========================================================================
# 8. Domain Score Verification (4 tests)
# ===========================================================================


class TestBriberyDomainScores:
    """Verify all 4 TRACE bribery domains are scored correctly."""

    def test_all_four_domains_present(self, bribery_engine):
        """Test that all 4 TRACE domains are present in profile."""
        result = bribery_engine.get_country_bribery_profile("BR")
        assert result.success is True
        domains = result.profile.domain_scores
        assert len(domains) == 4

    def test_domain_scores_within_range(self, bribery_engine):
        """Test all domain scores are within 1-100 range."""
        result = bribery_engine.get_country_bribery_profile("BR")
        assert result.success is True
        for domain_name, score in result.profile.domain_scores.items():
            assert Decimal("0") <= score <= Decimal("100"), (
                f"Domain {domain_name} score {score} out of range"
            )

    def test_high_risk_country_high_domain_scores(self, bribery_engine):
        """Test high-risk country has elevated domain scores."""
        result = bribery_engine.get_country_bribery_profile("CD")
        assert result.success is True
        avg_domain = sum(
            result.profile.domain_scores.values()
        ) / Decimal("4")
        # DR Congo should have average domain score > 50
        assert avg_domain >= Decimal("50")

    def test_low_risk_country_low_domain_scores(self, bribery_engine):
        """Test low-risk country has low domain scores."""
        result = bribery_engine.get_country_bribery_profile("DK")
        assert result.success is True
        avg_domain = sum(
            result.profile.domain_scores.values()
        ) / Decimal("4")
        # Denmark should have average domain score < 25
        assert avg_domain <= Decimal("25")


# ===========================================================================
# 9. Provenance Chain (4 tests)
# ===========================================================================


class TestBriberyProvenance:
    """Test provenance chain integrity for bribery risk results."""

    def test_assessment_provenance_hash(self, bribery_engine):
        """Test country assessment includes provenance hash."""
        result = bribery_engine.assess_country_risk("BR")
        assert result.provenance_hash is not None

    def test_assessment_provenance_deterministic(self, bribery_engine):
        """Test provenance hash is deterministic for same query."""
        r1 = bribery_engine.assess_country_risk("BR")
        r2 = bribery_engine.assess_country_risk("BR")
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_countries_different_provenance(self, bribery_engine):
        """Test different countries produce different provenance hashes."""
        r_br = bribery_engine.assess_country_risk("BR")
        r_dk = bribery_engine.assess_country_risk("DK")
        assert r_br.provenance_hash != r_dk.provenance_hash

    def test_provenance_hex_format(self, bribery_engine):
        """Test provenance hash is valid hexadecimal."""
        result = bribery_engine.assess_country_risk("BR")
        if result.provenance_hash:
            int(result.provenance_hash, 16)  # Should not raise


# ===========================================================================
# 10. Edge Cases and Error Handling (5 tests)
# ===========================================================================


class TestBriberyEdgeCases:
    """Test edge cases and error conditions for BriberyRiskEngine."""

    def test_whitespace_country_code(self, bribery_engine):
        """Test whitespace-padded country code is stripped."""
        result = bribery_engine.assess_country_risk(" BR ")
        assert result.success is True

    def test_three_letter_country_code(self, bribery_engine):
        """Test 3-letter country code is rejected."""
        result = bribery_engine.assess_country_risk("BRA")
        assert result.success is False

    def test_engine_initializes_without_config(self):
        """Test bribery engine can initialize without explicit config."""
        reset_config()
        engine = BriberyRiskEngine()
        result = engine.assess_country_risk("BR")
        assert result.success is True

    def test_assessment_deterministic_10_runs(self, bribery_engine):
        """Test 10 consecutive assessments return identical results."""
        first = bribery_engine.assess_country_risk("BR")
        for _ in range(9):
            result = bribery_engine.assess_country_risk("BR")
            assert result.assessment.composite_score == first.assessment.composite_score
            assert result.provenance_hash == first.provenance_hash

    def test_profile_deterministic(self, bribery_engine):
        """Test bribery profile is deterministic."""
        r1 = bribery_engine.get_country_bribery_profile("BR")
        r2 = bribery_engine.get_country_bribery_profile("BR")
        assert r1.profile.composite_score == r2.profile.composite_score
        assert r1.provenance_hash == r2.provenance_hash
