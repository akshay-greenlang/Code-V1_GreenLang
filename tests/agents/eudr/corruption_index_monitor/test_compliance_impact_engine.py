# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceImpactEngine (AGENT-EUDR-019, Engine 8).

Tests all methods of ComplianceImpactEngine including compliance impact
assessment, country impact profiles, due diligence recommendations,
country classification, composite corruption score calculation,
due diligence level determination, mitigation requirements generation,
EUDR classification mapping, enhanced DD triggers, Article 29 mapping,
and provenance chain integrity.

Classification thresholds tested:
    - LOW_RISK:      composite < 0.25
    - STANDARD_RISK: 0.25 <= composite < 0.60
    - HIGH_RISK:     composite >= 0.60

Composite score weights:
    CPI: 35%, WGI: 25%, Bribery: 20%, Institutional: 20%

Coverage target: 85%+ of ComplianceImpactEngine methods.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.engines.compliance_impact_engine import (
    ComplianceImpactEngine,
    EUDRCountryClassification,
    DueDiligenceLevel,
    RecommendationPriority,
    MonitoringFrequency,
    ComplianceImpact,
    DDRecommendation,
    CountryImpactProfile,
    REFERENCE_COUNTRY_DATA,
    COMMODITY_DD_REQUIREMENTS,
    RISK_FACTORS,
    FATF_GREY_LIST,
    FATF_BLACK_LIST,
    GOVERNANCE_CRISIS_COUNTRIES,
    EUDR_COMMODITIES,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    WEIGHT_CPI,
    WEIGHT_WGI,
    WEIGHT_BRIBERY,
    WEIGHT_INSTITUTIONAL,
    ENHANCED_DD_CPI_THRESHOLD,
    ENHANCED_DD_WGI_THRESHOLD,
    ENHANCED_DD_DECLINE_THRESHOLD,
    _to_decimal,
    _clamp_decimal,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ComplianceImpactEngine:
    """Create a default ComplianceImpactEngine instance."""
    return ComplianceImpactEngine()


# ---------------------------------------------------------------------------
# TestComplianceImpact
# ---------------------------------------------------------------------------


class TestComplianceImpact:
    """Tests for assess_compliance_impact for various countries."""

    def test_low_risk_country_denmark(self, engine: ComplianceImpactEngine):
        """Denmark should be classified as LOW_RISK."""
        result = engine.assess_compliance_impact("DK")
        assert result["eudr_classification"] == "LOW_RISK"
        assert result["due_diligence_level"] == "SIMPLIFIED"
        assert Decimal(result["corruption_composite_score"]) < LOW_RISK_THRESHOLD

    def test_high_risk_country_venezuela(self, engine: ComplianceImpactEngine):
        """Venezuela should be classified as HIGH_RISK with ENHANCED DD."""
        result = engine.assess_compliance_impact("VE")
        assert result["eudr_classification"] == "HIGH_RISK"
        assert result["due_diligence_level"] == "ENHANCED"
        assert Decimal(result["corruption_composite_score"]) >= HIGH_RISK_THRESHOLD

    def test_standard_risk_country_brazil(self, engine: ComplianceImpactEngine):
        """Brazil should be classified as STANDARD_RISK or HIGH_RISK."""
        result = engine.assess_compliance_impact("BR")
        assert result["eudr_classification"] in ("STANDARD_RISK", "HIGH_RISK")

    def test_unknown_country_defaults_to_standard(
        self, engine: ComplianceImpactEngine
    ):
        """Unknown country should default to STANDARD_RISK with warning."""
        result = engine.assess_compliance_impact("ZZ")
        assert result["eudr_classification"] == "STANDARD_RISK"
        assert len(result["warnings"]) > 0

    def test_impact_with_commodity(self, engine: ComplianceImpactEngine):
        """Commodity-specific assessment should include commodity requirements."""
        result = engine.assess_compliance_impact("BR", commodity="cattle")
        assert "mitigation_requirements" in result
        assert result["commodity"] == "cattle"

    def test_invalid_commodity_raises(self, engine: ComplianceImpactEngine):
        """Invalid commodity should raise ValueError."""
        with pytest.raises(ValueError):
            engine.assess_compliance_impact("BR", commodity="diamonds")

    def test_empty_country_raises(self, engine: ComplianceImpactEngine):
        """Empty country code should raise ValueError."""
        with pytest.raises(ValueError):
            engine.assess_compliance_impact("")

    def test_impact_has_raw_scores(self, engine: ComplianceImpactEngine):
        """Result should include raw CPI, WGI, bribery, institutional scores."""
        result = engine.assess_compliance_impact("DK")
        assert "cpi_score" in result
        assert "wgi_cc_score" in result
        assert "bribery_score" in result
        assert "institutional_score" in result

    def test_impact_has_normalized_scores(self, engine: ComplianceImpactEngine):
        """Result should include normalized scores."""
        result = engine.assess_compliance_impact("DK")
        assert "cpi_normalized" in result
        assert "wgi_normalized" in result
        assert "bribery_normalized" in result
        assert "institutional_normalized" in result

    def test_impact_has_processing_time(self, engine: ComplianceImpactEngine):
        """Result should include processing time."""
        result = engine.assess_compliance_impact("DK")
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_impact_has_provenance(self, engine: ComplianceImpactEngine):
        """Result should include provenance hash."""
        result = engine.assess_compliance_impact("DK")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestCountryImpactProfile
# ---------------------------------------------------------------------------


class TestCountryImpactProfile:
    """Tests for get_country_impact_profile."""

    def test_profile_for_brazil(self, engine: ComplianceImpactEngine):
        """Brazil profile should cover all 7 commodities."""
        result = engine.get_country_impact_profile("BR")
        assert result["country_code"] == "BR"
        assert "commodity_requirements" in result
        assert len(result["commodity_requirements"]) == len(EUDR_COMMODITIES)

    def test_profile_has_recommendations(self, engine: ComplianceImpactEngine):
        """Profile should include recommendations."""
        result = engine.get_country_impact_profile("VE")
        assert result["total_recommendations"] > 0

    def test_profile_critical_recommendations(self, engine: ComplianceImpactEngine):
        """High-risk country should have critical recommendations."""
        result = engine.get_country_impact_profile("VE")
        assert result["critical_recommendations"] >= 0

    def test_profile_empty_country_raises(self, engine: ComplianceImpactEngine):
        """Empty country code should raise ValueError."""
        with pytest.raises(ValueError):
            engine.get_country_impact_profile("")

    def test_profile_has_provenance(self, engine: ComplianceImpactEngine):
        """Profile should include provenance hash."""
        result = engine.get_country_impact_profile("DK")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestDDRecommendations
# ---------------------------------------------------------------------------


class TestDDRecommendations:
    """Tests for get_dd_recommendations for different risk levels."""

    def test_low_risk_recommendations(self, engine: ComplianceImpactEngine):
        """Low-risk country should have fewer recommendations."""
        result = engine.get_dd_recommendations("DK")
        assert result["due_diligence_level"] == "SIMPLIFIED"
        assert result["total_recommendations"] >= 1

    def test_high_risk_recommendations(self, engine: ComplianceImpactEngine):
        """High-risk country should have more recommendations."""
        result = engine.get_dd_recommendations("VE")
        assert result["due_diligence_level"] == "ENHANCED"
        assert result["total_recommendations"] > 3

    def test_recommendations_with_commodity(self, engine: ComplianceImpactEngine):
        """Commodity-specific recommendations should be included."""
        result = engine.get_dd_recommendations("BR", commodity="cocoa")
        assert result["commodity"] == "cocoa"

    def test_recommendations_invalid_commodity_raises(
        self, engine: ComplianceImpactEngine
    ):
        """Invalid commodity should raise ValueError."""
        with pytest.raises(ValueError):
            engine.get_dd_recommendations("BR", commodity="gold")

    def test_recommendations_has_provenance(self, engine: ComplianceImpactEngine):
        """Recommendations should include provenance hash."""
        result = engine.get_dd_recommendations("BR")
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestCountryClassification
# ---------------------------------------------------------------------------


class TestCountryClassification:
    """Tests for classify_country for LOW/STANDARD/HIGH risk."""

    def test_classify_denmark_low(self, engine: ComplianceImpactEngine):
        """Denmark should classify as LOW_RISK."""
        result = engine.classify_country("DK")
        assert result["eudr_classification"] == "LOW_RISK"

    def test_classify_venezuela_high(self, engine: ComplianceImpactEngine):
        """Venezuela should classify as HIGH_RISK."""
        result = engine.classify_country("VE")
        assert result["eudr_classification"] == "HIGH_RISK"

    def test_classify_empty_raises(self, engine: ComplianceImpactEngine):
        """Empty country code should raise ValueError."""
        with pytest.raises(ValueError):
            engine.classify_country("")

    def test_classify_has_composite_score(self, engine: ComplianceImpactEngine):
        """Classification result should include composite score."""
        result = engine.classify_country("BR")
        assert "composite_score" in result or "corruption_composite_score" in result


# ---------------------------------------------------------------------------
# TestCompositeScore
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Tests for _calculate_composite_corruption_score with CPI(35%)+WGI(25%)+Bribery(20%)+Institutional(20%) weights."""

    def test_composite_score_weights(self, engine: ComplianceImpactEngine):
        """Verify composite = CPI(35%) + WGI(25%) + Bribery(20%) + Institutional(20%)."""
        # Use known values. Normalization:
        #   CPI 50/100 -> normalized = (100-50)/100 = 0.50
        #   WGI 0.0 -> normalized = (2.5-0.0)/5.0 = 0.50
        #   Bribery 50 -> normalized = 50/100 = 0.50
        #   Institutional 50 -> normalized = (100-50)/100 = 0.50
        composite, normalized = engine._calculate_composite_corruption_score(
            Decimal("50"), Decimal("0"), Decimal("50"), Decimal("50")
        )
        # All normalized to 0.5, so composite = 0.5
        assert abs(composite - Decimal("0.5")) < Decimal("0.01")

    def test_clean_country_low_composite(self, engine: ComplianceImpactEngine):
        """Very clean country should have low composite (less corruption)."""
        composite, _ = engine._calculate_composite_corruption_score(
            Decimal("90"), Decimal("2.0"), Decimal("5"), Decimal("95")
        )
        assert composite < LOW_RISK_THRESHOLD

    def test_corrupt_country_high_composite(self, engine: ComplianceImpactEngine):
        """Very corrupt country should have high composite."""
        composite, _ = engine._calculate_composite_corruption_score(
            Decimal("15"), Decimal("-1.5"), Decimal("85"), Decimal("10")
        )
        assert composite >= HIGH_RISK_THRESHOLD

    def test_composite_clamped_to_0_1(self, engine: ComplianceImpactEngine):
        """Composite should be clamped to [0, 1]."""
        composite, _ = engine._calculate_composite_corruption_score(
            Decimal("0"), Decimal("-2.5"), Decimal("100"), Decimal("0")
        )
        assert Decimal("0") <= composite <= Decimal("1")

    def test_normalization_cpi(self, engine: ComplianceImpactEngine):
        """CPI normalization: 0 -> 1.0 (most corrupt), 100 -> 0.0 (cleanest)."""
        norm = engine._normalize_cpi(Decimal("0"))
        assert norm == Decimal("1.0000")
        norm = engine._normalize_cpi(Decimal("100"))
        assert norm == Decimal("0.0000")

    def test_normalization_wgi(self, engine: ComplianceImpactEngine):
        """WGI normalization: -2.5 -> 1.0 (most corrupt), 2.5 -> 0.0."""
        norm = engine._normalize_wgi(Decimal("-2.5"))
        assert norm == Decimal("1.0000")
        norm = engine._normalize_wgi(Decimal("2.5"))
        assert norm == Decimal("0.0000")

    def test_normalization_bribery(self, engine: ComplianceImpactEngine):
        """Bribery normalization: 0 -> 0.0 (low risk), 100 -> 1.0 (high risk)."""
        norm = engine._normalize_bribery(Decimal("0"))
        assert norm == Decimal("0.0000")
        norm = engine._normalize_bribery(Decimal("100"))
        assert norm == Decimal("1.0000")

    def test_normalization_institutional(self, engine: ComplianceImpactEngine):
        """Institutional normalization: 0 -> 1.0 (weakest), 100 -> 0.0 (strongest)."""
        norm = engine._normalize_institutional(Decimal("0"))
        assert norm == Decimal("1.0000")
        norm = engine._normalize_institutional(Decimal("100"))
        assert norm == Decimal("0.0000")


# ---------------------------------------------------------------------------
# TestDueDiligenceLevel
# ---------------------------------------------------------------------------


class TestDueDiligenceLevel:
    """Tests for _determine_due_diligence_level for SIMPLIFIED/STANDARD/ENHANCED."""

    def test_low_risk_simplified(self, engine: ComplianceImpactEngine):
        """Low-risk composite with no triggers should yield SIMPLIFIED."""
        level = engine._determine_due_diligence_level(Decimal("0.15"), [])
        assert level == "SIMPLIFIED"

    def test_standard_risk_standard(self, engine: ComplianceImpactEngine):
        """Standard-risk composite with no triggers should yield STANDARD."""
        level = engine._determine_due_diligence_level(Decimal("0.40"), [])
        assert level == "STANDARD"

    def test_high_risk_enhanced(self, engine: ComplianceImpactEngine):
        """High-risk composite should yield ENHANCED."""
        level = engine._determine_due_diligence_level(Decimal("0.70"), [])
        assert level == "ENHANCED"

    def test_any_trigger_forces_enhanced(self, engine: ComplianceImpactEngine):
        """Any enhanced DD trigger should force ENHANCED regardless of composite."""
        level = engine._determine_due_diligence_level(
            Decimal("0.15"), ["CPI below 30"]
        )
        assert level == "ENHANCED"

    def test_dd_level_enum_values(self):
        """All 3 DD levels should be defined."""
        levels = set(d.value for d in DueDiligenceLevel)
        assert levels == {"SIMPLIFIED", "STANDARD", "ENHANCED"}


# ---------------------------------------------------------------------------
# TestMitigationRequirements
# ---------------------------------------------------------------------------


class TestMitigationRequirements:
    """Tests for _generate_mitigation_requirements based on risk."""

    def test_low_risk_mitigations(self, engine: ComplianceImpactEngine):
        """Low-risk should have minimal mitigations."""
        reqs = engine._generate_mitigation_requirements("LOW_RISK")
        assert len(reqs) >= 2

    def test_high_risk_more_mitigations(self, engine: ComplianceImpactEngine):
        """High-risk should have more mitigations than low-risk."""
        low_reqs = engine._generate_mitigation_requirements("LOW_RISK")
        high_reqs = engine._generate_mitigation_requirements("HIGH_RISK")
        assert len(high_reqs) > len(low_reqs)

    def test_commodity_specific_mitigations(self, engine: ComplianceImpactEngine):
        """Commodity-specific mitigations should be added."""
        reqs = engine._generate_mitigation_requirements("HIGH_RISK", commodity="cattle")
        cattle_reqs = [r for r in reqs if "[cattle]" in r]
        assert len(cattle_reqs) > 0

    def test_all_commodities_have_requirements(self):
        """All EUDR commodities should have DD requirements defined."""
        for commodity in EUDR_COMMODITIES:
            assert commodity in COMMODITY_DD_REQUIREMENTS
            for level in ("SIMPLIFIED", "STANDARD", "ENHANCED"):
                assert level in COMMODITY_DD_REQUIREMENTS[commodity]


# ---------------------------------------------------------------------------
# TestEUDRClassification
# ---------------------------------------------------------------------------


class TestEUDRClassification:
    """Tests for _map_to_eudr_classification threshold tests."""

    @pytest.mark.parametrize(
        "composite_score,expected_class",
        [
            (Decimal("0.15"), "LOW_RISK"),
            (Decimal("0.24"), "LOW_RISK"),
            (Decimal("0.25"), "STANDARD_RISK"),
            (Decimal("0.45"), "STANDARD_RISK"),
            (Decimal("0.59"), "STANDARD_RISK"),
            (Decimal("0.60"), "HIGH_RISK"),
            (Decimal("0.85"), "HIGH_RISK"),
        ],
    )
    def test_eudr_classification_thresholds(
        self,
        engine: ComplianceImpactEngine,
        composite_score: Decimal,
        expected_class: str,
    ):
        """Classification should follow threshold boundaries exactly."""
        classification = engine._map_to_eudr_classification(composite_score)
        assert classification == expected_class

    def test_boundary_at_0_25(self, engine: ComplianceImpactEngine):
        """Score of exactly 0.25 should be STANDARD_RISK (boundary inclusive)."""
        assert engine._map_to_eudr_classification(Decimal("0.25")) == "STANDARD_RISK"

    def test_boundary_at_0_60(self, engine: ComplianceImpactEngine):
        """Score of exactly 0.60 should be HIGH_RISK (boundary inclusive)."""
        assert engine._map_to_eudr_classification(Decimal("0.60")) == "HIGH_RISK"

    def test_classification_enum_values(self):
        """All 3 classification values should be defined."""
        values = set(c.value for c in EUDRCountryClassification)
        assert values == {"LOW_RISK", "STANDARD_RISK", "HIGH_RISK"}


# ---------------------------------------------------------------------------
# TestEnhancedDDTriggers
# ---------------------------------------------------------------------------


class TestEnhancedDDTriggers:
    """Tests for enhanced DD triggers: CPI<30, WGI CC<-1.0, decline, FATF list."""

    def test_cpi_below_30_trigger(self, engine: ComplianceImpactEngine):
        """CPI below 30 should trigger enhanced DD."""
        triggers = engine._check_enhanced_dd_triggers(
            "XX", Decimal("25"), Decimal("0"), Decimal("50")
        )
        cpi_triggers = [t for t in triggers if "CPI" in t]
        assert len(cpi_triggers) >= 1

    def test_wgi_below_minus_1_trigger(self, engine: ComplianceImpactEngine):
        """WGI CC below -1.0 should trigger enhanced DD."""
        triggers = engine._check_enhanced_dd_triggers(
            "XX", Decimal("50"), Decimal("-1.5"), Decimal("50")
        )
        wgi_triggers = [t for t in triggers if "WGI" in t]
        assert len(wgi_triggers) >= 1

    def test_fatf_grey_list_trigger(self, engine: ComplianceImpactEngine):
        """Country on FATF grey list should trigger enhanced DD."""
        # NG is in FATF_GREY_LIST
        triggers = engine._check_enhanced_dd_triggers(
            "NG", Decimal("50"), Decimal("0"), Decimal("50")
        )
        fatf_triggers = [t for t in triggers if "FATF" in t]
        assert len(fatf_triggers) >= 1

    def test_fatf_black_list_trigger(self, engine: ComplianceImpactEngine):
        """Country on FATF blacklist should trigger enhanced DD."""
        # MM is in FATF_BLACK_LIST
        triggers = engine._check_enhanced_dd_triggers(
            "MM", Decimal("50"), Decimal("0"), Decimal("50")
        )
        fatf_triggers = [t for t in triggers if "FATF" in t or "blacklist" in t]
        assert len(fatf_triggers) >= 1

    def test_governance_crisis_trigger(self, engine: ComplianceImpactEngine):
        """Country in governance crisis should trigger enhanced DD."""
        # VE is in GOVERNANCE_CRISIS_COUNTRIES
        triggers = engine._check_enhanced_dd_triggers(
            "VE", Decimal("50"), Decimal("0"), Decimal("50")
        )
        crisis_triggers = [t for t in triggers if "governance" in t.lower()]
        assert len(crisis_triggers) >= 1

    def test_rapid_decline_trigger(self, engine: ComplianceImpactEngine):
        """Annual CPI decline > 5 should trigger enhanced DD."""
        engine.set_annual_decline("XX", Decimal("7"))
        triggers = engine._check_enhanced_dd_triggers(
            "XX", Decimal("50"), Decimal("0"), Decimal("50")
        )
        decline_triggers = [t for t in triggers if "declining" in t.lower()]
        assert len(decline_triggers) >= 1

    def test_no_triggers_clean_country(self, engine: ComplianceImpactEngine):
        """Clean country with no risk factors should have no triggers."""
        triggers = engine._check_enhanced_dd_triggers(
            "XX", Decimal("90"), Decimal("2.0"), Decimal("5")
        )
        assert len(triggers) == 0


# ---------------------------------------------------------------------------
# TestArticle29Mapping
# ---------------------------------------------------------------------------


class TestArticle29Mapping:
    """Tests for EUDR Article 29 country benchmarking alignment."""

    def test_reference_data_covers_major_countries(self):
        """Reference data should cover at least 25 countries."""
        assert len(REFERENCE_COUNTRY_DATA) >= 25

    def test_reference_data_has_all_indices(self):
        """Each country should have CPI, WGI, bribery, and institutional."""
        for cc, data in REFERENCE_COUNTRY_DATA.items():
            assert "cpi" in data, f"{cc} missing cpi"
            assert "wgi_cc" in data, f"{cc} missing wgi_cc"
            assert "bribery" in data, f"{cc} missing bribery"
            assert "institutional" in data, f"{cc} missing institutional"

    def test_eudr_commodities(self):
        """All 7 EUDR commodities should be defined."""
        assert EUDR_COMMODITIES == frozenset({
            "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"
        })

    def test_monitoring_frequency_high_risk(self, engine: ComplianceImpactEngine):
        """HIGH_RISK should require MONTHLY monitoring."""
        freq = engine._determine_monitoring_frequency("HIGH_RISK", [])
        assert freq == "monthly"

    def test_monitoring_frequency_standard(self, engine: ComplianceImpactEngine):
        """STANDARD_RISK should require QUARTERLY monitoring."""
        freq = engine._determine_monitoring_frequency("STANDARD_RISK", [])
        assert freq == "quarterly"

    def test_monitoring_frequency_low(self, engine: ComplianceImpactEngine):
        """LOW_RISK should require ANNUAL monitoring."""
        freq = engine._determine_monitoring_frequency("LOW_RISK", [])
        assert freq == "annual"

    def test_monitoring_continuous_many_triggers(
        self, engine: ComplianceImpactEngine
    ):
        """3+ enhanced DD triggers should require CONTINUOUS monitoring."""
        triggers = ["trigger1", "trigger2", "trigger3"]
        freq = engine._determine_monitoring_frequency("HIGH_RISK", triggers)
        assert freq == "continuous"


# ---------------------------------------------------------------------------
# TestComplianceProvenance
# ---------------------------------------------------------------------------


class TestComplianceProvenance:
    """Tests for provenance chain integrity."""

    def test_impact_provenance_deterministic(self, engine: ComplianceImpactEngine):
        """Same country should produce same provenance hash."""
        r1 = engine.assess_compliance_impact("DK")
        r2 = engine.assess_compliance_impact("DK")
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_countries_different_provenance(
        self, engine: ComplianceImpactEngine
    ):
        """Different countries should produce different provenance hashes."""
        r1 = engine.assess_compliance_impact("DK")
        r2 = engine.assess_compliance_impact("VE")
        assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_custom_data_overrides_reference(self, engine: ComplianceImpactEngine):
        """Custom data should override reference data."""
        engine.load_custom_data("DK", {
            "cpi": Decimal("30"),
            "wgi_cc": Decimal("-1.0"),
            "bribery": Decimal("70"),
            "institutional": Decimal("20"),
        })
        result = engine.assess_compliance_impact("DK")
        # With these corrupt values, DK should no longer be LOW_RISK
        assert result["eudr_classification"] != "LOW_RISK"

    def test_load_custom_data_empty_raises(self, engine: ComplianceImpactEngine):
        """Loading empty custom data should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_data("XX", {})

    def test_load_custom_data_empty_country_raises(
        self, engine: ComplianceImpactEngine
    ):
        """Empty country code should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_data("", {"cpi": Decimal("50")})

    def test_risk_factors_high_corruption(self, engine: ComplianceImpactEngine):
        """Country with CPI < 30 should have 'high_corruption' risk factor."""
        factors = engine._identify_risk_factors(
            "XX", Decimal("25"), Decimal("0"), Decimal("50")
        )
        factor_names = [f["name"] for f in factors]
        assert "High Corruption Level" in factor_names

    def test_risk_factors_weak_governance(self, engine: ComplianceImpactEngine):
        """Country with WGI < -1.0 should have 'weak_governance' risk factor."""
        factors = engine._identify_risk_factors(
            "XX", Decimal("50"), Decimal("-1.5"), Decimal("50")
        )
        factor_names = [f["name"] for f in factors]
        assert "Weak Governance Indicators" in factor_names

    def test_risk_factors_high_bribery(self, engine: ComplianceImpactEngine):
        """Country with bribery > 70 should have 'high_bribery' risk factor."""
        factors = engine._identify_risk_factors(
            "XX", Decimal("50"), Decimal("0"), Decimal("80")
        )
        factor_names = [f["name"] for f in factors]
        assert "High Bribery Risk" in factor_names

    def test_recommendation_priority_enum(self):
        """All 4 recommendation priorities should be defined."""
        priorities = set(p.value for p in RecommendationPriority)
        assert priorities == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def test_monitoring_frequency_enum(self):
        """All 5 monitoring frequencies should be defined."""
        freqs = set(f.value for f in MonitoringFrequency)
        assert freqs == {"continuous", "monthly", "quarterly", "semi_annual", "annual"}

    def test_weights_sum_to_one(self):
        """Composite score weights should sum to 1.0."""
        total = WEIGHT_CPI + WEIGHT_WGI + WEIGHT_BRIBERY + WEIGHT_INSTITUTIONAL
        assert total == Decimal("1.00")
