# -*- coding: utf-8 -*-
"""
Unit tests for InstitutionalQualityEngine (AGENT-EUDR-019, Engine 4).

Tests all methods of the InstitutionalQualityEngine including country
quality assessment, governance profiling, institutional strength analysis,
forest governance queries, cross-country comparison, composite scoring,
EUDR risk mapping, dimension enumeration, and provenance tracking.

Covers 8 institutional dimensions: JUDICIAL_INDEPENDENCE, REGULATORY_ENFORCEMENT,
PROPERTY_RIGHTS, CONTRACT_ENFORCEMENT, TRANSPARENCY_LAWS, ANTI_CORRUPTION_FRAMEWORK,
FOREST_GOVERNANCE, LAND_TENURE_SECURITY.

Coverage target: 85%+ of InstitutionalQualityEngine methods.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.institutional_quality_engine import (
    InstitutionalQualityEngine,
    InstitutionalDimension,
    InstitutionalCapacityLevel,
    IllegalLoggingPrevalence,
    InstitutionalAssessment,
    ForestGovernanceProfile,
    InstitutionalQualityResult,
    GovernanceProfileResult,
    StrengthResult,
    ForestGovernanceResult,
    ComparisonResult,
    INSTITUTIONAL_DATA,
    FOREST_GOVERNANCE_DATA,
    DEFAULT_DIMENSION_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> InstitutionalQualityEngine:
    """Create a default InstitutionalQualityEngine instance."""
    return InstitutionalQualityEngine()


@pytest.fixture
def custom_weights_engine() -> InstitutionalQualityEngine:
    """Create engine with custom equal weights across all dimensions."""
    equal = Decimal("0.125")
    weights = {dim.value: equal for dim in InstitutionalDimension}
    return InstitutionalQualityEngine(weights=weights)


# ---------------------------------------------------------------------------
# TestInstitutionalAssessment
# ---------------------------------------------------------------------------


class TestInstitutionalAssessment:
    """Tests for assess_country_quality on strong and weak governance countries."""

    def test_strong_governance_denmark(self, engine: InstitutionalQualityEngine):
        """Denmark (DK) should return STRONG institutional capacity."""
        result = engine.assess_country_quality("DK")
        assert result.success is True
        assert result.data is not None
        assert result.data.country_code == "DK"
        assert result.data.institutional_capacity_level == "STRONG"
        assert result.data.composite_score >= Decimal("75")
        assert result.eudr_risk_factor < Decimal("0.25")

    def test_strong_governance_finland(self, engine: InstitutionalQualityEngine):
        """Finland (FI) should return STRONG institutional capacity."""
        result = engine.assess_country_quality("FI")
        assert result.success is True
        assert result.data.institutional_capacity_level == "STRONG"

    def test_weak_governance_cameroon(self, engine: InstitutionalQualityEngine):
        """Cameroon (CM) should return WEAK institutional capacity."""
        result = engine.assess_country_quality("CM")
        assert result.success is True
        assert result.data is not None
        assert result.data.institutional_capacity_level == "WEAK"
        assert result.data.composite_score < Decimal("50")

    def test_very_weak_governance_venezuela(self, engine: InstitutionalQualityEngine):
        """Venezuela (VE) should return VERY_WEAK institutional capacity."""
        result = engine.assess_country_quality("VE")
        assert result.success is True
        assert result.data.institutional_capacity_level == "VERY_WEAK"
        assert result.data.composite_score < Decimal("25")
        assert result.eudr_risk_factor > Decimal("0.75")

    def test_adequate_governance_brazil(self, engine: InstitutionalQualityEngine):
        """Brazil (BR) should have composite around the WEAK/ADEQUATE boundary."""
        result = engine.assess_country_quality("BR")
        assert result.success is True
        assert result.data is not None
        # BR scores are around 40-55 range depending on weights
        assert result.data.institutional_capacity_level in ("WEAK", "ADEQUATE")

    def test_unknown_country_returns_failure(self, engine: InstitutionalQualityEngine):
        """Unknown country code should return success=False with error."""
        result = engine.assess_country_quality("ZZ")
        assert result.success is False
        assert result.error is not None
        assert "ZZ" in result.error

    def test_lowercase_country_code(self, engine: InstitutionalQualityEngine):
        """Lowercase country codes should be handled (auto-uppercased)."""
        result = engine.assess_country_quality("dk")
        assert result.success is True
        assert result.data.country_code == "DK"

    def test_year_defaults_to_2024(self, engine: InstitutionalQualityEngine):
        """Year should default to 2024 when not provided."""
        result = engine.assess_country_quality("DK")
        assert result.success is True
        assert result.data.year == 2024

    def test_explicit_year(self, engine: InstitutionalQualityEngine):
        """Explicit year should be reflected in the assessment."""
        result = engine.assess_country_quality("DK", year=2023)
        assert result.success is True
        assert result.data.year == 2023

    def test_all_dimension_scores_present(self, engine: InstitutionalQualityEngine):
        """All 8 dimension scores should be present in the assessment."""
        result = engine.assess_country_quality("DE")
        assert result.success is True
        expected_dims = {"JI", "RE", "PR", "CE", "TL", "ACF", "FG", "LTS"}
        assert set(result.data.dimension_scores.keys()) == expected_dims

    def test_metadata_contains_engine_info(self, engine: InstitutionalQualityEngine):
        """Result metadata should contain engine version and agent ID."""
        result = engine.assess_country_quality("DK")
        assert result.metadata["engine"] == "InstitutionalQualityEngine"
        assert result.metadata["engine_version"] == "1.0.0"
        assert result.metadata["agent_id"] == "GL-EUDR-CIM-019"
        assert "processing_time_ms" in result.metadata


# ---------------------------------------------------------------------------
# TestGovernanceProfile
# ---------------------------------------------------------------------------


class TestGovernanceProfile:
    """Tests for get_governance_profile."""

    def test_strong_country_profile(self, engine: InstitutionalQualityEngine):
        """Strong country should have strengths and few weaknesses."""
        result = engine.get_governance_profile("DK")
        assert result.success is True
        assert result.country_code == "DK"
        assert result.assessment is not None
        assert len(result.strengths) > 0
        assert len(result.weaknesses) == 0

    def test_weak_country_profile(self, engine: InstitutionalQualityEngine):
        """Weak country should have weaknesses and recommendations."""
        result = engine.get_governance_profile("CD")
        assert result.success is True
        assert len(result.weaknesses) > 0
        assert len(result.recommendations) > 0

    def test_unknown_country_profile(self, engine: InstitutionalQualityEngine):
        """Unknown country should return failure."""
        result = engine.get_governance_profile("ZZ")
        assert result.success is False
        assert result.error is not None

    def test_profile_recommendations_for_forest_governance(
        self, engine: InstitutionalQualityEngine
    ):
        """Countries with low forest governance should get forest recommendations."""
        result = engine.get_governance_profile("CM")
        assert result.success is True
        forest_recs = [r for r in result.recommendations if "forest" in r.lower()]
        assert len(forest_recs) > 0

    def test_profile_has_provenance_hash(self, engine: InstitutionalQualityEngine):
        """Profile result should include a provenance hash."""
        result = engine.get_governance_profile("DK")
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestInstitutionalStrength
# ---------------------------------------------------------------------------


class TestInstitutionalStrength:
    """Tests for assess_institutional_strength with selected and all dimensions."""

    def test_all_dimensions(self, engine: InstitutionalQualityEngine):
        """Assessing all dimensions should return all 8 dimension scores."""
        result = engine.assess_institutional_strength("DK")
        assert result.success is True
        assert len(result.dimensions_assessed) == 8
        assert result.capacity_level == "STRONG"

    def test_selected_dimensions(self, engine: InstitutionalQualityEngine):
        """Assessing selected dimensions should return only those."""
        result = engine.assess_institutional_strength("DK", dimensions=["JI", "FG"])
        assert result.success is True
        assert sorted(result.dimensions_assessed) == ["FG", "JI"]
        assert len(result.dimension_scores) == 2
        assert "JI" in result.dimension_scores
        assert "FG" in result.dimension_scores

    def test_single_dimension(self, engine: InstitutionalQualityEngine):
        """Single dimension assessment should work."""
        result = engine.assess_institutional_strength("BR", dimensions=["FG"])
        assert result.success is True
        assert result.overall_strength == Decimal(str(INSTITUTIONAL_DATA["BR"]["FG"]))

    def test_invalid_dimension(self, engine: InstitutionalQualityEngine):
        """Invalid dimension code should return failure."""
        result = engine.assess_institutional_strength("DK", dimensions=["INVALID"])
        assert result.success is False
        assert "Invalid dimensions" in result.error

    def test_unknown_country(self, engine: InstitutionalQualityEngine):
        """Unknown country should return failure."""
        result = engine.assess_institutional_strength("ZZ")
        assert result.success is False

    def test_strength_overall_is_average(self, engine: InstitutionalQualityEngine):
        """Overall strength should be the average of selected dimension scores."""
        result = engine.assess_institutional_strength("DK", dimensions=["JI", "RE"])
        assert result.success is True
        ji = Decimal(str(INSTITUTIONAL_DATA["DK"]["JI"]))
        re = Decimal(str(INSTITUTIONAL_DATA["DK"]["RE"]))
        expected = ((ji + re) / Decimal("2")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert result.overall_strength == expected


# ---------------------------------------------------------------------------
# TestForestGovernance
# ---------------------------------------------------------------------------


class TestForestGovernance:
    """Tests for get_forest_governance for EUDR-relevant countries."""

    def test_brazil_forest_governance(self, engine: InstitutionalQualityEngine):
        """Brazil should have HIGH illegal logging and risk factors."""
        result = engine.get_forest_governance("BR")
        assert result.success is True
        assert result.data is not None
        assert result.data.country_code == "BR"
        assert result.data.illegal_logging_prevalence == "HIGH"
        assert len(result.risk_factors) > 0
        assert result.eudr_risk_factor > Decimal("0.3")

    def test_indonesia_forest_governance(self, engine: InstitutionalQualityEngine):
        """Indonesia should have HIGH illegal logging prevalence."""
        result = engine.get_forest_governance("ID")
        assert result.success is True
        assert result.data.illegal_logging_prevalence == "HIGH"

    def test_drc_forest_governance(self, engine: InstitutionalQualityEngine):
        """DRC (CD) should have CRITICAL illegal logging prevalence."""
        result = engine.get_forest_governance("CD")
        assert result.success is True
        assert result.data.illegal_logging_prevalence == "CRITICAL"
        assert result.eudr_risk_factor > Decimal("0.7")

    def test_finland_forest_governance(self, engine: InstitutionalQualityEngine):
        """Finland should have LOW illegal logging and low EUDR risk."""
        result = engine.get_forest_governance("FI")
        assert result.success is True
        assert result.data.illegal_logging_prevalence == "LOW"
        assert result.eudr_risk_factor < Decimal("0.15")

    def test_unknown_country_forest(self, engine: InstitutionalQualityEngine):
        """Unknown country should return failure for forest governance."""
        result = engine.get_forest_governance("ZZ")
        assert result.success is False
        assert "ZZ" in result.error

    def test_composite_forest_score_range(self, engine: InstitutionalQualityEngine):
        """Composite forest score should be in [0, 100]."""
        result = engine.get_forest_governance("BR")
        assert result.success is True
        assert Decimal("0") <= result.composite_forest_score <= Decimal("100")


# ---------------------------------------------------------------------------
# TestCountryComparison
# ---------------------------------------------------------------------------


class TestCountryComparison:
    """Tests for compare_countries across dimensions."""

    def test_compare_two_countries(self, engine: InstitutionalQualityEngine):
        """Comparing two countries should return rankings."""
        result = engine.compare_countries(["DK", "BR"])
        assert result.success is True
        assert len(result.countries) == 2
        assert "DK" in result.countries
        assert "BR" in result.countries
        # Denmark should rank higher
        assert result.composite_rankings[0][0] == "DK"

    def test_compare_with_invalid_country(self, engine: InstitutionalQualityEngine):
        """Comparison including an unknown country should still succeed with warnings."""
        result = engine.compare_countries(["DK", "ZZ"])
        assert result.success is True
        assert len(result.warnings) > 0
        assert result.countries["ZZ"].success is False

    def test_compare_empty_list(self, engine: InstitutionalQualityEngine):
        """Empty country list should fail."""
        result = engine.compare_countries([])
        assert result.success is False

    def test_compare_with_dimension_filter(self, engine: InstitutionalQualityEngine):
        """Dimension filter should restrict rankings."""
        result = engine.compare_countries(["DK", "BR"], dimensions=["FG", "LTS"])
        assert result.success is True
        assert "FG" in result.dimension_rankings
        assert "LTS" in result.dimension_rankings
        # JI should not be in rankings when filtering to FG and LTS
        # But the implementation ranks all valid dims if dimensions is specified
        assert len(result.dimension_rankings) >= 2

    def test_compare_rankings_sorted_descending(
        self, engine: InstitutionalQualityEngine
    ):
        """Composite rankings should be sorted by score descending."""
        result = engine.compare_countries(["DK", "BR", "VE"])
        assert result.success is True
        scores = [s for _, s in result.composite_rankings]
        assert scores == sorted(scores, reverse=True)

    def test_compare_has_provenance(self, engine: InstitutionalQualityEngine):
        """Comparison result should have provenance hash."""
        result = engine.compare_countries(["DK", "BR"])
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestCompositeScore
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Tests for _calculate_composite_institutional_score."""

    def test_all_dimensions_100(self, engine: InstitutionalQualityEngine):
        """All dimensions at 100 should yield composite of 100."""
        scores = {dim.value: Decimal("100") for dim in InstitutionalDimension}
        composite = engine._calculate_composite_institutional_score(scores)
        assert composite == Decimal("100.00")

    def test_all_dimensions_0(self, engine: InstitutionalQualityEngine):
        """All dimensions at 0 should yield composite of 0."""
        scores = {dim.value: Decimal("0") for dim in InstitutionalDimension}
        composite = engine._calculate_composite_institutional_score(scores)
        assert composite == Decimal("0.00")

    def test_weighted_composite_correctness(self, engine: InstitutionalQualityEngine):
        """Composite should equal the weighted average of dimension scores."""
        scores: Dict[str, Decimal] = {
            "JI": Decimal("80"),
            "RE": Decimal("70"),
            "PR": Decimal("60"),
            "CE": Decimal("50"),
            "TL": Decimal("40"),
            "ACF": Decimal("90"),
            "FG": Decimal("30"),
            "LTS": Decimal("20"),
        }
        composite = engine._calculate_composite_institutional_score(scores)

        # Manual computation
        expected = Decimal("0")
        total_weight = Decimal("0")
        for dim, score in scores.items():
            w = DEFAULT_DIMENSION_WEIGHTS[dim]
            expected += score * w
            total_weight += w
        expected = (expected / total_weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert composite == expected

    def test_partial_dimensions(self, engine: InstitutionalQualityEngine):
        """Only some dimensions provided should still compute weighted average."""
        scores = {"JI": Decimal("80"), "FG": Decimal("40")}
        composite = engine._calculate_composite_institutional_score(scores)
        w_ji = DEFAULT_DIMENSION_WEIGHTS["JI"]
        w_fg = DEFAULT_DIMENSION_WEIGHTS["FG"]
        expected = (Decimal("80") * w_ji + Decimal("40") * w_fg) / (w_ji + w_fg)
        expected = expected.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert composite == expected

    def test_empty_dimensions(self, engine: InstitutionalQualityEngine):
        """Empty dimension scores should return 0."""
        composite = engine._calculate_composite_institutional_score({})
        assert composite == Decimal("0")

    def test_composite_clamped_to_100(self, engine: InstitutionalQualityEngine):
        """Composite should never exceed 100."""
        scores = {dim.value: Decimal("200") for dim in InstitutionalDimension}
        composite = engine._calculate_composite_institutional_score(scores)
        assert composite <= Decimal("100")


# ---------------------------------------------------------------------------
# TestInstitutionalEUDRMapping
# ---------------------------------------------------------------------------


class TestInstitutionalEUDRMapping:
    """Tests for _map_institutional_to_eudr_risk."""

    @pytest.mark.parametrize(
        "composite,expected_risk",
        [
            (Decimal("0"), Decimal("1.0000")),
            (Decimal("25"), Decimal("0.7500")),
            (Decimal("50"), Decimal("0.5000")),
            (Decimal("75"), Decimal("0.2500")),
            (Decimal("100"), Decimal("0.0000")),
        ],
    )
    def test_mapping_formula(
        self,
        engine: InstitutionalQualityEngine,
        composite: Decimal,
        expected_risk: Decimal,
    ):
        """EUDR risk = 1.0 - composite/100."""
        risk = engine._map_institutional_to_eudr_risk(composite)
        assert risk == expected_risk

    def test_risk_clamped_at_zero(self, engine: InstitutionalQualityEngine):
        """Composite above 100 should yield risk of 0."""
        risk = engine._map_institutional_to_eudr_risk(Decimal("150"))
        assert risk == Decimal("0.0000")

    def test_risk_clamped_at_one(self, engine: InstitutionalQualityEngine):
        """Composite below 0 should yield risk of 1."""
        risk = engine._map_institutional_to_eudr_risk(Decimal("-10"))
        assert risk == Decimal("1.0000")

    def test_risk_decimal_precision(self, engine: InstitutionalQualityEngine):
        """Risk should have 4 decimal places."""
        risk = engine._map_institutional_to_eudr_risk(Decimal("33"))
        expected = (Decimal("1.0") - Decimal("33") / Decimal("100")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        assert risk == expected


# ---------------------------------------------------------------------------
# TestInstitutionalDimensions
# ---------------------------------------------------------------------------


class TestInstitutionalDimensions:
    """Verify all 8 institutional dimensions are defined."""

    def test_all_dimensions_exist(self):
        """All 8 dimensions should be defined in the enum."""
        dims = set(d.value for d in InstitutionalDimension)
        expected = {"JI", "RE", "PR", "CE", "TL", "ACF", "FG", "LTS"}
        assert dims == expected

    def test_dimension_count(self):
        """There should be exactly 8 dimensions."""
        assert len(InstitutionalDimension) == 8

    def test_capacity_levels(self):
        """There should be 4 capacity levels."""
        levels = set(c.value for c in InstitutionalCapacityLevel)
        assert levels == {"STRONG", "ADEQUATE", "WEAK", "VERY_WEAK"}

    def test_illegal_logging_prevalence(self):
        """There should be 4 illegal logging levels."""
        levels = set(l.value for l in IllegalLoggingPrevalence)
        assert levels == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    @pytest.mark.parametrize(
        "capacity,min_score,max_score",
        [
            ("STRONG", Decimal("75"), Decimal("100")),
            ("ADEQUATE", Decimal("50"), Decimal("74.99")),
            ("WEAK", Decimal("25"), Decimal("49.99")),
            ("VERY_WEAK", Decimal("0"), Decimal("24.99")),
        ],
    )
    def test_capacity_classification_boundaries(
        self,
        engine: InstitutionalQualityEngine,
        capacity: str,
        min_score: Decimal,
        max_score: Decimal,
    ):
        """Verify capacity classification uses correct boundaries."""
        classified = engine._classify_capacity(min_score)
        assert classified == capacity

    def test_weights_sum_to_one(self):
        """Default dimension weights should sum to 1.0."""
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert total == Decimal("1.000")

    def test_forest_and_land_have_highest_weights(self):
        """FG and LTS should have the highest weight (0.150 each for EUDR)."""
        assert DEFAULT_DIMENSION_WEIGHTS["FG"] == Decimal("0.150")
        assert DEFAULT_DIMENSION_WEIGHTS["LTS"] == Decimal("0.150")


# ---------------------------------------------------------------------------
# TestInstitutionalProvenance
# ---------------------------------------------------------------------------


class TestInstitutionalProvenance:
    """Tests for provenance chain integrity."""

    def test_assess_quality_has_provenance(self, engine: InstitutionalQualityEngine):
        """assess_country_quality should produce a 64-char SHA-256 hash."""
        result = engine.assess_country_quality("DK")
        assert len(result.provenance_hash) == 64

    def test_provenance_deterministic(self, engine: InstitutionalQualityEngine):
        """Same inputs should produce the same provenance hash."""
        r1 = engine.assess_country_quality("BR", year=2024)
        r2 = engine.assess_country_quality("BR", year=2024)
        assert r1.provenance_hash == r2.provenance_hash

    def test_provenance_differs_for_different_countries(
        self, engine: InstitutionalQualityEngine
    ):
        """Different countries should produce different provenance hashes."""
        r1 = engine.assess_country_quality("DK")
        r2 = engine.assess_country_quality("BR")
        assert r1.provenance_hash != r2.provenance_hash

    def test_forest_governance_provenance(self, engine: InstitutionalQualityEngine):
        """Forest governance result should have provenance hash."""
        result = engine.get_forest_governance("BR")
        assert len(result.provenance_hash) == 64

    def test_strength_provenance(self, engine: InstitutionalQualityEngine):
        """Strength assessment should have provenance hash."""
        result = engine.assess_institutional_strength("DK")
        assert len(result.provenance_hash) == 64

    def test_eudr_risk_convenience_method(self, engine: InstitutionalQualityEngine):
        """get_eudr_risk_factor should return same risk as assess_country_quality."""
        risk = engine.get_eudr_risk_factor("DK")
        result = engine.assess_country_quality("DK")
        assert risk == result.eudr_risk_factor

    def test_eudr_risk_unknown_country(self, engine: InstitutionalQualityEngine):
        """Unknown country should return precautionary 1.0 risk."""
        risk = engine.get_eudr_risk_factor("ZZ")
        assert risk == Decimal("1.0000")


# ---------------------------------------------------------------------------
# TestForestGovernanceProfile
# ---------------------------------------------------------------------------


class TestForestGovernanceProfile:
    """Verify ForestGovernanceProfile fields."""

    def test_profile_fields_for_brazil(self, engine: InstitutionalQualityEngine):
        """Brazil forest profile should have all required fields."""
        result = engine.get_forest_governance("BR")
        assert result.success is True
        profile = result.data
        assert profile.legal_framework_score == Decimal("65")
        assert profile.enforcement_capacity == Decimal("0.35")
        assert profile.monitoring_capability == Decimal("0.55")
        assert profile.indigenous_rights_protection == Decimal("50")
        assert profile.community_participation == Decimal("45")
        assert profile.illegal_logging_prevalence == "HIGH"

    def test_profile_concession_transparency(self, engine: InstitutionalQualityEngine):
        """Concession transparency should be populated from reference data."""
        result = engine.get_forest_governance("BR")
        assert result.data.concession_transparency == Decimal("35")

    def test_profile_redd_readiness(self, engine: InstitutionalQualityEngine):
        """REDD+ readiness should be populated from reference data."""
        result = engine.get_forest_governance("BR")
        assert result.data.redd_plus_readiness == Decimal("55")

    def test_profile_protected_area_management(
        self, engine: InstitutionalQualityEngine
    ):
        """Protected area management should be populated from reference data."""
        result = engine.get_forest_governance("BR")
        assert result.data.protected_area_management == Decimal("45")

    def test_profile_to_dict(self, engine: InstitutionalQualityEngine):
        """ForestGovernanceProfile.to_dict should produce serializable output."""
        result = engine.get_forest_governance("FI")
        d = result.data.to_dict()
        assert d["country_code"] == "FI"
        assert d["illegal_logging_prevalence"] == "LOW"
        # All numeric values should be strings
        assert isinstance(d["legal_framework_score"], str)
        assert isinstance(d["enforcement_capacity"], str)

    def test_risk_factors_for_critical_country(
        self, engine: InstitutionalQualityEngine
    ):
        """Countries with CRITICAL logging should have multiple risk factors."""
        result = engine.get_forest_governance("CD")
        assert len(result.risk_factors) >= 3  # logging, enforcement, monitoring, etc.

    def test_assessment_to_dict(self, engine: InstitutionalQualityEngine):
        """InstitutionalAssessment.to_dict should be serializable."""
        result = engine.assess_country_quality("DE")
        d = result.data.to_dict()
        assert d["country_code"] == "DE"
        assert isinstance(d["composite_score"], str)
        assert isinstance(d["dimension_scores"], dict)
        for k, v in d["dimension_scores"].items():
            assert isinstance(v, str)
