# -*- coding: utf-8 -*-
"""
Deep tests for PartnershipScoringEngine (Engine 7 of 10).

Covers: 6-dimension scoring, dimension weights, partner database
validation, synergy analysis, collaboration impact calculation,
R2Z criteria coverage, governance maturity, Decimal arithmetic,
SHA-256 provenance.

Target: ~50 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.partnership_scoring_engine import (
    PartnershipScoringEngine,
    DimensionId,
    GovernanceMaturity,
    DIMENSION_WEIGHTS,
    DIMENSION_LABELS,
    R2Z_CRITERIA,
    PARTNER_DB,
)

from conftest import assert_decimal_close, assert_provenance_hash


# ========================================================================
# Dimension Constants
# ========================================================================


class TestPartnershipDimensionConstants:
    """Validate 6-dimension scoring constants."""

    def test_exactly_6_dimensions(self):
        assert len(DimensionId) == 6

    def test_dimension_values(self):
        assert DimensionId.REQUIREMENT_ALIGNMENT.value == "requirement_alignment"
        assert DimensionId.REPORTING_EFFICIENCY.value == "reporting_efficiency"
        assert DimensionId.ENGAGEMENT_QUALITY.value == "engagement_quality"
        assert DimensionId.CREDIBILITY_CONTRIBUTION.value == "credibility_contribution"
        assert DimensionId.COVERAGE_COMPLETENESS.value == "coverage_completeness"
        assert DimensionId.TIMELINE_ALIGNMENT.value == "timeline_alignment"

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert_decimal_close(total, Decimal("1.00"), Decimal("0.001"))

    def test_requirement_alignment_highest_weight(self):
        assert DIMENSION_WEIGHTS["requirement_alignment"] == Decimal("0.25")

    def test_coverage_completeness_10_pct(self):
        assert DIMENSION_WEIGHTS["coverage_completeness"] == Decimal("0.10")

    def test_timeline_alignment_10_pct(self):
        assert DIMENSION_WEIGHTS["timeline_alignment"] == Decimal("0.10")

    def test_all_dimensions_have_labels(self):
        for dim in DimensionId:
            assert dim.value in DIMENSION_LABELS

    def test_all_dimensions_have_weights(self):
        for dim in DimensionId:
            assert dim.value in DIMENSION_WEIGHTS


# ========================================================================
# Enums
# ========================================================================


class TestPartnershipEnums:
    """Validate partnership enums."""

    def test_governance_maturity_4_values(self):
        assert len(GovernanceMaturity) == 4

    def test_governance_values(self):
        assert GovernanceMaturity.EXEMPLARY.value == "exemplary"
        assert GovernanceMaturity.MATURE.value == "mature"
        assert GovernanceMaturity.DEVELOPING.value == "developing"
        assert GovernanceMaturity.NASCENT.value == "nascent"


# ========================================================================
# R2Z Criteria
# ========================================================================


class TestR2ZCriteria:
    """Validate Race to Zero criteria list."""

    def test_exactly_8_criteria(self):
        assert len(R2Z_CRITERIA) == 8

    @pytest.mark.parametrize("criterion", [
        "net_zero_pledge", "interim_target", "action_plan",
        "annual_reporting", "scope_coverage", "science_based",
        "governance", "transparency",
    ])
    def test_criterion_present(self, criterion):
        assert criterion in R2Z_CRITERIA


# ========================================================================
# Partner Database
# ========================================================================


class TestPartnerDatabase:
    """Validate partner initiative database."""

    def test_partner_count_at_least_10(self):
        assert len(PARTNER_DB) >= 10

    @pytest.mark.parametrize("partner", [
        "sbti", "cdp", "c40", "iclei", "gfanz", "wmb",
        "sme_climate_hub", "under2", "nzba", "nzam",
    ])
    def test_key_partners_present(self, partner):
        assert partner in PARTNER_DB

    def test_each_partner_has_name(self):
        for pid, data in PARTNER_DB.items():
            assert "name" in data, f"{pid} missing name"

    def test_each_partner_has_criteria_covered(self):
        for pid, data in PARTNER_DB.items():
            assert "criteria_covered" in data, f"{pid} missing criteria_covered"
            assert len(data["criteria_covered"]) > 0

    def test_each_partner_has_credibility_score(self):
        for pid, data in PARTNER_DB.items():
            assert "credibility_score" in data
            assert isinstance(data["credibility_score"], Decimal)
            assert Decimal("0") <= data["credibility_score"] <= Decimal("100")

    def test_each_partner_has_actor_types(self):
        for pid, data in PARTNER_DB.items():
            assert "actor_types" in data
            assert len(data["actor_types"]) > 0

    def test_sbti_criteria_coverage(self):
        sbti = PARTNER_DB["sbti"]
        assert "net_zero_pledge" in sbti["criteria_covered"]
        assert "interim_target" in sbti["criteria_covered"]
        assert "science_based" in sbti["criteria_covered"]

    def test_sbti_credibility_high(self):
        assert PARTNER_DB["sbti"]["credibility_score"] >= Decimal("90")

    def test_cdp_covers_transparency(self):
        assert "transparency" in PARTNER_DB["cdp"]["criteria_covered"]

    def test_gfanz_for_financial_institutions(self):
        assert "financial_institution" in PARTNER_DB["gfanz"]["actor_types"]

    def test_c40_for_cities(self):
        assert "city" in PARTNER_DB["c40"]["actor_types"]


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestPartnershipEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, partnership_engine):
        assert partnership_engine is not None

    def test_engine_has_calculate(self, partnership_engine):
        assert callable(getattr(partnership_engine, "assess", None))

    def test_engine_class_name(self):
        assert PartnershipScoringEngine.__name__ == "PartnershipScoringEngine"
