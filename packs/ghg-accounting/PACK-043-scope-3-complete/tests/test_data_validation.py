# -*- coding: utf-8 -*-
"""
Unit tests for Data Validation -- PACK-043
=============================================

Tests fixture data consistency across entity hierarchy, BOM mass balance,
SBTi target arithmetic, scenario budget constraints, PCAF attribution,
provenance hash format, and cross-fixture consistency.

Coverage target: 85%+
Total tests: ~45
"""

import hashlib
import json
from decimal import Decimal

import pytest

from tests.conftest import compute_provenance_hash, SCOPE3_CATEGORIES


# =============================================================================
# Entity Hierarchy Consistency
# =============================================================================


class TestEntityHierarchyConsistency:
    """Validate entity hierarchy fixture data consistency."""

    def test_hierarchy_has_group_id(self, sample_entity_hierarchy):
        assert sample_entity_hierarchy["group_id"] == "GRP-APEX-001"

    def test_hierarchy_has_group_name(self, sample_entity_hierarchy):
        assert len(sample_entity_hierarchy["group_name"]) > 0

    def test_parent_entity_present(self, sample_entity_hierarchy):
        assert "parent" in sample_entity_hierarchy

    def test_parent_100pct_equity(self, sample_entity_hierarchy):
        assert sample_entity_hierarchy["parent"]["equity_pct"] == Decimal("100")

    def test_all_entities_have_scope3(self, sample_entity_hierarchy):
        assert sample_entity_hierarchy["parent"]["scope3_tco2e"] >= Decimal("0")
        for sub in sample_entity_hierarchy["subsidiaries"]:
            assert sub["scope3_tco2e"] >= Decimal("0")
        for jv in sample_entity_hierarchy["joint_ventures"]:
            assert jv["scope3_tco2e"] >= Decimal("0")

    def test_entity_ids_unique(self, sample_entity_hierarchy):
        ids = [sample_entity_hierarchy["parent"]["entity_id"]]
        ids.extend(s["entity_id"] for s in sample_entity_hierarchy["subsidiaries"])
        ids.extend(j["entity_id"] for j in sample_entity_hierarchy["joint_ventures"])
        assert len(ids) == len(set(ids))

    def test_jv_equity_below_51(self, sample_entity_hierarchy):
        for jv in sample_entity_hierarchy["joint_ventures"]:
            assert jv["equity_pct"] <= Decimal("50")

    def test_subsidiary_equity_100(self, sample_entity_hierarchy):
        for sub in sample_entity_hierarchy["subsidiaries"]:
            assert sub["equity_pct"] == Decimal("100")


# =============================================================================
# BOM Mass Balance
# =============================================================================


class TestBOMMassBalance:
    """Validate BOM mass balance across all products."""

    @pytest.mark.parametrize("prod_idx", [0, 1, 2])
    def test_component_mass_equals_product(self, prod_idx, sample_product_bom):
        prod = sample_product_bom[prod_idx]
        component_mass = sum(c["mass_kg"] for c in prod["components"])
        assert component_mass == prod["mass_kg"]

    @pytest.mark.parametrize("prod_idx", [0, 1, 2])
    def test_all_masses_positive(self, prod_idx, sample_product_bom):
        prod = sample_product_bom[prod_idx]
        assert prod["mass_kg"] > Decimal("0")
        for c in prod["components"]:
            assert c["mass_kg"] > Decimal("0")

    @pytest.mark.parametrize("prod_idx", [0, 1, 2])
    def test_emission_factors_positive(self, prod_idx, sample_product_bom):
        for c in sample_product_bom[prod_idx]["components"]:
            assert c["emission_factor_kgco2e_per_kg"] > Decimal("0")

    @pytest.mark.parametrize("prod_idx", [0, 1, 2])
    def test_recycled_content_range(self, prod_idx, sample_product_bom):
        for c in sample_product_bom[prod_idx]["components"]:
            assert Decimal("0") <= c["recycled_content_pct"] <= Decimal("100")


# =============================================================================
# SBTi Target Arithmetic
# =============================================================================


class TestSBTiTargetArithmetic:
    """Validate SBTi target fixture arithmetic."""

    def test_scope3_pct_calculation(self, sample_sbti_targets):
        s3 = sample_sbti_targets["base_year_scope3_tco2e"]
        total = sample_sbti_targets["base_year_total_tco2e"]
        expected = s3 / total * Decimal("100")
        assert expected == pytest.approx(
            sample_sbti_targets["scope3_pct_of_total"], abs=Decimal("0.01")
        )

    def test_total_equals_sum_of_scopes(self, sample_sbti_targets):
        s1 = sample_sbti_targets["base_year_scope1_tco2e"]
        s2 = sample_sbti_targets["base_year_scope2_tco2e"]
        s3 = sample_sbti_targets["base_year_scope3_tco2e"]
        assert s1 + s2 + s3 == sample_sbti_targets["base_year_total_tco2e"]

    def test_near_term_target_arithmetic(self, sample_sbti_targets):
        nt = sample_sbti_targets["near_term"]
        base = sample_sbti_targets["base_year_scope3_tco2e"]
        expected = base * (Decimal("1") - nt["target_reduction_pct"] / Decimal("100"))
        assert nt["target_absolute_tco2e"] == expected

    def test_coverage_arithmetic(self, sample_sbti_targets):
        nt = sample_sbti_targets["near_term"]
        base = sample_sbti_targets["base_year_scope3_tco2e"]
        expected = nt["covered_emissions_tco2e"] / base * Decimal("100")
        assert expected == nt["scope3_coverage_pct"]


# =============================================================================
# Scenario Budget Constraints
# =============================================================================


class TestScenarioBudgetConstraints:
    """Validate scenario budget constraint data."""

    def test_budget_positive(self, sample_scenario_config):
        assert sample_scenario_config["budget_constraint_usd"] > Decimal("0")

    def test_baseline_positive(self, sample_scenario_config):
        assert sample_scenario_config["baseline_scope3_tco2e"] > Decimal("0")

    def test_all_interventions_positive_abatement(self, sample_macc_interventions):
        for intv in sample_macc_interventions:
            assert intv["abatement_tco2e"] > Decimal("0")

    def test_confidence_range(self, sample_macc_interventions):
        for intv in sample_macc_interventions:
            assert Decimal("0") < intv["confidence"] <= Decimal("1")


# =============================================================================
# PCAF Attribution Factors Sum
# =============================================================================


class TestPCAFAttributionFactors:
    """Validate PCAF attribution factor data."""

    def test_attribution_factors_below_1(self, sample_pcaf_portfolio):
        for asset_class in sample_pcaf_portfolio["asset_classes"].values():
            for inv in asset_class["investments"]:
                attr = inv["invested_amount_usd"] / inv["evic_usd"]
                assert attr <= Decimal("1")

    def test_portfolio_value_matches_sum(self, sample_pcaf_portfolio):
        total = Decimal("0")
        for asset_class in sample_pcaf_portfolio["asset_classes"].values():
            for inv in asset_class["investments"]:
                total += inv["invested_amount_usd"]
        assert total == sample_pcaf_portfolio["total_portfolio_value_usd"]

    def test_data_quality_scores_valid(self, sample_pcaf_portfolio):
        for asset_class in sample_pcaf_portfolio["asset_classes"].values():
            for inv in asset_class["investments"]:
                assert 1 <= inv["data_quality_score"] <= 5


# =============================================================================
# Provenance Hash Format
# =============================================================================


class TestProvenanceHashFormat:
    """Validate provenance hash format and behavior."""

    def test_hash_is_sha256(self, sample_scope3_screening):
        h = compute_provenance_hash(sample_scope3_screening)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self, sample_scope3_screening):
        hashes = [compute_provenance_hash(sample_scope3_screening) for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_hash_changes_with_data(self, sample_scope3_screening):
        h1 = compute_provenance_hash(sample_scope3_screening)
        modified = dict(sample_scope3_screening)
        modified["reporting_year"] = 9999
        h2 = compute_provenance_hash(modified)
        assert h1 != h2

    def test_hash_order_independent(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert compute_provenance_hash(d1) == compute_provenance_hash(d2)


# =============================================================================
# Cross-Fixture Consistency
# =============================================================================


class TestCrossFixtureConsistency:
    """Validate consistency across multiple fixtures."""

    def test_screening_total_matches_maturity(
        self, sample_scope3_screening, sample_maturity_assessment
    ):
        """Screening total should equal maturity assessment total."""
        screening_total = sample_scope3_screening["total_scope3_tco2e"]
        maturity_total = sample_maturity_assessment["total_scope3_tco2e"]
        assert screening_total == maturity_total

    def test_sbti_base_year_consistent(
        self, sample_sbti_targets, sample_base_year_data
    ):
        """SBTi base year scope 3 should match base year data."""
        sbti_s3 = sample_sbti_targets["base_year_scope3_tco2e"]
        base_s3 = sample_base_year_data["scope3_total_tco2e"]
        assert sbti_s3 == base_s3

    def test_scenario_baseline_matches_screening(
        self, sample_scenario_config, sample_scope3_screening
    ):
        """Scenario baseline should match screening total."""
        assert (
            sample_scenario_config["baseline_scope3_tco2e"]
            == sample_scope3_screening["total_scope3_tco2e"]
        )

    def test_base_year_2025_matches_screening(
        self, sample_base_year_data, sample_scope3_screening
    ):
        """2025 actual should match screening total."""
        actual_2025 = sample_base_year_data["yearly_actuals"][2025]
        screening_total = sample_scope3_screening["total_scope3_tco2e"]
        assert actual_2025 == screening_total

    def test_supplier_programme_org_matches(
        self, sample_supplier_programme, sample_entity_hierarchy
    ):
        """Supplier programme org should match entity hierarchy."""
        assert (
            sample_supplier_programme["org_id"]
            == sample_entity_hierarchy["group_id"]
        )
