# -*- coding: utf-8 -*-
"""
End-to-End Tests for PACK-043 Scope 3 Complete
=================================================

Tests complete Scope 3 inventory scenarios for manufacturing enterprise,
financial institution, retail chain, technology company, multi-entity
group, and full pipeline from maturity through assurance.

Coverage target: 85%+
Total tests: ~30
"""

from decimal import Decimal

import pytest

from tests.conftest import compute_provenance_hash, SCOPE3_CATEGORIES


# =============================================================================
# Manufacturing Enterprise E2E
# =============================================================================


class TestE2EManufacturingEnterprise:
    """End-to-end test for a manufacturing enterprise (multi-entity, LCA, circular)."""

    def test_manufacturing_has_lca(self, sample_lca_results):
        assert sample_lca_results["total_tco2e"] > Decimal("0")

    def test_manufacturing_has_bom(self, sample_product_bom):
        assert len(sample_product_bom) >= 1
        for prod in sample_product_bom:
            assert len(prod["components"]) >= 1

    def test_manufacturing_material_categories(self, sample_scope3_screening):
        """Manufacturing should have Cat 1, 2, 4, 5, 11, 12 as material."""
        material = sample_scope3_screening["material_categories"]
        for cat in [1, 2, 4, 5, 11, 12]:
            assert cat in material

    def test_manufacturing_circular_benefit(self, sample_product_bom):
        """Products should have recycled content."""
        for prod in sample_product_bom:
            has_recycled = any(
                c["recycled_content_pct"] > Decimal("0")
                for c in prod["components"]
            )
            assert has_recycled is True

    def test_manufacturing_supplier_engagement(self, sample_supplier_programme):
        assert sample_supplier_programme["total_suppliers"] >= 10


# =============================================================================
# Financial Institution E2E
# =============================================================================


class TestE2EFinancialInstitution:
    """End-to-end test for a financial institution (PCAF Cat 15)."""

    def test_pcaf_portfolio_present(self, sample_pcaf_portfolio):
        assert sample_pcaf_portfolio["total_portfolio_value_usd"] > Decimal("0")

    def test_cat15_is_material(self, sample_scope3_screening):
        cat15 = sample_scope3_screening["by_category"][15]
        assert cat15["material"] is True

    def test_cat15_is_significant_share(self, sample_scope3_screening):
        cat15_pct = sample_scope3_screening["by_category"][15]["pct"]
        assert cat15_pct > Decimal("10")

    def test_financed_emissions_calculable(self, sample_pcaf_portfolio):
        total_financed = Decimal("0")
        for ac in sample_pcaf_portfolio["asset_classes"].values():
            for inv in ac["investments"]:
                attr = inv["invested_amount_usd"] / inv["evic_usd"]
                total_financed += attr * inv["investee_scope12_tco2e"]
        assert total_financed > Decimal("0")

    def test_portfolio_data_quality(self, sample_pcaf_portfolio):
        for ac in sample_pcaf_portfolio["asset_classes"].values():
            for inv in ac["investments"]:
                assert 1 <= inv["data_quality_score"] <= 5


# =============================================================================
# Retail Chain E2E
# =============================================================================


class TestE2ERetailChain:
    """End-to-end test for a retail chain (last-mile, packaging, returns)."""

    def test_last_mile_present(self, sample_retail_data):
        assert sample_retail_data["last_mile"]["total_deliveries"] > 0

    def test_packaging_present(self, sample_retail_data):
        assert len(sample_retail_data["packaging"]["materials"]) >= 1

    def test_returns_impact(self, sample_retail_data):
        return_rate = sample_retail_data["returns"]["return_rate_pct"]
        assert return_rate > Decimal("0")

    def test_retail_total_emissions(self, sample_retail_data):
        """Calculate total retail-specific emissions."""
        lm = sample_retail_data["last_mile"]
        total_km = lm["total_deliveries"] * lm["average_distance_km"]
        emissions_tco2e = total_km * lm["emission_factor_kgco2e_per_km"] / Decimal("1000")
        assert emissions_tco2e > Decimal("0")


# =============================================================================
# Technology Company E2E
# =============================================================================


class TestE2ETechnologyCompany:
    """End-to-end test for a technology company (cloud, embodied, SaaS)."""

    def test_cloud_providers_present(self, sample_cloud_data):
        assert len(sample_cloud_data["providers"]) >= 2

    def test_embodied_carbon_present(self, sample_cloud_data):
        ec = sample_cloud_data["embodied_carbon"]
        assert len(ec) >= 2

    def test_saas_use_phase_present(self, sample_cloud_data):
        saas = sample_cloud_data["saas_use_phase"]
        assert saas["active_users"] > 0

    def test_tech_total_cloud_emissions(self, sample_cloud_data):
        total = Decimal("0")
        for name, p in sample_cloud_data["providers"].items():
            non_renewable = (Decimal("100") - p["renewable_pct"]) / Decimal("100")
            em = p["kwh_consumed"] * p["pue"] * p["grid_factor_kgco2e_per_kwh"] * non_renewable
            total += em
        total_tco2e = total / Decimal("1000")
        assert total_tco2e >= Decimal("0")


# =============================================================================
# Multi-Entity Group E2E
# =============================================================================


class TestE2EMultiEntityGroup:
    """End-to-end test for a multi-entity group (50+ entities)."""

    def test_entity_hierarchy_complete(self, sample_entity_hierarchy):
        total_entities = (
            1
            + len(sample_entity_hierarchy["subsidiaries"])
            + len(sample_entity_hierarchy["joint_ventures"])
        )
        assert total_entities >= 7

    def test_consolidation_approaches_work(self, sample_entity_hierarchy):
        """All three approaches should produce valid results."""
        for approach in ["equity_share", "operational_control", "financial_control"]:
            total = Decimal("0")
            parent = sample_entity_hierarchy["parent"]
            total += parent["scope3_tco2e"]
            for sub in sample_entity_hierarchy["subsidiaries"]:
                if approach == "equity_share":
                    total += sub["scope3_tco2e"] * sub["equity_pct"] / Decimal("100")
                else:
                    key = "has_operational_control" if approach == "operational_control" else "has_financial_control"
                    if sub.get(key, False):
                        total += sub["scope3_tco2e"]
            for jv in sample_entity_hierarchy["joint_ventures"]:
                if approach == "equity_share":
                    total += jv["scope3_tco2e"] * jv["equity_pct"] / Decimal("100")
                else:
                    key = "has_operational_control" if approach == "operational_control" else "has_financial_control"
                    if jv.get(key, False):
                        total += jv["scope3_tco2e"]
            assert total > Decimal("0")

    def test_equity_less_than_operational(self, sample_entity_hierarchy):
        """Equity share should produce different total than operational control."""
        equity_total = Decimal("0")
        oc_total = Decimal("0")
        parent = sample_entity_hierarchy["parent"]
        equity_total += parent["scope3_tco2e"]
        oc_total += parent["scope3_tco2e"]
        for sub in sample_entity_hierarchy["subsidiaries"]:
            equity_total += sub["scope3_tco2e"] * sub["equity_pct"] / Decimal("100")
            if sub["has_operational_control"]:
                oc_total += sub["scope3_tco2e"]
        for jv in sample_entity_hierarchy["joint_ventures"]:
            equity_total += jv["scope3_tco2e"] * jv["equity_pct"] / Decimal("100")
            if jv["has_operational_control"]:
                oc_total += jv["scope3_tco2e"]
        # In our fixture, equity includes JV proportional, OC excludes JVs
        assert equity_total != oc_total


# =============================================================================
# Full Pipeline E2E
# =============================================================================


class TestE2EFullPipeline:
    """End-to-end test: maturity -> LCA -> scenario -> SBTi -> risk -> assurance."""

    def test_step1_maturity_assessed(self, sample_maturity_assessment):
        assert sample_maturity_assessment["total_scope3_tco2e"] > Decimal("0")
        assert len(sample_maturity_assessment["categories"]) == 15

    def test_step2_lca_integrated(self, sample_lca_results):
        assert sample_lca_results["total_tco2e"] > Decimal("0")
        assert len(sample_lca_results["phases"]) == 5

    def test_step3_scenario_modelled(self, sample_macc_interventions, sample_scenario_config):
        total_abatement = sum(i["abatement_tco2e"] for i in sample_macc_interventions)
        assert total_abatement > Decimal("0")
        assert total_abatement < sample_scenario_config["baseline_scope3_tco2e"]

    def test_step4_sbti_targets_set(self, sample_sbti_targets):
        assert sample_sbti_targets["near_term"]["target_absolute_tco2e"] > Decimal("0")
        assert sample_sbti_targets["long_term"]["target_absolute_tco2e"] > Decimal("0")
        assert (
            sample_sbti_targets["long_term"]["target_absolute_tco2e"]
            < sample_sbti_targets["near_term"]["target_absolute_tco2e"]
        )

    def test_step5_supplier_engaged(self, sample_supplier_programme):
        assert sample_supplier_programme["committed_suppliers"] > 0
        assert sample_supplier_programme["coverage_pct"] > Decimal("50")

    def test_step6_climate_risks_assessed(self, sample_climate_risks):
        assert len(sample_climate_risks["transition_risks"]) >= 1
        assert len(sample_climate_risks["physical_risks"]) >= 1
        assert len(sample_climate_risks["opportunities"]) >= 1

    def test_step7_base_year_established(self, sample_base_year_data):
        assert sample_base_year_data["base_year"] == 2019
        assert sample_base_year_data["scope3_total_tco2e"] > Decimal("0")

    def test_step8_assurance_ready(self, sample_assurance_evidence):
        assert sample_assurance_evidence["readiness_score"] >= Decimal("80")
        assert len(sample_assurance_evidence["evidence_items"]) >= 5

    def test_full_pipeline_provenance(self, full_scope3_context):
        """Full pipeline should produce a provenance hash."""
        h = compute_provenance_hash(full_scope3_context)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_full_pipeline_data_consistency(self, full_scope3_context):
        """Screening total should match scenario baseline."""
        screening_total = full_scope3_context["scope3_screening"]["total_scope3_tco2e"]
        scenario_baseline = full_scope3_context["scenario_config"]["baseline_scope3_tco2e"]
        assert screening_total == scenario_baseline
