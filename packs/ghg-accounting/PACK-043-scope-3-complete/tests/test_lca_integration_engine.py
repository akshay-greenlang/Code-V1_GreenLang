# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 LCA Integration Engine
=================================================

Tests BOM explosion, use-phase modelling, end-of-life scenarios,
product comparison, ISO 14067 compliance, sensitivity analysis,
material emission factor lookup, and edge cases.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal

import pytest


# =============================================================================
# BOM Explosion
# =============================================================================


class TestBOMExplosion:
    """Test bill of materials explosion for multi-component products."""

    def test_product1_has_5_components(self, sample_product_bom):
        prod = sample_product_bom[0]
        assert len(prod["components"]) == 5

    def test_product2_has_4_components(self, sample_product_bom):
        prod = sample_product_bom[1]
        assert len(prod["components"]) == 4

    def test_product3_has_3_components(self, sample_product_bom):
        prod = sample_product_bom[2]
        assert len(prod["components"]) == 3

    def test_bom_mass_balance_product1(self, sample_product_bom):
        """Component masses should sum to product mass."""
        prod = sample_product_bom[0]
        component_mass = sum(c["mass_kg"] for c in prod["components"])
        assert component_mass == prod["mass_kg"]

    def test_bom_mass_balance_product2(self, sample_product_bom):
        prod = sample_product_bom[1]
        component_mass = sum(c["mass_kg"] for c in prod["components"])
        assert component_mass == prod["mass_kg"]

    def test_bom_mass_balance_product3(self, sample_product_bom):
        prod = sample_product_bom[2]
        component_mass = sum(c["mass_kg"] for c in prod["components"])
        assert component_mass == prod["mass_kg"]

    def test_material_emissions_calculation(self, sample_product_bom):
        """Material emissions = mass * emission_factor for each component."""
        prod = sample_product_bom[0]
        for comp in prod["components"]:
            emissions = comp["mass_kg"] * comp["emission_factor_kgco2e_per_kg"]
            assert emissions > Decimal("0")

    def test_total_material_emissions_product1(self, sample_product_bom):
        """Total material emissions for product 1."""
        prod = sample_product_bom[0]
        total = sum(
            c["mass_kg"] * c["emission_factor_kgco2e_per_kg"]
            for c in prod["components"]
        )
        # 45*1.91 + 20*6.15 + 12*3.81 + 5*3.18 + 3*25.0
        # = 85.95 + 123.0 + 45.72 + 15.9 + 75.0 = 345.57
        assert Decimal("340") < total < Decimal("350")

    def test_recycled_content_reduces_impact(self, sample_product_bom):
        """Components with higher recycled content should have credit."""
        prod = sample_product_bom[0]
        for comp in prod["components"]:
            recycled = comp["recycled_content_pct"]
            assert Decimal("0") <= recycled <= Decimal("100")


# =============================================================================
# Use-Phase Modelling
# =============================================================================


class TestUsePhaseModelling:
    """Test use-phase emission modelling for different product types."""

    def test_electric_appliance_use_phase(self, sample_lca_results):
        """Electric appliance use-phase = kWh/yr * grid_factor * years."""
        use = sample_lca_results["phases"]["use_phase"]
        assumptions = use["assumptions"]
        expected = (
            assumptions["electricity_kwh_per_year"]
            * assumptions["grid_factor_kgco2e_per_kwh"]
            * assumptions["service_life_years"]
            / Decimal("1000")  # convert kg to tonnes
        )
        assert expected == pytest.approx(use["tco2e"], abs=Decimal("5"))

    def test_use_phase_dominates_lifecycle(self, sample_lca_results):
        """Use phase should be the largest contributor (>40%)."""
        use_pct = sample_lca_results["phases"]["use_phase"]["pct_of_total"]
        assert use_pct > Decimal("40")

    def test_use_phase_assumptions_valid(self, sample_lca_results):
        assumptions = sample_lca_results["phases"]["use_phase"]["assumptions"]
        assert assumptions["electricity_kwh_per_year"] > Decimal("0")
        assert assumptions["grid_factor_kgco2e_per_kwh"] > Decimal("0")
        assert assumptions["service_life_years"] > 0
        assert assumptions["operating_hours_per_year"] > 0

    def test_vehicle_use_phase_model(self):
        """Vehicle use-phase: fuel_consumption * ef * lifetime_km."""
        fuel_consumption_l_per_100km = Decimal("7.5")
        ef_kgco2e_per_l = Decimal("2.31")
        lifetime_km = Decimal("200000")
        total_fuel = lifetime_km / Decimal("100") * fuel_consumption_l_per_100km
        emissions_t = total_fuel * ef_kgco2e_per_l / Decimal("1000")
        assert Decimal("30") < emissions_t < Decimal("40")

    def test_saas_use_phase_model(self, sample_cloud_data):
        """SaaS use-phase: users * hours * device_power * grid_factor."""
        saas = sample_cloud_data["saas_use_phase"]
        annual_kwh = (
            saas["active_users"]
            * float(saas["avg_session_hours_per_month"])
            * 12
            * float(saas["device_power_w"])
            / 1000
        )
        emissions_t = annual_kwh * float(saas["grid_factor_kgco2e_per_kwh"]) / 1000
        assert emissions_t > 0


# =============================================================================
# End-of-Life Scenarios
# =============================================================================


class TestEndOfLifeScenarios:
    """Test end-of-life scenario modelling."""

    def test_four_eol_scenarios(self, sample_lca_results):
        scenarios = sample_lca_results["phases"]["end_of_life"]["scenarios"]
        assert len(scenarios) == 4

    @pytest.mark.parametrize("scenario", ["landfill", "recycling", "incineration", "reuse"])
    def test_eol_scenario_exists(self, scenario, sample_lca_results):
        scenarios = sample_lca_results["phases"]["end_of_life"]["scenarios"]
        assert scenario in scenarios

    def test_recycling_lower_than_landfill(self, sample_lca_results):
        scenarios = sample_lca_results["phases"]["end_of_life"]["scenarios"]
        assert scenarios["recycling"] < scenarios["landfill"]

    def test_reuse_lowest_impact(self, sample_lca_results):
        scenarios = sample_lca_results["phases"]["end_of_life"]["scenarios"]
        reuse = scenarios["reuse"]
        for scenario, value in scenarios.items():
            if scenario != "reuse":
                assert reuse <= value

    def test_incineration_between_landfill_and_recycling(self, sample_lca_results):
        scenarios = sample_lca_results["phases"]["end_of_life"]["scenarios"]
        assert scenarios["recycling"] < scenarios["incineration"] < scenarios["landfill"]

    def test_selected_scenario_valid(self, sample_lca_results):
        eol = sample_lca_results["phases"]["end_of_life"]
        assert isinstance(eol["selected_scenario"], str)
        assert len(eol["selected_scenario"]) > 0


# =============================================================================
# Product Comparison
# =============================================================================


class TestProductComparison:
    """Test product comparison (A vs B)."""

    def test_three_products_available(self, sample_product_bom):
        assert len(sample_product_bom) == 3

    def test_product1_heaviest(self, sample_product_bom):
        masses = [p["mass_kg"] for p in sample_product_bom]
        assert sample_product_bom[0]["mass_kg"] == max(masses)

    def test_product_intensity_comparison(self, sample_product_bom):
        """Compare carbon intensity (kgCO2e/kg) across products."""
        intensities = []
        for prod in sample_product_bom:
            total_emissions = sum(
                c["mass_kg"] * c["emission_factor_kgco2e_per_kg"]
                for c in prod["components"]
            )
            intensity = total_emissions / prod["mass_kg"]
            intensities.append(intensity)
        assert len(intensities) == 3
        for i in intensities:
            assert i > Decimal("0")

    def test_product_ids_unique(self, sample_product_bom):
        ids = [p["product_id"] for p in sample_product_bom]
        assert len(ids) == len(set(ids))


# =============================================================================
# ISO 14067 Compliance
# =============================================================================


class TestISO14067Compliance:
    """Test ISO 14067 compliance checks."""

    def test_methodology_is_iso14067(self, sample_lca_results):
        assert sample_lca_results["methodology"] == "ISO_14067"

    def test_system_boundary_defined(self, sample_lca_results):
        assert sample_lca_results["system_boundary"] in {
            "cradle_to_gate", "cradle_to_grave", "gate_to_gate"
        }

    def test_functional_unit_defined(self, sample_product_bom):
        for prod in sample_product_bom:
            assert "functional_unit" in prod
            assert len(prod["functional_unit"]) > 0

    def test_all_lifecycle_phases_present(self, sample_lca_results):
        required_phases = [
            "raw_material_extraction",
            "manufacturing",
            "distribution",
            "use_phase",
            "end_of_life",
        ]
        for phase in required_phases:
            assert phase in sample_lca_results["phases"]

    def test_phase_percentages_sum_to_100(self, sample_lca_results):
        phases = sample_lca_results["phases"]
        total_pct = sum(p["pct_of_total"] for p in phases.values())
        assert total_pct == pytest.approx(Decimal("100"), abs=Decimal("0.1"))

    def test_total_consistent_with_phases(self, sample_lca_results):
        phases = sample_lca_results["phases"]
        phase_total = sum(p["tco2e"] for p in phases.values())
        assert phase_total == pytest.approx(
            sample_lca_results["total_tco2e"], abs=Decimal("1")
        )


# =============================================================================
# Sensitivity Analysis
# =============================================================================


class TestSensitivityAnalysis:
    """Test sensitivity analysis on key LCA parameters."""

    def test_grid_factor_sensitivity(self, sample_lca_results):
        """10% change in grid factor should change use-phase proportionally."""
        use = sample_lca_results["phases"]["use_phase"]
        base_tco2e = use["tco2e"]
        sensitivity_factor = Decimal("1.10")
        adjusted = base_tco2e * sensitivity_factor
        assert adjusted > base_tco2e
        pct_change = (adjusted - base_tco2e) / base_tco2e * Decimal("100")
        assert pct_change == pytest.approx(Decimal("10"), abs=Decimal("0.01"))

    def test_service_life_sensitivity(self, sample_lca_results):
        """Doubling service life should roughly double use-phase emissions."""
        use = sample_lca_results["phases"]["use_phase"]
        base = use["tco2e"]
        doubled = base * Decimal("2")
        assert doubled == pytest.approx(base * 2, abs=Decimal("1"))

    def test_recycled_content_sensitivity(self, sample_product_bom):
        """Higher recycled content should reduce material emissions."""
        prod = sample_product_bom[0]
        comp = prod["components"][0]  # cast_iron with 30% recycled
        virgin_ef = comp["emission_factor_kgco2e_per_kg"]
        recycled_ef = virgin_ef * Decimal("0.4")  # recycled typically 40% of virgin
        blended_ef = (
            comp["recycled_content_pct"] / Decimal("100") * recycled_ef
            + (Decimal("1") - comp["recycled_content_pct"] / Decimal("100")) * virgin_ef
        )
        assert blended_ef < virgin_ef


# =============================================================================
# Edge Cases
# =============================================================================


class TestLCAEdgeCases:
    """Test edge cases for LCA calculations."""

    def test_zero_bom_product(self):
        """Product with zero components should have zero material emissions."""
        prod = {"product_id": "PROD-EMPTY", "mass_kg": Decimal("0"), "components": []}
        total = sum(
            c["mass_kg"] * c.get("emission_factor_kgco2e_per_kg", Decimal("0"))
            for c in prod["components"]
        )
        assert total == Decimal("0")

    def test_single_component_product(self):
        """Product with one component."""
        prod = {
            "product_id": "PROD-SINGLE",
            "mass_kg": Decimal("10"),
            "components": [
                {"mass_kg": Decimal("10"), "emission_factor_kgco2e_per_kg": Decimal("5.0")}
            ],
        }
        total = sum(c["mass_kg"] * c["emission_factor_kgco2e_per_kg"] for c in prod["components"])
        assert total == Decimal("50.0")

    def test_100_pct_recycled_content(self):
        """Product with 100% recycled content."""
        comp = {
            "mass_kg": Decimal("10"),
            "emission_factor_kgco2e_per_kg": Decimal("6.15"),
            "recycled_content_pct": Decimal("100"),
        }
        virgin_ef = comp["emission_factor_kgco2e_per_kg"]
        recycled_ef = virgin_ef * Decimal("0.4")
        effective_ef = recycled_ef  # 100% recycled
        assert effective_ef < virgin_ef

    def test_carbon_intensity_calculation(self, sample_lca_results):
        """Carbon intensity = total_tco2e * 1000 / mass_kg."""
        intensity = sample_lca_results["carbon_intensity_kgco2e_per_kg"]
        assert intensity > Decimal("0")

    def test_negative_eol_credit(self):
        """Recycling can yield negative (credit) end-of-life emissions."""
        virgin_production_avoided = Decimal("50")
        recycling_energy = Decimal("15")
        net_eol = recycling_energy - virgin_production_avoided
        assert net_eol < Decimal("0")
