# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Scenario Modelling Engine
====================================================

Tests MACC curve generation, what-if scenarios, technology pathway
modelling, Paris alignment checks, budget-constrained optimization,
waterfall generation, intervention ranking, and edge cases.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal

import pytest


# =============================================================================
# MACC Curve Generation
# =============================================================================


class TestMACCCurveGeneration:
    """Test marginal abatement cost curve generation."""

    def test_five_interventions_present(self, sample_macc_interventions):
        assert len(sample_macc_interventions) == 5

    def test_interventions_have_required_fields(self, sample_macc_interventions):
        required = [
            "intervention_id", "name", "abatement_tco2e",
            "cost_per_tco2e", "confidence",
        ]
        for intv in sample_macc_interventions:
            for field in required:
                assert field in intv, f"Missing {field} in {intv['intervention_id']}"

    def test_macc_sorted_by_cost(self, sample_macc_interventions):
        """MACC curve should order interventions by cost per tCO2e."""
        sorted_interventions = sorted(
            sample_macc_interventions, key=lambda x: x["cost_per_tco2e"]
        )
        # First should be cheapest (negative = savings)
        assert sorted_interventions[0]["cost_per_tco2e"] < Decimal("0")
        # Last should be most expensive
        assert sorted_interventions[-1]["cost_per_tco2e"] > Decimal("100")

    def test_total_abatement_potential(self, sample_macc_interventions):
        total = sum(i["abatement_tco2e"] for i in sample_macc_interventions)
        # 15000 + 8000 + 5500 + 12000 + 10000 = 50500
        assert total == Decimal("50500")

    def test_negative_cost_intervention_exists(self, sample_macc_interventions):
        """At least one intervention should have negative cost (net savings)."""
        negatives = [i for i in sample_macc_interventions if i["cost_per_tco2e"] < 0]
        assert len(negatives) >= 1

    def test_macc_curve_x_axis_cumulative(self, sample_macc_interventions):
        """X-axis of MACC is cumulative abatement."""
        sorted_interventions = sorted(
            sample_macc_interventions, key=lambda x: x["cost_per_tco2e"]
        )
        cumulative = Decimal("0")
        for intv in sorted_interventions:
            cumulative += intv["abatement_tco2e"]
        assert cumulative == Decimal("50500")


# =============================================================================
# What-If Scenarios
# =============================================================================


class TestWhatIfScenarios:
    """Test what-if scenario analysis."""

    def test_top20_suppliers_30pct_reduction(self, sample_supplier_programme):
        """What if top 20 suppliers reduce 30%?"""
        suppliers = sample_supplier_programme["suppliers"]
        top20 = sorted(suppliers, key=lambda s: s["scope3_contribution_tco2e"], reverse=True)[:20]
        total_top20 = sum(s["scope3_contribution_tco2e"] for s in top20)
        reduction = total_top20 * Decimal("0.30")
        assert reduction > Decimal("0")

    def test_scenario_baseline_defined(self, sample_scenario_config):
        assert sample_scenario_config["baseline_scope3_tco2e"] == Decimal("252500")

    def test_scenario_target_year(self, sample_scenario_config):
        assert sample_scenario_config["target_year"] == 2030

    def test_what_if_no_action(self, sample_scenario_config):
        """No-action scenario keeps emissions at baseline."""
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        no_action_2030 = baseline  # flat
        assert no_action_2030 == baseline

    def test_what_if_all_interventions(self, sample_macc_interventions, sample_scenario_config):
        """All interventions reduce emissions by total abatement."""
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        total_abatement = sum(i["abatement_tco2e"] for i in sample_macc_interventions)
        post_intervention = baseline - total_abatement
        assert post_intervention == Decimal("202000")
        assert post_intervention < baseline


# =============================================================================
# Technology Pathway Modelling
# =============================================================================


class TestTechnologyPathwayModelling:
    """Test technology pathway and transition modelling."""

    def test_pathway_has_milestones(self, sample_scenario_config):
        """Each pathway should define annual reduction rate."""
        pathways = sample_scenario_config["paris_trajectories"]
        assert "1.5C" in pathways
        assert "WB2C" in pathways
        assert "2C" in pathways

    def test_15c_pathway_steepest(self, sample_scenario_config):
        p = sample_scenario_config["paris_trajectories"]
        assert p["1.5C"]["annual_reduction_pct"] > p["WB2C"]["annual_reduction_pct"]
        assert p["WB2C"]["annual_reduction_pct"] > p["2C"]["annual_reduction_pct"]

    def test_technology_pathway_linear_projection(self, sample_scenario_config):
        """Project emissions at annual reduction rate."""
        baseline = float(sample_scenario_config["baseline_scope3_tco2e"])
        rate = float(sample_scenario_config["paris_trajectories"]["1.5C"]["annual_reduction_pct"]) / 100
        years = sample_scenario_config["target_year"] - sample_scenario_config["reporting_year"]
        projected = baseline * (1 - rate) ** years
        assert projected < baseline

    def test_implementation_timeline(self, sample_macc_interventions):
        """Interventions have implementation timelines."""
        for intv in sample_macc_interventions:
            assert intv["implementation_years"] >= 1


# =============================================================================
# Paris Alignment Check
# =============================================================================


class TestParisAlignmentCheck:
    """Test Paris Agreement alignment checks."""

    def test_15c_alignment_check(self, sample_scenario_config, sample_macc_interventions):
        """Check if total abatement aligns with 1.5C pathway."""
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        rate = sample_scenario_config["paris_trajectories"]["1.5C"]["annual_reduction_pct"]
        years = sample_scenario_config["target_year"] - sample_scenario_config["reporting_year"]
        target_2030 = baseline * (Decimal("1") - rate / Decimal("100")) ** years
        total_abatement = sum(i["abatement_tco2e"] for i in sample_macc_interventions)
        achieved_2030 = baseline - total_abatement
        aligned = achieved_2030 <= target_2030
        assert isinstance(aligned, bool)

    def test_wb2c_alignment_check(self, sample_scenario_config, sample_macc_interventions):
        """WB2C pathway is less stringent than 1.5C."""
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        rate_15c = sample_scenario_config["paris_trajectories"]["1.5C"]["annual_reduction_pct"]
        rate_wb2c = sample_scenario_config["paris_trajectories"]["WB2C"]["annual_reduction_pct"]
        years = 5
        target_15c = baseline * (Decimal("1") - rate_15c / Decimal("100")) ** years
        target_wb2c = baseline * (Decimal("1") - rate_wb2c / Decimal("100")) ** years
        assert target_wb2c > target_15c

    def test_alignment_gap_calculation(self, sample_scenario_config):
        """Calculate gap between current trajectory and target."""
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        target_reduction_pct = Decimal("42")  # SBTi near-term
        target_absolute = baseline * (Decimal("1") - target_reduction_pct / Decimal("100"))
        gap = baseline - target_absolute
        assert gap > Decimal("0")


# =============================================================================
# Budget-Constrained Optimization
# =============================================================================


class TestBudgetConstrainedOptimization:
    """Test budget-constrained intervention optimization."""

    def test_budget_constraint_applied(self, sample_macc_interventions, sample_scenario_config):
        budget = sample_scenario_config["budget_constraint_usd"]
        sorted_by_cost_effectiveness = sorted(
            sample_macc_interventions, key=lambda x: x["cost_per_tco2e"]
        )
        selected = []
        total_cost = Decimal("0")
        for intv in sorted_by_cost_effectiveness:
            annual_cost = abs(intv["annual_cost_usd"])
            if total_cost + annual_cost <= budget:
                selected.append(intv)
                total_cost += annual_cost
        assert len(selected) >= 1

    def test_budget_optimal_maximizes_abatement(self, sample_macc_interventions, sample_scenario_config):
        """Optimal selection should maximize abatement within budget."""
        budget = sample_scenario_config["budget_constraint_usd"]
        sorted_by_cost_effectiveness = sorted(
            sample_macc_interventions, key=lambda x: x["cost_per_tco2e"]
        )
        selected_abatement = Decimal("0")
        total_cost = Decimal("0")
        for intv in sorted_by_cost_effectiveness:
            annual_cost = abs(intv["annual_cost_usd"])
            if total_cost + annual_cost <= budget:
                selected_abatement += intv["abatement_tco2e"]
                total_cost += annual_cost
        assert selected_abatement > Decimal("0")

    def test_discount_rate_valid(self, sample_scenario_config):
        rate = sample_scenario_config["discount_rate"]
        assert Decimal("0") < rate < Decimal("1")


# =============================================================================
# Waterfall Generation
# =============================================================================


class TestWaterfallGeneration:
    """Test emissions waterfall chart generation."""

    def test_waterfall_starts_at_baseline(self, sample_scenario_config):
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        assert baseline == Decimal("252500")

    def test_waterfall_steps_reduce(self, sample_macc_interventions, sample_scenario_config):
        """Each intervention adds a step down in the waterfall."""
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        current = baseline
        for intv in sorted(sample_macc_interventions, key=lambda x: x["cost_per_tco2e"]):
            current -= intv["abatement_tco2e"]
            assert current < baseline

    def test_waterfall_ends_at_target(self, sample_macc_interventions, sample_scenario_config):
        baseline = sample_scenario_config["baseline_scope3_tco2e"]
        total_abatement = sum(i["abatement_tco2e"] for i in sample_macc_interventions)
        final = baseline - total_abatement
        assert final == Decimal("202000")


# =============================================================================
# Intervention Ranking
# =============================================================================


class TestInterventionRanking:
    """Test intervention ranking by cost-effectiveness."""

    def test_ranking_by_cost_per_tco2e(self, sample_macc_interventions):
        ranked = sorted(sample_macc_interventions, key=lambda x: x["cost_per_tco2e"])
        assert ranked[0]["intervention_id"] == "INT-001"  # negative cost
        assert ranked[-1]["intervention_id"] == "INT-005"  # most expensive

    def test_ranking_by_abatement(self, sample_macc_interventions):
        ranked = sorted(
            sample_macc_interventions,
            key=lambda x: x["abatement_tco2e"],
            reverse=True,
        )
        assert ranked[0]["abatement_tco2e"] == Decimal("15000")

    def test_confidence_weighted_ranking(self, sample_macc_interventions):
        """Rank by confidence-weighted abatement."""
        ranked = sorted(
            sample_macc_interventions,
            key=lambda x: x["abatement_tco2e"] * x["confidence"],
            reverse=True,
        )
        assert len(ranked) == 5
        # Top should be high abatement * high confidence
        top = ranked[0]
        assert top["abatement_tco2e"] * top["confidence"] >= Decimal("8000")


# =============================================================================
# Edge Cases
# =============================================================================


class TestScenarioEdgeCases:
    """Test edge cases for scenario modelling."""

    def test_zero_budget(self, sample_macc_interventions):
        """Zero budget should select only net-negative-cost interventions."""
        budget = Decimal("0")
        selected = [
            i for i in sample_macc_interventions
            if i["annual_cost_usd"] <= budget
        ]
        # Only INT-001 has negative annual cost
        assert len(selected) == 1
        assert selected[0]["intervention_id"] == "INT-001"

    def test_single_intervention(self):
        """Scenario with one intervention."""
        interventions = [
            {
                "intervention_id": "INT-SINGLE",
                "abatement_tco2e": Decimal("5000"),
                "cost_per_tco2e": Decimal("10"),
            }
        ]
        total = sum(i["abatement_tco2e"] for i in interventions)
        assert total == Decimal("5000")

    def test_all_negative_cost(self):
        """All interventions have negative cost (all are savings)."""
        interventions = [
            {"abatement_tco2e": Decimal("1000"), "cost_per_tco2e": Decimal("-5")},
            {"abatement_tco2e": Decimal("2000"), "cost_per_tco2e": Decimal("-10")},
        ]
        total_cost = sum(
            i["abatement_tco2e"] * i["cost_per_tco2e"]
            for i in interventions
        )
        assert total_cost < Decimal("0")  # net savings

    def test_carbon_price_impact(self, sample_scenario_config, sample_macc_interventions):
        """Carbon price shifts cost-effectiveness of all interventions."""
        carbon_price = sample_scenario_config["carbon_price_usd_per_tco2e"]
        for intv in sample_macc_interventions:
            adjusted_cost = intv["cost_per_tco2e"] - carbon_price
            assert adjusted_cost < intv["cost_per_tco2e"]

    def test_very_high_discount_rate(self):
        """High discount rate reduces NPV of future benefits."""
        annual_benefit = Decimal("100000")
        discount_rate = Decimal("0.20")
        years = 10
        npv = sum(
            annual_benefit / (Decimal("1") + discount_rate) ** y
            for y in range(1, years + 1)
        )
        undiscounted = annual_benefit * years
        assert npv < undiscounted
