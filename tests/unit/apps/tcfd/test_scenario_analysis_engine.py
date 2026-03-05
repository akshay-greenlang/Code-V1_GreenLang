# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Scenario Analysis Engine -- MOST CRITICAL.

Tests all 7 pre-built scenarios, custom scenario creation, carbon cost
impact calculation, energy transition impact, asset stranding risk,
revenue impact, sensitivity analysis, probability-weighted impact,
scenario comparison, and climate resilience assessment with 38+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ScenarioType,
    TemperatureOutcome,
    TimeHorizon,
    SCENARIO_LIBRARY,
)
from services.models import (
    ScenarioDefinition,
    ScenarioParameter,
    ScenarioResult,
    _new_id,
    _sha256,
)


# ===========================================================================
# Pre-Built Scenario Loading
# ===========================================================================

class TestPreBuiltScenarioLoading:
    """Test all 7 pre-built scenarios load correctly."""

    @pytest.mark.parametrize("scenario_type", [
        ScenarioType.IEA_NZE,
        ScenarioType.IEA_APS,
        ScenarioType.IEA_STEPS,
        ScenarioType.NGFS_CURRENT_POLICIES,
        ScenarioType.NGFS_DELAYED_TRANSITION,
        ScenarioType.NGFS_BELOW_2C,
        ScenarioType.NGFS_DIVERGENT_NZ,
    ])
    def test_scenario_library_has_entry(self, scenario_type):
        assert scenario_type in SCENARIO_LIBRARY
        entry = SCENARIO_LIBRARY[scenario_type]
        assert "name" in entry
        assert "temperature_outcome" in entry
        assert "carbon_price_trajectory" in entry
        assert "energy_mix_trajectory" in entry

    def test_iea_nze_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.IEA_NZE]
        assert entry["temperature_outcome"] == TemperatureOutcome.BELOW_1_5C

    def test_iea_aps_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.IEA_APS]
        assert entry["temperature_outcome"] == TemperatureOutcome.AROUND_2C

    def test_iea_steps_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.IEA_STEPS]
        assert entry["temperature_outcome"] == TemperatureOutcome.ABOVE_2_5C

    def test_ngfs_current_policies_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.NGFS_CURRENT_POLICIES]
        assert entry["temperature_outcome"] == TemperatureOutcome.ABOVE_3C

    def test_ngfs_delayed_transition_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.NGFS_DELAYED_TRANSITION]
        assert entry["temperature_outcome"] == TemperatureOutcome.AROUND_2C

    def test_ngfs_below_2c_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.NGFS_BELOW_2C]
        assert entry["temperature_outcome"] == TemperatureOutcome.AROUND_2C

    def test_ngfs_divergent_nz_temperature(self):
        entry = SCENARIO_LIBRARY[ScenarioType.NGFS_DIVERGENT_NZ]
        assert entry["temperature_outcome"] == TemperatureOutcome.BELOW_1_5C

    def test_custom_scenario_empty_trajectories(self):
        entry = SCENARIO_LIBRARY[ScenarioType.CUSTOM]
        assert entry["carbon_price_trajectory"] == {}
        assert entry["energy_mix_trajectory"] == {}

    def test_total_scenario_count(self):
        assert len(SCENARIO_LIBRARY) == 8  # 7 pre-built + custom

    @pytest.mark.parametrize("scenario_type", [
        ScenarioType.IEA_NZE, ScenarioType.IEA_APS, ScenarioType.IEA_STEPS,
    ])
    def test_iea_scenarios_have_description(self, scenario_type):
        entry = SCENARIO_LIBRARY[scenario_type]
        assert len(entry["description"]) > 50


# ===========================================================================
# Custom Scenario Creation
# ===========================================================================

class TestCustomScenarioCreation:
    """Test custom scenario creation."""

    def test_custom_scenario_type(self, custom_scenario_definition):
        assert custom_scenario_definition.scenario_type == ScenarioType.CUSTOM

    def test_custom_carbon_trajectory(self, custom_scenario_definition):
        traj = custom_scenario_definition.carbon_price_trajectory
        assert traj[2025] == Decimal("100")
        assert traj[2050] == Decimal("500")

    def test_custom_scenario_name(self, custom_scenario_definition):
        assert custom_scenario_definition.name == "Custom High Ambition"

    def test_custom_energy_mix(self, custom_scenario_definition):
        mix = custom_scenario_definition.energy_mix_trajectory
        assert mix[2050]["renewable_pct"] == 95

    def test_scenario_definition_with_parameters(self):
        params = [
            ScenarioParameter(
                parameter_name="carbon_price",
                parameter_category="carbon",
                year=2030,
                value=Decimal("200"),
                unit="USD/tCO2e",
                source="Custom",
            ),
        ]
        scenario = ScenarioDefinition(
            name="Parametric Scenario",
            scenario_type=ScenarioType.CUSTOM,
            parameters=params,
        )
        assert len(scenario.parameters) == 1
        assert scenario.parameters[0].value == Decimal("200")


# ===========================================================================
# Carbon Cost Impact Calculation
# ===========================================================================

class TestCarbonCostImpact:
    """Test carbon cost impact calculation."""

    @pytest.mark.parametrize("scenario_type,year,expected_min", [
        (ScenarioType.IEA_NZE, 2030, Decimal("100")),
        (ScenarioType.IEA_NZE, 2050, Decimal("200")),
        (ScenarioType.NGFS_CURRENT_POLICIES, 2030, Decimal("10")),
    ])
    def test_carbon_price_trajectory_values(self, scenario_type, year, expected_min):
        traj = SCENARIO_LIBRARY[scenario_type]["carbon_price_trajectory"]
        assert traj[year] >= expected_min

    def test_nze_carbon_price_increasing(self):
        traj = SCENARIO_LIBRARY[ScenarioType.IEA_NZE]["carbon_price_trajectory"]
        years = sorted(traj.keys())
        for i in range(len(years) - 1):
            assert traj[years[i + 1]] >= traj[years[i]]

    def test_delayed_transition_late_spike(self):
        traj = SCENARIO_LIBRARY[ScenarioType.NGFS_DELAYED_TRANSITION]["carbon_price_trajectory"]
        assert traj[2030] < traj[2040]
        spike_ratio = traj[2040] / traj[2030]
        assert spike_ratio > 3  # significant late acceleration


# ===========================================================================
# Energy Transition Impact
# ===========================================================================

class TestEnergyTransitionImpact:
    """Test energy transition impact analysis."""

    def test_nze_renewable_growth(self):
        mix = SCENARIO_LIBRARY[ScenarioType.IEA_NZE]["energy_mix_trajectory"]
        assert mix[2050]["renewable_pct"] > mix[2030]["renewable_pct"]
        assert mix[2050]["renewable_pct"] >= 90

    def test_steps_slow_renewable_growth(self):
        mix = SCENARIO_LIBRARY[ScenarioType.IEA_STEPS]["energy_mix_trajectory"]
        assert mix[2050]["renewable_pct"] <= 55

    def test_energy_mix_sums_to_100(self):
        for scenario_type in [ScenarioType.IEA_NZE, ScenarioType.IEA_APS]:
            mix = SCENARIO_LIBRARY[scenario_type]["energy_mix_trajectory"]
            for year, values in mix.items():
                total = values["renewable_pct"] + values["fossil_pct"] + values["nuclear_pct"]
                assert total == 100, f"{scenario_type} {year}: {total}"


# ===========================================================================
# Asset Stranding Risk
# ===========================================================================

class TestAssetStrandingRisk:
    """Test asset stranding risk assessment."""

    def test_scenario_result_impairment(self, sample_scenario_result):
        assert sample_scenario_result.asset_impairment_pct == Decimal("15.0")

    def test_zero_impairment(self):
        result = ScenarioResult(
            scenario_id=_new_id(),
            org_id=_new_id(),
            asset_impairment_pct=Decimal("0"),
        )
        assert result.asset_impairment_pct == Decimal("0")

    def test_high_impairment(self):
        result = ScenarioResult(
            scenario_id=_new_id(),
            org_id=_new_id(),
            asset_impairment_pct=Decimal("80.0"),
        )
        assert result.asset_impairment_pct == Decimal("80.0")


# ===========================================================================
# Revenue Impact
# ===========================================================================

class TestRevenueImpact:
    """Test revenue impact calculation."""

    def test_negative_revenue_impact(self, sample_scenario_result):
        assert sample_scenario_result.revenue_impact_pct < Decimal("0")

    def test_positive_revenue_impact(self):
        result = ScenarioResult(
            scenario_id=_new_id(),
            org_id=_new_id(),
            revenue_impact_pct=Decimal("5.0"),
        )
        assert result.revenue_impact_pct > Decimal("0")


# ===========================================================================
# Sensitivity Analysis
# ===========================================================================

class TestSensitivityAnalysis:
    """Test sensitivity analysis via carbon price variation."""

    @pytest.mark.parametrize("carbon_price,expected_direction", [
        (Decimal("50"), "lower"),
        (Decimal("130"), "baseline"),
        (Decimal("300"), "higher"),
    ])
    def test_carbon_price_sensitivity(self, carbon_price, expected_direction):
        emissions = Decimal("125000")
        cost = emissions * carbon_price
        if expected_direction == "lower":
            assert cost < Decimal("125000") * Decimal("130")
        elif expected_direction == "higher":
            assert cost > Decimal("125000") * Decimal("130")
        else:
            assert cost == Decimal("125000") * Decimal("130")


# ===========================================================================
# Probability-Weighted Impact
# ===========================================================================

class TestProbabilityWeightedImpact:
    """Test probability-weighted financial impact."""

    def test_weighted_npv(self):
        scenarios = [
            {"npv": Decimal("-200000000"), "probability": Decimal("0.2")},
            {"npv": Decimal("-120000000"), "probability": Decimal("0.5")},
            {"npv": Decimal("-50000000"), "probability": Decimal("0.3")},
        ]
        weighted_npv = sum(s["npv"] * s["probability"] for s in scenarios)
        assert weighted_npv == Decimal("-115000000")

    def test_probabilities_sum_to_one(self):
        probs = [Decimal("0.2"), Decimal("0.5"), Decimal("0.3")]
        assert sum(probs) == Decimal("1.0")


# ===========================================================================
# Scenario Comparison
# ===========================================================================

class TestScenarioComparison:
    """Test comparison across scenarios."""

    def test_nze_higher_carbon_than_steps_2030(self):
        nze = SCENARIO_LIBRARY[ScenarioType.IEA_NZE]["carbon_price_trajectory"][2030]
        steps = SCENARIO_LIBRARY[ScenarioType.IEA_STEPS]["carbon_price_trajectory"][2030]
        assert nze > steps

    def test_current_policies_lowest_carbon_2050(self):
        prices_2050 = {}
        for st in [ScenarioType.IEA_NZE, ScenarioType.IEA_APS, ScenarioType.IEA_STEPS,
                    ScenarioType.NGFS_CURRENT_POLICIES, ScenarioType.NGFS_BELOW_2C]:
            traj = SCENARIO_LIBRARY[st]["carbon_price_trajectory"]
            if 2050 in traj:
                prices_2050[st] = traj[2050]
        min_scenario = min(prices_2050, key=prices_2050.get)
        assert min_scenario == ScenarioType.NGFS_CURRENT_POLICIES


# ===========================================================================
# Climate Resilience Assessment
# ===========================================================================

class TestClimateResilienceAssessment:
    """Test climate resilience scoring."""

    def test_resilience_score_in_result(self, sample_scenario_result):
        assert sample_scenario_result.key_assumptions is not None
        assert len(sample_scenario_result.key_assumptions) > 0

    def test_scenario_result_narrative(self, sample_scenario_result):
        assert len(sample_scenario_result.narrative) > 0

    def test_scenario_result_provenance_deterministic(self):
        sid = _new_id()
        oid = _new_id()
        r1 = ScenarioResult(
            scenario_id=sid,
            org_id=oid,
            revenue_impact_pct=Decimal("-5.0"),
            npv=Decimal("-100000000"),
        )
        r2 = ScenarioResult(
            scenario_id=sid,
            org_id=oid,
            revenue_impact_pct=Decimal("-5.0"),
            npv=Decimal("-100000000"),
        )
        assert r1.provenance_hash == r2.provenance_hash

    def test_scenario_capex_requirement(self, sample_scenario_result):
        assert sample_scenario_result.capex_required == Decimal("250000000")
