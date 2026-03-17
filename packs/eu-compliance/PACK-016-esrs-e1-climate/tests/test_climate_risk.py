# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Climate Risk Engine Tests
============================================================

Unit tests for ClimateRiskEngine (Engine 8) covering physical risk
assessment, transition risk assessment, opportunity assessment,
build_risk_assessment, scenario impact, time horizon breakdown,
completeness validation, and E1-9 data point extraction.

ESRS E1-9: Anticipated financial effects from material physical and
transition risks and potential climate-related opportunities.

Target: 55+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the climate_risk engine module."""
    return _load_engine("climate_risk")


@pytest.fixture
def engine(mod):
    """Create a fresh ClimateRiskEngine instance."""
    return mod.ClimateRiskEngine()


@pytest.fixture
def sample_physical_risk(mod):
    """Create a sample physical climate risk (acute flooding)."""
    return mod.PhysicalRisk(
        risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
        name="Coastal Facility Flood Risk",
        affected_assets_value=Decimal("50000000.00"),
        affected_assets_pct=Decimal("15.0"),
        affected_revenue=Decimal("8000000.00"),
        affected_revenue_pct=Decimal("10.0"),
        likelihood=mod.LikelihoodLevel.HIGH,
        time_horizon=mod.RiskTimeHorizon.MEDIUM_TERM_3_10Y,
        scenario=mod.ClimateScenario.RCP_4_5,
        adaptation_cost=Decimal("500000.00"),
        location="Rotterdam, Netherlands",
    )


@pytest.fixture
def sample_transition_risk(mod):
    """Create a sample transition climate risk (carbon pricing)."""
    return mod.TransitionRisk(
        risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
        name="EU ETS Carbon Price Increase",
        affected_assets_value=Decimal("30000000.00"),
        affected_assets_pct=Decimal("10.0"),
        affected_revenue_pct=Decimal("12.0"),
        likelihood=mod.LikelihoodLevel.VERY_HIGH,
        time_horizon=mod.RiskTimeHorizon.SHORT_TERM_0_3Y,
        scenario=mod.ClimateScenario.IEA_NZE,
        mitigation_cost=Decimal("2000000.00"),
    )


@pytest.fixture
def sample_opportunity(mod):
    """Create a sample climate-related opportunity."""
    return mod.ClimateOpportunity(
        opportunity_type=mod.ClimateOpportunityType.ENERGY_SOURCE,
        name="On-site Solar PV Installation",
        estimated_revenue_impact=Decimal("500000.00"),
        estimated_cost_savings=Decimal("300000.00"),
        investment_required=Decimal("2000000.00"),
        time_horizon=mod.RiskTimeHorizon.MEDIUM_TERM_3_10Y,
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestRiskEnums:
    """Tests for climate risk enums."""

    def test_physical_risk_type_count(self, mod):
        """PhysicalRiskType has at least 8 values."""
        assert len(mod.PhysicalRiskType) >= 8
        values = {m.value for m in mod.PhysicalRiskType}
        assert "acute_flooding" in values
        assert "acute_wildfire" in values
        assert "acute_storm" in values
        assert "acute_heatwave" in values
        assert "chronic_sea_level" in values
        assert "chronic_temperature" in values

    def test_transition_risk_type_count(self, mod):
        """TransitionRiskType has at least 6 values."""
        assert len(mod.TransitionRiskType) >= 6
        values = {m.value for m in mod.TransitionRiskType}
        assert "policy_carbon_pricing" in values
        assert "policy_regulation" in values
        assert "technology_disruption" in values
        assert "market_shift" in values
        assert "reputation" in values
        assert "legal_liability" in values

    def test_climate_opportunity_type_count(self, mod):
        """ClimateOpportunityType has at least 5 values."""
        assert len(mod.ClimateOpportunityType) >= 5
        values = {m.value for m in mod.ClimateOpportunityType}
        assert "resource_efficiency" in values
        assert "energy_source" in values
        assert "products_services" in values
        assert "markets" in values
        assert "resilience" in values

    def test_risk_time_horizon_values(self, mod):
        """RiskTimeHorizon has 3 values (short/medium/long)."""
        assert len(mod.RiskTimeHorizon) == 3
        values = {m.value for m in mod.RiskTimeHorizon}
        assert "short_term_0_3y" in values
        assert "medium_term_3_10y" in values
        assert "long_term_10_plus" in values

    def test_climate_scenario_count(self, mod):
        """ClimateScenario has at least 8 values."""
        assert len(mod.ClimateScenario) >= 8
        values = {m.value for m in mod.ClimateScenario}
        assert "rcp_2_6" in values
        assert "rcp_4_5" in values
        assert "rcp_8_5" in values
        assert "iea_nze" in values
        assert "iea_steps" in values
        assert "ngfs_orderly" in values
        assert "ngfs_disorderly" in values
        assert "ngfs_hot_house" in values

    def test_likelihood_level_count(self, mod):
        """LikelihoodLevel has 5 values."""
        assert len(mod.LikelihoodLevel) == 5
        values = {m.value for m in mod.LikelihoodLevel}
        assert values == {"very_low", "low", "medium", "high", "very_high"}


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestRiskConstants:
    """Tests for climate risk constants."""

    def test_e1_9_datapoints_exist(self, mod):
        """E1_9_DATAPOINTS is a non-empty dict with at least 18 entries."""
        assert len(mod.E1_9_DATAPOINTS) >= 18

    def test_physical_risk_descriptions(self, mod):
        """PHYSICAL_RISK_DESCRIPTIONS has entries for all physical risk types."""
        assert len(mod.PHYSICAL_RISK_DESCRIPTIONS) >= 8
        for rt in mod.PhysicalRiskType:
            assert rt.value in mod.PHYSICAL_RISK_DESCRIPTIONS

    def test_transition_risk_descriptions(self, mod):
        """TRANSITION_RISK_DESCRIPTIONS has entries for all transition risk types."""
        assert len(mod.TRANSITION_RISK_DESCRIPTIONS) >= 6
        for rt in mod.TransitionRiskType:
            assert rt.value in mod.TRANSITION_RISK_DESCRIPTIONS

    def test_likelihood_probabilities(self, mod):
        """LIKELIHOOD_PROBABILITIES has entries for all likelihood levels."""
        assert len(mod.LIKELIHOOD_PROBABILITIES) == 5
        for ll in mod.LikelihoodLevel:
            assert ll.value in mod.LIKELIHOOD_PROBABILITIES
            entry = mod.LIKELIHOOD_PROBABILITIES[ll.value]
            assert "midpoint" in entry
            assert "weight" in entry

    def test_scenario_descriptions(self, mod):
        """SCENARIO_DESCRIPTIONS has entries for all climate scenarios."""
        assert len(mod.SCENARIO_DESCRIPTIONS) >= 8
        for sc in mod.ClimateScenario:
            assert sc.value in mod.SCENARIO_DESCRIPTIONS
            entry = mod.SCENARIO_DESCRIPTIONS[sc.value]
            assert "name" in entry
            assert "warming" in entry
            assert "description" in entry

    def test_damage_function_params(self, mod):
        """DAMAGE_FUNCTION_PARAMS has entries for all physical risk types."""
        assert len(mod.DAMAGE_FUNCTION_PARAMS) >= 8
        for rt in mod.PhysicalRiskType:
            assert rt.value in mod.DAMAGE_FUNCTION_PARAMS
            factors = mod.DAMAGE_FUNCTION_PARAMS[rt.value]
            assert "rcp_2_6" in factors
            assert "rcp_4_5" in factors
            assert "rcp_8_5" in factors

    def test_damage_function_increasing_with_warming(self, mod):
        """Damage factors increase with warming scenario severity."""
        for rt in mod.PhysicalRiskType:
            factors = mod.DAMAGE_FUNCTION_PARAMS[rt.value]
            assert factors["rcp_2_6"] <= factors["rcp_4_5"]
            assert factors["rcp_4_5"] <= factors["rcp_8_5"]

    def test_opportunity_descriptions(self, mod):
        """OPPORTUNITY_DESCRIPTIONS has entries for all opportunity types."""
        assert len(mod.OPPORTUNITY_DESCRIPTIONS) >= 5
        for ot in mod.ClimateOpportunityType:
            assert ot.value in mod.OPPORTUNITY_DESCRIPTIONS

    def test_likelihood_weights_ordered(self, mod):
        """Likelihood weights increase from very_low to very_high."""
        probs = mod.LIKELIHOOD_PROBABILITIES
        assert probs["very_low"]["weight"] < probs["low"]["weight"]
        assert probs["low"]["weight"] < probs["medium"]["weight"]
        assert probs["medium"]["weight"] < probs["high"]["weight"]
        assert probs["high"]["weight"] < probs["very_high"]["weight"]


# ===========================================================================
# Physical Risk Model Tests
# ===========================================================================


class TestPhysicalRiskModel:
    """Tests for PhysicalRisk Pydantic model."""

    def test_create_valid_physical_risk(self, mod):
        """Create a valid PhysicalRisk with minimal fields."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
        )
        assert risk.risk_type == mod.PhysicalRiskType.ACUTE_FLOODING
        assert len(risk.risk_id) > 0

    def test_default_values(self, mod):
        """PhysicalRisk defaults are sensible."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.CHRONIC_WATER_STRESS,
        )
        assert risk.affected_assets_value == Decimal("0.00")
        assert risk.affected_assets_pct == Decimal("0.00")
        assert risk.likelihood == mod.LikelihoodLevel.MEDIUM
        assert risk.time_horizon == mod.RiskTimeHorizon.MEDIUM_TERM_3_10Y
        assert risk.scenario == mod.ClimateScenario.RCP_4_5
        assert risk.currency == "EUR"

    def test_physical_risk_with_all_fields(self, mod):
        """PhysicalRisk with all fields set."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_WILDFIRE,
            name="Southern Europe Wildfire Risk",
            affected_assets_value=Decimal("25000000.00"),
            affected_assets_pct=Decimal("8.0"),
            affected_revenue=Decimal("5000000.00"),
            affected_revenue_pct=Decimal("6.0"),
            likelihood=mod.LikelihoodLevel.HIGH,
            time_horizon=mod.RiskTimeHorizon.SHORT_TERM_0_3Y,
            scenario=mod.ClimateScenario.RCP_8_5,
            estimated_annual_loss=Decimal("750000.00"),
            adaptation_cost=Decimal("200000.00"),
            residual_risk_value=Decimal("550000.00"),
            location="Andalusia, Spain",
        )
        assert risk.name == "Southern Europe Wildfire Risk"
        assert risk.affected_assets_value == Decimal("25000000.00")
        assert risk.location == "Andalusia, Spain"

    def test_physical_risk_unique_ids(self, mod):
        """Each PhysicalRisk gets a unique risk_id."""
        r1 = mod.PhysicalRisk(risk_type=mod.PhysicalRiskType.ACUTE_STORM)
        r2 = mod.PhysicalRisk(risk_type=mod.PhysicalRiskType.ACUTE_STORM)
        assert r1.risk_id != r2.risk_id


# ===========================================================================
# Transition Risk Model Tests
# ===========================================================================


class TestTransitionRiskModel:
    """Tests for TransitionRisk Pydantic model."""

    def test_create_valid_transition_risk(self, mod):
        """Create a valid TransitionRisk with minimal fields."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
        )
        assert risk.risk_type == mod.TransitionRiskType.POLICY_CARBON_PRICING
        assert len(risk.risk_id) > 0

    def test_default_scenario_is_iea_nze(self, mod):
        """Default scenario for TransitionRisk is IEA_NZE."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.TECHNOLOGY_DISRUPTION,
        )
        assert risk.scenario == mod.ClimateScenario.IEA_NZE

    def test_transition_risk_with_all_fields(self, mod):
        """TransitionRisk with all fields set."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.MARKET_SHIFT,
            name="Demand Shift to EVs",
            affected_assets_value=Decimal("40000000.00"),
            affected_assets_pct=Decimal("20.0"),
            affected_revenue_pct=Decimal("15.0"),
            likelihood=mod.LikelihoodLevel.HIGH,
            time_horizon=mod.RiskTimeHorizon.MEDIUM_TERM_3_10Y,
            scenario=mod.ClimateScenario.NGFS_ORDERLY,
            estimated_financial_impact=Decimal("6000000.00"),
            mitigation_cost=Decimal("1500000.00"),
            residual_risk_value=Decimal("4500000.00"),
        )
        assert risk.affected_revenue_pct == Decimal("15.0")
        assert risk.mitigation_cost == Decimal("1500000.00")

    def test_transition_risk_unique_ids(self, mod):
        """Each TransitionRisk gets a unique risk_id."""
        r1 = mod.TransitionRisk(risk_type=mod.TransitionRiskType.REPUTATION)
        r2 = mod.TransitionRisk(risk_type=mod.TransitionRiskType.REPUTATION)
        assert r1.risk_id != r2.risk_id


# ===========================================================================
# Climate Opportunity Model Tests
# ===========================================================================


class TestOpportunityModel:
    """Tests for ClimateOpportunity Pydantic model."""

    def test_create_valid_opportunity(self, mod):
        """Create a valid ClimateOpportunity with minimal fields."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESOURCE_EFFICIENCY,
        )
        assert opp.opportunity_type == mod.ClimateOpportunityType.RESOURCE_EFFICIENCY
        assert len(opp.opportunity_id) > 0

    def test_opportunity_default_values(self, mod):
        """ClimateOpportunity defaults are sensible."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.MARKETS,
        )
        assert opp.estimated_revenue_impact == Decimal("0.00")
        assert opp.estimated_cost_savings == Decimal("0.00")
        assert opp.investment_required == Decimal("0.00")
        assert opp.time_horizon == mod.RiskTimeHorizon.MEDIUM_TERM_3_10Y
        assert opp.currency == "EUR"

    def test_opportunity_with_all_fields(self, mod):
        """ClimateOpportunity with all fields set."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.PRODUCTS_SERVICES,
            name="Low-Carbon Product Line",
            estimated_revenue_impact=Decimal("2000000.00"),
            estimated_cost_savings=Decimal("0.00"),
            investment_required=Decimal("5000000.00"),
            time_horizon=mod.RiskTimeHorizon.LONG_TERM_10_PLUS,
        )
        assert opp.name == "Low-Carbon Product Line"
        assert opp.estimated_revenue_impact == Decimal("2000000.00")

    def test_opportunity_unique_ids(self, mod):
        """Each ClimateOpportunity gets a unique opportunity_id."""
        o1 = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESILIENCE,
        )
        o2 = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESILIENCE,
        )
        assert o1.opportunity_id != o2.opportunity_id


# ===========================================================================
# Assess Physical Risk Tests
# ===========================================================================


class TestAssessPhysicalRisk:
    """Tests for assess_physical_risk method."""

    def test_assess_calculates_loss(self, engine, mod):
        """assess_physical_risk calculates estimated_annual_loss from damage function."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            scenario=mod.ClimateScenario.RCP_4_5,
        )
        assessed = engine.assess_physical_risk(risk)
        # loss = 50000000 * 0.012 (rcp_4_5 flooding) * 0.65 (high) = 390000
        assert assessed.estimated_annual_loss > Decimal("0")
        assert float(assessed.estimated_annual_loss) == pytest.approx(390000.0, abs=1.0)

    def test_assess_preserves_provided_loss(self, engine, mod):
        """assess_physical_risk preserves estimated_annual_loss if already set."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            estimated_annual_loss=Decimal("999999.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            scenario=mod.ClimateScenario.RCP_4_5,
        )
        assessed = engine.assess_physical_risk(risk)
        assert assessed.estimated_annual_loss == Decimal("999999.00")

    def test_assess_calculates_residual_risk(self, engine, mod):
        """Residual risk = estimated_annual_loss - adaptation_cost, floored at 0."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_STORM,
            affected_assets_value=Decimal("10000000.00"),
            likelihood=mod.LikelihoodLevel.MEDIUM,
            scenario=mod.ClimateScenario.RCP_4_5,
            adaptation_cost=Decimal("10000.00"),
        )
        assessed = engine.assess_physical_risk(risk)
        expected_residual = assessed.estimated_annual_loss - Decimal("10000.00")
        assert assessed.residual_risk_value == max(expected_residual, Decimal("0.00"))

    def test_assess_residual_floors_at_zero(self, engine, mod):
        """Residual risk is floored at zero when adaptation exceeds loss."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_HEATWAVE,
            affected_assets_value=Decimal("100000.00"),
            likelihood=mod.LikelihoodLevel.VERY_LOW,
            scenario=mod.ClimateScenario.RCP_2_6,
            adaptation_cost=Decimal("99999999.00"),
        )
        assessed = engine.assess_physical_risk(risk)
        assert assessed.residual_risk_value == Decimal("0.00")

    def test_assess_generates_provenance_hash(self, engine, mod):
        """assess_physical_risk sets a 64-char provenance hash."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.CHRONIC_SEA_LEVEL,
            affected_assets_value=Decimal("20000000.00"),
        )
        assessed = engine.assess_physical_risk(risk)
        assert len(assessed.provenance_hash) == 64
        int(assessed.provenance_hash, 16)  # valid hex

    def test_assess_auto_fills_name(self, engine, mod):
        """assess_physical_risk fills name from description if empty."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.CHRONIC_WATER_STRESS,
        )
        assessed = engine.assess_physical_risk(risk)
        assert len(assessed.name) > 0

    def test_assess_rcp85_higher_loss_than_rcp26(self, engine, mod):
        """RCP 8.5 produces higher loss than RCP 2.6 for same risk."""
        risk_26 = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            scenario=mod.ClimateScenario.RCP_2_6,
        )
        eng1 = mod.ClimateRiskEngine()
        assessed_26 = eng1.assess_physical_risk(risk_26)

        risk_85 = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            scenario=mod.ClimateScenario.RCP_8_5,
        )
        eng2 = mod.ClimateRiskEngine()
        assessed_85 = eng2.assess_physical_risk(risk_85)

        assert assessed_85.estimated_annual_loss > assessed_26.estimated_annual_loss


# ===========================================================================
# Assess Transition Risk Tests
# ===========================================================================


class TestAssessTransitionRisk:
    """Tests for assess_transition_risk method."""

    def test_assess_calculates_impact(self, engine, mod):
        """assess_transition_risk calculates financial impact."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            affected_assets_value=Decimal("30000000.00"),
            likelihood=mod.LikelihoodLevel.VERY_HIGH,
        )
        assessed = engine.assess_transition_risk(risk)
        # impact = 30000000 * 0.90 (very_high weight) = 27000000
        assert assessed.estimated_financial_impact > Decimal("0")
        assert float(assessed.estimated_financial_impact) == pytest.approx(
            27000000.0, abs=1.0
        )

    def test_assess_preserves_provided_impact(self, engine, mod):
        """assess_transition_risk preserves estimated_financial_impact if set."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.TECHNOLOGY_DISRUPTION,
            affected_assets_value=Decimal("10000000.00"),
            estimated_financial_impact=Decimal("555555.00"),
        )
        assessed = engine.assess_transition_risk(risk)
        assert assessed.estimated_financial_impact == Decimal("555555.00")

    def test_assess_calculates_residual(self, engine, mod):
        """Residual risk = impact - mitigation_cost, floored at 0."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.MARKET_SHIFT,
            affected_assets_value=Decimal("20000000.00"),
            likelihood=mod.LikelihoodLevel.MEDIUM,
            mitigation_cost=Decimal("1000000.00"),
        )
        assessed = engine.assess_transition_risk(risk)
        expected_residual = assessed.estimated_financial_impact - Decimal("1000000.00")
        assert assessed.residual_risk_value == max(expected_residual, Decimal("0.00"))

    def test_assess_generates_provenance_hash(self, engine, mod):
        """assess_transition_risk sets a 64-char provenance hash."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.LEGAL_LIABILITY,
            affected_assets_value=Decimal("5000000.00"),
        )
        assessed = engine.assess_transition_risk(risk)
        assert len(assessed.provenance_hash) == 64

    def test_assess_auto_fills_name(self, engine, mod):
        """assess_transition_risk fills name from description if empty."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.REPUTATION,
        )
        assessed = engine.assess_transition_risk(risk)
        assert len(assessed.name) > 0


# ===========================================================================
# Assess Opportunity Tests
# ===========================================================================


class TestAssessOpportunity:
    """Tests for assess_opportunity method."""

    def test_assess_opportunity_basic(self, engine, mod):
        """assess_opportunity registers opportunity and sets provenance."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.ENERGY_SOURCE,
            estimated_revenue_impact=Decimal("500000.00"),
            estimated_cost_savings=Decimal("200000.00"),
        )
        assessed = engine.assess_opportunity(opp)
        assert len(assessed.provenance_hash) == 64
        int(assessed.provenance_hash, 16)

    def test_assess_opportunity_auto_fills_name(self, engine, mod):
        """assess_opportunity fills name from description if empty."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESOURCE_EFFICIENCY,
        )
        assessed = engine.assess_opportunity(opp)
        assert len(assessed.name) > 0

    def test_assess_opportunity_preserves_fields(self, engine, mod):
        """assess_opportunity preserves all provided fields."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.PRODUCTS_SERVICES,
            name="Green Product Line",
            estimated_revenue_impact=Decimal("1000000.00"),
            estimated_cost_savings=Decimal("50000.00"),
            investment_required=Decimal("3000000.00"),
            time_horizon=mod.RiskTimeHorizon.LONG_TERM_10_PLUS,
        )
        assessed = engine.assess_opportunity(opp)
        assert assessed.name == "Green Product Line"
        assert assessed.estimated_revenue_impact == Decimal("1000000.00")
        assert assessed.investment_required == Decimal("3000000.00")


# ===========================================================================
# Build Risk Assessment Tests
# ===========================================================================


class TestBuildAssessment:
    """Tests for build_risk_assessment method."""

    def test_basic_assessment(
        self, engine, sample_physical_risk, sample_transition_risk, sample_opportunity
    ):
        """Build a basic risk assessment with all three categories."""
        p = engine.assess_physical_risk(sample_physical_risk)
        t = engine.assess_transition_risk(sample_transition_risk)
        o = engine.assess_opportunity(sample_opportunity)
        result = engine.build_risk_assessment()
        assert result is not None
        assert result.total_physical_risks >= 1
        assert result.total_transition_risks >= 1
        assert result.total_opportunities >= 1
        assert result.processing_time_ms >= 0.0

    def test_assessment_aggregates_physical(self, mod):
        """Assessment sums physical risk exposures."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("10000000.00"),
            estimated_annual_loss=Decimal("100000.00"),
        ))
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_WILDFIRE,
            affected_assets_value=Decimal("5000000.00"),
            estimated_annual_loss=Decimal("50000.00"),
        ))
        result = eng.build_risk_assessment()
        assert float(result.total_physical_risk_exposure) == pytest.approx(
            150000.0, abs=1.0
        )

    def test_assessment_aggregates_transition(self, mod):
        """Assessment sums transition risk exposures."""
        eng = mod.ClimateRiskEngine()
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            estimated_financial_impact=Decimal("200000.00"),
        ))
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.MARKET_SHIFT,
            estimated_financial_impact=Decimal("300000.00"),
        ))
        result = eng.build_risk_assessment()
        assert float(result.total_transition_risk_exposure) == pytest.approx(
            500000.0, abs=1.0
        )

    def test_assessment_aggregates_opportunities(self, mod):
        """Assessment sums opportunity values (revenue + savings)."""
        eng = mod.ClimateRiskEngine()
        eng.assess_opportunity(mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.ENERGY_SOURCE,
            estimated_revenue_impact=Decimal("100000.00"),
            estimated_cost_savings=Decimal("50000.00"),
        ))
        eng.assess_opportunity(mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESOURCE_EFFICIENCY,
            estimated_revenue_impact=Decimal("0.00"),
            estimated_cost_savings=Decimal("80000.00"),
        ))
        result = eng.build_risk_assessment()
        # Total = (100000 + 50000) + (0 + 80000) = 230000
        assert float(result.total_opportunity_value) == pytest.approx(
            230000.0, abs=1.0
        )

    def test_assessment_provenance_hash(self, mod):
        """Assessment result has a 64-char provenance hash."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            estimated_annual_loss=Decimal("100000.00"),
        ))
        result = eng.build_risk_assessment()
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_assessment_scenarios_used(self, mod):
        """Assessment tracks which scenarios were used."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            scenario=mod.ClimateScenario.RCP_4_5,
        ))
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            scenario=mod.ClimateScenario.IEA_NZE,
        ))
        result = eng.build_risk_assessment()
        assert "rcp_4_5" in result.scenarios_used
        assert "iea_nze" in result.scenarios_used

    def test_assessment_risks_by_type(self, mod):
        """Assessment counts risks by type."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
        ))
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
        ))
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_WILDFIRE,
        ))
        result = eng.build_risk_assessment()
        assert result.physical_risks_by_type.get("acute_flooding", 0) == 2
        assert result.physical_risks_by_type.get("acute_wildfire", 0) == 1

    def test_empty_assessment(self, mod):
        """Assessment with no risks or opportunities."""
        eng = mod.ClimateRiskEngine()
        result = eng.build_risk_assessment()
        assert result.total_physical_risks == 0
        assert result.total_transition_risks == 0
        assert result.total_opportunities == 0
        assert result.total_physical_risk_exposure == Decimal("0.00")
        assert result.net_climate_financial_impact == Decimal("0.00")

    def test_assessment_uses_explicit_lists(self, mod):
        """build_risk_assessment accepts explicit lists instead of registry."""
        eng = mod.ClimateRiskEngine()
        physical = [mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.CHRONIC_TEMPERATURE,
            estimated_annual_loss=Decimal("50000.00"),
        )]
        result = eng.build_risk_assessment(physical_risks=physical)
        assert result.total_physical_risks == 1
        assert float(result.total_physical_risk_exposure) == pytest.approx(
            50000.0, abs=1.0
        )


# ===========================================================================
# Net Impact Tests
# ===========================================================================


class TestNetImpact:
    """Tests for net climate financial impact calculation."""

    def test_net_impact_risks_minus_opportunities(self, mod):
        """Net impact = total risks - total opportunities."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            estimated_annual_loss=Decimal("200000.00"),
        ))
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            estimated_financial_impact=Decimal("300000.00"),
        ))
        eng.assess_opportunity(mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.ENERGY_SOURCE,
            estimated_revenue_impact=Decimal("100000.00"),
            estimated_cost_savings=Decimal("50000.00"),
        ))
        result = eng.build_risk_assessment()
        # net = (200000 + 300000) - (100000 + 50000) = 350000
        assert float(result.net_climate_financial_impact) == pytest.approx(
            350000.0, abs=1.0
        )

    def test_net_impact_negative_when_opportunities_exceed_risks(self, mod):
        """Net impact is negative when opportunities exceed risks."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_HEATWAVE,
            estimated_annual_loss=Decimal("10000.00"),
        ))
        eng.assess_opportunity(mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESOURCE_EFFICIENCY,
            estimated_revenue_impact=Decimal("0.00"),
            estimated_cost_savings=Decimal("500000.00"),
        ))
        result = eng.build_risk_assessment()
        assert result.net_climate_financial_impact < Decimal("0")

    def test_adaptation_mitigation_totals(self, mod):
        """Assessment sums adaptation and mitigation costs."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            estimated_annual_loss=Decimal("100000.00"),
            adaptation_cost=Decimal("25000.00"),
        ))
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_REGULATION,
            estimated_financial_impact=Decimal("200000.00"),
            mitigation_cost=Decimal("50000.00"),
        ))
        result = eng.build_risk_assessment()
        assert float(result.total_adaptation_cost) == pytest.approx(25000.0, abs=1.0)
        assert float(result.total_mitigation_cost) == pytest.approx(50000.0, abs=1.0)


# ===========================================================================
# Scenario Impact Tests
# ===========================================================================


class TestScenarioImpact:
    """Tests for calculate_scenario_impact method."""

    def test_scenario_impact_filters_by_scenario(self, mod):
        """calculate_scenario_impact returns only risks for the given scenario."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            estimated_annual_loss=Decimal("100000.00"),
            scenario=mod.ClimateScenario.RCP_4_5,
        ))
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_WILDFIRE,
            estimated_annual_loss=Decimal("50000.00"),
            scenario=mod.ClimateScenario.RCP_8_5,
        ))
        result = eng.build_risk_assessment()
        impact = eng.calculate_scenario_impact(result, mod.ClimateScenario.RCP_4_5)
        assert impact["physical_risks_count"] == 1
        assert "100000" in impact["total_physical_exposure"]

    def test_scenario_impact_includes_description(self, mod):
        """Scenario impact includes scenario name and warming details."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            scenario=mod.ClimateScenario.RCP_8_5,
        ))
        result = eng.build_risk_assessment()
        impact = eng.calculate_scenario_impact(result, mod.ClimateScenario.RCP_8_5)
        assert impact["scenario"] == "rcp_8_5"
        assert "RCP 8.5" in impact["scenario_name"]
        assert "4.3" in impact["warming"]

    def test_scenario_impact_provenance_hash(self, mod):
        """Scenario impact has a 64-char provenance hash."""
        eng = mod.ClimateRiskEngine()
        result = eng.build_risk_assessment()
        impact = eng.calculate_scenario_impact(result, mod.ClimateScenario.IEA_NZE)
        assert len(impact["provenance_hash"]) == 64


# ===========================================================================
# Time Horizon Breakdown Tests
# ===========================================================================


class TestTimeHorizonBreakdown:
    """Tests for calculate_time_horizon_breakdown method."""

    def test_breakdown_groups_by_horizon(self, mod):
        """Time horizon breakdown groups risks by short/medium/long-term."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            estimated_annual_loss=Decimal("100000.00"),
            time_horizon=mod.RiskTimeHorizon.SHORT_TERM_0_3Y,
        ))
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.CHRONIC_SEA_LEVEL,
            estimated_annual_loss=Decimal("200000.00"),
            time_horizon=mod.RiskTimeHorizon.LONG_TERM_10_PLUS,
        ))
        result = eng.build_risk_assessment()
        breakdown = eng.calculate_time_horizon_breakdown(result)
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_breakdown_returns_dict(self, mod):
        """Time horizon breakdown returns a dict."""
        eng = mod.ClimateRiskEngine()
        result = eng.build_risk_assessment()
        breakdown = eng.calculate_time_horizon_breakdown(result)
        assert isinstance(breakdown, dict)


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestCompleteness:
    """Tests for E1-9 completeness validation."""

    def test_complete_assessment(self, mod):
        """Fully populated assessment has high completeness."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            affected_assets_pct=Decimal("15.0"),
            affected_revenue=Decimal("8000000.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            time_horizon=mod.RiskTimeHorizon.MEDIUM_TERM_3_10Y,
            scenario=mod.ClimateScenario.RCP_4_5,
            adaptation_cost=Decimal("500000.00"),
        ))
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            affected_assets_value=Decimal("30000000.00"),
            affected_assets_pct=Decimal("10.0"),
            affected_revenue_pct=Decimal("12.0"),
            likelihood=mod.LikelihoodLevel.VERY_HIGH,
            time_horizon=mod.RiskTimeHorizon.SHORT_TERM_0_3Y,
            scenario=mod.ClimateScenario.IEA_NZE,
            mitigation_cost=Decimal("2000000.00"),
        ))
        eng.assess_opportunity(mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.ENERGY_SOURCE,
            estimated_revenue_impact=Decimal("500000.00"),
            estimated_cost_savings=Decimal("300000.00"),
        ))
        result = eng.build_risk_assessment()
        completeness = eng.validate_completeness(result)
        assert isinstance(completeness, dict)
        assert completeness["total_datapoints"] >= 18
        assert completeness["covered_datapoints"] > 0
        assert completeness["completeness_score"] > 0.0

    def test_incomplete_assessment(self, mod):
        """Empty assessment has low completeness."""
        eng = mod.ClimateRiskEngine()
        result = eng.build_risk_assessment()
        completeness = eng.validate_completeness(result)
        assert isinstance(completeness, dict)
        assert completeness["covered_datapoints"] < completeness["total_datapoints"]

    def test_completeness_has_provenance(self, mod):
        """Completeness result has a provenance hash."""
        eng = mod.ClimateRiskEngine()
        result = eng.build_risk_assessment()
        completeness = eng.validate_completeness(result)
        assert len(completeness["provenance_hash"]) == 64

    def test_completeness_datapoints_detail(self, mod):
        """Completeness includes detail per data point."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("1000000.00"),
        ))
        result = eng.build_risk_assessment()
        completeness = eng.validate_completeness(result)
        assert "datapoints" in completeness
        dp = completeness["datapoints"]
        assert "e1_9_dp01" in dp
        assert dp["e1_9_dp01"]["status"] in ("COMPLETE", "MISSING")


# ===========================================================================
# E1-9 Data Points Tests
# ===========================================================================


class TestE19Datapoints:
    """Tests for E1-9 required data point extraction."""

    def test_returns_datapoints(self, mod):
        """get_e1_9_datapoints returns required data points."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            affected_assets_pct=Decimal("15.0"),
            affected_revenue=Decimal("8000000.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            scenario=mod.ClimateScenario.RCP_4_5,
            adaptation_cost=Decimal("500000.00"),
        ))
        eng.assess_transition_risk(mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            affected_assets_value=Decimal("30000000.00"),
            affected_assets_pct=Decimal("10.0"),
            affected_revenue_pct=Decimal("12.0"),
            scenario=mod.ClimateScenario.IEA_NZE,
            mitigation_cost=Decimal("2000000.00"),
        ))
        eng.assess_opportunity(mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.ENERGY_SOURCE,
            estimated_revenue_impact=Decimal("500000.00"),
            estimated_cost_savings=Decimal("300000.00"),
        ))
        result = eng.build_risk_assessment()
        datapoints = eng.get_e1_9_datapoints(result)
        assert isinstance(datapoints, dict)
        # 18 data points + provenance_hash
        assert len(datapoints) >= 18

    def test_e1_9_datapoints_constant(self, mod):
        """E1_9_DATAPOINTS dict has at least 18 entries."""
        assert len(mod.E1_9_DATAPOINTS) >= 18

    def test_datapoints_include_xbrl_elements(self, mod):
        """Data points include XBRL element references."""
        eng = mod.ClimateRiskEngine()
        eng.assess_physical_risk(mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("10000000.00"),
        ))
        result = eng.build_risk_assessment()
        datapoints = eng.get_e1_9_datapoints(result)
        for dp_id in ["e1_9_dp01", "e1_9_dp03", "e1_9_dp11"]:
            assert "xbrl_element" in datapoints[dp_id]

    def test_datapoints_provenance_hash(self, mod):
        """Data points include a provenance hash."""
        eng = mod.ClimateRiskEngine()
        result = eng.build_risk_assessment()
        datapoints = eng.get_e1_9_datapoints(result)
        assert "provenance_hash" in datapoints
        assert len(datapoints["provenance_hash"]) == 64


# ===========================================================================
# Summary Utility Tests
# ===========================================================================


class TestSummaryUtilities:
    """Tests for summary and reporting utility methods."""

    def test_physical_risk_summary(self, engine, mod):
        """get_physical_risk_summary returns structured dict."""
        risk = mod.PhysicalRisk(
            risk_type=mod.PhysicalRiskType.ACUTE_FLOODING,
            affected_assets_value=Decimal("50000000.00"),
            likelihood=mod.LikelihoodLevel.HIGH,
            location="Rotterdam, NL",
        )
        assessed = engine.assess_physical_risk(risk)
        summary = engine.get_physical_risk_summary(assessed)
        assert isinstance(summary, dict)
        assert summary["risk_type"] == "acute_flooding"
        assert summary["likelihood"] == "high"
        assert summary["location"] == "Rotterdam, NL"
        assert "risk_description" in summary

    def test_transition_risk_summary(self, engine, mod):
        """get_transition_risk_summary returns structured dict."""
        risk = mod.TransitionRisk(
            risk_type=mod.TransitionRiskType.POLICY_CARBON_PRICING,
            affected_assets_value=Decimal("30000000.00"),
        )
        assessed = engine.assess_transition_risk(risk)
        summary = engine.get_transition_risk_summary(assessed)
        assert isinstance(summary, dict)
        assert summary["risk_type"] == "policy_carbon_pricing"
        assert "risk_description" in summary

    def test_opportunity_summary(self, engine, mod):
        """get_opportunity_summary returns structured dict."""
        opp = mod.ClimateOpportunity(
            opportunity_type=mod.ClimateOpportunityType.RESILIENCE,
            estimated_revenue_impact=Decimal("200000.00"),
            estimated_cost_savings=Decimal("100000.00"),
            investment_required=Decimal("1000000.00"),
        )
        assessed = engine.assess_opportunity(opp)
        summary = engine.get_opportunity_summary(assessed)
        assert isinstance(summary, dict)
        assert summary["opportunity_type"] == "resilience"
        # total = 200000 + 100000 = 300000
        assert "300000" in summary["total_estimated_value"]
        assert "description" in summary

    def test_scenario_descriptions(self, engine, mod):
        """get_scenario_descriptions returns all scenario descriptions."""
        descriptions = engine.get_scenario_descriptions()
        assert isinstance(descriptions, dict)
        assert len(descriptions) >= 8
        for sc in mod.ClimateScenario:
            assert sc.value in descriptions
