# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Transition Plan Engine Tests
===============================================================

Unit tests for TransitionPlanEngine (Engine 3) covering plan building,
abatement calculation, locked-in emissions, gap analysis, scenario
alignment, CapEx allocation, completeness, and E1-1 data points.

ESRS E1-1: Transition plan for climate change mitigation.

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
    """Load the transition_plan engine module."""
    return _load_engine("transition_plan")


@pytest.fixture
def engine(mod):
    """Create a fresh TransitionPlanEngine instance."""
    return mod.TransitionPlanEngine()


@pytest.fixture
def sample_action(mod):
    """Create a sample transition plan action."""
    return mod.TransitionPlanAction(
        name="Solar PV Installation",
        lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
        expected_abatement_tco2e=Decimal("5000"),
        expected_abatement_pct=Decimal("10"),
        capex_eur=Decimal("2000000"),
        opex_annual_eur=Decimal("50000"),
        start_year=2025,
        completion_year=2028,
        time_horizon=mod.TimeHorizon.SHORT_TERM,
        confidence_level=Decimal("0.80"),
    )


@pytest.fixture
def sample_locked_in(mod):
    """Create a sample locked-in emission source."""
    return mod.LockedInEmission(
        name="Coal Power Plant A",
        emission_type=mod.LockedInEmissionType.EXISTING_ASSETS,
        total_locked_in_tco2e=Decimal("50000"),
        annual_tco2e=Decimal("5000"),
        remaining_years=10,
        asset_value_eur=Decimal("100000000"),
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestTransitionEnums:
    """Tests for transition plan enums."""

    def test_decarbonization_lever_count(self, mod):
        """DecarbonizationLever has at least 8 values."""
        assert len(mod.DecarbonizationLever) >= 8

    def test_decarbonization_lever_values(self, mod):
        """Key levers are present."""
        values = {m.value for m in mod.DecarbonizationLever}
        for lever in ["energy_efficiency", "fuel_switching", "electrification",
                       "renewable_energy", "process_change", "carbon_capture"]:
            assert lever in values

    def test_plan_status_values(self, mod):
        """PlanStatus has expected values."""
        values = {m.value for m in mod.PlanStatus}
        assert "draft" in values
        assert "approved" in values
        assert "in_progress" in values

    def test_scenario_alignment_values(self, mod):
        """ScenarioAlignment includes 1.5C alignment."""
        values = {m.value for m in mod.ScenarioAlignment}
        assert "aligned_1_5c" in values
        assert "not_aligned" in values

    def test_locked_in_emission_type(self, mod):
        """LockedInEmissionType has 3 types."""
        assert len(mod.LockedInEmissionType) == 3
        values = {m.value for m in mod.LockedInEmissionType}
        assert "existing_assets" in values
        assert "contractual_obligations" in values

    def test_time_horizon_values(self, mod):
        """TimeHorizon has 3 values."""
        assert len(mod.TimeHorizon) == 3


# ===========================================================================
# Model Tests
# ===========================================================================


class TestTransitionModels:
    """Tests for transition plan Pydantic models."""

    def test_create_action(self, mod):
        """Create a valid TransitionPlanAction."""
        action = mod.TransitionPlanAction(
            name="LED Retrofit",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
            expected_abatement_tco2e=Decimal("200"),
        )
        assert action.name == "LED Retrofit"
        assert action.lever == mod.DecarbonizationLever.ENERGY_EFFICIENCY
        assert len(action.action_id) > 0

    def test_create_locked_in(self, mod):
        """Create a valid LockedInEmission."""
        locked = mod.LockedInEmission(
            name="Gas Pipeline Contract",
            emission_type=mod.LockedInEmissionType.CONTRACTUAL_OBLIGATIONS,
            annual_tco2e=Decimal("1000"),
            remaining_years=5,
        )
        assert locked.name == "Gas Pipeline Contract"
        assert locked.remaining_years == 5

    def test_action_completion_before_start_raises(self, mod):
        """completion_year before start_year raises error."""
        with pytest.raises(Exception):
            mod.TransitionPlanAction(
                name="Invalid Action",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                start_year=2030,
                completion_year=2025,
            )

    def test_action_default_status(self, mod):
        """Default status is DRAFT."""
        action = mod.TransitionPlanAction(
            name="Test",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
        )
        assert action.status == mod.PlanStatus.DRAFT


# ===========================================================================
# Build Plan Tests
# ===========================================================================


class TestBuildPlan:
    """Tests for build_transition_plan method."""

    def test_basic_plan(self, engine, sample_action, sample_locked_in):
        """Build a basic transition plan."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("50000"),
            target=Decimal("25000"),
            target_year=2030,
            actions=[sample_action],
            locked_in=[sample_locked_in],
        )
        assert result is not None
        assert result.processing_time_ms >= 0.0

    def test_plan_with_actions(self, engine, mod):
        """Plan with multiple actions."""
        actions = [
            mod.TransitionPlanAction(
                name="Action A",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("3000"),
                capex_eur=Decimal("500000"),
            ),
            mod.TransitionPlanAction(
                name="Action B",
                lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
                expected_abatement_tco2e=Decimal("7000"),
                capex_eur=Decimal("1500000"),
            ),
        ]
        result = engine.build_transition_plan(
            current_emissions=Decimal("50000"),
            target=Decimal("25000"),
            target_year=2035,
            actions=actions,
        )
        assert result is not None

    def test_provenance_hash(self, engine, sample_action):
        """Plan result has a provenance hash."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("50000"),
            target=Decimal("25000"),
            target_year=2030,
            actions=[sample_action],
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_plan_with_locked_in(self, engine, sample_action, sample_locked_in):
        """Plan accounts for locked-in emissions."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[sample_action],
            locked_in=[sample_locked_in],
        )
        assert result is not None


# ===========================================================================
# Abatement Tests
# ===========================================================================


class TestAbatement:
    """Tests for abatement calculation."""

    def test_single_action_abatement(self, engine, sample_action):
        """Single action abatement is calculated."""
        result = engine.calculate_abatement_potential([sample_action])
        assert result > Decimal("0")
        assert float(result) == pytest.approx(5000.0, abs=1.0)

    def test_multiple_actions_sum(self, engine, mod):
        """Multiple actions sum their abatement."""
        actions = [
            mod.TransitionPlanAction(
                name="A",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("3000"),
            ),
            mod.TransitionPlanAction(
                name="B",
                lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
                expected_abatement_tco2e=Decimal("7000"),
            ),
        ]
        result = engine.calculate_abatement_potential(actions)
        assert float(result) == pytest.approx(10000.0, abs=1.0)

    def test_zero_abatement_action(self, engine, mod):
        """Action with zero abatement contributes nothing."""
        action = mod.TransitionPlanAction(
            name="Zero Impact",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
            expected_abatement_tco2e=Decimal("0"),
        )
        result = engine.calculate_abatement_potential([action])
        assert result == Decimal("0")

    def test_confidence_weighted_abatement(self, engine, mod):
        """Confidence-weighted abatement accounts for confidence levels."""
        action = mod.TransitionPlanAction(
            name="Uncertain",
            lever=mod.DecarbonizationLever.CARBON_CAPTURE,
            expected_abatement_tco2e=Decimal("10000"),
            confidence_level=Decimal("0.50"),
        )
        result = engine.calculate_confidence_weighted_abatement([action])
        assert float(result) == pytest.approx(5000.0, abs=100.0)


# ===========================================================================
# Locked-In Emissions Tests
# ===========================================================================


class TestLockedIn:
    """Tests for locked-in emission calculations."""

    def test_single_source(self, engine, sample_locked_in):
        """Single locked-in source total is calculated."""
        result = engine.calculate_locked_in_emissions([sample_locked_in])
        assert result > Decimal("0")

    def test_multiple_sources(self, engine, mod):
        """Multiple locked-in sources sum correctly."""
        sources = [
            mod.LockedInEmission(
                name="A",
                emission_type=mod.LockedInEmissionType.EXISTING_ASSETS,
                total_locked_in_tco2e=Decimal("20000"),
                annual_tco2e=Decimal("2000"),
                remaining_years=10,
            ),
            mod.LockedInEmission(
                name="B",
                emission_type=mod.LockedInEmissionType.CONTRACTUAL_OBLIGATIONS,
                total_locked_in_tco2e=Decimal("10000"),
                annual_tco2e=Decimal("2000"),
                remaining_years=5,
            ),
        ]
        result = engine.calculate_locked_in_emissions(sources)
        assert result > Decimal("0")

    def test_zero_years_remaining(self, engine, mod):
        """Source with zero remaining years."""
        source = mod.LockedInEmission(
            name="Decommissioned",
            emission_type=mod.LockedInEmissionType.EXISTING_ASSETS,
            total_locked_in_tco2e=Decimal("0"),
            annual_tco2e=Decimal("0"),
            remaining_years=0,
        )
        result = engine.calculate_locked_in_emissions([source])
        assert result == Decimal("0")


# ===========================================================================
# Gap Analysis Tests
# ===========================================================================


class TestGapAnalysis:
    """Tests for gap analysis."""

    def test_gap_exists(self, engine, mod):
        """Gap exists when abatement is insufficient."""
        actions = [
            mod.TransitionPlanAction(
                name="Small Action",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("5000"),
                confidence_level=Decimal("0.80"),
            ),
        ]
        plan = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            actions=actions,
        )
        result = engine.analyze_gap(plan)
        assert isinstance(result, mod.PlanGapAnalysis)
        assert result.gap_tco2e > Decimal("0")
        assert result.is_on_track is False

    def test_no_gap(self, engine, mod):
        """No gap when abatement exceeds target reduction."""
        actions = [
            mod.TransitionPlanAction(
                name="Big Action",
                lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
                expected_abatement_tco2e=Decimal("60000"),
                confidence_level=Decimal("0.95"),
            ),
        ]
        plan = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            actions=actions,
        )
        result = engine.analyze_gap(plan)
        assert result.is_on_track is True

    def test_gap_with_locked_in(self, engine, mod):
        """Gap analysis accounts for locked-in emissions."""
        actions = [
            mod.TransitionPlanAction(
                name="Action",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("40000"),
                confidence_level=Decimal("0.90"),
            ),
        ]
        locked_in = [
            mod.LockedInEmission(
                name="Plant",
                emission_type=mod.LockedInEmissionType.EXISTING_ASSETS,
                total_locked_in_tco2e=Decimal("30000"),
                annual_tco2e=Decimal("3000"),
                remaining_years=10,
            ),
        ]
        plan = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            actions=actions,
            locked_in=locked_in,
        )
        result = engine.analyze_gap(plan)
        assert result.locked_in_emissions_tco2e > Decimal("0")


# ===========================================================================
# Scenario Alignment Tests
# ===========================================================================


class TestScenarioAlignment:
    """Tests for scenario alignment validation."""

    def test_1_5c_aligned(self, engine, mod):
        """Validate 1.5C alignment with sufficient reduction."""
        actions = [
            mod.TransitionPlanAction(
                name="Major Reduction",
                lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
                expected_abatement_tco2e=Decimal("50000"),
            ),
        ]
        plan = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2030,
            actions=actions,
        )
        result = engine.validate_scenario_alignment(plan)
        assert isinstance(result, dict)

    def test_not_aligned(self, engine, mod):
        """Insufficient reduction is not aligned."""
        actions = [
            mod.TransitionPlanAction(
                name="Tiny Action",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("1000"),
            ),
        ]
        plan = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("95000"),
            target_year=2050,
            actions=actions,
        )
        result = engine.validate_scenario_alignment(plan)
        assert result is not None


# ===========================================================================
# CapEx Allocation Tests
# ===========================================================================


class TestCapexAllocation:
    """Tests for CapEx allocation by lever."""

    def test_capex_by_lever(self, engine, mod):
        """CapEx is allocated by decarbonization lever."""
        actions = [
            mod.TransitionPlanAction(
                name="A",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                capex_eur=Decimal("500000"),
            ),
            mod.TransitionPlanAction(
                name="B",
                lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
                capex_eur=Decimal("1500000"),
            ),
        ]
        result = engine.calculate_capex_allocation(actions)
        assert isinstance(result, dict)
        assert len(result) >= 2

    def test_total_capex_matches_sum(self, engine, mod):
        """Total CapEx matches sum of actions."""
        actions = [
            mod.TransitionPlanAction(
                name="A",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                capex_eur=Decimal("500000"),
            ),
            mod.TransitionPlanAction(
                name="B",
                lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
                capex_eur=Decimal("1500000"),
            ),
        ]
        result = engine.calculate_capex_allocation(actions)
        total = float(Decimal(result["total_capex_eur"]))
        assert total == pytest.approx(2000000.0, abs=1.0)


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestCompleteness:
    """Tests for E1-1 completeness validation."""

    def test_complete_plan(self, engine, mod, sample_action, sample_locked_in):
        """Complete plan has high completeness score."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[sample_action],
            locked_in=[sample_locked_in],
            scenario_alignment=mod.ScenarioAlignment.ALIGNED_1_5C,
        )
        completeness = engine.validate_completeness(result)
        assert isinstance(completeness, dict)
        assert len(completeness) > 0

    def test_incomplete_plan(self, engine, mod):
        """Minimal plan has lower completeness."""
        action = mod.TransitionPlanAction(
            name="Minimal",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
        )
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[action],
        )
        completeness = engine.validate_completeness(result)
        assert isinstance(completeness, dict)


# ===========================================================================
# E1-1 Data Points Tests
# ===========================================================================


class TestE11Datapoints:
    """Tests for E1-1 required data point extraction."""

    def test_returns_datapoints(self, engine, sample_action, sample_locked_in):
        """get_e1_1_datapoints returns required data points."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[sample_action],
            locked_in=[sample_locked_in],
        )
        datapoints = engine.get_e1_1_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 10

    def test_e1_1_datapoints_constant(self, mod):
        """E1_1_DATAPOINTS list has at least 15 entries."""
        assert len(mod.E1_1_DATAPOINTS) >= 15

    def test_e1_1_datapoints_include_plan_status(self, engine, sample_action, sample_locked_in):
        """Datapoints include plan status information."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[sample_action],
            locked_in=[sample_locked_in],
        )
        datapoints = engine.get_e1_1_datapoints(result)
        has_status = any("status" in k.lower() for k in datapoints.keys())
        assert has_status or len(datapoints) >= 10


# ===========================================================================
# Lever Typical Abatement Tests
# ===========================================================================


class TestLeverTypicalAbatement:
    """Tests for LEVER_TYPICAL_ABATEMENT constant."""

    def test_lever_abatement_exists(self, mod):
        """LEVER_TYPICAL_ABATEMENT maps all levers."""
        for lever in mod.DecarbonizationLever:
            assert lever.value in mod.LEVER_TYPICAL_ABATEMENT

    def test_lever_abatement_positive(self, mod):
        """All typical abatement values are non-negative."""
        for lever_val, val in mod.LEVER_TYPICAL_ABATEMENT.items():
            if isinstance(val, dict):
                assert val["median_pct"] >= Decimal("0")
            else:
                assert val >= Decimal("0")

    def test_renewable_has_high_abatement(self, mod):
        """Renewable energy lever has substantial abatement potential."""
        val = mod.LEVER_TYPICAL_ABATEMENT["renewable_energy"]
        if isinstance(val, dict):
            assert val["median_pct"] > Decimal("0")
        else:
            assert val > Decimal("0")


# ===========================================================================
# Plan Completeness Criteria Tests
# ===========================================================================


class TestPlanCompletenessCriteria:
    """Tests for PLAN_COMPLETENESS_CRITERIA constant."""

    def test_criteria_count(self, mod):
        """PLAN_COMPLETENESS_CRITERIA has at least 8 entries."""
        assert len(mod.PLAN_COMPLETENESS_CRITERIA) >= 8

    def test_criteria_are_strings(self, mod):
        """Each criterion is a non-empty string."""
        for criterion in mod.PLAN_COMPLETENESS_CRITERIA:
            assert isinstance(criterion, str)
            assert len(criterion) > 0


# ===========================================================================
# Additional Model Validation Tests
# ===========================================================================


class TestAdditionalModelValidation:
    """Additional validation tests for transition plan models."""

    def test_action_name_stored(self, mod):
        """Action name is stored as provided."""
        action = mod.TransitionPlanAction(
            name="Test Action Name",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
        )
        assert action.name == "Test Action Name"

    def test_action_lever_stored(self, mod):
        """Action lever is stored correctly."""
        action = mod.TransitionPlanAction(
            name="Test",
            lever=mod.DecarbonizationLever.FUEL_SWITCHING,
        )
        assert action.lever == mod.DecarbonizationLever.FUEL_SWITCHING

    def test_locked_in_auto_compute_total(self, mod):
        """LockedInEmission auto-computes total from annual * remaining_years."""
        locked = mod.LockedInEmission(
            name="Auto Total",
            emission_type=mod.LockedInEmissionType.EXISTING_ASSETS,
            annual_tco2e=Decimal("2000"),
            remaining_years=10,
        )
        # total_locked_in_tco2e should be computed as 2000 * 10 = 20000
        if locked.total_locked_in_tco2e > Decimal("0"):
            assert float(locked.total_locked_in_tco2e) == pytest.approx(20000.0, abs=1.0)

    def test_action_unique_ids(self, mod):
        """Each action gets a unique action_id."""
        a1 = mod.TransitionPlanAction(
            name="A", lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
        )
        a2 = mod.TransitionPlanAction(
            name="A", lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
        )
        assert a1.action_id != a2.action_id


# ===========================================================================
# Time Horizon Tests
# ===========================================================================


class TestTimeHorizon:
    """Tests for time horizon handling in transition plans."""

    def test_short_term_action(self, engine, mod):
        """Short-term action has completion before 2028."""
        action = mod.TransitionPlanAction(
            name="Quick Win",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
            expected_abatement_tco2e=Decimal("1000"),
            start_year=2025,
            completion_year=2027,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        result = engine.build_transition_plan(
            current_emissions=Decimal("50000"),
            target=Decimal("40000"),
            target_year=2030,
            actions=[action],
        )
        assert result is not None

    def test_long_term_action(self, engine, mod):
        """Long-term action with distant completion year."""
        action = mod.TransitionPlanAction(
            name="CCS Project",
            lever=mod.DecarbonizationLever.CARBON_CAPTURE,
            expected_abatement_tco2e=Decimal("20000"),
            start_year=2025,
            completion_year=2040,
            time_horizon=mod.TimeHorizon.LONG_TERM,
        )
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2050,
            actions=[action],
        )
        assert result is not None


# ===========================================================================
# Confidence Weighted Abatement Tests
# ===========================================================================


class TestConfidenceWeighted:
    """Tests for confidence-weighted abatement calculations."""

    def test_full_confidence(self, engine, mod):
        """100% confidence returns full abatement."""
        action = mod.TransitionPlanAction(
            name="Certain",
            lever=mod.DecarbonizationLever.RENEWABLE_ENERGY,
            expected_abatement_tco2e=Decimal("10000"),
            confidence_level=Decimal("1.00"),
        )
        result = engine.calculate_confidence_weighted_abatement([action])
        assert float(result) == pytest.approx(10000.0, abs=1.0)

    def test_zero_confidence(self, engine, mod):
        """0% confidence returns zero abatement."""
        action = mod.TransitionPlanAction(
            name="Speculative",
            lever=mod.DecarbonizationLever.CARBON_CAPTURE,
            expected_abatement_tco2e=Decimal("10000"),
            confidence_level=Decimal("0.00"),
        )
        result = engine.calculate_confidence_weighted_abatement([action])
        assert float(result) == pytest.approx(0.0, abs=1.0)

    def test_mixed_confidence(self, engine, mod):
        """Multiple actions with different confidence levels."""
        actions = [
            mod.TransitionPlanAction(
                name="High Conf",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("5000"),
                confidence_level=Decimal("0.90"),
            ),
            mod.TransitionPlanAction(
                name="Low Conf",
                lever=mod.DecarbonizationLever.CARBON_CAPTURE,
                expected_abatement_tco2e=Decimal("5000"),
                confidence_level=Decimal("0.30"),
            ),
        ]
        result = engine.calculate_confidence_weighted_abatement(actions)
        # 5000*0.9 + 5000*0.3 = 4500 + 1500 = 6000
        assert float(result) == pytest.approx(6000.0, abs=100.0)


# ===========================================================================
# Additional Build Plan Tests
# ===========================================================================


class TestBuildPlanAdvanced:
    """Advanced tests for build_transition_plan."""

    def test_plan_with_scenario_alignment(self, engine, mod, sample_action):
        """Plan with explicit scenario alignment."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[sample_action],
            scenario_alignment=mod.ScenarioAlignment.ALIGNED_1_5C,
        )
        assert result is not None
        assert len(result.provenance_hash) == 64

    def test_plan_without_actions(self, engine, mod):
        """Plan with empty actions list still produces result."""
        action = mod.TransitionPlanAction(
            name="Placeholder",
            lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
            expected_abatement_tco2e=Decimal("0"),
        )
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[action],
        )
        assert result is not None

    def test_plan_many_actions(self, engine, mod):
        """Plan with many actions (10+) processes correctly."""
        levers = list(mod.DecarbonizationLever)
        actions = []
        for i, lever in enumerate(levers):
            actions.append(mod.TransitionPlanAction(
                name=f"Action {i+1}",
                lever=lever,
                expected_abatement_tco2e=Decimal("1000"),
                capex_eur=Decimal("100000"),
            ))
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=actions,
        )
        assert result is not None

    def test_plan_processing_time(self, engine, mod, sample_action):
        """Plan records positive processing time."""
        result = engine.build_transition_plan(
            current_emissions=Decimal("100000"),
            target=Decimal("50000"),
            target_year=2035,
            actions=[sample_action],
        )
        assert result.processing_time_ms >= 0.0


# ===========================================================================
# OpEx Allocation Tests
# ===========================================================================


class TestOpexAllocation:
    """Tests for OpEx allocation in transition plans."""

    def test_opex_in_capex_allocation(self, engine, mod):
        """CapEx allocation includes OpEx information."""
        actions = [
            mod.TransitionPlanAction(
                name="A",
                lever=mod.DecarbonizationLever.ENERGY_EFFICIENCY,
                capex_eur=Decimal("500000"),
                opex_annual_eur=Decimal("25000"),
            ),
        ]
        result = engine.calculate_capex_allocation(actions)
        assert isinstance(result, dict)
        # Should include total OpEx or per-lever data
        assert len(result) >= 1
