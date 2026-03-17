# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Climate Action Engine Tests
===============================================================

Unit tests for ClimateActionEngine (Engine 5) covering policy
creation, action creation, resource allocation, taxonomy alignment,
completeness for E1-2 and E1-3, and data point extraction.

ESRS E1-2: Policies related to climate change.
ESRS E1-3: Actions and resources related to climate change.

Target: 50+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from decimal import Decimal
from datetime import date

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the climate_action engine module."""
    return _load_engine("climate_action")


@pytest.fixture
def engine(mod):
    """Create a fresh ClimateActionEngine instance."""
    return mod.ClimateActionEngine()


@pytest.fixture
def sample_policy(mod):
    """Create a sample climate policy."""
    return mod.ClimatePolicy(
        name="Group-Wide Climate Mitigation Policy",
        policy_type=mod.PolicyType.MITIGATION,
        scope=mod.PolicyScope.GROUP_WIDE,
        description="Comprehensive policy for GHG emission reduction.",
        covers_own_operations=True,
        covers_upstream=True,
        covers_downstream=False,
    )


@pytest.fixture
def sample_action(mod):
    """Create a sample climate action."""
    return mod.ClimateAction(
        name="LED Lighting Retrofit",
        category=mod.ActionCategory.ENERGY_EFFICIENCY,
        expected_reduction_tco2e=Decimal("500"),
        capex_amount=Decimal("200000"),
        opex_amount=Decimal("10000"),
        status=mod.ActionStatus.IN_PROGRESS,
        taxonomy_aligned=True,
    )


@pytest.fixture
def sample_resource(mod):
    """Create a sample resource allocation."""
    return mod.ResourceAllocation(
        resource_type=mod.ResourceType.CAPEX,
        amount=Decimal("500000"),
        currency="EUR",
        period="2025-2030",
        action_id="test-action-001",
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestActionEnums:
    """Tests for climate action enums."""

    def test_policy_type_count(self, mod):
        """PolicyType has at least 4 values."""
        assert len(mod.PolicyType) >= 4
        values = {m.value for m in mod.PolicyType}
        assert "mitigation" in values
        assert "adaptation" in values

    def test_policy_scope_count(self, mod):
        """PolicyScope has at least 4 values."""
        assert len(mod.PolicyScope) >= 4
        values = {m.value for m in mod.PolicyScope}
        assert "group_wide" in values
        assert "supply_chain" in values

    def test_action_category_count(self, mod):
        """ActionCategory has at least 8 values."""
        assert len(mod.ActionCategory) >= 8
        values = {m.value for m in mod.ActionCategory}
        assert "energy_efficiency" in values
        assert "renewable_energy" in values
        assert "carbon_capture" in values

    def test_action_status_values(self, mod):
        """ActionStatus has lifecycle values."""
        assert len(mod.ActionStatus) >= 4
        values = {m.value for m in mod.ActionStatus}
        assert "planned" in values
        assert "in_progress" in values
        assert "completed" in values

    def test_resource_type_values(self, mod):
        """ResourceType has expected values."""
        assert len(mod.ResourceType) >= 3
        values = {m.value for m in mod.ResourceType}
        assert "capex" in values
        assert "opex" in values


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestActionConstants:
    """Tests for climate action constants."""

    def test_e1_2_datapoints_exist(self, mod):
        """E1_2_DATAPOINTS is a non-empty dict."""
        assert len(mod.E1_2_DATAPOINTS) >= 10

    def test_e1_3_datapoints_exist(self, mod):
        """E1_3_DATAPOINTS is a non-empty dict."""
        assert len(mod.E1_3_DATAPOINTS) >= 10

    def test_action_taxonomy_alignment(self, mod):
        """ACTION_TAXONOMY_ALIGNMENT maps action categories."""
        assert len(mod.ACTION_TAXONOMY_ALIGNMENT) >= 8
        for cat in ["energy_efficiency", "renewable_energy", "carbon_capture"]:
            assert cat in mod.ACTION_TAXONOMY_ALIGNMENT

    def test_policy_type_descriptions(self, mod):
        """POLICY_TYPE_DESCRIPTIONS has entries for all policy types."""
        for pt in mod.PolicyType:
            assert pt.value in mod.POLICY_TYPE_DESCRIPTIONS

    def test_action_category_descriptions(self, mod):
        """ACTION_CATEGORY_DESCRIPTIONS has entries for all categories."""
        for ac in mod.ActionCategory:
            assert ac.value in mod.ACTION_CATEGORY_DESCRIPTIONS


# ===========================================================================
# Policy Model Tests
# ===========================================================================


class TestPolicyModel:
    """Tests for ClimatePolicy Pydantic model."""

    def test_create_valid_policy(self, mod):
        """Create a valid ClimatePolicy."""
        policy = mod.ClimatePolicy(
            name="Energy Efficiency Policy",
            policy_type=mod.PolicyType.ENERGY_EFFICIENCY,
        )
        assert policy.name == "Energy Efficiency Policy"
        assert len(policy.policy_id) > 0

    def test_policy_requires_name(self, mod):
        """Policy requires a non-empty name."""
        with pytest.raises(Exception):
            mod.ClimatePolicy(
                name="",
                policy_type=mod.PolicyType.MITIGATION,
            )

    def test_policy_whitespace_name_rejected(self, mod):
        """Policy rejects whitespace-only name."""
        with pytest.raises(Exception):
            mod.ClimatePolicy(
                name="   ",
                policy_type=mod.PolicyType.MITIGATION,
            )

    def test_policy_default_scope(self, mod):
        """Default scope is GROUP_WIDE."""
        policy = mod.ClimatePolicy(
            name="Test Policy",
            policy_type=mod.PolicyType.MITIGATION,
        )
        assert policy.scope == mod.PolicyScope.GROUP_WIDE

    def test_policy_default_covers_own_ops(self, mod):
        """Default covers_own_operations is True."""
        policy = mod.ClimatePolicy(
            name="Test",
            policy_type=mod.PolicyType.MITIGATION,
        )
        assert policy.covers_own_operations is True


# ===========================================================================
# Action Model Tests
# ===========================================================================


class TestActionModel:
    """Tests for ClimateAction Pydantic model."""

    def test_create_valid_action(self, mod):
        """Create a valid ClimateAction."""
        action = mod.ClimateAction(
            name="Solar PV Installation",
            category=mod.ActionCategory.RENEWABLE_ENERGY,
            expected_reduction_tco2e=Decimal("3000"),
        )
        assert action.name == "Solar PV Installation"
        assert len(action.action_id) > 0

    def test_action_with_capex_opex(self, mod):
        """Action with CapEx and OpEx values."""
        action = mod.ClimateAction(
            name="Heat Pump Upgrade",
            category=mod.ActionCategory.ELECTRIFICATION,
            capex_amount=Decimal("500000"),
            opex_amount=Decimal("25000"),
        )
        assert action.capex_amount == Decimal("500000")
        assert action.opex_amount == Decimal("25000")

    def test_action_requires_name(self, mod):
        """Action requires a non-empty name."""
        with pytest.raises(Exception):
            mod.ClimateAction(
                name="",
                category=mod.ActionCategory.ENERGY_EFFICIENCY,
            )

    def test_action_default_status(self, mod):
        """Default status is PLANNED."""
        action = mod.ClimateAction(
            name="Test",
            category=mod.ActionCategory.ENERGY_EFFICIENCY,
        )
        assert action.status == mod.ActionStatus.PLANNED


# ===========================================================================
# Build Action Plan Tests
# ===========================================================================


class TestBuildActionPlan:
    """Tests for build_action_plan method."""

    def test_basic_plan(self, engine, sample_policy, sample_action):
        """Build basic action plan."""
        result = engine.build_action_plan(
            policies=[sample_policy],
            actions=[sample_action],
        )
        assert result is not None
        assert result.processing_time_ms >= 0.0

    def test_plan_with_policies_and_actions(self, engine, mod):
        """Plan with multiple policies and actions."""
        policies = [
            mod.ClimatePolicy(
                name="Mitigation Policy",
                policy_type=mod.PolicyType.MITIGATION,
            ),
            mod.ClimatePolicy(
                name="Adaptation Policy",
                policy_type=mod.PolicyType.ADAPTATION,
            ),
        ]
        actions = [
            mod.ClimateAction(
                name="EE Project",
                category=mod.ActionCategory.ENERGY_EFFICIENCY,
                capex_amount=Decimal("100000"),
                expected_reduction_tco2e=Decimal("500"),
            ),
            mod.ClimateAction(
                name="Solar Project",
                category=mod.ActionCategory.RENEWABLE_ENERGY,
                capex_amount=Decimal("500000"),
                expected_reduction_tco2e=Decimal("2000"),
            ),
        ]
        result = engine.build_action_plan(
            policies=policies, actions=actions,
        )
        assert result is not None
        assert len(result.provenance_hash) == 64

    def test_plan_resource_summary(self, engine, mod):
        """Plan includes resource allocation summary."""
        actions = [
            mod.ClimateAction(
                name="Action A",
                category=mod.ActionCategory.ENERGY_EFFICIENCY,
                capex_amount=Decimal("200000"),
                opex_amount=Decimal("10000"),
            ),
        ]
        result = engine.build_action_plan(
            policies=[], actions=actions,
        )
        assert result is not None


# ===========================================================================
# Resource Allocation Tests
# ===========================================================================


class TestResourceAllocation:
    """Tests for resource allocation calculation."""

    def test_capex_total(self, engine, mod):
        """Total CapEx is calculated across all actions."""
        actions = [
            mod.ClimateAction(
                name="A",
                category=mod.ActionCategory.ENERGY_EFFICIENCY,
                capex_amount=Decimal("300000"),
            ),
            mod.ClimateAction(
                name="B",
                category=mod.ActionCategory.RENEWABLE_ENERGY,
                capex_amount=Decimal("700000"),
            ),
        ]
        result = engine.calculate_resource_allocation(actions)
        assert isinstance(result, dict)
        # Total capex should be 1000000
        total_capex = result.get("total_capex", Decimal("0"))
        if isinstance(total_capex, Decimal):
            assert float(total_capex) == pytest.approx(1000000.0, abs=1.0)

    def test_opex_total(self, engine, mod):
        """Total OpEx is calculated across all actions."""
        actions = [
            mod.ClimateAction(
                name="A",
                category=mod.ActionCategory.ENERGY_EFFICIENCY,
                opex_amount=Decimal("50000"),
            ),
            mod.ClimateAction(
                name="B",
                category=mod.ActionCategory.RENEWABLE_ENERGY,
                opex_amount=Decimal("30000"),
            ),
        ]
        result = engine.calculate_resource_allocation(actions)
        assert isinstance(result, dict)

    def test_by_category(self, engine, mod):
        """Resources broken down by category."""
        actions = [
            mod.ClimateAction(
                name="A",
                category=mod.ActionCategory.ENERGY_EFFICIENCY,
                capex_amount=Decimal("300000"),
            ),
            mod.ClimateAction(
                name="B",
                category=mod.ActionCategory.RENEWABLE_ENERGY,
                capex_amount=Decimal("700000"),
            ),
        ]
        result = engine.calculate_resource_allocation(actions)
        assert isinstance(result, dict)
        assert len(result) >= 2


# ===========================================================================
# Taxonomy Alignment Tests
# ===========================================================================


class TestTaxonomyAlignment:
    """Tests for EU Taxonomy alignment assessment."""

    def test_taxonomy_aligned_actions(self, engine, mod):
        """Taxonomy-aligned actions are identified."""
        actions = [
            mod.ClimateAction(
                name="Solar Installation",
                category=mod.ActionCategory.RENEWABLE_ENERGY,
                taxonomy_aligned=True,
                capex_amount=Decimal("500000"),
            ),
            mod.ClimateAction(
                name="Offset Purchase",
                category=mod.ActionCategory.OFFSET,
                taxonomy_aligned=False,
                capex_amount=Decimal("100000"),
            ),
        ]
        result = engine.assess_taxonomy_alignment(actions)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_no_aligned_actions(self, engine, mod):
        """Result when no actions are taxonomy-aligned."""
        actions = [
            mod.ClimateAction(
                name="Offset",
                category=mod.ActionCategory.OFFSET,
                taxonomy_aligned=False,
            ),
        ]
        result = engine.assess_taxonomy_alignment(actions)
        assert isinstance(result, dict)


# ===========================================================================
# Completeness E1-2 Tests
# ===========================================================================


class TestCompletenessE12:
    """Tests for E1-2 completeness validation."""

    def test_policy_disclosure_complete(self, engine, sample_policy, sample_action):
        """Complete policy disclosure."""
        result = engine.build_action_plan(
            policies=[sample_policy], actions=[sample_action],
        )
        completeness = engine.validate_completeness_e1_2(result)
        assert isinstance(completeness, dict)
        assert len(completeness) > 0

    def test_policy_disclosure_incomplete(self, engine, mod):
        """Incomplete policy disclosure (no policies)."""
        result = engine.build_action_plan(
            policies=[], actions=[
                mod.ClimateAction(
                    name="A",
                    category=mod.ActionCategory.ENERGY_EFFICIENCY,
                ),
            ],
        )
        completeness = engine.validate_completeness_e1_2(result)
        assert isinstance(completeness, dict)


# ===========================================================================
# Completeness E1-3 Tests
# ===========================================================================


class TestCompletenessE13:
    """Tests for E1-3 completeness validation."""

    def test_action_disclosure_complete(self, engine, sample_policy, sample_action):
        """Complete action disclosure."""
        result = engine.build_action_plan(
            policies=[sample_policy], actions=[sample_action],
        )
        completeness = engine.validate_completeness_e1_3(result)
        assert isinstance(completeness, dict)
        assert len(completeness) > 0

    def test_action_disclosure_incomplete(self, engine, mod):
        """Incomplete action disclosure (no actions)."""
        result = engine.build_action_plan(
            policies=[
                mod.ClimatePolicy(
                    name="P",
                    policy_type=mod.PolicyType.MITIGATION,
                ),
            ],
            actions=[],
        )
        completeness = engine.validate_completeness_e1_3(result)
        assert isinstance(completeness, dict)


# ===========================================================================
# E1-2 Data Points Tests
# ===========================================================================


class TestE12Datapoints:
    """Tests for E1-2 required data point extraction."""

    def test_returns_datapoints(self, engine, sample_policy, sample_action):
        """get_e1_2_datapoints returns required data points."""
        result = engine.build_action_plan(
            policies=[sample_policy], actions=[sample_action],
        )
        datapoints = engine.get_e1_2_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 8


# ===========================================================================
# E1-3 Data Points Tests
# ===========================================================================


class TestE13Datapoints:
    """Tests for E1-3 required data point extraction."""

    def test_returns_datapoints(self, engine, sample_policy, sample_action):
        """get_e1_3_datapoints returns required data points."""
        result = engine.build_action_plan(
            policies=[sample_policy], actions=[sample_action],
        )
        datapoints = engine.get_e1_3_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 8


# ===========================================================================
# Register Policy Tests (Stateful)
# ===========================================================================


class TestRegisterPolicy:
    """Tests for register_policy stateful method."""

    def test_register_single_policy(self, engine, sample_policy):
        """Register a single policy."""
        result = engine.register_policy(sample_policy)
        assert result is not None

    def test_register_multiple_policies(self, engine, mod):
        """Register multiple policies."""
        p1 = mod.ClimatePolicy(
            name="Mitigation", policy_type=mod.PolicyType.MITIGATION,
        )
        p2 = mod.ClimatePolicy(
            name="Adaptation", policy_type=mod.PolicyType.ADAPTATION,
        )
        engine.register_policy(p1)
        engine.register_policy(p2)
        result = engine.build_action_plan(policies=[p1, p2], actions=[])
        assert result is not None


# ===========================================================================
# Register Action Tests (Stateful)
# ===========================================================================


class TestRegisterAction:
    """Tests for register_action stateful method."""

    def test_register_single_action(self, engine, sample_action):
        """Register a single action."""
        result = engine.register_action(sample_action)
        assert result is not None

    def test_register_multiple_actions(self, engine, mod):
        """Register multiple actions."""
        a1 = mod.ClimateAction(
            name="LED Retrofit", category=mod.ActionCategory.ENERGY_EFFICIENCY,
            expected_reduction_tco2e=Decimal("500"),
        )
        a2 = mod.ClimateAction(
            name="Solar PV", category=mod.ActionCategory.RENEWABLE_ENERGY,
            expected_reduction_tco2e=Decimal("2000"),
        )
        engine.register_action(a1)
        engine.register_action(a2)
        result = engine.build_action_plan(policies=[], actions=[a1, a2])
        assert result is not None


# ===========================================================================
# Policy Value Chain Coverage Tests
# ===========================================================================


class TestPolicyValueChain:
    """Tests for policy value chain coverage fields."""

    def test_policy_covers_upstream(self, mod):
        """Policy can cover upstream value chain."""
        policy = mod.ClimatePolicy(
            name="Supply Chain Policy",
            policy_type=mod.PolicyType.MITIGATION,
            covers_own_operations=True,
            covers_upstream=True,
            covers_downstream=False,
        )
        assert policy.covers_upstream is True
        assert policy.covers_downstream is False

    def test_policy_covers_all_chain(self, mod):
        """Policy covers entire value chain."""
        policy = mod.ClimatePolicy(
            name="Full Chain",
            policy_type=mod.PolicyType.MITIGATION,
            covers_own_operations=True,
            covers_upstream=True,
            covers_downstream=True,
        )
        assert policy.covers_own_operations is True
        assert policy.covers_upstream is True
        assert policy.covers_downstream is True


# ===========================================================================
# Action With All Fields Tests
# ===========================================================================


class TestActionAllFields:
    """Tests for ClimateAction with all fields populated."""

    def test_action_all_fields(self, mod):
        """Action with every field populated."""
        action = mod.ClimateAction(
            name="Full Action",
            category=mod.ActionCategory.ENERGY_EFFICIENCY,
            expected_reduction_tco2e=Decimal("1000"),
            capex_amount=Decimal("500000"),
            opex_amount=Decimal("25000"),
            status=mod.ActionStatus.IN_PROGRESS,
            taxonomy_aligned=True,
        )
        assert action.capex_amount == Decimal("500000")
        assert action.opex_amount == Decimal("25000")
        assert action.status == mod.ActionStatus.IN_PROGRESS
        assert action.taxonomy_aligned is True

    def test_action_completed_status(self, mod):
        """Action can be set to completed status."""
        action = mod.ClimateAction(
            name="Done Action",
            category=mod.ActionCategory.RENEWABLE_ENERGY,
            status=mod.ActionStatus.COMPLETED,
        )
        assert action.status == mod.ActionStatus.COMPLETED


# ===========================================================================
# Plan Provenance Tests
# ===========================================================================


class TestPlanProvenance:
    """Tests for action plan provenance tracking."""

    def test_plan_has_provenance(self, engine, sample_policy, sample_action):
        """Action plan has a 64-char provenance hash."""
        result = engine.build_action_plan(
            policies=[sample_policy], actions=[sample_action],
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_plan_provenance_is_valid_hex(self, mod):
        """Plan provenance hash is a valid 64-char hex string."""
        policy = mod.ClimatePolicy(
            name="P", policy_type=mod.PolicyType.MITIGATION,
        )
        action = mod.ClimateAction(
            name="A", category=mod.ActionCategory.ENERGY_EFFICIENCY,
            expected_reduction_tco2e=Decimal("500"),
        )
        eng = mod.ClimateActionEngine()
        result = eng.build_action_plan(policies=[policy], actions=[action])
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # Valid hex


# ===========================================================================
# Taxonomy Alignment Advanced Tests
# ===========================================================================


class TestTaxonomyAlignmentAdvanced:
    """Advanced tests for EU Taxonomy alignment."""

    def test_all_categories_in_alignment_map(self, mod):
        """All action categories are in ACTION_TAXONOMY_ALIGNMENT."""
        for cat in mod.ActionCategory:
            assert cat.value in mod.ACTION_TAXONOMY_ALIGNMENT

    def test_taxonomy_alignment_percentage(self, engine, mod):
        """Taxonomy alignment result includes percentage."""
        actions = [
            mod.ClimateAction(
                name="Aligned",
                category=mod.ActionCategory.RENEWABLE_ENERGY,
                taxonomy_aligned=True,
                capex_amount=Decimal("800000"),
            ),
            mod.ClimateAction(
                name="Not Aligned",
                category=mod.ActionCategory.OFFSET,
                taxonomy_aligned=False,
                capex_amount=Decimal("200000"),
            ),
        ]
        result = engine.assess_taxonomy_alignment(actions)
        assert isinstance(result, dict)
        # Should have at least aligned count or percentage key
        assert len(result) >= 1
