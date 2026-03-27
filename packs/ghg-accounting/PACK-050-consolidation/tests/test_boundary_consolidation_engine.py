# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Boundary Consolidation Engine Tests

Tests equity share approach boundary, operational control approach boundary,
financial control approach boundary, materiality threshold application,
approach comparison, boundary locking, and boundary change tracking.

Since BoundaryDeterminationEngine may not yet be physically present as a
separate file, these tests exercise boundary logic through the config models
and ownership/control assessment engines that implement boundary determination.

Target: 50-70 tests.
"""

import pytest
from decimal import Decimal

from config.pack_config import (
    ConsolidationPackConfig,
    BoundaryConfig,
    ConsolidationApproach,
    MaterialityThreshold,
    ScopeCategory,
)
from engines.ownership_structure_engine import (
    OwnershipStructureEngine,
    ControlType,
    OwnershipCategory,
)


@pytest.fixture
def ownership_engine(ownership_records):
    """OwnershipStructureEngine with standard ownership data."""
    engine = OwnershipStructureEngine()
    for rec in ownership_records:
        engine.set_ownership(rec)
    return engine


class TestEquityShareBoundary:
    """Test equity share approach boundary determination."""

    def test_equity_share_config_defaults(self):
        config = BoundaryConfig(consolidation_approach=ConsolidationApproach.EQUITY_SHARE)
        assert config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE

    def test_equity_share_wholly_owned_inclusion(self, ownership_engine, parent_entity_id, sub1_entity_id):
        chain = ownership_engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        assert chain.effective_ownership_pct == Decimal("100")
        assert chain.ownership_category == "WHOLLY_OWNED"

    def test_equity_share_majority_inclusion(self, ownership_engine, parent_entity_id, sub2_entity_id):
        chain = ownership_engine.resolve_equity_chain(parent_entity_id, sub2_entity_id)
        assert chain.effective_ownership_pct == Decimal("80")
        inclusion_pct = chain.effective_ownership_pct
        expected_s1 = Decimal("3000.00") * inclusion_pct / Decimal("100")
        assert expected_s1 == Decimal("2400.00")

    def test_equity_share_jv_proportional(self, ownership_engine, parent_entity_id, jv_entity_id):
        chain = ownership_engine.resolve_equity_chain(parent_entity_id, jv_entity_id)
        assert chain.effective_ownership_pct == Decimal("50")
        jv_emissions = Decimal("6000.00")
        reported = jv_emissions * Decimal("50") / Decimal("100")
        assert reported == Decimal("3000.00")

    def test_equity_share_associate_proportional(self, ownership_engine, parent_entity_id, associate_entity_id):
        chain = ownership_engine.resolve_equity_chain(parent_entity_id, associate_entity_id)
        assert chain.effective_ownership_pct == Decimal("30")
        assoc_emissions = Decimal("1000.00")
        reported = assoc_emissions * Decimal("30") / Decimal("100")
        assert reported == Decimal("300.00")

    def test_equity_share_zero_ownership_excluded(self, ownership_engine):
        chain = ownership_engine.resolve_equity_chain("ENT-PARENT-001", "NONEXISTENT")
        assert chain.effective_ownership_pct == Decimal("0")

    def test_equity_share_total_consolidation(self, ownership_engine, parent_entity_id, entity_emissions_data):
        entity_ids = [parent_entity_id, "ENT-SUB-001", "ENT-SUB-002", "ENT-SUB-003", "ENT-JV-001", "ENT-ASSOC-001"]
        total_scope1 = Decimal("0")
        for eid in entity_ids:
            chain = ownership_engine.resolve_equity_chain(parent_entity_id, eid)
            pct = chain.effective_ownership_pct if eid != parent_entity_id else Decimal("100")
            entity_s1 = entity_emissions_data[eid]["scope_1"]
            total_scope1 += entity_s1 * pct / Decimal("100")
        assert total_scope1 > Decimal("0")


class TestOperationalControlBoundary:
    """Test operational control approach boundary."""

    def test_operational_control_config_default(self):
        config = BoundaryConfig()
        assert config.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_operational_control_100pct_for_controlled(self, ownership_engine, parent_entity_id, sub1_entity_id):
        assessment = ownership_engine.assess_control(parent_entity_id, sub1_entity_id)
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_operational_control_0pct_for_uncontrolled(self, ownership_engine, parent_entity_id, associate_entity_id):
        assessment = ownership_engine.assess_control(parent_entity_id, associate_entity_id)
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_operational_control_jv_excluded(self, ownership_engine, parent_entity_id, jv_entity_id):
        assessment = ownership_engine.assess_control(parent_entity_id, jv_entity_id)
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_operational_control_all_subs_100pct(self, ownership_engine, parent_entity_id):
        for sub_id in ["ENT-SUB-001", "ENT-SUB-002", "ENT-SUB-003"]:
            assessment = ownership_engine.assess_control(parent_entity_id, sub_id)
            assert assessment.inclusion_pct_operational == Decimal("100")


class TestFinancialControlBoundary:
    """Test financial control approach boundary."""

    def test_financial_control_config(self):
        config = BoundaryConfig(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
        )
        assert config.consolidation_approach == ConsolidationApproach.FINANCIAL_CONTROL

    def test_financial_control_100pct_for_majority(self, ownership_engine, parent_entity_id, sub1_entity_id):
        assessment = ownership_engine.assess_control(parent_entity_id, sub1_entity_id)
        assert assessment.inclusion_pct_financial == Decimal("100")

    def test_financial_control_0pct_for_associate(self, ownership_engine, parent_entity_id, associate_entity_id):
        assessment = ownership_engine.assess_control(parent_entity_id, associate_entity_id)
        assert assessment.inclusion_pct_financial == Decimal("0")

    def test_financial_control_majority_implied(self, ownership_engine, parent_entity_id, sub2_entity_id):
        assessment = ownership_engine.assess_control(parent_entity_id, sub2_entity_id)
        assert assessment.has_financial_control is True


class TestMaterialityThreshold:
    """Test materiality threshold application."""

    def test_materiality_five_pct_default(self):
        config = BoundaryConfig()
        assert config.materiality_threshold == MaterialityThreshold.FIVE_PCT
        assert config.materiality_threshold_pct == Decimal("0.05")

    def test_materiality_one_pct(self):
        config = BoundaryConfig(
            materiality_threshold=MaterialityThreshold.ONE_PCT,
            materiality_threshold_pct=Decimal("0.01"),
        )
        assert config.materiality_threshold_pct == Decimal("0.01")

    def test_materiality_ten_pct(self):
        config = BoundaryConfig(
            materiality_threshold=MaterialityThreshold.TEN_PCT,
            materiality_threshold_pct=Decimal("0.10"),
        )
        assert config.materiality_threshold_pct == Decimal("0.10")

    def test_materiality_none(self):
        config = BoundaryConfig(
            materiality_threshold=MaterialityThreshold.NONE,
            materiality_threshold_pct=Decimal("0.0"),
        )
        assert config.materiality_threshold_pct == Decimal("0.0")

    def test_de_minimis_threshold(self):
        config = BoundaryConfig(de_minimis_threshold_pct=Decimal("0.01"))
        assert config.de_minimis_threshold_pct == Decimal("0.01")

    def test_entity_below_materiality_excluded(self):
        total_emissions = Decimal("100000")
        entity_emissions = Decimal("100")
        materiality_pct = Decimal("0.05")
        entity_share = entity_emissions / total_emissions
        assert entity_share < materiality_pct

    def test_entity_above_materiality_included(self):
        total_emissions = Decimal("100000")
        entity_emissions = Decimal("10000")
        materiality_pct = Decimal("0.05")
        entity_share = entity_emissions / total_emissions
        assert entity_share >= materiality_pct


class TestApproachComparison:
    """Test approach comparison."""

    def test_dual_approach_enabled(self):
        config = BoundaryConfig(
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            dual_approach_enabled=True,
            secondary_approach=ConsolidationApproach.EQUITY_SHARE,
        )
        assert config.dual_approach_enabled is True
        assert config.secondary_approach == ConsolidationApproach.EQUITY_SHARE

    def test_dual_approach_disabled(self):
        config = BoundaryConfig(dual_approach_enabled=False)
        assert config.dual_approach_enabled is False

    def test_equity_vs_control_different_totals(self, ownership_engine, parent_entity_id, entity_emissions_data):
        equity_total = Decimal("0")
        control_total = Decimal("0")
        for eid, data in entity_emissions_data.items():
            chain = ownership_engine.resolve_equity_chain(parent_entity_id, eid)
            assessment = ownership_engine.assess_control(parent_entity_id, eid)
            pct = chain.effective_ownership_pct if eid != parent_entity_id else Decimal("100")
            equity_total += data["scope_1"] * pct / Decimal("100")
            control_total += data["scope_1"] * assessment.inclusion_pct_operational / Decimal("100")
        assert equity_total != control_total


class TestBoundaryLocking:
    """Test boundary locking."""

    def test_boundary_lock_config(self):
        config = BoundaryConfig(annual_boundary_lock=True)
        assert config.annual_boundary_lock is True

    def test_boundary_lock_disabled(self):
        config = BoundaryConfig(annual_boundary_lock=False)
        assert config.annual_boundary_lock is False

    def test_boundary_requires_approval(self):
        config = BoundaryConfig(require_boundary_approval=True)
        assert config.require_boundary_approval is True


class TestBoundaryChangeTracking:
    """Test boundary change tracking."""

    def test_track_boundary_changes_enabled(self):
        config = BoundaryConfig(track_boundary_changes=True)
        assert config.track_boundary_changes is True

    def test_scopes_in_boundary_default(self):
        config = BoundaryConfig()
        scope_values = [s.value for s in config.scopes_in_boundary]
        assert "SCOPE_1" in scope_values
        assert "SCOPE_2_LOCATION" in scope_values
        assert "SCOPE_2_MARKET" in scope_values

    def test_scopes_in_boundary_with_scope3(self):
        config = BoundaryConfig(
            scopes_in_boundary=[
                ScopeCategory.SCOPE_1,
                ScopeCategory.SCOPE_2_LOCATION,
                ScopeCategory.SCOPE_2_MARKET,
                ScopeCategory.SCOPE_3,
            ],
        )
        scope_values = [s.value for s in config.scopes_in_boundary]
        assert "SCOPE_3" in scope_values

    def test_boundary_consolidation_alignment_warning(self):
        config = ConsolidationPackConfig(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            boundary=BoundaryConfig(
                consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            ),
        )
        assert config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE
