# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Control Approach Engine Tests

Tests operational control 100%/0%, financial control 100%/0%,
franchise boundary decisions, lease boundary decisions, and
outsourcing boundary decisions.

Target: 40-60 tests.
"""

import pytest
from decimal import Decimal

from config.pack_config import (
    ControlApproachConfig,
    ConsolidationApproach,
)
from engines.ownership_structure_engine import (
    OwnershipStructureEngine,
    ControlType,
    OwnershipCategory,
)


@pytest.fixture
def engine():
    """Fresh OwnershipStructureEngine."""
    return OwnershipStructureEngine()


class TestOperationalControl100_0:
    """Test operational control 100/0 inclusion logic."""

    def test_operational_control_100pct(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "SUB",
            "ownership_pct": Decimal("100"),
            "manages_operations": True,
            "has_operational_control": True,
        })
        assessment = engine.assess_control("PARENT", "SUB")
        assert assessment.control_type == ControlType.OPERATIONAL_CONTROL.value
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_operational_control_0pct(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "ASSOC",
            "ownership_pct": Decimal("30"),
            "manages_operations": False,
            "has_operational_control": False,
        })
        assessment = engine.assess_control("PARENT", "ASSOC")
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_operational_control_binary_with_80pct_ownership(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "SUB",
            "ownership_pct": Decimal("80"),
            "manages_operations": True,
        })
        assessment = engine.assess_control("PARENT", "SUB")
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_operational_control_binary_with_minority_but_ops(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "FRANCHISE",
            "ownership_pct": Decimal("0"),
            "manages_operations": True,
            "has_operational_control": True,
        })
        assessment = engine.assess_control("PARENT", "FRANCHISE")
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_no_operational_control_jv(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "JV",
            "ownership_pct": Decimal("50"),
            "manages_operations": False,
        })
        assessment = engine.assess_control("PARENT", "JV")
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_all_subs_100pct_under_operational(self, engine, ownership_records, parent_entity_id):
        for rec in ownership_records:
            engine.set_ownership(rec)
        for sub_id in ["ENT-SUB-001", "ENT-SUB-002", "ENT-SUB-003"]:
            assessment = engine.assess_control(parent_entity_id, sub_id)
            assert assessment.inclusion_pct_operational == Decimal("100")


class TestFinancialControl100_0:
    """Test financial control 100/0 inclusion logic."""

    def test_financial_control_majority_ownership(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "SUB",
            "ownership_pct": Decimal("60"),
            "directs_policies": True,
        })
        assessment = engine.assess_control("PARENT", "SUB")
        assert assessment.has_financial_control is True
        assert assessment.inclusion_pct_financial == Decimal("100")

    def test_financial_control_implied_by_majority(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "SUB",
            "ownership_pct": Decimal("51"),
        })
        assessment = engine.assess_control("PARENT", "SUB")
        assert assessment.has_financial_control is True
        assert assessment.inclusion_pct_financial == Decimal("100")

    def test_financial_control_board_majority(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "SUB",
            "ownership_pct": Decimal("40"),
            "has_board_majority": True,
        })
        assessment = engine.assess_control("PARENT", "SUB")
        assert assessment.has_financial_control is True

    def test_no_financial_control_minority(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "ASSOC",
            "ownership_pct": Decimal("30"),
        })
        assessment = engine.assess_control("PARENT", "ASSOC")
        assert assessment.inclusion_pct_financial == Decimal("0")

    def test_financial_control_0pct_for_no_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "MINORITY",
            "ownership_pct": Decimal("10"),
        })
        assessment = engine.assess_control("PARENT", "MINORITY")
        assert assessment.inclusion_pct_financial == Decimal("0")

    def test_spv_financial_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "SPV",
            "ownership_pct": Decimal("100"),
            "has_financial_control": True,
            "directs_policies": True,
        })
        assessment = engine.assess_control("PARENT", "SPV")
        assert assessment.has_financial_control is True
        assert assessment.inclusion_pct_financial == Decimal("100")


class TestFranchiseBoundary:
    """Test franchise boundary decisions."""

    def test_franchise_inclusion_config(self):
        config = ControlApproachConfig(franchise_inclusion=True)
        assert config.franchise_inclusion is True

    def test_franchise_exclusion_config(self):
        config = ControlApproachConfig(franchise_inclusion=False)
        assert config.franchise_inclusion is False

    def test_franchise_operational_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "FRANCHISOR",
            "target_entity_id": "FRANCHISE-01",
            "ownership_pct": Decimal("0"),
            "has_operational_control": True,
            "manages_operations": True,
        })
        assessment = engine.assess_control("FRANCHISOR", "FRANCHISE-01")
        assert assessment.has_operational_control is True
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_franchise_no_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "FRANCHISOR",
            "target_entity_id": "FRANCHISE-02",
            "ownership_pct": Decimal("0"),
            "manages_operations": False,
        })
        assessment = engine.assess_control("FRANCHISOR", "FRANCHISE-02")
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_franchise_equity_share_zero(self, engine):
        engine.set_ownership({
            "owner_entity_id": "FRANCHISOR",
            "target_entity_id": "FRANCHISE-01",
            "ownership_pct": Decimal("0"),
        })
        chain = engine.resolve_equity_chain("FRANCHISOR", "FRANCHISE-01")
        assert chain.effective_ownership_pct == Decimal("0")


class TestLeaseBoundary:
    """Test lease boundary decisions."""

    def test_lease_config_finance_include(self):
        config = ControlApproachConfig(leased_asset_treatment="FINANCE_LEASE_INCLUDE")
        assert config.leased_asset_treatment == "FINANCE_LEASE_INCLUDE"

    def test_lease_config_all_exclude(self):
        config = ControlApproachConfig(leased_asset_treatment="ALL_EXCLUDE")
        assert config.leased_asset_treatment == "ALL_EXCLUDE"

    def test_lease_config_case_by_case(self):
        config = ControlApproachConfig(leased_asset_treatment="CASE_BY_CASE")
        assert config.leased_asset_treatment == "CASE_BY_CASE"

    def test_lease_invalid_treatment_raises(self):
        with pytest.raises(ValueError, match="leased_asset_treatment"):
            ControlApproachConfig(leased_asset_treatment="INVALID")


class TestOutsourcingBoundary:
    """Test outsourcing boundary decisions."""

    def test_outsourced_ops_with_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "OUTSOURCED",
            "ownership_pct": Decimal("0"),
            "manages_operations": True,
            "has_operational_control": True,
        })
        assessment = engine.assess_control("PARENT", "OUTSOURCED")
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_outsourced_ops_without_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "PARENT",
            "target_entity_id": "CONTRACTOR",
            "ownership_pct": Decimal("0"),
        })
        assessment = engine.assess_control("PARENT", "CONTRACTOR")
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_jv_control_override_config(self):
        config = ControlApproachConfig(jv_control_override=True)
        assert config.jv_control_override is True

    def test_jv_control_override_disabled(self):
        config = ControlApproachConfig(jv_control_override=False)
        assert config.jv_control_override is False


class TestControlTestMethods:
    """Test control test method configuration."""

    @pytest.mark.parametrize("method", ["POLICY_AUTHORITY", "MAJORITY_BOARD", "COMBINED"])
    def test_valid_control_test_methods(self, method):
        config = ControlApproachConfig(control_test_method=method)
        assert config.control_test_method == method

    def test_invalid_control_test_method_raises(self):
        with pytest.raises(ValueError, match="control_test_method"):
            ControlApproachConfig(control_test_method="INVALID_METHOD")

    def test_document_control_basis(self):
        config = ControlApproachConfig(document_control_basis=True)
        assert config.document_control_basis is True


class TestVetoRights:
    """Test veto rights impact on control assessment."""

    def test_veto_rights_noted(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("40"),
            "has_veto_rights": True,
        })
        assessment = engine.assess_control("A", "B")
        assert any("veto" in basis.lower() for basis in assessment.assessment_basis)

    def test_veto_without_majority_no_control(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("30"),
            "has_veto_rights": True,
        })
        assessment = engine.assess_control("A", "B")
        assert assessment.control_type == ControlType.NO_CONTROL.value
