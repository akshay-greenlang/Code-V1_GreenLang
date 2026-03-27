# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Equity Share Engine Tests

Tests proportional allocation calculation, multi-tier equity chain
consolidation, JV equity split, associate inclusion, scope-level equity
adjustment, portfolio consolidated total, reconciliation, and Decimal
precision.

Target: 50-70 tests.
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP

from engines.ownership_structure_engine import (
    OwnershipStructureEngine,
    OwnershipCategory,
    _round2,
    _round4,
)


@pytest.fixture
def engine(ownership_records):
    """OwnershipStructureEngine with standard data."""
    eng = OwnershipStructureEngine()
    for rec in ownership_records:
        eng.set_ownership(rec)
    return eng


class TestProportionalAllocation:
    """Test proportional allocation calculation."""

    def test_wholly_owned_100pct(self, engine, parent_entity_id, sub1_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        s1 = Decimal("15000.00")
        allocated = _round2(s1 * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("15000.00")

    def test_80pct_subsidiary(self, engine, parent_entity_id, sub2_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, sub2_entity_id)
        s1 = Decimal("3000.00")
        allocated = _round2(s1 * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("2400.00")

    def test_60pct_subsidiary(self, engine, parent_entity_id, sub3_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, sub3_entity_id)
        s1 = Decimal("8000.00")
        allocated = _round2(s1 * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("4800.00")

    def test_50pct_jv(self, engine, parent_entity_id, jv_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, jv_entity_id)
        s1 = Decimal("6000.00")
        allocated = _round2(s1 * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("3000.00")

    def test_30pct_associate(self, engine, parent_entity_id, associate_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, associate_entity_id)
        s1 = Decimal("1000.00")
        allocated = _round2(s1 * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("300.00")

    def test_zero_emissions_zero_allocation(self, engine, parent_entity_id, sub1_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        allocated = _round2(Decimal("0") * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("0.00")

    def test_fractional_ownership_precision(self, engine):
        eng = OwnershipStructureEngine()
        eng.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("33.33"),
        })
        chain = eng.resolve_equity_chain("A", "B")
        emissions = Decimal("10000.00")
        allocated = _round2(emissions * chain.effective_ownership_pct / Decimal("100"))
        assert allocated == Decimal("3333.00")


class TestMultiTierEquityConsolidation:
    """Test multi-tier equity chain consolidation."""

    def test_two_tier_chain_60pct(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("80")})
        eng.set_ownership({"owner_entity_id": "B", "target_entity_id": "C", "ownership_pct": Decimal("75")})
        chain = eng.resolve_equity_chain("A", "C")
        assert chain.effective_ownership_pct == Decimal("60.0000")

    def test_three_tier_chain(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("100")})
        eng.set_ownership({"owner_entity_id": "B", "target_entity_id": "C", "ownership_pct": Decimal("50")})
        eng.set_ownership({"owner_entity_id": "C", "target_entity_id": "D", "ownership_pct": Decimal("80")})
        chain = eng.resolve_equity_chain("A", "D")
        expected = _round4(Decimal("100") * Decimal("50") * Decimal("80") / Decimal("10000"))
        assert chain.effective_ownership_pct == expected

    def test_chain_effective_below_20pct_is_minority(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("30")})
        eng.set_ownership({"owner_entity_id": "B", "target_entity_id": "C", "ownership_pct": Decimal("50")})
        chain = eng.resolve_equity_chain("A", "C")
        assert chain.effective_ownership_pct == Decimal("15.0000")
        assert chain.ownership_category == OwnershipCategory.MINORITY.value


class TestJVEquitySplit:
    """Test JV equity split calculations."""

    def test_50_50_jv_split(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "PARENT", "target_entity_id": "JV", "ownership_pct": Decimal("50")})
        eng.set_ownership({"owner_entity_id": "PARTNER", "target_entity_id": "JV", "ownership_pct": Decimal("50")})
        partners = eng.get_jv_partners("JV")
        total = sum(Decimal(p["ownership_pct"]) for p in partners)
        assert total == Decimal("100")

    def test_60_40_jv_split(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "PARENT", "target_entity_id": "JV", "ownership_pct": Decimal("60")})
        eng.set_ownership({"owner_entity_id": "PARTNER", "target_entity_id": "JV", "ownership_pct": Decimal("40")})
        partners = eng.get_jv_partners("JV")
        total = sum(Decimal(p["ownership_pct"]) for p in partners)
        assert total == Decimal("100")

    def test_three_way_jv(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "JV", "ownership_pct": Decimal("40")})
        eng.set_ownership({"owner_entity_id": "B", "target_entity_id": "JV", "ownership_pct": Decimal("35")})
        eng.set_ownership({"owner_entity_id": "C", "target_entity_id": "JV", "ownership_pct": Decimal("25")})
        partners = eng.get_jv_partners("JV")
        total = sum(Decimal(p["ownership_pct"]) for p in partners)
        assert total == Decimal("100")

    def test_jv_emissions_allocation(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "PARENT", "target_entity_id": "JV", "ownership_pct": Decimal("50")})
        chain = eng.resolve_equity_chain("PARENT", "JV")
        jv_total = Decimal("10000.00")
        parent_share = _round2(jv_total * chain.effective_ownership_pct / Decimal("100"))
        assert parent_share == Decimal("5000.00")


class TestAssociateInclusion:
    """Test associate entity inclusion."""

    def test_associate_30pct_included(self, engine, parent_entity_id, associate_entity_id):
        chain = engine.resolve_equity_chain(parent_entity_id, associate_entity_id)
        assert chain.effective_ownership_pct == Decimal("30")
        assert chain.ownership_category == OwnershipCategory.ASSOCIATE.value

    def test_associate_20pct_boundary(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("20")})
        chain = eng.resolve_equity_chain("A", "B")
        assert chain.ownership_category == OwnershipCategory.ASSOCIATE.value

    def test_below_associate_is_minority(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("19")})
        chain = eng.resolve_equity_chain("A", "B")
        assert chain.ownership_category == OwnershipCategory.MINORITY.value


class TestScopeLevelEquityAdjustment:
    """Test scope-level equity adjustment."""

    @pytest.mark.parametrize("scope_key,expected_pct", [
        ("scope_1", Decimal("80")),
        ("scope_2_location", Decimal("80")),
        ("scope_2_market", Decimal("80")),
        ("scope_3", Decimal("80")),
    ])
    def test_scope_level_adjustment_80pct(self, engine, parent_entity_id, sub2_entity_id, entity_emissions_data, scope_key, expected_pct):
        chain = engine.resolve_equity_chain(parent_entity_id, sub2_entity_id)
        assert chain.effective_ownership_pct == expected_pct
        raw = entity_emissions_data[sub2_entity_id][scope_key]
        adjusted = _round2(raw * expected_pct / Decimal("100"))
        expected_val = _round2(raw * Decimal("80") / Decimal("100"))
        assert adjusted == expected_val


class TestPortfolioConsolidatedTotal:
    """Test portfolio consolidated total calculation."""

    def test_full_equity_share_consolidation(self, engine, parent_entity_id, entity_emissions_data):
        total_s1 = Decimal("0")
        for eid, data in entity_emissions_data.items():
            if eid == parent_entity_id:
                pct = Decimal("100")
            else:
                chain = engine.resolve_equity_chain(parent_entity_id, eid)
                pct = chain.effective_ownership_pct
            total_s1 += _round2(data["scope_1"] * pct / Decimal("100"))
        expected = (
            Decimal("500.00") +
            Decimal("15000.00") +
            Decimal("2400.00") +
            Decimal("4800.00") +
            Decimal("3000.00") +
            Decimal("300.00")
        )
        assert total_s1 == expected

    def test_portfolio_total_nonzero(self, engine, parent_entity_id, entity_emissions_data):
        total = Decimal("0")
        for eid, data in entity_emissions_data.items():
            if eid == parent_entity_id:
                pct = Decimal("100")
            else:
                chain = engine.resolve_equity_chain(parent_entity_id, eid)
                pct = chain.effective_ownership_pct
            total += _round2(data["total_location"] * pct / Decimal("100"))
        assert total > Decimal("0")


class TestReconciliation:
    """Test reconciliation (sum of partners = 100%)."""

    def test_partner_sum_100(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "JV", "ownership_pct": Decimal("50")})
        eng.set_ownership({"owner_entity_id": "B", "target_entity_id": "JV", "ownership_pct": Decimal("50")})
        partners = eng.get_jv_partners("JV")
        total = sum(Decimal(p["ownership_pct"]) for p in partners)
        assert total == Decimal("100")

    def test_partner_sum_less_than_100(self):
        eng = OwnershipStructureEngine()
        eng.set_ownership({"owner_entity_id": "A", "target_entity_id": "TARGET", "ownership_pct": Decimal("30")})
        eng.set_ownership({"owner_entity_id": "B", "target_entity_id": "TARGET", "ownership_pct": Decimal("25")})
        partners = eng.get_jv_partners("TARGET")
        total = sum(Decimal(p["ownership_pct"]) for p in partners)
        assert total == Decimal("55")
        assert total < Decimal("100")


class TestDecimalPrecision:
    """Test Decimal precision (no floating point errors)."""

    def test_no_floating_point_error_33_33(self):
        pct = Decimal("33.33")
        emissions = Decimal("100.00")
        result = _round2(emissions * pct / Decimal("100"))
        assert result == Decimal("33.33")

    def test_no_floating_point_error_66_67(self):
        pct = Decimal("66.67")
        emissions = Decimal("100.00")
        result = _round2(emissions * pct / Decimal("100"))
        assert result == Decimal("66.67")

    def test_chain_product_no_float_errors(self):
        p1 = Decimal("33.33")
        p2 = Decimal("66.67")
        effective = _round4(p1 * p2 / Decimal("100"))
        assert isinstance(effective, Decimal)
        assert "E" not in str(effective)

    def test_round2_half_up(self):
        assert _round2(Decimal("1.005")) == Decimal("1.01")
        assert _round2(Decimal("1.004")) == Decimal("1.00")
        assert _round2(Decimal("1.015")) == Decimal("1.02")

    def test_round4_half_up(self):
        assert _round4(Decimal("1.00005")) == Decimal("1.0001")

    @pytest.mark.parametrize("val", [
        Decimal("0.01"),
        Decimal("10.50"),
        Decimal("99.99"),
        Decimal("100.00"),
        Decimal("0.00"),
    ])
    def test_round2_preserves_type(self, val):
        result = _round2(val)
        assert isinstance(result, Decimal)
