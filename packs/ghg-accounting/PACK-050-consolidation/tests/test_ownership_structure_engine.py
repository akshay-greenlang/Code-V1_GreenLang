# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Ownership Structure Engine Tests

Tests ownership percentage setting, multi-tier equity chain resolution,
control type assessment, minority interest identification, ownership
change tracking, and effective ownership calculation accuracy.

Target: 60-80 tests.
"""

import pytest
from decimal import Decimal
from datetime import date

from engines.ownership_structure_engine import (
    OwnershipStructureEngine,
    OwnershipRecord,
    EquityChain,
    ControlAssessment,
    OwnershipChange,
    ControlType,
    OwnershipCategory,
    ChangeReason,
    _classify_ownership,
    _round2,
    _round4,
)


@pytest.fixture
def engine():
    """Fresh OwnershipStructureEngine instance."""
    return OwnershipStructureEngine()


@pytest.fixture
def populated_engine(engine, ownership_records):
    """Engine pre-populated with standard ownership records."""
    for rec in ownership_records:
        engine.set_ownership(rec)
    return engine


class TestSetOwnership:
    """Test ownership percentage setting."""

    def test_set_ownership_basic(self, engine, parent_entity_id, sub1_entity_id):
        record = engine.set_ownership({
            "owner_entity_id": parent_entity_id,
            "target_entity_id": sub1_entity_id,
            "ownership_pct": Decimal("100"),
        })
        assert isinstance(record, OwnershipRecord)
        assert record.ownership_pct == Decimal("100")

    def test_set_ownership_decimal_precision(self, engine):
        record = engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("33.33"),
        })
        assert record.ownership_pct == Decimal("33.33")

    def test_set_ownership_zero_percent(self, engine):
        record = engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("0"),
        })
        assert record.ownership_pct == Decimal("0")

    def test_set_ownership_hundred_percent(self, engine):
        record = engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("100"),
        })
        assert record.ownership_pct == Decimal("100")

    def test_set_ownership_self_reference_raises(self, engine):
        with pytest.raises(ValueError, match="cannot own itself"):
            engine.set_ownership({
                "owner_entity_id": "A",
                "target_entity_id": "A",
                "ownership_pct": Decimal("100"),
            })

    def test_set_ownership_with_control_flags(self, engine):
        record = engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("80"),
            "has_operational_control": True,
            "has_financial_control": True,
            "manages_operations": True,
            "directs_policies": True,
            "has_board_majority": True,
        })
        assert record.has_operational_control is True
        assert record.has_financial_control is True
        assert record.manages_operations is True

    def test_set_ownership_update_replaces(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("60"),
        })
        record = engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("80"),
        })
        assert record.ownership_pct == Decimal("80")

    def test_set_ownership_update_creates_change(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("60"),
        })
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("80"),
            "change_reason": "PARTIAL_ACQUISITION",
        })
        history = engine.get_ownership_history()
        assert len(history) == 1
        assert history[0].previous_pct == Decimal("60")
        assert history[0].new_pct == Decimal("80")

    def test_remove_ownership(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("100"),
        })
        engine.remove_ownership("A", "B")
        assert len(engine.get_all_records()) == 0

    def test_remove_nonexistent_raises(self, engine):
        with pytest.raises(KeyError, match="No ownership link"):
            engine.remove_ownership("A", "B")

    def test_populated_engine_record_count(self, populated_engine):
        records = populated_engine.get_all_records()
        assert len(records) == 5


class TestEquityChainResolution:
    """Test multi-tier equity chain resolution."""

    def test_direct_equity_chain(self, populated_engine, parent_entity_id, sub1_entity_id):
        chain = populated_engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        assert isinstance(chain, EquityChain)
        assert chain.effective_ownership_pct == Decimal("100")
        assert chain.chain_depth == 1
        assert chain.ownership_category == "WHOLLY_OWNED"

    def test_direct_80pct_chain(self, populated_engine, parent_entity_id, sub2_entity_id):
        chain = populated_engine.resolve_equity_chain(parent_entity_id, sub2_entity_id)
        assert chain.effective_ownership_pct == Decimal("80")
        assert chain.ownership_category == "MAJORITY"

    def test_direct_60pct_chain(self, populated_engine, parent_entity_id, sub3_entity_id):
        chain = populated_engine.resolve_equity_chain(parent_entity_id, sub3_entity_id)
        assert chain.effective_ownership_pct == Decimal("60")
        assert chain.ownership_category == "MAJORITY"

    def test_direct_50pct_jv(self, populated_engine, parent_entity_id, jv_entity_id):
        chain = populated_engine.resolve_equity_chain(parent_entity_id, jv_entity_id)
        assert chain.effective_ownership_pct == Decimal("50")
        assert chain.ownership_category == "JOINT_VENTURE"

    def test_direct_30pct_associate(self, populated_engine, parent_entity_id, associate_entity_id):
        chain = populated_engine.resolve_equity_chain(parent_entity_id, associate_entity_id)
        assert chain.effective_ownership_pct == Decimal("30")
        assert chain.ownership_category == "ASSOCIATE"

    def test_multi_tier_chain(self, engine, multi_tier_ownership):
        for link in multi_tier_ownership["links"]:
            engine.set_ownership(link)
        chain = engine.resolve_equity_chain("ENT-A", "ENT-C")
        assert chain.effective_ownership_pct == multi_tier_ownership["expected_effective_a_to_c"]
        assert chain.chain_depth == 2

    def test_multi_tier_decimal_precision(self, engine):
        engine.set_ownership({
            "owner_entity_id": "X",
            "target_entity_id": "Y",
            "ownership_pct": Decimal("33.33"),
        })
        engine.set_ownership({
            "owner_entity_id": "Y",
            "target_entity_id": "Z",
            "ownership_pct": Decimal("66.67"),
        })
        chain = engine.resolve_equity_chain("X", "Z")
        expected = _round4(Decimal("33.33") * Decimal("66.67") / Decimal("100"))
        assert chain.effective_ownership_pct == expected

    def test_no_path_returns_zero(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("100"),
        })
        chain = engine.resolve_equity_chain("A", "NONEXISTENT")
        assert chain.effective_ownership_pct == Decimal("0")
        assert chain.ownership_category == "NO_STAKE"

    def test_chain_provenance_hash(self, populated_engine, parent_entity_id, sub1_entity_id):
        chain = populated_engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        assert chain.provenance_hash != ""
        assert len(chain.provenance_hash) == 64

    def test_chain_provenance_deterministic(self, populated_engine, parent_entity_id, sub1_entity_id):
        c1 = populated_engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        c2 = populated_engine.resolve_equity_chain(parent_entity_id, sub1_entity_id)
        assert c1.provenance_hash == c2.provenance_hash

    def test_effective_ownership_convenience(self, populated_engine, parent_entity_id, sub2_entity_id):
        pct = populated_engine.get_effective_ownership(parent_entity_id, sub2_entity_id)
        assert pct == Decimal("80")

    def test_three_tier_chain(self, engine):
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("90")})
        engine.set_ownership({"owner_entity_id": "B", "target_entity_id": "C", "ownership_pct": Decimal("80")})
        engine.set_ownership({"owner_entity_id": "C", "target_entity_id": "D", "ownership_pct": Decimal("70")})
        chain = engine.resolve_equity_chain("A", "D")
        expected = _round4(Decimal("90") * Decimal("80") * Decimal("70") / Decimal("10000"))
        assert chain.effective_ownership_pct == expected
        assert chain.chain_depth == 3


class TestControlAssessment:
    """Test control type assessment."""

    def test_operational_control_subsidiary(self, populated_engine, parent_entity_id, sub1_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, sub1_entity_id)
        assert isinstance(assessment, ControlAssessment)
        assert assessment.has_operational_control is True
        assert assessment.control_type == ControlType.OPERATIONAL_CONTROL.value

    def test_no_control_associate(self, populated_engine, parent_entity_id, associate_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, associate_entity_id)
        assert assessment.has_operational_control is False

    def test_jv_joint_control(self, populated_engine, parent_entity_id, jv_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, jv_entity_id)
        assert assessment.control_type == ControlType.JOINT_CONTROL.value

    def test_financial_control_majority(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("60"),
            "directs_policies": True,
        })
        assessment = engine.assess_control("A", "B")
        assert assessment.has_financial_control is True

    def test_inclusion_pct_equity(self, populated_engine, parent_entity_id, sub2_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, sub2_entity_id)
        assert assessment.inclusion_pct_equity == Decimal("80")

    def test_inclusion_pct_operational_with_control(self, populated_engine, parent_entity_id, sub1_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, sub1_entity_id)
        assert assessment.inclusion_pct_operational == Decimal("100")

    def test_inclusion_pct_operational_without_control(self, populated_engine, parent_entity_id, associate_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, associate_entity_id)
        assert assessment.inclusion_pct_operational == Decimal("0")

    def test_control_assessment_provenance_hash(self, populated_engine, parent_entity_id, sub1_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, sub1_entity_id)
        assert len(assessment.provenance_hash) == 64

    def test_control_assessment_basis_populated(self, populated_engine, parent_entity_id, sub1_entity_id):
        assessment = populated_engine.assess_control(parent_entity_id, sub1_entity_id)
        assert len(assessment.assessment_basis) > 0

    def test_no_control_low_ownership(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("10"),
        })
        assessment = engine.assess_control("A", "B")
        assert assessment.control_type == ControlType.NO_CONTROL.value


class TestMinorityInterests:
    """Test minority interest identification."""

    def test_get_minority_interests(self, populated_engine, parent_entity_id):
        minorities = populated_engine.get_minority_interests(parent_entity_id)
        assert len(minorities) >= 2
        minority_ids = [m.target_entity_id for m in minorities]
        assert any("JV" in mid or "ASSOC" in mid for mid in minority_ids)

    def test_minority_interest_includes_jv(self, populated_engine, parent_entity_id, jv_entity_id):
        minorities = populated_engine.get_minority_interests(parent_entity_id)
        target_ids = [m.target_entity_id for m in minorities]
        assert jv_entity_id in target_ids

    def test_minority_interest_includes_associate(self, populated_engine, parent_entity_id, associate_entity_id):
        minorities = populated_engine.get_minority_interests(parent_entity_id)
        target_ids = [m.target_entity_id for m in minorities]
        assert associate_entity_id in target_ids

    def test_no_minority_interests_for_empty(self, engine):
        minorities = engine.get_minority_interests("NOBODY")
        assert len(minorities) == 0


class TestJVPartners:
    """Test JV partner identification."""

    def test_get_jv_partners(self, engine, parent_entity_id, jv_entity_id):
        engine.set_ownership({
            "owner_entity_id": parent_entity_id,
            "target_entity_id": jv_entity_id,
            "ownership_pct": Decimal("50"),
        })
        engine.set_ownership({
            "owner_entity_id": "EXT-PARTNER-001",
            "target_entity_id": jv_entity_id,
            "ownership_pct": Decimal("50"),
        })
        partners = engine.get_jv_partners(jv_entity_id)
        assert len(partners) == 2
        total_pct = sum(Decimal(p["ownership_pct"]) for p in partners)
        assert total_pct == Decimal("100")

    def test_jv_partners_empty(self, engine):
        partners = engine.get_jv_partners("NOBODY")
        assert len(partners) == 0


class TestOwnershipChangeTracking:
    """Test ownership change history."""

    def test_ownership_change_logged(self, engine):
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("60"),
        })
        engine.set_ownership({
            "owner_entity_id": "A",
            "target_entity_id": "B",
            "ownership_pct": Decimal("80"),
            "change_reason": "PARTIAL_ACQUISITION",
        })
        history = engine.get_ownership_history(owner_entity_id="A")
        assert len(history) == 1
        assert history[0].previous_pct == Decimal("60")
        assert history[0].new_pct == Decimal("80")

    def test_ownership_change_filter_by_target(self, engine):
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("60")})
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("80")})
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "C", "ownership_pct": Decimal("50")})
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "C", "ownership_pct": Decimal("70")})
        history = engine.get_ownership_history(target_entity_id="B")
        assert len(history) == 1

    def test_no_change_for_first_set(self, engine):
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("60")})
        history = engine.get_ownership_history()
        assert len(history) == 0


class TestOwnershipClassification:
    """Test ownership classification helper."""

    @pytest.mark.parametrize("pct,expected", [
        (Decimal("100"), "WHOLLY_OWNED"),
        (Decimal("75"), "MAJORITY"),
        (Decimal("51"), "MAJORITY"),
        (Decimal("50"), "JOINT_VENTURE"),
        (Decimal("30"), "ASSOCIATE"),
        (Decimal("20"), "ASSOCIATE"),
        (Decimal("10"), "MINORITY"),
        (Decimal("1"), "MINORITY"),
        (Decimal("0"), "NO_STAKE"),
    ])
    def test_classify_ownership(self, pct, expected):
        assert _classify_ownership(pct) == expected


class TestAccessors:
    """Test accessor methods."""

    def test_get_all_records(self, populated_engine):
        records = populated_engine.get_all_records()
        assert len(records) == 5

    def test_get_direct_ownership(self, populated_engine, parent_entity_id):
        direct = populated_engine.get_direct_ownership(parent_entity_id)
        assert len(direct) == 5

    def test_get_change_log(self, populated_engine):
        log = populated_engine.get_change_log()
        assert len(log) >= 5
