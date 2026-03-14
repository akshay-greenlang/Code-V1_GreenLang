# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Deep Consolidation Tests
============================================================

Comprehensive tests for multi-entity consolidation logic including
all three approaches, intercompany elimination, minority interest,
reconciliation, currency handling, edge cases, and provenance.

Test count: 25
Author: GreenLang QA Team
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

from consolidation_engine import (
    ConsolidationApproach,
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationMethod,
    EntityDefinition,
    EntityESRSData,
    IntercompanyTransaction,
    TransactionType,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_5_entity_engine() -> ConsolidationEngine:
    """Create a ConsolidationEngine with parent + 5 subsidiaries and data."""
    engine = ConsolidationEngine()

    # Parent
    engine.add_entity(EntityDefinition(
        entity_id="parent", name="Parent AG", country="DE",
        ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        parent_entity_id=None, employee_count=8000,
    ))
    engine.set_entity_data("parent", EntityESRSData(
        entity_id="parent",
        data_points={"scope1": 12500, "scope2": 8200, "scope3": 45000, "revenue": 2100000000},
        reporting_period="2025-01-01/2025-12-31", quality_score=92.5,
    ))

    # 5 subsidiaries
    subs = [
        ("sub-fr", "France SAS", "FR", 100.0, ConsolidationMethod.OPERATIONAL_CONTROL, 4800, 3100, 18500, 380000000, 89.0),
        ("sub-it", "Italia S.r.l.", "IT", 100.0, ConsolidationMethod.OPERATIONAL_CONTROL, 3200, 2400, 12000, 245000000, 87.5),
        ("sub-es", "Espana S.L.", "ES", 80.0, ConsolidationMethod.FINANCIAL_CONTROL, 2100, 1800, 8500, 175000000, 85.0),
        ("sub-nl", "Nederland B.V.", "NL", 100.0, ConsolidationMethod.OPERATIONAL_CONTROL, 1500, 1200, 5800, 120000000, 91.0),
        ("sub-pl", "Polska Sp.z.o.o", "PL", 100.0, ConsolidationMethod.OPERATIONAL_CONTROL, 950, 1600, 3200, 65000000, 83.5),
    ]
    for eid, name, country, own, method, s1, s2, s3, rev, qs in subs:
        engine.add_entity(EntityDefinition(
            entity_id=eid, name=name, country=country,
            ownership_pct=Decimal(str(own)), consolidation_method=method,
            parent_entity_id="parent", employee_count=500,
        ))
        engine.set_entity_data(eid, EntityESRSData(
            entity_id=eid,
            data_points={"scope1": s1, "scope2": s2, "scope3": s3, "revenue": rev},
            reporting_period="2025-01-01/2025-12-31", quality_score=qs,
        ))

    return engine


# ===========================================================================
# Core Consolidation Tests
# ===========================================================================

class TestDeepConsolidation:
    """Deep tests for multi-entity consolidation."""

    @pytest.mark.asyncio
    async def test_5_entity_operational_control(self):
        """5-entity operational control consolidation sums all ops-controlled entities."""
        engine = _setup_5_entity_engine()
        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.entity_count == 6

        # sub-es uses FINANCIAL_CONTROL -> factor=0 for OPERATIONAL_CONTROL
        # scope1 = 12500 + 4800 + 3200 + 0 + 1500 + 950 = 22950
        scope1 = Decimal(result.consolidated_data["scope1"])
        assert scope1 == Decimal("22950.0000")

    @pytest.mark.asyncio
    async def test_5_entity_financial_control(self):
        """5-entity financial control includes both operational and financial entities."""
        engine = _setup_5_entity_engine()
        result = await engine.consolidate(ConsolidationApproach.FINANCIAL_CONTROL)

        # All entities included at 100%: 12500+4800+3200+2100+1500+950 = 25050
        scope1 = Decimal(result.consolidated_data["scope1"])
        assert scope1 == Decimal("25050.0000")

    @pytest.mark.asyncio
    async def test_5_entity_equity_share(self):
        """5-entity equity share applies ownership percentages."""
        engine = _setup_5_entity_engine()
        result = await engine.consolidate(ConsolidationApproach.EQUITY_SHARE)

        # scope1: 12500*1.0 + 4800*1.0 + 3200*1.0 + 2100*0.8 + 1500*1.0 + 950*1.0
        # = 12500 + 4800 + 3200 + 1680 + 1500 + 950 = 24630
        scope1 = Decimal(result.consolidated_data["scope1"])
        assert scope1 == Decimal("24630.0000")

    @pytest.mark.asyncio
    async def test_intercompany_scope3_elimination(self):
        """Intercompany Cat 1 emission transfer is eliminated to avoid double counting."""
        engine = _setup_5_entity_engine()
        txn = IntercompanyTransaction(
            from_entity="sub-it", to_entity="sub-es",
            transaction_type=TransactionType.EMISSION_TRANSFER,
            amount=Decimal("1250"), scope3_category=1,
        )
        engine.add_intercompany_transaction(txn)
        result = await engine.consolidate(ConsolidationApproach.FINANCIAL_CONTROL)

        assert len(result.eliminations_applied) == 1
        assert result.eliminations_applied[0]["scope3_category"] == 1
        assert "_elimination_scope3_cat1_eliminated" in result.consolidated_data

    @pytest.mark.asyncio
    async def test_intercompany_revenue_elimination(self):
        """Intercompany revenue is fully eliminated."""
        engine = _setup_5_entity_engine()
        txn = IntercompanyTransaction(
            from_entity="parent", to_entity="sub-fr",
            transaction_type=TransactionType.REVENUE,
            amount=Decimal("45000000"),
        )
        engine.add_intercompany_transaction(txn)
        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)

        assert len(result.eliminations_applied) == 1
        assert result.eliminations_applied[0]["transaction_type"] == "revenue"
        assert result.eliminations_applied[0]["elimination_amount"] == "45000000.0000"

    @pytest.mark.asyncio
    async def test_minority_interest_calculation(self):
        """80% owned subsidiary has 20% minority interest disclosure."""
        engine = _setup_5_entity_engine()
        adjustments = engine.calculate_minority_interest()

        # Only sub-es (80%) has minority interest
        assert len(adjustments) == 1
        adj = adjustments[0]
        assert adj["entity_id"] == "sub-es"
        assert Decimal(adj["minority_pct"]) == Decimal("20")
        # scope1: 2100 * 0.20 = 420
        assert adj["minority_data_points"]["scope1"] == "420.0000"
        assert adj["provenance_hash"] != ""

    @pytest.mark.asyncio
    async def test_mixed_consolidation_approaches(self):
        """Entities with different methods produce different results per approach."""
        engine = _setup_5_entity_engine()

        result_oc = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        result_fc = await engine.consolidate(ConsolidationApproach.FINANCIAL_CONTROL)
        result_eq = await engine.consolidate(ConsolidationApproach.EQUITY_SHARE)

        scope1_oc = Decimal(result_oc.consolidated_data["scope1"])
        scope1_fc = Decimal(result_fc.consolidated_data["scope1"])
        scope1_eq = Decimal(result_eq.consolidated_data["scope1"])

        # FC > EQ > OC because sub-es is only in FC and partially in EQ
        assert scope1_fc > scope1_eq
        assert scope1_eq > scope1_oc

    def test_entity_hierarchy_validation(self):
        """Verify entity hierarchy is correctly structured as a tree."""
        engine = _setup_5_entity_engine()
        hierarchy = engine.get_entity_hierarchy()

        assert hierarchy["total_entities"] == 6
        assert len(hierarchy["group"]) == 1
        root = hierarchy["group"][0]
        assert root["entity_id"] == "parent"
        assert len(root["children"]) == 5
        child_ids = {c["entity_id"] for c in root["children"]}
        assert child_ids == {"sub-fr", "sub-it", "sub-es", "sub-nl", "sub-pl"}

    @pytest.mark.asyncio
    async def test_reconciliation_zero_variance(self):
        """Operational control with 100% ownership should have zero variance."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.set_entity_data("parent", EntityESRSData(
            entity_id="parent", data_points={"scope1": 5000},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))
        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.reconciliation_variance == Decimal("0.0000")

    @pytest.mark.asyncio
    async def test_reconciliation_with_adjustments(self):
        """Equity share approach creates adjustments tracked in reconciliation."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.add_entity(EntityDefinition(
            entity_id="sub-jv", name="JV Co", country="DE",
            ownership_pct=Decimal("50"), consolidation_method=ConsolidationMethod.FINANCIAL_CONTROL,
            parent_entity_id="parent",
        ))
        engine.set_entity_data("parent", EntityESRSData(
            entity_id="parent", data_points={"scope1": 1000},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))
        engine.set_entity_data("sub-jv", EntityESRSData(
            entity_id="sub-jv", data_points={"scope1": 1000},
            reporting_period="2025-01-01/2025-12-31", quality_score=90.0,
        ))

        result = await engine.consolidate(ConsolidationApproach.EQUITY_SHARE)
        entries = engine.generate_reconciliation(result)

        jv_entries = [e for e in entries if e.entity_id == "sub-jv"]
        assert len(jv_entries) == 1
        assert jv_entries[0].adjustment == Decimal("-500.0000")  # 1000 -> 500 (50%)

    @pytest.mark.asyncio
    async def test_consolidation_provenance_hash(self):
        """Verify consolidation result has a non-empty SHA-256 provenance hash."""
        engine = _setup_5_entity_engine()
        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    @pytest.mark.asyncio
    async def test_consolidation_with_missing_entity_data(self):
        """Consolidation skips entities with no data when not required."""
        engine = ConsolidationEngine(ConsolidationConfig(require_all_entities_data=False))
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.add_entity(EntityDefinition(
            entity_id="sub-x", name="Sub X", country="FR",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
            parent_entity_id="parent",
        ))
        engine.set_entity_data("parent", EntityESRSData(
            entity_id="parent", data_points={"scope1": 5000},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))
        # No data for sub-x

        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.entity_count == 2
        scope1 = Decimal(result.consolidated_data["scope1"])
        assert scope1 == Decimal("5000.0000")  # Only parent data
        assert result.per_entity_results["sub-x"]["status"] == "no_data"

    @pytest.mark.asyncio
    async def test_consolidation_currency_handling(self):
        """Verify currency field is preserved in entity definitions."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
            currency="EUR",
        ))
        engine.add_entity(EntityDefinition(
            entity_id="sub-pl", name="PL Sub", country="PL",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
            parent_entity_id="parent", currency="PLN",
        ))
        assert engine.entities["parent"].currency == "EUR"
        assert engine.entities["sub-pl"].currency == "PLN"

    def test_cross_border_consolidation(self):
        """Verify entity hierarchy distinguishes EU and non-EU entities."""
        engine = _setup_5_entity_engine()
        hierarchy = engine.get_entity_hierarchy()
        assert hierarchy["eu_entities"] >= 5  # All are EU
        assert hierarchy["non_eu_entities"] == 0

    def test_entity_data_quality_scoring(self):
        """Verify entity data quality scores are within valid range."""
        engine = _setup_5_entity_engine()
        for eid, data in engine.entity_data.items():
            assert 0.0 <= data.quality_score <= 100.0

    # -- Edge Cases --

    @pytest.mark.asyncio
    async def test_empty_entities_raises(self):
        """Consolidation with no entities raises ValueError."""
        engine = ConsolidationEngine()
        with pytest.raises(ValueError, match="No entities registered"):
            await engine.consolidate()

    @pytest.mark.asyncio
    async def test_single_entity_consolidation(self):
        """Single entity consolidation returns its own values unchanged."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="solo", name="Solo Corp", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.set_entity_data("solo", EntityESRSData(
            entity_id="solo", data_points={"scope1": 1234.5678},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))
        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.entity_count == 1
        assert Decimal(result.consolidated_data["scope1"]) == Decimal("1234.5678")

    @pytest.mark.asyncio
    async def test_100_pct_ownership(self):
        """100% owned subsidiary contributes full value in all approaches."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.add_entity(EntityDefinition(
            entity_id="sub", name="Sub", country="FR",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
            parent_entity_id="parent",
        ))
        engine.set_entity_data("parent", EntityESRSData(
            entity_id="parent", data_points={"v": 100},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))
        engine.set_entity_data("sub", EntityESRSData(
            entity_id="sub", data_points={"v": 100},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))

        for approach in ConsolidationApproach:
            result = await engine.consolidate(approach)
            val = Decimal(result.consolidated_data["v"])
            assert val == Decimal("200.0000"), (
                f"100% ownership should give 200 for {approach.value}, got {val}"
            )

    def test_zero_ownership_excluded(self):
        """0% owned entity contributes nothing in equity share."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.add_entity(EntityDefinition(
            entity_id="assoc", name="Associate", country="IT",
            ownership_pct=Decimal("0"), consolidation_method=ConsolidationMethod.EQUITY_SHARE,
            parent_entity_id="parent",
        ))
        engine.set_entity_data("assoc", EntityESRSData(
            entity_id="assoc", data_points={"v": 999},
            reporting_period="2025-01-01/2025-12-31", quality_score=80.0,
        ))
        factor = engine._get_consolidation_factor(
            engine.entities["assoc"], ConsolidationApproach.EQUITY_SHARE,
        )
        assert factor == Decimal("0.0000")

    def test_duplicate_entity_rejected(self):
        """Adding same entity_id twice raises ValueError."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="dup", name="Dup", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        with pytest.raises(ValueError, match="already registered"):
            engine.add_entity(EntityDefinition(
                entity_id="dup", name="Dup2", country="DE",
                ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
            ))

    def test_set_data_unregistered_entity_rejected(self):
        """Setting data for unregistered entity raises ValueError."""
        engine = ConsolidationEngine()
        with pytest.raises(ValueError, match="not registered"):
            engine.set_entity_data("ghost", EntityESRSData(
                entity_id="ghost", data_points={"v": 1},
                reporting_period="2025-01-01/2025-12-31", quality_score=50.0,
            ))

    def test_intercompany_same_entity_rejected(self):
        """Intercompany transaction from entity to itself is rejected."""
        engine = ConsolidationEngine()
        engine.add_entity(EntityDefinition(
            entity_id="self", name="Self", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        with pytest.raises(ValueError, match="must be different"):
            engine.add_intercompany_transaction(IntercompanyTransaction(
                from_entity="self", to_entity="self",
                transaction_type=TransactionType.REVENUE,
                amount=Decimal("100"),
            ))

    @pytest.mark.asyncio
    async def test_require_all_entities_data(self):
        """require_all_entities_data=True raises when data is missing."""
        engine = ConsolidationEngine(ConsolidationConfig(require_all_entities_data=True))
        engine.add_entity(EntityDefinition(
            entity_id="parent", name="Parent", country="DE",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))
        engine.add_entity(EntityDefinition(
            entity_id="sub", name="Sub", country="FR",
            ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
            parent_entity_id="parent",
        ))
        engine.set_entity_data("parent", EntityESRSData(
            entity_id="parent", data_points={"v": 100},
            reporting_period="2025-01-01/2025-12-31", quality_score=95.0,
        ))
        with pytest.raises(ValueError, match="Missing data"):
            await engine.consolidate()

    def test_engine_reset(self):
        """Reset clears all entities, data, and transactions."""
        engine = _setup_5_entity_engine()
        assert len(engine.entities) == 6
        engine.reset()
        assert len(engine.entities) == 0
        assert len(engine.entity_data) == 0
        assert len(engine.transactions) == 0
