# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Intercompany Elimination Engine Tests

Tests energy transfer registration, double-counting elimination calculation,
waste transfer elimination, reconciliation, partial elimination, net
consolidated calculation, and elimination log generation.

Target: 60-80 tests.
"""

import pytest
from decimal import Decimal

from engines.intercompany_elimination_engine import (
    IntercompanyEliminationEngine,
    TransferRecord,
    EliminationEntry,
    EliminationResult,
    TransferReconciliation,
    TransferType,
    EliminationScope,
    ReconciliationStatus,
    DEFAULT_RECONCILIATION_TOLERANCE_PCT,
)


@pytest.fixture
def engine():
    """Fresh IntercompanyEliminationEngine."""
    return IntercompanyEliminationEngine()


@pytest.fixture
def populated_engine(engine, transfer_records):
    """Engine pre-populated with standard transfer records."""
    for tr in transfer_records:
        engine.register_transfer(tr)
    return engine


class TestTransferRegistration:
    """Test energy transfer registration."""

    def test_register_single_transfer(self, engine):
        transfer = engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "ENT-A",
            "buyer_entity_id": "ENT-B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("50000"),
            "seller_emissions_tco2e": Decimal("12500"),
            "buyer_emissions_tco2e": Decimal("12500"),
        })
        assert isinstance(transfer, TransferRecord)
        assert transfer.seller_entity_id == "ENT-A"
        assert transfer.buyer_entity_id == "ENT-B"
        assert transfer.quantity == Decimal("50000")

    def test_register_transfer_generates_id(self, engine):
        transfer = engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "STEAM",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
        })
        assert transfer.transfer_id is not None
        assert len(transfer.transfer_id) > 0

    def test_register_transfer_provenance_hash(self, engine):
        transfer = engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
        })
        assert transfer.provenance_hash != ""
        assert len(transfer.provenance_hash) == 64

    def test_register_same_entity_raises(self, engine):
        with pytest.raises(ValueError, match="same entity"):
            engine.register_transfer({
                "reporting_year": 2026,
                "seller_entity_id": "A",
                "buyer_entity_id": "A",
                "transfer_type": "ELECTRICITY",
                "quantity": Decimal("1000"),
                "seller_emissions_tco2e": Decimal("100"),
                "buyer_emissions_tco2e": Decimal("100"),
            })

    def test_register_all_transfer_types(self, engine):
        for tt in TransferType:
            transfer = engine.register_transfer({
                "reporting_year": 2026,
                "seller_entity_id": "SELLER",
                "buyer_entity_id": f"BUYER-{tt.value}",
                "transfer_type": tt.value,
                "quantity": Decimal("100"),
                "seller_emissions_tco2e": Decimal("10"),
                "buyer_emissions_tco2e": Decimal("10"),
            })
            assert transfer.transfer_type == tt.value

    def test_register_batch_transfers(self, engine, transfer_records):
        results = engine.register_transfers_batch(transfer_records)
        assert len(results) == 3

    def test_default_buyer_scope_electricity(self, engine):
        transfer = engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
        })
        assert transfer.buyer_scope == EliminationScope.SCOPE_2_LOCATION.value

    def test_default_buyer_scope_waste(self, engine):
        transfer = engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "WASTE",
            "quantity": Decimal("100"),
            "seller_emissions_tco2e": Decimal("10"),
            "buyer_emissions_tco2e": Decimal("10"),
        })
        assert transfer.buyer_scope == EliminationScope.SCOPE_3.value

    def test_populated_engine_transfer_count(self, populated_engine):
        transfers = populated_engine.get_transfers_for_year(2026)
        assert len(transfers) == 3


class TestEliminationCalculation:
    """Test double-counting elimination calculation."""

    def test_basic_elimination(self, engine, entity_total_emissions):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "ENT-SUB-001",
            "buyer_entity_id": "ENT-SUB-002",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("50000"),
            "seller_emissions_tco2e": Decimal("12500"),
            "buyer_emissions_tco2e": Decimal("12500"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        assert isinstance(result, EliminationResult)
        assert result.total_eliminations == Decimal("12500.00")
        assert result.elimination_count == 1

    def test_elimination_uses_min_of_seller_buyer(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("120"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("500"), "B": Decimal("500")},
        )
        assert result.total_eliminations == Decimal("100.00")

    def test_net_consolidated_calculation(self, engine, entity_total_emissions):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "ENT-SUB-001",
            "buyer_entity_id": "ENT-SUB-002",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("50000"),
            "seller_emissions_tco2e": Decimal("12500"),
            "buyer_emissions_tco2e": Decimal("12500"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        total = sum(entity_total_emissions.values())
        assert result.net_consolidated == Decimal(str(total)) - Decimal("12500.00")

    def test_elimination_provenance_hash(self, engine, entity_total_emissions):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "ENT-SUB-001",
            "buyer_entity_id": "ENT-SUB-002",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("50000"),
            "seller_emissions_tco2e": Decimal("12500"),
            "buyer_emissions_tco2e": Decimal("12500"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        assert len(result.provenance_hash) == 64

    def test_no_transfers_no_eliminations(self, engine):
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("1000")},
        )
        assert result.total_eliminations == Decimal("0")
        assert result.net_consolidated == Decimal("1000.00")

    def test_multiple_transfers_accumulated(self, populated_engine, entity_total_emissions):
        result = populated_engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        assert result.elimination_count == 3
        assert result.total_eliminations > Decimal("0")

    def test_eliminations_by_type_populated(self, populated_engine, entity_total_emissions):
        result = populated_engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        assert "ELECTRICITY" in result.eliminations_by_type
        assert "WASTE" in result.eliminations_by_type

    def test_eliminations_by_scope_populated(self, populated_engine, entity_total_emissions):
        result = populated_engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        assert len(result.eliminations_by_scope) > 0


class TestWasteTransferElimination:
    """Test waste transfer elimination."""

    def test_waste_transfer_elimination(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "WASTE",
            "quantity": Decimal("500"),
            "quantity_unit": "tonnes",
            "seller_emissions_tco2e": Decimal("250"),
            "buyer_emissions_tco2e": Decimal("250"),
            "buyer_scope": "SCOPE_3",
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("1000"), "B": Decimal("1000")},
        )
        assert result.total_eliminations == Decimal("250.00")
        assert "WASTE" in result.eliminations_by_type


class TestReconciliation:
    """Test reconciliation (seller matches buyer)."""

    def test_reconciled_transfer(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
        })
        recons = engine.reconcile_transfers(2026)
        assert len(recons) == 1
        assert recons[0].status == ReconciliationStatus.RECONCILED.value
        assert recons[0].variance == Decimal("0.00")

    def test_variance_within_tolerance(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("99"),
        })
        recons = engine.reconcile_transfers(2026, tolerance_pct=Decimal("2"))
        assert recons[0].status == ReconciliationStatus.RECONCILED.value

    def test_variance_outside_tolerance(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("80"),
        })
        recons = engine.reconcile_transfers(2026, tolerance_pct=Decimal("2"))
        assert recons[0].status == ReconciliationStatus.VARIANCE.value

    def test_reconciliation_variance_calculation(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("110"),
            "buyer_emissions_tco2e": Decimal("100"),
        })
        recons = engine.reconcile_transfers(2026)
        assert recons[0].variance == Decimal("10.00")

    def test_reconciliation_provenance_hash(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
        })
        recons = engine.reconcile_transfers(2026)
        assert len(recons[0].provenance_hash) == 64


class TestPartialElimination:
    """Test partial elimination."""

    def test_partial_elimination_50pct(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "STEAM",
            "quantity": Decimal("10000"),
            "seller_emissions_tco2e": Decimal("2000"),
            "buyer_emissions_tco2e": Decimal("2000"),
            "intra_group_pct": Decimal("50"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("5000"), "B": Decimal("5000")},
        )
        assert result.total_eliminations == Decimal("1000.00")

    def test_partial_elimination_75pct(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("400"),
            "buyer_emissions_tco2e": Decimal("400"),
            "intra_group_pct": Decimal("75"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("1000"), "B": Decimal("1000")},
        )
        assert result.total_eliminations == Decimal("300.00")

    def test_partial_elimination_flagged(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
            "intra_group_pct": Decimal("50"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("500"), "B": Decimal("500")},
        )
        assert result.elimination_entries[0].is_partial is True

    def test_full_elimination_not_flagged_partial(self, engine):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "A",
            "buyer_entity_id": "B",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("1000"),
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
            "intra_group_pct": Decimal("100"),
        })
        result = engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions={"A": Decimal("500"), "B": Decimal("500")},
        )
        assert result.elimination_entries[0].is_partial is False


class TestNetConsolidatedCalculation:
    """Test net consolidated calculation."""

    def test_get_net_consolidated(self, engine, entity_total_emissions):
        engine.register_transfer({
            "reporting_year": 2026,
            "seller_entity_id": "ENT-SUB-001",
            "buyer_entity_id": "ENT-SUB-002",
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("50000"),
            "seller_emissions_tco2e": Decimal("12500"),
            "buyer_emissions_tco2e": Decimal("12500"),
        })
        summary = engine.get_net_consolidated(2026, entity_total_emissions)
        assert "net_consolidated" in summary
        assert "total_eliminations" in summary
        assert "provenance_hash" in summary

    def test_net_consolidated_no_transfers(self, engine):
        summary = engine.get_net_consolidated(
            2026,
            {"A": Decimal("1000"), "B": Decimal("2000")},
        )
        assert summary["net_consolidated"] == "3000.00"


class TestEliminationLog:
    """Test elimination log generation."""

    def test_elimination_log_after_calc(self, populated_engine, entity_total_emissions):
        populated_engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        log = populated_engine.get_elimination_log()
        assert len(log) == 3

    def test_elimination_log_filter_by_year(self, populated_engine, entity_total_emissions):
        populated_engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        log = populated_engine.get_elimination_log(reporting_year=2026)
        assert len(log) == 3

    def test_elimination_log_filter_by_entity(self, populated_engine, entity_total_emissions, sub1_entity_id):
        populated_engine.calculate_eliminations(
            reporting_year=2026,
            entity_emissions=entity_total_emissions,
        )
        log = populated_engine.get_elimination_log(entity_id=sub1_entity_id)
        assert len(log) >= 2

    def test_elimination_log_empty_before_calc(self, populated_engine):
        log = populated_engine.get_elimination_log()
        assert len(log) == 0


class TestAccessors:
    """Test accessor methods."""

    def test_get_transfer(self, populated_engine):
        transfers = populated_engine.get_transfers_for_year(2026)
        transfer = populated_engine.get_transfer(transfers[0].transfer_id)
        assert transfer is not None

    def test_get_transfer_nonexistent_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_transfer("NONEXISTENT")

    def test_get_transfers_for_year(self, populated_engine):
        transfers = populated_engine.get_transfers_for_year(2026)
        assert len(transfers) == 3

    def test_get_transfers_for_entity(self, populated_engine, sub1_entity_id):
        transfers = populated_engine.get_transfers_for_entity(sub1_entity_id)
        assert len(transfers) == 2

    def test_get_result_nonexistent_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_result("NONEXISTENT")

    def test_get_all_results_empty(self, engine):
        assert len(engine.get_all_results()) == 0

    def test_get_transfer_summary(self, populated_engine):
        summary = populated_engine.get_transfer_summary(2026)
        assert summary["total_transfers"] == 3
        assert "by_type" in summary
        assert "provenance_hash" in summary
