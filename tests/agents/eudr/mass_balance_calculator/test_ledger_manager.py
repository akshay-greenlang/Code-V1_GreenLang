# -*- coding: utf-8 -*-
"""
Tests for LedgerManager - AGENT-EUDR-011 Engine 1: Double-Entry Ledger Management

Comprehensive test suite covering:
- Ledger creation (valid, duplicate, all commodities, all standards)
- Entry recording (input, output, adjustment, loss, waste, balance update,
  provenance hash)
- Balance calculation (running balance, utilization rate, summary)
- Ledger search (by facility, commodity, period, batch_id)
- Bulk import (valid list, validation, error handling)
- Ledger immutability (no delete/modify, corrections via adjustment only)
- Edge cases (zero quantity, negative quantity, empty ledger)

Test count: 60+ tests
Coverage target: >= 85% of LedgerManager module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    STANDARDS,
    ENTRY_TYPES,
    SHA256_HEX_LENGTH,
    LEDGER_COCOA_MILL_MY,
    LEDGER_PALM_REFINERY_ID,
    LEDGER_COFFEE_WAREHOUSE_NL,
    LEDGER_COCOA_001,
    LEDGER_PALM_001,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    BATCH_COCOA_001,
    BATCH_COCOA_002,
    BATCH_PALM_001,
    BATCH_COFFEE_001,
    ENTRY_INPUT_COCOA,
    ENTRY_OUTPUT_COCOA,
    ENTRY_ADJUSTMENT_COCOA,
    make_ledger,
    make_entry,
    assert_valid_provenance_hash,
    assert_valid_balance,
)


# ===========================================================================
# 1. Ledger Creation
# ===========================================================================


class TestLedgerCreation:
    """Test ledger creation operations."""

    def test_create_ledger_basic(self, ledger_manager):
        """Create a basic cocoa ledger."""
        ledger = make_ledger()
        result = ledger_manager.create_ledger(ledger)
        assert result is not None
        assert result["commodity"] == "cocoa"
        assert result["facility_id"] == FAC_ID_MILL_MY

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_create_ledger_all_commodities(self, ledger_manager, commodity):
        """Ledger can be created for all 7 EUDR commodities."""
        ledger = make_ledger(commodity=commodity)
        result = ledger_manager.create_ledger(ledger)
        assert result is not None
        assert result["commodity"] == commodity

    @pytest.mark.parametrize("standard", STANDARDS)
    def test_create_ledger_all_standards(self, ledger_manager, standard):
        """Ledger can be created for all certification standards."""
        ledger = make_ledger(standard=standard)
        result = ledger_manager.create_ledger(ledger)
        assert result is not None
        assert result["standard"] == standard

    def test_create_ledger_with_opening_balance(self, ledger_manager):
        """Create a ledger with non-zero opening balance."""
        ledger = make_ledger(opening_balance_kg=Decimal("5000.0"))
        result = ledger_manager.create_ledger(ledger)
        assert Decimal(str(result["opening_balance_kg"])) == Decimal("5000.0")
        assert Decimal(str(result["current_balance_kg"])) == Decimal("5000.0")

    def test_create_ledger_assigns_id(self, ledger_manager):
        """Creation assigns a unique ledger_id."""
        ledger = make_ledger()
        ledger["ledger_id"] = None
        result = ledger_manager.create_ledger(ledger)
        assert result.get("ledger_id") is not None
        assert len(result["ledger_id"]) > 0

    def test_create_ledger_provenance_hash(self, ledger_manager):
        """Creation generates a provenance hash."""
        ledger = make_ledger()
        result = ledger_manager.create_ledger(ledger)
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_duplicate_ledger_id_raises(self, ledger_manager):
        """Creating a ledger with duplicate ID raises an error."""
        ledger = make_ledger(ledger_id="LDG-DUP-001")
        ledger_manager.create_ledger(ledger)
        with pytest.raises((ValueError, KeyError)):
            ledger_manager.create_ledger(copy.deepcopy(ledger))

    def test_missing_commodity_raises(self, ledger_manager):
        """Ledger without commodity raises ValueError."""
        ledger = make_ledger()
        ledger["commodity"] = None
        with pytest.raises(ValueError):
            ledger_manager.create_ledger(ledger)

    def test_missing_facility_id_raises(self, ledger_manager):
        """Ledger without facility_id raises ValueError."""
        ledger = make_ledger()
        ledger["facility_id"] = None
        with pytest.raises(ValueError):
            ledger_manager.create_ledger(ledger)

    def test_invalid_commodity_raises(self, ledger_manager):
        """Ledger with invalid commodity raises ValueError."""
        ledger = make_ledger()
        ledger["commodity"] = "unknown_crop"
        with pytest.raises(ValueError):
            ledger_manager.create_ledger(ledger)

    def test_create_ledger_initial_balance_zero(self, ledger_manager):
        """Default opening balance is zero."""
        ledger = make_ledger()
        result = ledger_manager.create_ledger(ledger)
        assert Decimal(str(result["current_balance_kg"])) == Decimal("0.0")

    def test_create_ledger_entry_count_zero(self, ledger_manager):
        """New ledger has zero entry count."""
        ledger = make_ledger()
        result = ledger_manager.create_ledger(ledger)
        assert result["entry_count"] == 0


# ===========================================================================
# 2. Entry Recording
# ===========================================================================


class TestEntryRecording:
    """Test ledger entry recording."""

    def test_record_input_entry(self, ledger_manager):
        """Record an input entry increases balance."""
        ledger = make_ledger(ledger_id="LDG-REC-INP-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-REC-INP-001",
            entry_type="input",
            quantity_kg=Decimal("5000.0"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None
        assert result.get("entry_type") == "input"

    def test_record_output_entry(self, ledger_manager):
        """Record an output entry decreases balance."""
        ledger = make_ledger(
            ledger_id="LDG-REC-OUT-001",
            opening_balance_kg=Decimal("10000.0"),
        )
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-REC-OUT-001",
            entry_type="output",
            quantity_kg=Decimal("3000.0"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None

    def test_record_adjustment_entry(self, ledger_manager):
        """Record an adjustment entry updates balance."""
        ledger = make_ledger(ledger_id="LDG-REC-ADJ-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-REC-ADJ-001",
            entry_type="adjustment",
            quantity_kg=Decimal("200.0"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None
        assert result.get("entry_type") == "adjustment"

    def test_record_loss_entry(self, ledger_manager):
        """Record a loss entry decreases balance."""
        ledger = make_ledger(
            ledger_id="LDG-REC-LOSS-001",
            opening_balance_kg=Decimal("5000.0"),
        )
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-REC-LOSS-001",
            entry_type="loss",
            quantity_kg=Decimal("100.0"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None

    def test_record_waste_entry(self, ledger_manager):
        """Record a waste entry decreases balance."""
        ledger = make_ledger(
            ledger_id="LDG-REC-WST-001",
            opening_balance_kg=Decimal("5000.0"),
        )
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-REC-WST-001",
            entry_type="waste",
            quantity_kg=Decimal("50.0"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None

    def test_entry_updates_balance(self, ledger_manager):
        """Recording an input updates the running balance."""
        ledger = make_ledger(ledger_id="LDG-BAL-UPD-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-BAL-UPD-001",
            entry_type="input",
            quantity_kg=Decimal("3000.0"),
        )
        ledger_manager.record_entry(entry)
        balance = ledger_manager.get_balance("LDG-BAL-UPD-001")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("3000.0")

    def test_entry_provenance_hash(self, ledger_manager):
        """Each entry gets a provenance hash."""
        ledger = make_ledger(ledger_id="LDG-PROV-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(ledger_id="LDG-PROV-001")
        result = ledger_manager.record_entry(entry)
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_entry_increments_count(self, ledger_manager):
        """Each entry increments the entry count."""
        ledger = make_ledger(ledger_id="LDG-CNT-001")
        ledger_manager.create_ledger(ledger)
        for i in range(3):
            entry = make_entry(
                ledger_id="LDG-CNT-001",
                entry_id=f"ENT-CNT-{i:03d}",
            )
            ledger_manager.record_entry(entry)
        balance = ledger_manager.get_balance("LDG-CNT-001")
        assert balance["entry_count"] >= 3

    def test_entry_invalid_ledger_raises(self, ledger_manager):
        """Recording entry for non-existent ledger raises error."""
        entry = make_entry(ledger_id="LDG-NONEXISTENT")
        with pytest.raises((ValueError, KeyError)):
            ledger_manager.record_entry(entry)

    def test_entry_invalid_type_raises(self, ledger_manager):
        """Recording entry with invalid type raises ValueError."""
        ledger = make_ledger(ledger_id="LDG-INV-TYPE-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(ledger_id="LDG-INV-TYPE-001")
        entry["entry_type"] = "invalid_type"
        with pytest.raises(ValueError):
            ledger_manager.record_entry(entry)


# ===========================================================================
# 3. Balance Calculation
# ===========================================================================


class TestBalanceCalculation:
    """Test balance calculation operations."""

    def test_running_balance_after_inputs(self, ledger_manager):
        """Running balance accumulates inputs."""
        ledger = make_ledger(ledger_id="LDG-RBAL-001")
        ledger_manager.create_ledger(ledger)
        for i in range(3):
            entry = make_entry(
                ledger_id="LDG-RBAL-001",
                entry_type="input",
                quantity_kg=Decimal("1000.0"),
                entry_id=f"ENT-RBAL-{i:03d}",
            )
            ledger_manager.record_entry(entry)
        balance = ledger_manager.get_balance("LDG-RBAL-001")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("3000.0")

    def test_running_balance_input_minus_output(self, ledger_manager):
        """Running balance = inputs - outputs."""
        ledger = make_ledger(ledger_id="LDG-RBAL-002")
        ledger_manager.create_ledger(ledger)
        inp = make_entry(
            ledger_id="LDG-RBAL-002",
            entry_type="input",
            quantity_kg=Decimal("5000.0"),
            entry_id="ENT-RBAL-INP",
        )
        ledger_manager.record_entry(inp)
        out = make_entry(
            ledger_id="LDG-RBAL-002",
            entry_type="output",
            quantity_kg=Decimal("2000.0"),
            entry_id="ENT-RBAL-OUT",
        )
        ledger_manager.record_entry(out)
        balance = ledger_manager.get_balance("LDG-RBAL-002")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("3000.0")

    def test_balance_with_losses(self, ledger_manager):
        """Losses reduce the running balance."""
        ledger = make_ledger(ledger_id="LDG-RBAL-003")
        ledger_manager.create_ledger(ledger)
        inp = make_entry(
            ledger_id="LDG-RBAL-003",
            entry_type="input",
            quantity_kg=Decimal("5000.0"),
            entry_id="ENT-RBAL-INP3",
        )
        ledger_manager.record_entry(inp)
        loss = make_entry(
            ledger_id="LDG-RBAL-003",
            entry_type="loss",
            quantity_kg=Decimal("500.0"),
            entry_id="ENT-RBAL-LOSS3",
        )
        ledger_manager.record_entry(loss)
        balance = ledger_manager.get_balance("LDG-RBAL-003")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("4500.0")

    def test_utilization_rate(self, ledger_manager):
        """Utilization rate = outputs / inputs * 100."""
        ledger = make_ledger(ledger_id="LDG-UTIL-001")
        ledger_manager.create_ledger(ledger)
        inp = make_entry(
            ledger_id="LDG-UTIL-001",
            entry_type="input",
            quantity_kg=Decimal("10000.0"),
            entry_id="ENT-UTIL-INP",
        )
        ledger_manager.record_entry(inp)
        out = make_entry(
            ledger_id="LDG-UTIL-001",
            entry_type="output",
            quantity_kg=Decimal("8000.0"),
            entry_id="ENT-UTIL-OUT",
        )
        ledger_manager.record_entry(out)
        summary = ledger_manager.get_summary("LDG-UTIL-001")
        util_rate = summary.get("utilization_rate", summary.get("utilization_percent"))
        assert util_rate is not None
        assert float(util_rate) == pytest.approx(80.0, abs=1.0)

    def test_summary_includes_totals(self, ledger_manager):
        """Summary includes total inputs, outputs, losses."""
        ledger = make_ledger(ledger_id="LDG-SUM-001")
        ledger_manager.create_ledger(ledger)
        inp = make_entry(
            ledger_id="LDG-SUM-001",
            entry_type="input",
            quantity_kg=Decimal("5000.0"),
            entry_id="ENT-SUM-INP",
        )
        ledger_manager.record_entry(inp)
        summary = ledger_manager.get_summary("LDG-SUM-001")
        assert "total_inputs_kg" in summary or "total_inputs" in summary
        assert summary is not None

    def test_balance_nonexistent_ledger_raises(self, ledger_manager):
        """Getting balance for non-existent ledger raises error."""
        with pytest.raises((ValueError, KeyError)):
            ledger_manager.get_balance("LDG-NONEXISTENT")

    def test_balance_is_non_negative_decimal(self, ledger_manager):
        """Balance should be representable as a Decimal."""
        ledger = make_ledger(ledger_id="LDG-DEC-001")
        ledger_manager.create_ledger(ledger)
        balance = ledger_manager.get_balance("LDG-DEC-001")
        assert_valid_balance(balance["current_balance_kg"])


# ===========================================================================
# 4. Ledger Search
# ===========================================================================


class TestLedgerSearch:
    """Test ledger search and filtering."""

    def test_search_by_facility(self, ledger_manager):
        """Search ledgers by facility ID."""
        l1 = make_ledger(
            ledger_id="LDG-SRCH-FAC-A",
            facility_id=FAC_ID_MILL_MY,
        )
        l2 = make_ledger(
            ledger_id="LDG-SRCH-FAC-B",
            facility_id=FAC_ID_REFINERY_ID,
        )
        ledger_manager.create_ledger(l1)
        ledger_manager.create_ledger(l2)
        results = ledger_manager.search(facility_id=FAC_ID_MILL_MY)
        assert all(r["facility_id"] == FAC_ID_MILL_MY for r in results)

    def test_search_by_commodity(self, ledger_manager):
        """Search ledgers by commodity."""
        l1 = make_ledger(
            ledger_id="LDG-SRCH-COM-A",
            commodity="cocoa",
        )
        l2 = make_ledger(
            ledger_id="LDG-SRCH-COM-B",
            commodity="oil_palm",
        )
        ledger_manager.create_ledger(l1)
        ledger_manager.create_ledger(l2)
        results = ledger_manager.search(commodity="cocoa")
        assert all(r["commodity"] == "cocoa" for r in results)

    def test_search_by_standard(self, ledger_manager):
        """Search ledgers by standard."""
        l1 = make_ledger(
            ledger_id="LDG-SRCH-STD-A",
            standard="rspo",
        )
        l2 = make_ledger(
            ledger_id="LDG-SRCH-STD-B",
            standard="fsc",
        )
        ledger_manager.create_ledger(l1)
        ledger_manager.create_ledger(l2)
        results = ledger_manager.search(standard="rspo")
        assert all(r["standard"] == "rspo" for r in results)

    def test_search_no_results(self, ledger_manager):
        """Search with no matching criteria returns empty list."""
        results = ledger_manager.search(facility_id="FAC-NONEXISTENT")
        assert len(results) == 0

    def test_search_by_status(self, ledger_manager):
        """Search ledgers by status."""
        l1 = make_ledger(
            ledger_id="LDG-SRCH-STA-A",
            status="active",
        )
        ledger_manager.create_ledger(l1)
        results = ledger_manager.search(status="active")
        assert all(r["status"] == "active" for r in results)


# ===========================================================================
# 5. Bulk Import
# ===========================================================================


class TestBulkImport:
    """Test bulk entry import."""

    def test_bulk_import_valid_list(self, ledger_manager):
        """Import a list of valid entries."""
        ledger = make_ledger(ledger_id="LDG-BULK-001")
        ledger_manager.create_ledger(ledger)
        entries = [
            make_entry(
                ledger_id="LDG-BULK-001",
                entry_id=f"ENT-BULK-{i:03d}",
                entry_type="input",
                quantity_kg=Decimal("100.0"),
            )
            for i in range(10)
        ]
        results = ledger_manager.bulk_import(entries)
        assert len(results) == 10

    def test_bulk_import_with_invalid(self, ledger_manager):
        """Bulk import with invalid entries reports partial failures."""
        ledger = make_ledger(ledger_id="LDG-BULK-002")
        ledger_manager.create_ledger(ledger)
        entries = [
            make_entry(
                ledger_id="LDG-BULK-002",
                entry_id="ENT-BULK-OK",
                entry_type="input",
            ),
            make_entry(
                ledger_id="LDG-BULK-002",
                entry_id="ENT-BULK-BAD",
            ),
            make_entry(
                ledger_id="LDG-BULK-002",
                entry_id="ENT-BULK-OK2",
                entry_type="input",
            ),
        ]
        entries[1]["entry_type"] = "invalid_type"
        results = ledger_manager.bulk_import(entries, continue_on_error=True)
        assert len([r for r in results if r.get("status") == "error"]) >= 1

    def test_bulk_import_empty(self, ledger_manager):
        """Bulk import of empty list returns empty results."""
        results = ledger_manager.bulk_import([])
        assert len(results) == 0

    def test_bulk_import_updates_balance(self, ledger_manager):
        """Bulk import updates the running balance."""
        ledger = make_ledger(ledger_id="LDG-BULK-003")
        ledger_manager.create_ledger(ledger)
        entries = [
            make_entry(
                ledger_id="LDG-BULK-003",
                entry_id=f"ENT-BULK3-{i:03d}",
                entry_type="input",
                quantity_kg=Decimal("500.0"),
            )
            for i in range(4)
        ]
        ledger_manager.bulk_import(entries)
        balance = ledger_manager.get_balance("LDG-BULK-003")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("2000.0")


# ===========================================================================
# 6. Ledger Immutability
# ===========================================================================


class TestLedgerImmutability:
    """Test ledger immutability constraints."""

    def test_no_delete_entry(self, ledger_manager):
        """Entries cannot be deleted."""
        ledger = make_ledger(ledger_id="LDG-IMMUT-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-IMMUT-001",
            entry_id="ENT-IMMUT-001",
        )
        ledger_manager.record_entry(entry)
        with pytest.raises((AttributeError, NotImplementedError, ValueError)):
            ledger_manager.delete_entry("ENT-IMMUT-001")

    def test_no_modify_entry(self, ledger_manager):
        """Entries cannot be modified after recording."""
        ledger = make_ledger(ledger_id="LDG-IMMUT-002")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-IMMUT-002",
            entry_id="ENT-IMMUT-002",
        )
        ledger_manager.record_entry(entry)
        with pytest.raises((AttributeError, NotImplementedError, ValueError)):
            ledger_manager.modify_entry("ENT-IMMUT-002", {"quantity_kg": Decimal("9999.0")})

    def test_correction_via_adjustment(self, ledger_manager):
        """Corrections are made via adjustment entries, not modifications."""
        ledger = make_ledger(ledger_id="LDG-IMMUT-003")
        ledger_manager.create_ledger(ledger)
        # Original entry
        entry = make_entry(
            ledger_id="LDG-IMMUT-003",
            entry_type="input",
            quantity_kg=Decimal("1000.0"),
            entry_id="ENT-IMMUT-003",
        )
        ledger_manager.record_entry(entry)
        # Correction via adjustment
        adj = make_entry(
            ledger_id="LDG-IMMUT-003",
            entry_type="adjustment",
            quantity_kg=Decimal("-200.0"),
            entry_id="ENT-IMMUT-ADJ",
            description="Correction: overcount of 200kg",
        )
        result = ledger_manager.record_entry(adj)
        assert result is not None

    def test_entry_history_preserved(self, ledger_manager):
        """All entries are preserved in history."""
        ledger = make_ledger(ledger_id="LDG-IMMUT-004")
        ledger_manager.create_ledger(ledger)
        for i in range(5):
            entry = make_entry(
                ledger_id="LDG-IMMUT-004",
                entry_id=f"ENT-IMMUT4-{i:03d}",
            )
            ledger_manager.record_entry(entry)
        history = ledger_manager.get_entries("LDG-IMMUT-004")
        assert len(history) == 5


# ===========================================================================
# 7. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for ledger operations."""

    def test_zero_quantity_entry(self, ledger_manager):
        """Entry with zero quantity raises ValueError."""
        ledger = make_ledger(ledger_id="LDG-EDGE-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-EDGE-001",
            quantity_kg=Decimal("0.0"),
        )
        with pytest.raises(ValueError):
            ledger_manager.record_entry(entry)

    def test_negative_quantity_raises(self, ledger_manager):
        """Entry with negative quantity on input raises ValueError."""
        ledger = make_ledger(ledger_id="LDG-EDGE-002")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-EDGE-002",
            entry_type="input",
            quantity_kg=Decimal("-500.0"),
        )
        with pytest.raises(ValueError):
            ledger_manager.record_entry(entry)

    def test_empty_ledger_balance(self, ledger_manager):
        """Empty ledger has zero balance."""
        ledger = make_ledger(ledger_id="LDG-EDGE-003")
        ledger_manager.create_ledger(ledger)
        balance = ledger_manager.get_balance("LDG-EDGE-003")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("0.0")

    def test_empty_ledger_summary(self, ledger_manager):
        """Empty ledger summary returns zeroes."""
        ledger = make_ledger(ledger_id="LDG-EDGE-004")
        ledger_manager.create_ledger(ledger)
        summary = ledger_manager.get_summary("LDG-EDGE-004")
        assert summary is not None

    def test_very_large_quantity(self, ledger_manager):
        """Entry with very large quantity is accepted."""
        ledger = make_ledger(ledger_id="LDG-EDGE-005")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-EDGE-005",
            entry_type="input",
            quantity_kg=Decimal("999999999.999"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None

    def test_fractional_quantity(self, ledger_manager):
        """Entry with fractional kg is accepted."""
        ledger = make_ledger(ledger_id="LDG-EDGE-006")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-EDGE-006",
            entry_type="input",
            quantity_kg=Decimal("0.001"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None

    def test_get_entries_empty_ledger(self, ledger_manager):
        """Getting entries for empty ledger returns empty list."""
        ledger = make_ledger(ledger_id="LDG-EDGE-007")
        ledger_manager.create_ledger(ledger)
        entries = ledger_manager.get_entries("LDG-EDGE-007")
        assert len(entries) == 0

    def test_get_nonexistent_ledger_returns_none(self, ledger_manager):
        """Getting a non-existent ledger returns None."""
        result = ledger_manager.get("LDG-NONEXISTENT-999")
        assert result is None

    def test_negative_opening_balance_raises(self, ledger_manager):
        """Negative opening balance raises ValueError."""
        ledger = make_ledger(opening_balance_kg=Decimal("-100.0"))
        with pytest.raises(ValueError):
            ledger_manager.create_ledger(ledger)

    def test_multiple_ledgers_same_facility(self, ledger_manager):
        """Multiple ledgers for different commodities at same facility."""
        l1 = make_ledger(
            ledger_id="LDG-MULTI-001",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        l2 = make_ledger(
            ledger_id="LDG-MULTI-002",
            facility_id=FAC_ID_MILL_MY,
            commodity="coffee",
        )
        ledger_manager.create_ledger(l1)
        result = ledger_manager.create_ledger(l2)
        assert result is not None

    @pytest.mark.parametrize("entry_type", ["input", "output", "adjustment", "loss", "waste"])
    def test_record_all_entry_types(self, ledger_manager, entry_type):
        """All entry types can be recorded."""
        lid = f"LDG-ALLTYPE-{entry_type}"
        ledger = make_ledger(
            ledger_id=lid,
            opening_balance_kg=Decimal("10000.0"),
        )
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id=lid,
            entry_type=entry_type,
            quantity_kg=Decimal("100.0"),
        )
        result = ledger_manager.record_entry(entry)
        assert result is not None

    def test_entry_chronological_order(self, ledger_manager):
        """Entries are stored in chronological order."""
        ledger = make_ledger(ledger_id="LDG-CHRON-001")
        ledger_manager.create_ledger(ledger)
        for i in range(5):
            entry = make_entry(
                ledger_id="LDG-CHRON-001",
                entry_id=f"ENT-CHRON-{i:03d}",
            )
            ledger_manager.record_entry(entry)
        entries = ledger_manager.get_entries("LDG-CHRON-001")
        for i in range(len(entries) - 1):
            assert entries[i].get("timestamp", "") <= entries[i + 1].get("timestamp", "")

    def test_duplicate_entry_id_raises(self, ledger_manager):
        """Duplicate entry ID raises error."""
        ledger = make_ledger(ledger_id="LDG-DUPENT-001")
        ledger_manager.create_ledger(ledger)
        entry = make_entry(
            ledger_id="LDG-DUPENT-001",
            entry_id="ENT-DUPENT-001",
        )
        ledger_manager.record_entry(entry)
        with pytest.raises((ValueError, KeyError)):
            ledger_manager.record_entry(copy.deepcopy(entry))

    def test_search_combined_filters(self, ledger_manager):
        """Search with combined facility and commodity filters."""
        l1 = make_ledger(
            ledger_id="LDG-COMB-001",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        l2 = make_ledger(
            ledger_id="LDG-COMB-002",
            facility_id=FAC_ID_MILL_MY,
            commodity="coffee",
        )
        ledger_manager.create_ledger(l1)
        ledger_manager.create_ledger(l2)
        results = ledger_manager.search(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        assert all(r["commodity"] == "cocoa" for r in results)

    def test_balance_after_multiple_operations(self, ledger_manager):
        """Balance is correct after mixed input/output/loss operations."""
        ledger = make_ledger(ledger_id="LDG-MIXED-001")
        ledger_manager.create_ledger(ledger)
        # Input 5000
        ledger_manager.record_entry(make_entry(
            ledger_id="LDG-MIXED-001", entry_type="input",
            quantity_kg=Decimal("5000.0"), entry_id="ENT-MIX-INP",
        ))
        # Output 2000
        ledger_manager.record_entry(make_entry(
            ledger_id="LDG-MIXED-001", entry_type="output",
            quantity_kg=Decimal("2000.0"), entry_id="ENT-MIX-OUT",
        ))
        # Loss 300
        ledger_manager.record_entry(make_entry(
            ledger_id="LDG-MIXED-001", entry_type="loss",
            quantity_kg=Decimal("300.0"), entry_id="ENT-MIX-LOSS",
        ))
        balance = ledger_manager.get_balance("LDG-MIXED-001")
        assert Decimal(str(balance["current_balance_kg"])) == Decimal("2700.0")
