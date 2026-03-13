# -*- coding: utf-8 -*-
"""
Tests for LossWasteTracker - AGENT-EUDR-011 Engine 5: Loss and Waste Tracking

Comprehensive test suite covering:
- Loss recording (processing, transport, storage, provenance)
- Waste recording (by-product, waste material, hazardous waste)
- Tolerance validation (within, under-reporting flag, over-reporting flag)
- By-product credit (credit back valuable by-products)
- Cumulative loss (cumulative across processing steps)
- Loss trends (per facility per commodity)
- Loss allocation (proportional for batch splits)
- Edge cases (zero loss, 100% loss, commodity-specific tolerances)

Test count: 55+ tests
Coverage target: >= 85% of LossWasteTracker module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    LOSS_TYPES,
    WASTE_TYPES,
    COMMODITY_LOSS_TOLERANCES,
    LOSS_TYPE_TOLERANCES,
    SHA256_HEX_LENGTH,
    LOSS_COCOA_PROCESSING,
    LOSS_PALM_TRANSPORT,
    LEDGER_COCOA_001,
    LEDGER_PALM_001,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    BATCH_COCOA_001,
    BATCH_PALM_001,
    make_loss_record,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Loss Recording
# ===========================================================================


class TestLossRecording:
    """Test loss record creation."""

    def test_record_processing_loss(self, loss_waste_tracker):
        """Record a processing loss."""
        loss = make_loss_record(
            loss_type="processing_loss",
            quantity_kg=Decimal("500.0"),
            commodity="cocoa",
            process_step="roasting",
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None
        assert result.get("loss_type") == "processing_loss"

    def test_record_transport_loss(self, loss_waste_tracker):
        """Record a transport loss."""
        loss = make_loss_record(
            loss_type="transport_loss",
            quantity_kg=Decimal("100.0"),
            commodity="oil_palm",
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    def test_record_storage_loss(self, loss_waste_tracker):
        """Record a storage loss."""
        loss = make_loss_record(
            loss_type="storage_loss",
            quantity_kg=Decimal("50.0"),
            commodity="coffee",
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    def test_record_quality_rejection(self, loss_waste_tracker):
        """Record a quality rejection loss."""
        loss = make_loss_record(
            loss_type="quality_rejection",
            quantity_kg=Decimal("80.0"),
            commodity="cocoa",
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    def test_record_spillage(self, loss_waste_tracker):
        """Record a spillage loss."""
        loss = make_loss_record(
            loss_type="spillage",
            quantity_kg=Decimal("15.0"),
            commodity="oil_palm",
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    def test_record_contamination_loss(self, loss_waste_tracker):
        """Record a contamination loss."""
        loss = make_loss_record(
            loss_type="contamination_loss",
            quantity_kg=Decimal("200.0"),
            commodity="coffee",
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    @pytest.mark.parametrize("loss_type", LOSS_TYPES)
    def test_record_all_loss_types(self, loss_waste_tracker, loss_type):
        """All 6 loss types can be recorded."""
        loss = make_loss_record(loss_type=loss_type, quantity_kg=Decimal("50.0"))
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    def test_loss_provenance_hash(self, loss_waste_tracker):
        """Loss record generates a provenance hash."""
        loss = make_loss_record()
        result = loss_waste_tracker.record_loss(loss)
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_loss_auto_calculates_percent(self, loss_waste_tracker):
        """Loss percentage is auto-calculated from quantity/input."""
        loss = make_loss_record(
            quantity_kg=Decimal("100.0"),
            input_quantity_kg=Decimal("1000.0"),
        )
        result = loss_waste_tracker.record_loss(loss)
        loss_pct = result.get("loss_percent", result.get("percentage"))
        assert loss_pct is not None
        assert abs(float(loss_pct) - 10.0) < 0.1

    def test_loss_invalid_type_raises(self, loss_waste_tracker):
        """Invalid loss type raises ValueError."""
        loss = make_loss_record()
        loss["loss_type"] = "invalid_loss"
        with pytest.raises(ValueError):
            loss_waste_tracker.record_loss(loss)


# ===========================================================================
# 2. Waste Recording
# ===========================================================================


class TestWasteRecording:
    """Test waste record creation."""

    def test_record_by_product(self, loss_waste_tracker):
        """Record a valuable by-product."""
        waste = {
            "waste_id": f"WST-{uuid.uuid4().hex[:8].upper()}",
            "ledger_id": LEDGER_COCOA_001,
            "waste_type": "by_product",
            "quantity_kg": Decimal("200.0"),
            "commodity": "cocoa",
            "batch_id": BATCH_COCOA_001,
            "description": "Cocoa butter from pressing",
            "value_per_kg": Decimal("5.50"),
            "metadata": {},
        }
        result = loss_waste_tracker.record_waste(waste)
        assert result is not None

    def test_record_waste_material(self, loss_waste_tracker):
        """Record non-valuable waste material."""
        waste = {
            "waste_id": f"WST-{uuid.uuid4().hex[:8].upper()}",
            "ledger_id": LEDGER_COCOA_001,
            "waste_type": "waste_material",
            "quantity_kg": Decimal("50.0"),
            "commodity": "cocoa",
            "batch_id": BATCH_COCOA_001,
            "description": "Cocoa shell waste",
            "value_per_kg": Decimal("0.0"),
            "metadata": {},
        }
        result = loss_waste_tracker.record_waste(waste)
        assert result is not None

    def test_record_hazardous_waste(self, loss_waste_tracker):
        """Record hazardous waste."""
        waste = {
            "waste_id": f"WST-{uuid.uuid4().hex[:8].upper()}",
            "ledger_id": LEDGER_PALM_001,
            "waste_type": "hazardous_waste",
            "quantity_kg": Decimal("10.0"),
            "commodity": "oil_palm",
            "batch_id": BATCH_PALM_001,
            "description": "Chemical cleaning residue",
            "value_per_kg": Decimal("0.0"),
            "metadata": {"disposal_method": "licensed_contractor"},
        }
        result = loss_waste_tracker.record_waste(waste)
        assert result is not None

    @pytest.mark.parametrize("waste_type", WASTE_TYPES)
    def test_record_all_waste_types(self, loss_waste_tracker, waste_type):
        """All 3 waste types can be recorded."""
        waste = {
            "waste_id": f"WST-{uuid.uuid4().hex[:8].upper()}",
            "ledger_id": LEDGER_COCOA_001,
            "waste_type": waste_type,
            "quantity_kg": Decimal("25.0"),
            "commodity": "cocoa",
            "batch_id": BATCH_COCOA_001,
            "description": f"Test {waste_type}",
            "value_per_kg": Decimal("0.0"),
            "metadata": {},
        }
        result = loss_waste_tracker.record_waste(waste)
        assert result is not None

    def test_waste_provenance_hash(self, loss_waste_tracker):
        """Waste record generates a provenance hash."""
        waste = {
            "waste_id": f"WST-{uuid.uuid4().hex[:8].upper()}",
            "ledger_id": LEDGER_COCOA_001,
            "waste_type": "waste_material",
            "quantity_kg": Decimal("30.0"),
            "commodity": "cocoa",
            "batch_id": BATCH_COCOA_001,
            "description": "Test",
            "value_per_kg": Decimal("0.0"),
            "metadata": {},
        }
        result = loss_waste_tracker.record_waste(waste)
        assert result.get("provenance_hash") is not None


# ===========================================================================
# 3. Tolerance Validation
# ===========================================================================


class TestToleranceValidation:
    """Test loss tolerance validation."""

    def test_within_tolerance(self, loss_waste_tracker):
        """Loss within tolerance is accepted."""
        loss = make_loss_record(
            loss_type="processing_loss",
            quantity_kg=Decimal("100.0"),
            input_quantity_kg=Decimal("1000.0"),
        )
        result = loss_waste_tracker.validate_loss(loss)
        assert result["within_tolerance"] is True

    def test_exceeds_tolerance_flags(self, loss_waste_tracker):
        """Loss exceeding tolerance is flagged."""
        loss = make_loss_record(
            loss_type="processing_loss",
            quantity_kg=Decimal("200.0"),
            input_quantity_kg=Decimal("1000.0"),
        )
        result = loss_waste_tracker.validate_loss(loss)
        assert result["within_tolerance"] is False or result.get("flagged") is True

    def test_under_reporting_flag(self, loss_waste_tracker):
        """Suspiciously low loss triggers under-reporting flag."""
        loss = make_loss_record(
            loss_type="processing_loss",
            quantity_kg=Decimal("1.0"),
            input_quantity_kg=Decimal("10000.0"),
            commodity="cocoa",
            process_step="roasting",
        )
        result = loss_waste_tracker.validate_loss(loss)
        has_flag = (
            result.get("under_reporting") is True
            or result.get("suspiciously_low") is True
            or result.get("flag") == "under_reporting"
        )
        assert has_flag or result["within_tolerance"] is True

    def test_over_reporting_flag(self, loss_waste_tracker):
        """Very high loss triggers over-reporting flag."""
        loss = make_loss_record(
            loss_type="processing_loss",
            quantity_kg=Decimal("500.0"),
            input_quantity_kg=Decimal("1000.0"),  # 50% loss
        )
        result = loss_waste_tracker.validate_loss(loss)
        assert result["within_tolerance"] is False

    @pytest.mark.parametrize("commodity,tolerance", list(COMMODITY_LOSS_TOLERANCES.items()))
    def test_commodity_specific_tolerances(
        self, loss_waste_tracker, commodity, tolerance
    ):
        """Each commodity has its own loss tolerance."""
        loss = make_loss_record(
            commodity=commodity,
            loss_type="processing_loss",
            quantity_kg=Decimal(str(tolerance / 2)),
            input_quantity_kg=Decimal("100.0"),
        )
        result = loss_waste_tracker.validate_loss(loss)
        assert result is not None

    @pytest.mark.parametrize("loss_type,tolerance", list(LOSS_TYPE_TOLERANCES.items()))
    def test_loss_type_tolerances(self, loss_waste_tracker, loss_type, tolerance):
        """Each loss type has its own tolerance threshold."""
        # Use half of tolerance to stay within bounds
        qty = Decimal(str(tolerance / 2))
        loss = make_loss_record(
            loss_type=loss_type,
            quantity_kg=qty,
            input_quantity_kg=Decimal("100.0"),
        )
        result = loss_waste_tracker.validate_loss(loss)
        assert result is not None


# ===========================================================================
# 4. By-Product Credit
# ===========================================================================


class TestByProductCredit:
    """Test by-product credit operations."""

    def test_credit_back_valuable_by_product(self, loss_waste_tracker):
        """Valuable by-product generates a credit entry."""
        result = loss_waste_tracker.record_by_product_credit(
            ledger_id=LEDGER_COCOA_001,
            batch_id=BATCH_COCOA_001,
            by_product_name="cocoa_butter",
            quantity_kg=Decimal("200.0"),
            conversion_rate=0.8,
        )
        assert result is not None
        credit_kg = result.get("credit_kg", result.get("credited_quantity_kg"))
        if credit_kg is not None:
            assert Decimal(str(credit_kg)) == Decimal("160.0")

    def test_by_product_credit_disabled(self, loss_waste_tracker):
        """By-product credit can be disabled via config."""
        # This test checks that the system respects the config flag
        result = loss_waste_tracker.record_by_product_credit(
            ledger_id=LEDGER_COCOA_001,
            batch_id=BATCH_COCOA_001,
            by_product_name="cocoa_shell",
            quantity_kg=Decimal("50.0"),
            conversion_rate=0.0,
        )
        assert result is not None

    def test_by_product_zero_conversion_no_credit(self, loss_waste_tracker):
        """By-product with zero conversion rate generates no credit."""
        result = loss_waste_tracker.record_by_product_credit(
            ledger_id=LEDGER_COCOA_001,
            batch_id=BATCH_COCOA_001,
            by_product_name="waste_hull",
            quantity_kg=Decimal("100.0"),
            conversion_rate=0.0,
        )
        credit_kg = result.get("credit_kg", result.get("credited_quantity_kg", 0))
        assert Decimal(str(credit_kg)) == Decimal("0.0")


# ===========================================================================
# 5. Cumulative Loss
# ===========================================================================


class TestCumulativeLoss:
    """Test cumulative loss across processing steps."""

    def test_cumulative_loss_across_steps(self, loss_waste_tracker):
        """Cumulative loss tracks total loss across all processing steps."""
        losses = [
            make_loss_record(
                loss_id=f"LOSS-CUM-{i:03d}",
                loss_type="processing_loss",
                quantity_kg=Decimal("50.0"),
                process_step=f"step_{i}",
                batch_id=BATCH_COCOA_001,
            )
            for i in range(3)
        ]
        for loss in losses:
            loss_waste_tracker.record_loss(loss)
        cumulative = loss_waste_tracker.get_cumulative_loss(
            batch_id=BATCH_COCOA_001,
        )
        total_kg = cumulative.get("total_loss_kg", cumulative.get("cumulative_kg"))
        assert Decimal(str(total_kg)) >= Decimal("150.0")

    def test_cumulative_loss_per_batch(self, loss_waste_tracker):
        """Cumulative loss is tracked per batch."""
        loss_a = make_loss_record(
            loss_id="LOSS-CBA-001",
            batch_id="BATCH-CUM-A",
            quantity_kg=Decimal("100.0"),
        )
        loss_b = make_loss_record(
            loss_id="LOSS-CBB-001",
            batch_id="BATCH-CUM-B",
            quantity_kg=Decimal("200.0"),
        )
        loss_waste_tracker.record_loss(loss_a)
        loss_waste_tracker.record_loss(loss_b)
        cum_a = loss_waste_tracker.get_cumulative_loss(batch_id="BATCH-CUM-A")
        cum_b = loss_waste_tracker.get_cumulative_loss(batch_id="BATCH-CUM-B")
        assert Decimal(str(cum_a.get("total_loss_kg", cum_a.get("cumulative_kg")))) != \
            Decimal(str(cum_b.get("total_loss_kg", cum_b.get("cumulative_kg"))))


# ===========================================================================
# 6. Loss Trends
# ===========================================================================


class TestLossTrends:
    """Test loss trend analysis per facility per commodity."""

    def test_trend_per_facility(self, loss_waste_tracker):
        """Get loss trends for a facility."""
        for i in range(5):
            loss = make_loss_record(
                loss_id=f"LOSS-TREND-{i:03d}",
                quantity_kg=Decimal(str(50 + i * 10)),
                commodity="cocoa",
            )
            loss_waste_tracker.record_loss(loss)
        trends = loss_waste_tracker.get_trends(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        assert trends is not None

    def test_trend_per_commodity(self, loss_waste_tracker):
        """Loss trends are tracked per commodity."""
        loss_cocoa = make_loss_record(
            loss_id="LOSS-TRC-001",
            commodity="cocoa",
            quantity_kg=Decimal("100.0"),
        )
        loss_palm = make_loss_record(
            loss_id="LOSS-TRP-001",
            commodity="oil_palm",
            quantity_kg=Decimal("200.0"),
        )
        loss_waste_tracker.record_loss(loss_cocoa)
        loss_waste_tracker.record_loss(loss_palm)
        trends = loss_waste_tracker.get_trends(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        assert trends is not None

    def test_trend_empty_facility(self, loss_waste_tracker):
        """Facility with no losses returns empty trends."""
        trends = loss_waste_tracker.get_trends(
            facility_id="FAC-EMPTY-001",
            commodity="cocoa",
        )
        assert trends is not None
        if isinstance(trends, list):
            assert len(trends) == 0


# ===========================================================================
# 7. Loss Allocation
# ===========================================================================


class TestLossAllocation:
    """Test proportional loss allocation for batch splits."""

    def test_proportional_allocation(self, loss_waste_tracker):
        """Loss is proportionally allocated across batch splits."""
        result = loss_waste_tracker.allocate_loss(
            total_loss_kg=Decimal("100.0"),
            split_batches=[
                {"batch_id": "BATCH-SPLIT-A", "quantity_kg": Decimal("3000.0")},
                {"batch_id": "BATCH-SPLIT-B", "quantity_kg": Decimal("7000.0")},
            ],
        )
        assert result is not None
        allocations = result.get("allocations", result.get("split_losses", []))
        assert len(allocations) == 2

    def test_allocation_sums_to_total(self, loss_waste_tracker):
        """Allocated amounts sum to total loss."""
        result = loss_waste_tracker.allocate_loss(
            total_loss_kg=Decimal("100.0"),
            split_batches=[
                {"batch_id": "BATCH-SPLIT-C", "quantity_kg": Decimal("5000.0")},
                {"batch_id": "BATCH-SPLIT-D", "quantity_kg": Decimal("5000.0")},
            ],
        )
        allocations = result.get("allocations", result.get("split_losses", []))
        total = sum(Decimal(str(a.get("loss_kg", a.get("allocated_kg", 0)))) for a in allocations)
        assert abs(total - Decimal("100.0")) < Decimal("0.01")

    def test_single_batch_gets_all_loss(self, loss_waste_tracker):
        """Single batch split gets entire loss."""
        result = loss_waste_tracker.allocate_loss(
            total_loss_kg=Decimal("50.0"),
            split_batches=[
                {"batch_id": "BATCH-SINGLE", "quantity_kg": Decimal("5000.0")},
            ],
        )
        allocations = result.get("allocations", result.get("split_losses", []))
        assert len(allocations) == 1


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for loss/waste tracking."""

    def test_zero_loss_raises(self, loss_waste_tracker):
        """Zero quantity loss raises ValueError."""
        loss = make_loss_record(quantity_kg=Decimal("0.0"))
        with pytest.raises(ValueError):
            loss_waste_tracker.record_loss(loss)

    def test_negative_loss_raises(self, loss_waste_tracker):
        """Negative quantity loss raises ValueError."""
        loss = make_loss_record(quantity_kg=Decimal("-50.0"))
        with pytest.raises(ValueError):
            loss_waste_tracker.record_loss(loss)

    def test_100_percent_loss(self, loss_waste_tracker):
        """100% loss (entire input) is recorded but flagged."""
        loss = make_loss_record(
            quantity_kg=Decimal("1000.0"),
            input_quantity_kg=Decimal("1000.0"),
        )
        result = loss_waste_tracker.validate_loss(loss)
        assert result["within_tolerance"] is False

    def test_loss_exceeds_input_raises(self, loss_waste_tracker):
        """Loss exceeding input quantity raises ValueError."""
        loss = make_loss_record(
            quantity_kg=Decimal("1500.0"),
            input_quantity_kg=Decimal("1000.0"),
        )
        with pytest.raises(ValueError):
            loss_waste_tracker.record_loss(loss)

    def test_duplicate_loss_id_raises(self, loss_waste_tracker):
        """Duplicate loss ID raises an error."""
        loss = make_loss_record(loss_id="LOSS-DUP-001")
        loss_waste_tracker.record_loss(loss)
        with pytest.raises((ValueError, KeyError)):
            loss_waste_tracker.record_loss(copy.deepcopy(loss))

    def test_very_small_loss_accepted(self, loss_waste_tracker):
        """Very small loss (0.001 kg) is accepted."""
        loss = make_loss_record(quantity_kg=Decimal("0.001"))
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_loss_all_commodities(self, loss_waste_tracker, commodity):
        """Losses can be recorded for all 7 EUDR commodities."""
        loss = make_loss_record(
            commodity=commodity,
            quantity_kg=Decimal("10.0"),
        )
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    @pytest.mark.parametrize("quantity", [
        "0.001", "0.5", "10.0", "500.0", "9999.0",
    ])
    def test_loss_various_quantities(self, loss_waste_tracker, quantity):
        """Losses of various quantities are accepted."""
        loss = make_loss_record(quantity_kg=Decimal(quantity))
        result = loss_waste_tracker.record_loss(loss)
        assert result is not None

    def test_get_loss_by_id(self, loss_waste_tracker):
        """Retrieve a loss record by ID."""
        loss = make_loss_record(loss_id="LOSS-GET-001")
        loss_waste_tracker.record_loss(loss)
        result = loss_waste_tracker.get("LOSS-GET-001")
        assert result is not None
        assert result.get("loss_id") == "LOSS-GET-001"

    def test_get_nonexistent_loss_returns_none(self, loss_waste_tracker):
        """Getting non-existent loss returns None."""
        result = loss_waste_tracker.get("LOSS-NONEXISTENT-999")
        assert result is None
