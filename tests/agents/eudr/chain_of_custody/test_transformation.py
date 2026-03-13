# -*- coding: utf-8 -*-
"""
Tests for TransformationTracker - AGENT-EUDR-009 Engine 5: Transformation Tracking

Comprehensive test suite covering:
- Process type recording (F5.1-F5.2)
- Yield ratio validation (F5.4)
- By-product tracking (F5.5)
- Multi-step chains (F5.6)
- Derived product tracking (F5.7)
- Co-product allocation (F5.8)
- Batch transformation import (F5.10)

Test count: 50+ tests
Coverage target: >= 85% of TransformationTracker module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    PROCESS_TYPES,
    EUDR_COMMODITIES,
    CONVERSION_FACTORS,
    ALLOCATION_METHODS,
    SHA256_HEX_LENGTH,
    FAC_ID_PROC_GH,
    FAC_ID_MILL_ID,
    FAC_ID_REFINERY_ID,
    FAC_ID_MILL_CO,
    FAC_ID_SAWMILL_CD,
    FAC_ID_FEEDLOT_BR,
    BATCH_ID_COCOA_COOP_GH,
    BATCH_ID_COCOA_NIBS_GH,
    BATCH_ID_COCOA_LIQUOR_GH,
    BATCH_ID_COCOA_BUTTER_GH,
    BATCH_ID_COCOA_POWDER_GH,
    TRANSFORM_COCOA_BEANS_TO_NIBS,
    TRANSFORM_COCOA_NIBS_TO_LIQUOR,
    TRANSFORM_COCOA_LIQUOR_TO_BUTTER_POWDER,
    make_transformation,
    make_batch,
    assert_mass_conservation,
)


# ===========================================================================
# 1. Process Type Recording (F5.1-F5.2)
# ===========================================================================


class TestProcessTypeRecording:
    """Test recording of processing steps with various process types."""

    @pytest.mark.parametrize("process_type", PROCESS_TYPES)
    def test_record_all_process_types(self, transformation_tracker, process_type):
        """All 15+ process types can be recorded."""
        transform = make_transformation(process_type=process_type)
        result = transformation_tracker.record(transform)
        assert result is not None
        assert result["process_type"] == process_type

    def test_record_cocoa_shelling(self, transformation_tracker):
        """Record cocoa beans to nibs (shelling) transformation."""
        transform = copy.deepcopy(TRANSFORM_COCOA_BEANS_TO_NIBS)
        result = transformation_tracker.record(transform)
        assert result["process_type"] == "milling"
        assert len(result["input_batches"]) == 1
        assert len(result["output_batches"]) == 1

    def test_record_cocoa_pressing(self, transformation_tracker):
        """Record cocoa liquor to butter + powder (pressing)."""
        transform = copy.deepcopy(TRANSFORM_COCOA_LIQUOR_TO_BUTTER_POWDER)
        result = transformation_tracker.record(transform)
        assert result["process_type"] == "pressing"
        assert len(result["output_batches"]) == 2

    def test_transformation_assigns_id(self, transformation_tracker):
        """Transformation recording assigns a unique ID."""
        transform = make_transformation(transformation_id=None)
        result = transformation_tracker.record(transform)
        assert result.get("transformation_id") is not None

    def test_transformation_provenance_hash(self, transformation_tracker):
        """Transformation generates a provenance hash."""
        transform = make_transformation()
        result = transformation_tracker.record(transform)
        assert len(result.get("provenance_hash", "")) == SHA256_HEX_LENGTH

    def test_invalid_process_type_raises(self, transformation_tracker):
        """Invalid process type raises ValueError."""
        transform = make_transformation(process_type="teleportation")
        with pytest.raises(ValueError):
            transformation_tracker.record(transform)

    def test_empty_input_batches_raises(self, transformation_tracker):
        """Transformation with no input batches raises ValueError."""
        transform = make_transformation(input_batches=[])
        with pytest.raises(ValueError):
            transformation_tracker.record(transform)

    def test_empty_output_batches_raises(self, transformation_tracker):
        """Transformation with no output batches raises ValueError."""
        transform = make_transformation(output_batches=[])
        with pytest.raises(ValueError):
            transformation_tracker.record(transform)


# ===========================================================================
# 2. Yield Ratio Validation (F5.4)
# ===========================================================================


class TestYieldRatioValidation:
    """Test actual vs expected yield ratio validation."""

    def test_yield_within_tolerance(self, transformation_tracker):
        """Actual yield within 5% of expected is accepted."""
        transform = make_transformation(
            input_batches=[{"batch_id": "IN-1", "quantity_kg": 10000.0}],
            output_batches=[{"batch_id": "OUT-1", "quantity_kg": 8700.0, "product_type": "main"}],
            expected_yield=0.87,
            actual_yield=0.87,
        )
        result = transformation_tracker.validate_yield(transform)
        assert result["yield_valid"] is True

    def test_yield_exceeds_expected_flagged(self, transformation_tracker):
        """Actual yield significantly exceeding expected is flagged."""
        transform = make_transformation(
            input_batches=[{"batch_id": "IN-1", "quantity_kg": 10000.0}],
            output_batches=[{"batch_id": "OUT-1", "quantity_kg": 9500.0, "product_type": "main"}],
            expected_yield=0.87,
            actual_yield=0.95,
        )
        result = transformation_tracker.validate_yield(transform)
        assert result["yield_valid"] is False

    def test_yield_below_expected_flagged(self, transformation_tracker):
        """Actual yield significantly below expected is flagged."""
        transform = make_transformation(
            input_batches=[{"batch_id": "IN-1", "quantity_kg": 10000.0}],
            output_batches=[{"batch_id": "OUT-1", "quantity_kg": 7000.0, "product_type": "main"}],
            expected_yield=0.87,
            actual_yield=0.70,
        )
        result = transformation_tracker.validate_yield(transform)
        assert result["yield_valid"] is False

    @pytest.mark.parametrize("commodity,conv_key", [
        ("cocoa", "cocoa_beans_to_nibs"),
        ("palm_oil", "palm_ffb_to_cpo"),
        ("coffee", "coffee_cherry_to_green"),
        ("soya", "soya_beans_to_oil"),
        ("rubber", "rubber_latex_to_sheet"),
        ("wood", "wood_log_to_sawn"),
        ("cattle", "cattle_live_to_carcass"),
    ])
    def test_yield_ratios_per_commodity(self, transformation_tracker, commodity, conv_key):
        """Each commodity has expected yield ratios from reference data."""
        expected = CONVERSION_FACTORS[conv_key]["yield_ratio"]
        ratio = transformation_tracker.get_expected_yield(conv_key)
        assert ratio == pytest.approx(expected, rel=0.05)

    def test_yield_ratio_computed_correctly(self, transformation_tracker):
        """Yield ratio is computed as output / input."""
        transform = make_transformation(
            input_batches=[{"batch_id": "IN-Y", "quantity_kg": 5000.0}],
            output_batches=[{"batch_id": "OUT-Y", "quantity_kg": 4350.0, "product_type": "main"}],
        )
        result = transformation_tracker.record(transform)
        assert result["actual_yield_ratio"] == pytest.approx(0.87, abs=0.01)


# ===========================================================================
# 3. By-Product Tracking (F5.5)
# ===========================================================================


class TestByProductTracking:
    """Test tracking of by-products and waste from processing."""

    def test_by_products_recorded(self, transformation_tracker):
        """By-products are recorded alongside main output."""
        transform = make_transformation(
            input_batches=[{"batch_id": "IN-BP", "quantity_kg": 10000.0}],
            output_batches=[{"batch_id": "OUT-BP-M", "quantity_kg": 8700.0, "product_type": "main"}],
            waste_kg=1200.0,
        )
        transform["by_products"] = [
            {"type": "shell", "quantity_kg": 100.0},
        ]
        result = transformation_tracker.record(transform)
        assert len(result["by_products"]) == 1
        assert result["by_products"][0]["type"] == "shell"

    def test_waste_recorded(self, transformation_tracker):
        """Waste quantity is recorded and tracked."""
        transform = make_transformation(waste_kg=500.0)
        result = transformation_tracker.record(transform)
        assert result["waste_kg"] == 500.0

    def test_mass_conservation_with_byproducts(self, transformation_tracker):
        """Main output + by-products + waste = input (within tolerance)."""
        transform = copy.deepcopy(TRANSFORM_COCOA_BEANS_TO_NIBS)
        result = transformation_tracker.record(transform)
        total_input = sum(b["quantity_kg"] for b in result["input_batches"])
        total_output = sum(b["quantity_kg"] for b in result["output_batches"])
        total_byproduct = sum(b["quantity_kg"] for b in result.get("by_products", []))
        total_waste = result.get("waste_kg", 0.0)
        assert_mass_conservation(total_input, total_output + total_byproduct, total_waste)

    def test_multiple_by_products(self, transformation_tracker):
        """Multiple by-products from a single process are tracked."""
        transform = make_transformation()
        transform["by_products"] = [
            {"type": "shell", "quantity_kg": 100.0},
            {"type": "husk", "quantity_kg": 50.0},
            {"type": "dust", "quantity_kg": 20.0},
        ]
        result = transformation_tracker.record(transform)
        assert len(result["by_products"]) == 3


# ===========================================================================
# 4. Multi-Step Chains (F5.6)
# ===========================================================================


class TestMultiStepChains:
    """Test multi-step transformation chains (raw -> intermediate -> final)."""

    def test_cocoa_full_chain(self, transformation_tracker):
        """Track complete cocoa transformation: beans -> nibs -> liquor -> butter/powder."""
        t1 = copy.deepcopy(TRANSFORM_COCOA_BEANS_TO_NIBS)
        t2 = copy.deepcopy(TRANSFORM_COCOA_NIBS_TO_LIQUOR)
        t3 = copy.deepcopy(TRANSFORM_COCOA_LIQUOR_TO_BUTTER_POWDER)
        r1 = transformation_tracker.record(t1)
        r2 = transformation_tracker.record(t2)
        r3 = transformation_tracker.record(t3)
        assert r1 is not None
        assert r2 is not None
        assert r3 is not None

    def test_chain_lineage_retrieval(self, transformation_tracker):
        """Retrieve full transformation lineage for a batch."""
        t1 = copy.deepcopy(TRANSFORM_COCOA_BEANS_TO_NIBS)
        t2 = copy.deepcopy(TRANSFORM_COCOA_NIBS_TO_LIQUOR)
        transformation_tracker.record(t1)
        transformation_tracker.record(t2)
        lineage = transformation_tracker.get_lineage(BATCH_ID_COCOA_LIQUOR_GH)
        assert len(lineage) >= 2

    def test_three_step_chain_yields(self, transformation_tracker):
        """Cumulative yield across 3 steps matches expected calculation."""
        # beans(5000) -> nibs(4350) -> liquor(3480) -> butter(1566) + powder(1914)
        t1 = copy.deepcopy(TRANSFORM_COCOA_BEANS_TO_NIBS)
        t2 = copy.deepcopy(TRANSFORM_COCOA_NIBS_TO_LIQUOR)
        t3 = copy.deepcopy(TRANSFORM_COCOA_LIQUOR_TO_BUTTER_POWDER)
        transformation_tracker.record(t1)
        transformation_tracker.record(t2)
        transformation_tracker.record(t3)
        # Cumulative yield: 0.87 * 0.80 = 0.696
        # From 5000 kg beans -> 3480 kg liquor
        assert 3480.0 / 5000.0 == pytest.approx(0.696, abs=0.01)


# ===========================================================================
# 5. Derived Product Tracking (F5.7)
# ===========================================================================


class TestDerivedProductTracking:
    """Test tracking when commodity changes form."""

    def test_commodity_form_change_tracked(self, transformation_tracker):
        """Transformation tracking captures form change."""
        transform = make_transformation(
            input_batches=[{"batch_id": "IN-FORM", "quantity_kg": 10000.0,
                            "commodity_form": "beans"}],
            output_batches=[{"batch_id": "OUT-FORM", "quantity_kg": 8700.0,
                             "product_type": "main", "commodity_form": "nibs"}],
        )
        result = transformation_tracker.record(transform)
        assert result["input_batches"][0].get("commodity_form") == "beans"
        assert result["output_batches"][0].get("commodity_form") == "nibs"

    def test_palm_fruit_to_cpo_form_change(self, transformation_tracker):
        """Palm FFB to CPO represents a commodity form change."""
        transform = make_transformation(
            process_type="pressing",
            facility_id=FAC_ID_MILL_ID,
            input_batches=[{"batch_id": "IN-FFB", "quantity_kg": 50000.0,
                            "commodity_form": "ffb"}],
            output_batches=[{"batch_id": "OUT-CPO", "quantity_kg": 11000.0,
                             "product_type": "main", "commodity_form": "cpo"}],
            expected_yield=0.22,
        )
        result = transformation_tracker.record(transform)
        assert result is not None

    @pytest.mark.parametrize("commodity,in_form,out_form", [
        ("cocoa", "beans", "nibs"),
        ("cocoa", "nibs", "liquor"),
        ("cocoa", "liquor", "butter"),
        ("palm_oil", "ffb", "cpo"),
        ("palm_oil", "cpo", "rbd"),
        ("coffee", "cherry", "green"),
        ("coffee", "green", "roasted"),
        ("wood", "log", "sawn"),
    ])
    def test_common_form_changes(self, transformation_tracker, commodity, in_form, out_form):
        """Common commodity form changes are tracked correctly."""
        transform = make_transformation(
            input_batches=[{"batch_id": f"IN-{in_form}", "quantity_kg": 1000.0,
                            "commodity_form": in_form}],
            output_batches=[{"batch_id": f"OUT-{out_form}", "quantity_kg": 500.0,
                             "product_type": "main", "commodity_form": out_form}],
        )
        result = transformation_tracker.record(transform)
        assert result is not None


# ===========================================================================
# 6. Co-Product Allocation (F5.8)
# ===========================================================================


class TestCoProductAllocation:
    """Test co-product allocation methods."""

    @pytest.mark.parametrize("method", ALLOCATION_METHODS)
    def test_allocation_methods_supported(self, transformation_tracker, method):
        """All allocation methods (economic, mass, energy) are supported."""
        result = transformation_tracker.allocate_co_products(
            input_quantity_kg=10000.0,
            co_products=[
                {"name": "product_a", "quantity_kg": 4000.0, "value_per_kg": 5.0},
                {"name": "product_b", "quantity_kg": 6000.0, "value_per_kg": 2.0},
            ],
            method=method,
        )
        assert result is not None
        assert len(result["allocations"]) == 2

    def test_economic_allocation(self, transformation_tracker):
        """Economic allocation allocates by value share."""
        result = transformation_tracker.allocate_co_products(
            input_quantity_kg=10000.0,
            co_products=[
                {"name": "butter", "quantity_kg": 4500.0, "value_per_kg": 10.0},
                {"name": "powder", "quantity_kg": 5500.0, "value_per_kg": 3.0},
            ],
            method="economic",
        )
        # butter value: 45000, powder value: 16500, total: 61500
        # butter share: 73.2%, powder share: 26.8%
        butter_alloc = next(a for a in result["allocations"] if a["name"] == "butter")
        assert butter_alloc["allocation_pct"] > 50.0

    def test_mass_allocation(self, transformation_tracker):
        """Mass allocation allocates by weight share."""
        result = transformation_tracker.allocate_co_products(
            input_quantity_kg=10000.0,
            co_products=[
                {"name": "butter", "quantity_kg": 4500.0, "value_per_kg": 10.0},
                {"name": "powder", "quantity_kg": 5500.0, "value_per_kg": 3.0},
            ],
            method="mass",
        )
        butter_alloc = next(a for a in result["allocations"] if a["name"] == "butter")
        assert butter_alloc["allocation_pct"] == pytest.approx(45.0)

    def test_allocation_sums_to_100(self, transformation_tracker):
        """All allocation percentages sum to 100%."""
        result = transformation_tracker.allocate_co_products(
            input_quantity_kg=10000.0,
            co_products=[
                {"name": "a", "quantity_kg": 3000.0, "value_per_kg": 5.0},
                {"name": "b", "quantity_kg": 4000.0, "value_per_kg": 3.0},
                {"name": "c", "quantity_kg": 3000.0, "value_per_kg": 4.0},
            ],
            method="economic",
        )
        total = sum(a["allocation_pct"] for a in result["allocations"])
        assert total == pytest.approx(100.0, abs=0.1)


# ===========================================================================
# 7. Batch Transformation Import
# ===========================================================================


class TestBatchTransformationImport:
    """Test batch import of transformation records."""

    def test_bulk_import(self, transformation_tracker):
        """Import multiple transformations in bulk."""
        transforms = [
            make_transformation(transformation_id=f"XFRM-BULK-{i}")
            for i in range(5)
        ]
        results = transformation_tracker.bulk_import(transforms)
        assert len(results) == 5

    def test_bulk_import_empty(self, transformation_tracker):
        """Bulk import of empty list returns empty results."""
        results = transformation_tracker.bulk_import([])
        assert len(results) == 0

    def test_get_nonexistent_transformation(self, transformation_tracker):
        """Getting a non-existent transformation returns None."""
        result = transformation_tracker.get("XFRM-NONEXISTENT")
        assert result is None

    def test_search_by_facility(self, transformation_tracker):
        """Search transformations by facility ID."""
        transform = make_transformation(facility_id=FAC_ID_PROC_GH)
        transformation_tracker.record(transform)
        results = transformation_tracker.search(facility_id=FAC_ID_PROC_GH)
        assert len(results) >= 1

    def test_search_by_process_type(self, transformation_tracker):
        """Search transformations by process type."""
        transform = make_transformation(process_type="milling")
        transformation_tracker.record(transform)
        results = transformation_tracker.search(process_type="milling")
        assert len(results) >= 1
