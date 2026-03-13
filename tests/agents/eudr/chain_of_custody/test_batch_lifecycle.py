# -*- coding: utf-8 -*-
"""
Tests for BatchLifecycleManager - AGENT-EUDR-009 Engine 2: Batch Lifecycle

Comprehensive test suite covering:
- Batch creation with origin plot linkage (F2.1-F2.2)
- Split operations (F2.3)
- Merge operations (F2.4)
- Blend operations (F2.5)
- Genealogy tree traversal (F2.6)
- Status transitions (F2.7)
- Origin allocation and quantity conservation (F2.8-F2.9)
- Search (F2.10)

Test count: 65+ tests
Coverage target: >= 85% of BatchLifecycleManager module

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
    EUDR_COMMODITIES,
    BATCH_STATUSES,
    VALID_BATCH_TRANSITIONS,
    INVALID_BATCH_TRANSITIONS,
    BATCH_COCOA_FARM_GH,
    BATCH_COCOA_COOP_GH,
    BATCH_COFFEE_FARM_CO,
    BATCH_PALM_MILL_ID,
    PLOT_ID_COCOA_GH_1,
    PLOT_ID_COCOA_GH_2,
    PLOT_ID_COCOA_GH_3,
    PLOT_ID_COFFEE_CO_1,
    SHA256_HEX_LENGTH,
    make_batch,
    assert_origin_preserved,
    assert_mass_conservation,
    build_linear_genealogy,
    build_split_genealogy,
    build_merge_genealogy,
    build_diamond_genealogy,
    build_blend_genealogy,
)


# ===========================================================================
# 1. Batch Creation (F2.1-F2.2)
# ===========================================================================


class TestBatchCreation:
    """Test batch creation with origin plot linkage."""

    def test_create_batch_with_single_origin(self, batch_lifecycle_manager):
        """Create a batch linked to a single origin plot."""
        batch = make_batch(
            commodity="cocoa", quantity_kg=500.0,
            origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}],
        )
        result = batch_lifecycle_manager.create(batch)
        assert result is not None
        assert result["commodity"] == "cocoa"
        assert len(result["origin_plots"]) == 1
        assert result["origin_plots"][0]["percentage"] == 100.0

    def test_create_batch_with_multiple_origins(self, batch_lifecycle_manager):
        """Create a batch with multiple origin plot allocations."""
        batch = make_batch(
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 40.0},
            ],
        )
        result = batch_lifecycle_manager.create(batch)
        assert len(result["origin_plots"]) == 2
        total = sum(p["percentage"] for p in result["origin_plots"])
        assert total == pytest.approx(100.0)

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_create_batch_all_commodities(self, batch_lifecycle_manager, commodity):
        """Batch creation works for each of the 7 EUDR commodities."""
        batch = make_batch(commodity=commodity)
        result = batch_lifecycle_manager.create(batch)
        assert result["commodity"] == commodity

    def test_create_batch_assigns_id(self, batch_lifecycle_manager):
        """Batch creation assigns a unique batch_id."""
        batch = make_batch(batch_id=None)
        result = batch_lifecycle_manager.create(batch)
        assert result.get("batch_id") is not None

    def test_create_batch_initial_status(self, batch_lifecycle_manager):
        """Newly created batch has status 'created'."""
        batch = make_batch()
        result = batch_lifecycle_manager.create(batch)
        assert result["status"] == "created"

    def test_create_batch_provenance_hash(self, batch_lifecycle_manager):
        """Batch creation generates a provenance hash."""
        batch = make_batch()
        result = batch_lifecycle_manager.create(batch)
        assert len(result.get("provenance_hash", "")) == SHA256_HEX_LENGTH

    def test_create_batch_zero_quantity_raises(self, batch_lifecycle_manager):
        """Batch with zero quantity raises ValueError."""
        batch = make_batch(quantity_kg=0.0)
        with pytest.raises(ValueError):
            batch_lifecycle_manager.create(batch)

    def test_create_batch_negative_quantity_raises(self, batch_lifecycle_manager):
        """Batch with negative quantity raises ValueError."""
        batch = make_batch(quantity_kg=-100.0)
        with pytest.raises(ValueError):
            batch_lifecycle_manager.create(batch)

    def test_create_batch_origin_not_sum_100_raises(self, batch_lifecycle_manager):
        """Origin allocations not summing to 100% raise ValueError."""
        batch = make_batch(
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 50.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 30.0},
            ],
        )
        with pytest.raises(ValueError):
            batch_lifecycle_manager.create(batch)

    def test_create_duplicate_batch_id_raises(self, batch_lifecycle_manager):
        """Creating two batches with same ID raises an error."""
        batch = make_batch(batch_id="BATCH-DUP-001")
        batch_lifecycle_manager.create(batch)
        with pytest.raises((ValueError, KeyError)):
            batch_lifecycle_manager.create(copy.deepcopy(batch))


# ===========================================================================
# 2. Split Operations (F2.3)
# ===========================================================================


class TestBatchSplit:
    """Test batch splitting into sub-batches."""

    def test_split_two_way(self, batch_lifecycle_manager):
        """Split a batch into 2 equal sub-batches."""
        parent = make_batch(batch_id="BATCH-SPLIT2-P", quantity_kg=10000.0)
        batch_lifecycle_manager.create(parent)
        result = batch_lifecycle_manager.split(
            "BATCH-SPLIT2-P",
            ratios=[0.5, 0.5],
        )
        assert len(result["children"]) == 2
        total = sum(c["quantity_kg"] for c in result["children"])
        assert total == pytest.approx(10000.0, abs=1.0)

    def test_split_three_way(self, batch_lifecycle_manager):
        """Split a batch into 3 sub-batches with unequal ratios."""
        parent = make_batch(batch_id="BATCH-SPLIT3-P", quantity_kg=9000.0)
        batch_lifecycle_manager.create(parent)
        result = batch_lifecycle_manager.split(
            "BATCH-SPLIT3-P",
            ratios=[0.5, 0.3, 0.2],
        )
        assert len(result["children"]) == 3
        assert result["children"][0]["quantity_kg"] == pytest.approx(4500.0)
        assert result["children"][1]["quantity_kg"] == pytest.approx(2700.0)
        assert result["children"][2]["quantity_kg"] == pytest.approx(1800.0)

    @pytest.mark.parametrize("n_splits", [2, 3, 5, 10])
    def test_split_n_way(self, batch_lifecycle_manager, n_splits):
        """Split a batch into N equal sub-batches."""
        parent = make_batch(batch_id=f"BATCH-SPLITN-{n_splits}", quantity_kg=10000.0)
        batch_lifecycle_manager.create(parent)
        ratios = [1.0 / n_splits] * n_splits
        result = batch_lifecycle_manager.split(f"BATCH-SPLITN-{n_splits}", ratios=ratios)
        assert len(result["children"]) == n_splits

    def test_split_preserves_origin_allocation(self, batch_lifecycle_manager):
        """Split sub-batches inherit parent origin allocations proportionally."""
        parent = make_batch(
            batch_id="BATCH-SPLIT-ORIG",
            quantity_kg=10000.0,
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 40.0},
            ],
        )
        batch_lifecycle_manager.create(parent)
        result = batch_lifecycle_manager.split("BATCH-SPLIT-ORIG", ratios=[0.5, 0.5])
        for child in result["children"]:
            assert_origin_preserved(child)

    def test_split_ratios_not_sum_one_raises(self, batch_lifecycle_manager):
        """Split ratios not summing to 1.0 raises ValueError."""
        parent = make_batch(batch_id="BATCH-SPLIT-BAD", quantity_kg=10000.0)
        batch_lifecycle_manager.create(parent)
        with pytest.raises(ValueError):
            batch_lifecycle_manager.split("BATCH-SPLIT-BAD", ratios=[0.5, 0.3])

    def test_split_updates_parent_status(self, batch_lifecycle_manager):
        """Splitting updates parent batch status to 'consumed'."""
        parent = make_batch(batch_id="BATCH-SPLIT-STAT", quantity_kg=10000.0)
        batch_lifecycle_manager.create(parent)
        batch_lifecycle_manager.split("BATCH-SPLIT-STAT", ratios=[0.5, 0.5])
        parent_after = batch_lifecycle_manager.get("BATCH-SPLIT-STAT")
        assert parent_after["status"] in ("consumed", "split")

    def test_split_nonexistent_batch_raises(self, batch_lifecycle_manager):
        """Splitting a non-existent batch raises an error."""
        with pytest.raises((ValueError, KeyError)):
            batch_lifecycle_manager.split("BATCH-NOSUCH", ratios=[0.5, 0.5])


# ===========================================================================
# 3. Merge Operations (F2.4)
# ===========================================================================


class TestBatchMerge:
    """Test batch merging operations."""

    def test_merge_two_batches(self, batch_lifecycle_manager):
        """Merge 2 batches into one with combined quantity."""
        b1 = make_batch(batch_id="BATCH-MRG-A", quantity_kg=3000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}])
        b2 = make_batch(batch_id="BATCH-MRG-B", quantity_kg=2000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 100.0}])
        batch_lifecycle_manager.create(b1)
        batch_lifecycle_manager.create(b2)
        result = batch_lifecycle_manager.merge(["BATCH-MRG-A", "BATCH-MRG-B"])
        assert result["quantity_kg"] == pytest.approx(5000.0)

    @pytest.mark.parametrize("n_inputs", [2, 3, 5])
    def test_merge_n_batches(self, batch_lifecycle_manager, n_inputs):
        """Merge N batches into one."""
        ids = []
        for i in range(n_inputs):
            bid = f"BATCH-MRGN-{n_inputs}-{i}"
            batch_lifecycle_manager.create(make_batch(batch_id=bid, quantity_kg=1000.0))
            ids.append(bid)
        result = batch_lifecycle_manager.merge(ids)
        assert result["quantity_kg"] == pytest.approx(1000.0 * n_inputs)

    def test_merge_combines_origins(self, batch_lifecycle_manager):
        """Merged batch has combined origin allocations."""
        b1 = make_batch(batch_id="BATCH-MRG-ORIG-A", quantity_kg=6000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}])
        b2 = make_batch(batch_id="BATCH-MRG-ORIG-B", quantity_kg=4000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 100.0}])
        batch_lifecycle_manager.create(b1)
        batch_lifecycle_manager.create(b2)
        result = batch_lifecycle_manager.merge(["BATCH-MRG-ORIG-A", "BATCH-MRG-ORIG-B"])
        assert_origin_preserved(result)
        plot_ids = [p["plot_id"] for p in result["origin_plots"]]
        assert PLOT_ID_COCOA_GH_1 in plot_ids
        assert PLOT_ID_COCOA_GH_2 in plot_ids

    def test_merge_single_batch_raises(self, batch_lifecycle_manager):
        """Merging a single batch raises ValueError."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-MRG-SINGLE"))
        with pytest.raises(ValueError):
            batch_lifecycle_manager.merge(["BATCH-MRG-SINGLE"])

    def test_merge_different_commodities_raises(self, batch_lifecycle_manager):
        """Merging batches of different commodities raises ValueError."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-MRG-COC", commodity="cocoa"))
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-MRG-COF", commodity="coffee"))
        with pytest.raises(ValueError):
            batch_lifecycle_manager.merge(["BATCH-MRG-COC", "BATCH-MRG-COF"])

    def test_merge_updates_parent_statuses(self, batch_lifecycle_manager):
        """Merged parent batches are marked as consumed."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-MRG-S-A", quantity_kg=1000.0))
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-MRG-S-B", quantity_kg=1000.0))
        batch_lifecycle_manager.merge(["BATCH-MRG-S-A", "BATCH-MRG-S-B"])
        a = batch_lifecycle_manager.get("BATCH-MRG-S-A")
        b = batch_lifecycle_manager.get("BATCH-MRG-S-B")
        assert a["status"] in ("consumed", "merged")
        assert b["status"] in ("consumed", "merged")


# ===========================================================================
# 4. Blend Operations (F2.5)
# ===========================================================================


class TestBatchBlend:
    """Test batch blending operations with percentage tracking."""

    def test_blend_two_batches(self, batch_lifecycle_manager):
        """Blend 2 batches with percentage tracking."""
        b1 = make_batch(batch_id="BATCH-BLD-A", quantity_kg=3000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}])
        b2 = make_batch(batch_id="BATCH-BLD-B", quantity_kg=2000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 100.0}])
        batch_lifecycle_manager.create(b1)
        batch_lifecycle_manager.create(b2)
        result = batch_lifecycle_manager.blend(["BATCH-BLD-A", "BATCH-BLD-B"])
        assert result["quantity_kg"] == pytest.approx(5000.0)
        # Origin percentages should be weighted by quantity
        gh1_pct = next(
            (p["percentage"] for p in result["origin_plots"]
             if p["plot_id"] == PLOT_ID_COCOA_GH_1), 0.0
        )
        assert gh1_pct == pytest.approx(60.0, abs=1.0)

    def test_blend_preserves_origin_sum(self, batch_lifecycle_manager):
        """Blended batch origin percentages sum to 100%."""
        b1 = make_batch(batch_id="BATCH-BLD-SUM-A", quantity_kg=5000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}])
        b2 = make_batch(batch_id="BATCH-BLD-SUM-B", quantity_kg=5000.0,
                        origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 100.0}])
        batch_lifecycle_manager.create(b1)
        batch_lifecycle_manager.create(b2)
        result = batch_lifecycle_manager.blend(["BATCH-BLD-SUM-A", "BATCH-BLD-SUM-B"])
        assert_origin_preserved(result)


# ===========================================================================
# 5. Genealogy Tree Traversal (F2.6)
# ===========================================================================


class TestBatchGenealogy:
    """Test batch genealogy tree traversal."""

    def test_linear_genealogy_upstream(self, batch_lifecycle_manager):
        """Traverse upstream through a linear genealogy."""
        batches = build_linear_genealogy(depth=5)
        for b in batches:
            batch_lifecycle_manager.create(b)
        upstream = batch_lifecycle_manager.get_upstream(batches[-1]["batch_id"])
        assert len(upstream) == 4

    def test_linear_genealogy_downstream(self, batch_lifecycle_manager):
        """Traverse downstream through a linear genealogy."""
        batches = build_linear_genealogy(depth=5)
        for b in batches:
            batch_lifecycle_manager.create(b)
        downstream = batch_lifecycle_manager.get_downstream(batches[0]["batch_id"])
        assert len(downstream) == 4

    def test_split_genealogy_tree(self, batch_lifecycle_manager):
        """Split genealogy shows all children."""
        batches = build_split_genealogy(n_splits=3)
        for b in batches:
            batch_lifecycle_manager.create(b)
        children = batch_lifecycle_manager.get_downstream(batches[0]["batch_id"])
        assert len(children) == 3

    def test_merge_genealogy_tree(self, batch_lifecycle_manager):
        """Merge genealogy shows all parents."""
        batches = build_merge_genealogy(n_inputs=3)
        for b in batches:
            batch_lifecycle_manager.create(b)
        parents = batch_lifecycle_manager.get_upstream(batches[-1]["batch_id"])
        assert len(parents) == 3

    def test_diamond_genealogy_tree(self, batch_lifecycle_manager):
        """Diamond genealogy traversal handles shared ancestors."""
        batches = build_diamond_genealogy()
        for b in batches:
            batch_lifecycle_manager.create(b)
        tree = batch_lifecycle_manager.get_genealogy_tree(batches[0]["batch_id"])
        assert tree is not None
        batch_ids = {n["batch_id"] for n in tree["nodes"]}
        assert len(batch_ids) == 4

    def test_full_genealogy_tree(self, batch_lifecycle_manager):
        """Get complete genealogy tree from any node."""
        batches = build_linear_genealogy(depth=4)
        for b in batches:
            batch_lifecycle_manager.create(b)
        tree = batch_lifecycle_manager.get_genealogy_tree(batches[1]["batch_id"])
        assert len(tree["nodes"]) == 4

    def test_root_batch_has_no_parents(self, batch_lifecycle_manager):
        """Root batch upstream traversal returns empty."""
        batch = make_batch(batch_id="BATCH-ROOT", parent_batch_ids=[])
        batch_lifecycle_manager.create(batch)
        upstream = batch_lifecycle_manager.get_upstream("BATCH-ROOT")
        assert len(upstream) == 0

    def test_leaf_batch_has_no_children(self, batch_lifecycle_manager):
        """Leaf batch downstream traversal returns empty."""
        batch = make_batch(batch_id="BATCH-LEAF")
        batch_lifecycle_manager.create(batch)
        downstream = batch_lifecycle_manager.get_downstream("BATCH-LEAF")
        assert len(downstream) == 0


# ===========================================================================
# 6. Status Transitions (F2.7)
# ===========================================================================


class TestBatchStatusTransitions:
    """Test batch status transitions."""

    @pytest.mark.parametrize("from_status,to_status", VALID_BATCH_TRANSITIONS)
    def test_valid_transitions(self, batch_lifecycle_manager, from_status, to_status):
        """All valid status transitions are accepted."""
        batch = make_batch(batch_id=f"BATCH-TR-{from_status}-{to_status}", status=from_status)
        batch_lifecycle_manager.create(batch)
        result = batch_lifecycle_manager.update_status(
            f"BATCH-TR-{from_status}-{to_status}", to_status
        )
        assert result["status"] == to_status

    @pytest.mark.parametrize("from_status,to_status", INVALID_BATCH_TRANSITIONS)
    def test_invalid_transitions(self, batch_lifecycle_manager, from_status, to_status):
        """All invalid status transitions raise ValueError."""
        batch = make_batch(batch_id=f"BATCH-ITR-{from_status}-{to_status}", status=from_status)
        batch_lifecycle_manager.create(batch)
        with pytest.raises(ValueError):
            batch_lifecycle_manager.update_status(
                f"BATCH-ITR-{from_status}-{to_status}", to_status
            )

    def test_all_statuses_recognized(self, batch_lifecycle_manager):
        """All 8 batch statuses are recognized by the system."""
        for status in BATCH_STATUSES:
            batch = make_batch(batch_id=f"BATCH-STAT-{status}", status=status)
            result = batch_lifecycle_manager.create(batch)
            assert result["status"] == status


# ===========================================================================
# 7. Origin Allocation (F2.9)
# ===========================================================================


class TestOriginAllocation:
    """Test origin plot percentage allocation."""

    def test_single_origin_100_percent(self, batch_lifecycle_manager):
        """Single origin always has 100% allocation."""
        batch = make_batch(
            batch_id="BATCH-ORIG-100",
            origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}],
        )
        result = batch_lifecycle_manager.create(batch)
        assert result["origin_plots"][0]["percentage"] == 100.0

    def test_three_origins_sum_100(self, batch_lifecycle_manager):
        """Three origins with different percentages sum to 100%."""
        batch = make_batch(
            batch_id="BATCH-ORIG-3WAY",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 50.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 30.0},
                {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 20.0},
            ],
        )
        result = batch_lifecycle_manager.create(batch)
        assert_origin_preserved(result)

    def test_duplicate_plot_in_origins_raises(self, batch_lifecycle_manager):
        """Same plot listed twice in origin allocations raises ValueError."""
        batch = make_batch(
            batch_id="BATCH-ORIG-DUP",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0},
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 40.0},
            ],
        )
        with pytest.raises(ValueError):
            batch_lifecycle_manager.create(batch)

    def test_negative_percentage_raises(self, batch_lifecycle_manager):
        """Negative origin percentage raises ValueError."""
        batch = make_batch(
            batch_id="BATCH-ORIG-NEG",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": -10.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 110.0},
            ],
        )
        with pytest.raises(ValueError):
            batch_lifecycle_manager.create(batch)


# ===========================================================================
# 8. Quantity Conservation (F2.8)
# ===========================================================================


class TestQuantityConservation:
    """Test input = outputs + waste conservation."""

    def test_split_conserves_quantity(self, batch_lifecycle_manager):
        """Split operation conserves total quantity."""
        parent = make_batch(batch_id="BATCH-QC-SPLIT", quantity_kg=10000.0)
        batch_lifecycle_manager.create(parent)
        result = batch_lifecycle_manager.split("BATCH-QC-SPLIT", ratios=[0.6, 0.4])
        total_out = sum(c["quantity_kg"] for c in result["children"])
        assert_mass_conservation(10000.0, total_out)

    def test_merge_conserves_quantity(self, batch_lifecycle_manager):
        """Merge operation conserves total quantity."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-QC-M-A", quantity_kg=3000.0))
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-QC-M-B", quantity_kg=2000.0))
        result = batch_lifecycle_manager.merge(["BATCH-QC-M-A", "BATCH-QC-M-B"])
        assert_mass_conservation(5000.0, result["quantity_kg"])


# ===========================================================================
# 9. Search (F2.10)
# ===========================================================================


class TestBatchSearch:
    """Test batch search by various criteria."""

    def test_search_by_commodity(self, batch_lifecycle_manager):
        """Search batches by commodity type."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-SRC-COC", commodity="cocoa"))
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-SRC-COF", commodity="coffee"))
        results = batch_lifecycle_manager.search(commodity="cocoa")
        assert all(r["commodity"] == "cocoa" for r in results)

    def test_search_by_origin_plot(self, batch_lifecycle_manager):
        """Search batches by origin plot ID."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-SRC-PLT"))
        results = batch_lifecycle_manager.search(origin_plot_id=PLOT_ID_COCOA_GH_1)
        assert len(results) >= 1

    def test_search_by_status(self, batch_lifecycle_manager):
        """Search batches by status."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-SRC-STAT", status="created"))
        results = batch_lifecycle_manager.search(status="created")
        assert all(r["status"] == "created" for r in results)

    def test_search_by_date_range(self, batch_lifecycle_manager):
        """Search batches by production date range."""
        batch_lifecycle_manager.create(make_batch(batch_id="BATCH-SRC-DATE"))
        results = batch_lifecycle_manager.search(
            date_from="2020-01-01", date_to="2030-12-31"
        )
        assert len(results) >= 1

    def test_search_no_results(self, batch_lifecycle_manager):
        """Search with no matches returns empty list."""
        results = batch_lifecycle_manager.search(commodity="nonexistent_commodity")
        assert len(results) == 0

    def test_search_combined_criteria(self, batch_lifecycle_manager):
        """Search with multiple criteria narrows results."""
        batch_lifecycle_manager.create(make_batch(
            batch_id="BATCH-SRC-COMBO", commodity="cocoa", status="created"
        ))
        results = batch_lifecycle_manager.search(commodity="cocoa", status="created")
        assert len(results) >= 1

    def test_get_nonexistent_batch_returns_none(self, batch_lifecycle_manager):
        """Getting a non-existent batch returns None."""
        result = batch_lifecycle_manager.get("BATCH-NONEXISTENT")
        assert result is None
