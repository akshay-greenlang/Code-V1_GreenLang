# -*- coding: utf-8 -*-
"""
Unit Tests for InventoryTracker (AGENT-DATA-003)

Tests inventory sync, snapshot, warehouse queries, material group queries,
valuation, and slow-moving item detection.

Coverage target: 85%+ of inventory tracking logic

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline InventoryTracker
# ---------------------------------------------------------------------------


class InventoryTracker:
    """Tracks inventory items synced from ERP warehouse management."""

    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def sync_items(self, items: List[Dict[str, Any]]) -> int:
        for item in items:
            self._items[item["item_id"]] = item
        return len(items)

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self._items.get(item_id)

    def get_all_items(self) -> List[Dict[str, Any]]:
        return list(self._items.values())

    def get_snapshot(self) -> Dict[str, Any]:
        total_items = len(self._items)
        total_value = sum(
            i.get("quantity", 0.0) * i.get("unit_cost", 0.0)
            for i in self._items.values()
        )
        warehouses = set(i.get("warehouse") for i in self._items.values())
        groups = set(i.get("material_group") for i in self._items.values())
        return {
            "total_items": total_items,
            "total_value": round(total_value, 2),
            "warehouse_count": len(warehouses),
            "material_group_count": len(groups),
        }

    def get_by_warehouse(self, warehouse: str) -> List[Dict[str, Any]]:
        return [i for i in self._items.values() if i.get("warehouse") == warehouse]

    def get_by_material_group(self, group: str) -> List[Dict[str, Any]]:
        return [i for i in self._items.values() if i.get("material_group") == group]

    def get_valuation(self) -> Dict[str, float]:
        """Returns total value per material group."""
        result: Dict[str, float] = {}
        for item in self._items.values():
            group = item.get("material_group", "other")
            value = item.get("quantity", 0.0) * item.get("unit_cost", 0.0)
            result[group] = result.get(group, 0.0) + value
        return {k: round(v, 2) for k, v in result.items()}

    def get_slow_moving(self, threshold_quantity: float = 10.0) -> List[Dict[str, Any]]:
        """Items with quantity below threshold."""
        return [i for i in self._items.values() if i.get("quantity", 0.0) <= threshold_quantity]

    def get_total_value(self) -> float:
        return round(sum(
            i.get("quantity", 0.0) * i.get("unit_cost", 0.0)
            for i in self._items.values()
        ), 2)

    def get_warehouse_summary(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for item in self._items.values():
            wh = item.get("warehouse", "unknown")
            result[wh] = result.get(wh, 0) + 1
        return result


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

SAMPLE_INVENTORY = [
    {"item_id": "INV-001", "material": "STEEL-HR-001", "warehouse": "WH-MAIN", "quantity": 120.0, "unit_cost": 4000.0, "material_group": "raw_materials"},
    {"item_id": "INV-002", "material": "PACK-BIO-001", "warehouse": "WH-MAIN", "quantity": 5000.0, "unit_cost": 2.50, "material_group": "packaging"},
    {"item_id": "INV-003", "material": "CHEM-SOL-001", "warehouse": "WH-CHEM", "quantity": 800.0, "unit_cost": 42.75, "material_group": "chemicals"},
    {"item_id": "INV-004", "material": "STEEL-CR-002", "warehouse": "WH-MAIN", "quantity": 45.0, "unit_cost": 5000.0, "material_group": "raw_materials"},
    {"item_id": "INV-005", "material": "ELEC-COMP-001", "warehouse": "WH-ELEC", "quantity": 5.0, "unit_cost": 1.20, "material_group": "components"},
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestSyncItems:
    def test_sync_items(self):
        tracker = InventoryTracker()
        count = tracker.sync_items(SAMPLE_INVENTORY)
        assert count == 5

    def test_sync_overwrites_existing(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        updated = [{"item_id": "INV-001", "material": "STEEL-HR-001", "warehouse": "WH-MAIN", "quantity": 150.0, "unit_cost": 4000.0, "material_group": "raw_materials"}]
        tracker.sync_items(updated)
        item = tracker.get_item("INV-001")
        assert item["quantity"] == 150.0

    def test_get_item_existing(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        item = tracker.get_item("INV-001")
        assert item is not None
        assert item["material"] == "STEEL-HR-001"

    def test_get_item_nonexistent(self):
        tracker = InventoryTracker()
        assert tracker.get_item("INV-999") is None

    def test_get_all_items(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        assert len(tracker.get_all_items()) == 5


class TestSnapshot:
    def test_snapshot_total_items(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        snap = tracker.get_snapshot()
        assert snap["total_items"] == 5

    def test_snapshot_total_value(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        snap = tracker.get_snapshot()
        expected = 120.0 * 4000.0 + 5000.0 * 2.50 + 800.0 * 42.75 + 45.0 * 5000.0 + 5.0 * 1.20
        assert snap["total_value"] == pytest.approx(expected)

    def test_snapshot_warehouse_count(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        snap = tracker.get_snapshot()
        assert snap["warehouse_count"] == 3

    def test_snapshot_material_group_count(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        snap = tracker.get_snapshot()
        assert snap["material_group_count"] == 4

    def test_snapshot_empty(self):
        tracker = InventoryTracker()
        snap = tracker.get_snapshot()
        assert snap["total_items"] == 0
        assert snap["total_value"] == 0.0


class TestByWarehouse:
    def test_by_warehouse_main(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        items = tracker.get_by_warehouse("WH-MAIN")
        assert len(items) == 3

    def test_by_warehouse_chem(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        items = tracker.get_by_warehouse("WH-CHEM")
        assert len(items) == 1

    def test_by_warehouse_nonexistent(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        items = tracker.get_by_warehouse("WH-UNKNOWN")
        assert len(items) == 0


class TestByMaterialGroup:
    def test_by_group_raw_materials(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        items = tracker.get_by_material_group("raw_materials")
        assert len(items) == 2

    def test_by_group_nonexistent(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        items = tracker.get_by_material_group("electronics")
        assert len(items) == 0


class TestValuation:
    def test_valuation_by_group(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        val = tracker.get_valuation()
        assert "raw_materials" in val
        assert val["raw_materials"] == pytest.approx(120.0 * 4000.0 + 45.0 * 5000.0)

    def test_valuation_all_groups(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        val = tracker.get_valuation()
        assert len(val) == 4

    def test_total_value(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        expected = 120.0 * 4000.0 + 5000.0 * 2.50 + 800.0 * 42.75 + 45.0 * 5000.0 + 5.0 * 1.20
        assert tracker.get_total_value() == pytest.approx(expected)


class TestSlowMoving:
    def test_slow_moving_default_threshold(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        slow = tracker.get_slow_moving()
        assert len(slow) == 1
        assert slow[0]["item_id"] == "INV-005"

    def test_slow_moving_custom_threshold(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        slow = tracker.get_slow_moving(threshold_quantity=50.0)
        assert len(slow) == 2

    def test_slow_moving_empty(self):
        tracker = InventoryTracker()
        assert tracker.get_slow_moving() == []


class TestWarehouseSummary:
    def test_warehouse_summary(self):
        tracker = InventoryTracker()
        tracker.sync_items(SAMPLE_INVENTORY)
        summary = tracker.get_warehouse_summary()
        assert summary["WH-MAIN"] == 3
        assert summary["WH-CHEM"] == 1
        assert summary["WH-ELEC"] == 1
