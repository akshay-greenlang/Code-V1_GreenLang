# -*- coding: utf-8 -*-
"""
Unit Tests for PurchaseOrderEngine (AGENT-DATA-003)

Tests PO extraction, single PO retrieval, summary, goods receipt matching,
open PO listing, value calculation, and line item handling.

Coverage target: 85%+ of purchase order logic

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline PurchaseOrderEngine
# ---------------------------------------------------------------------------


class PurchaseOrderEngine:
    """Manages purchase order extraction and analysis from ERP data."""

    def __init__(self):
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._receipts: Dict[str, List[Dict[str, Any]]] = {}

    def add_orders(self, orders: List[Dict[str, Any]]) -> int:
        for po in orders:
            self._orders[po["po_number"]] = po
        return len(orders)

    def get_order(self, po_number: str) -> Optional[Dict[str, Any]]:
        return self._orders.get(po_number)

    def get_all_orders(self) -> List[Dict[str, Any]]:
        return list(self._orders.values())

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return [po for po in self._orders.values() if po.get("status") == "open"]

    def get_closed_orders(self) -> List[Dict[str, Any]]:
        return [po for po in self._orders.values() if po.get("status") == "closed"]

    def get_by_vendor(self, vendor_id: str) -> List[Dict[str, Any]]:
        return [po for po in self._orders.values() if po.get("vendor_id") == vendor_id]

    def summarize(self) -> Dict[str, Any]:
        total_value = sum(po.get("total_value", 0.0) for po in self._orders.values())
        open_count = len(self.get_open_orders())
        closed_count = len(self.get_closed_orders())
        return {
            "total_orders": len(self._orders),
            "open_orders": open_count,
            "closed_orders": closed_count,
            "total_value": round(total_value, 2),
        }

    def get_total_value(self) -> float:
        return round(sum(po.get("total_value", 0.0) for po in self._orders.values()), 2)

    def get_line_items(self, po_number: str) -> List[Dict[str, Any]]:
        po = self._orders.get(po_number)
        if po is None:
            return []
        return po.get("line_items", [])

    def get_line_item_count(self, po_number: str) -> int:
        return len(self.get_line_items(po_number))

    def record_goods_receipt(self, po_number: str, receipt: Dict[str, Any]) -> bool:
        if po_number not in self._orders:
            return False
        if po_number not in self._receipts:
            self._receipts[po_number] = []
        self._receipts[po_number].append(receipt)
        return True

    def get_goods_receipts(self, po_number: str) -> List[Dict[str, Any]]:
        return list(self._receipts.get(po_number, []))

    def is_fully_received(self, po_number: str) -> bool:
        po = self._orders.get(po_number)
        if po is None:
            return False
        receipts = self._receipts.get(po_number, [])
        received_total = sum(r.get("quantity", 0.0) for r in receipts)
        ordered_total = sum(li.get("quantity", 0.0) for li in po.get("line_items", []))
        return received_total >= ordered_total if ordered_total > 0 else False

    def calculate_line_total(self, po_number: str) -> float:
        items = self.get_line_items(po_number)
        return round(sum(li.get("amount", 0.0) for li in items), 2)


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

SAMPLE_POS = [
    {
        "po_number": "PO-2025-0001", "vendor_id": "V-SAP-10001", "status": "open",
        "total_value": 250000.0, "currency": "EUR",
        "line_items": [
            {"item_number": 10, "material": "STEEL-HR-001", "quantity": 50.0, "unit_price": 4000.0, "amount": 200000.0},
            {"item_number": 20, "material": "STEEL-CR-002", "quantity": 10.0, "unit_price": 5000.0, "amount": 50000.0},
        ],
    },
    {
        "po_number": "PO-2025-0002", "vendor_id": "V-ORA-20001", "status": "closed",
        "total_value": 78500.0, "currency": "USD",
        "line_items": [
            {"item_number": 10, "material": "REC-SOLAR-001", "quantity": 500.0, "unit_price": 157.0, "amount": 78500.0},
        ],
    },
    {
        "po_number": "PO-2025-0003", "vendor_id": "V-SAP-10001", "status": "open",
        "total_value": 35000.0, "currency": "EUR",
        "line_items": [
            {"item_number": 10, "material": "PACK-BIO-001", "quantity": 10000.0, "unit_price": 3.50, "amount": 35000.0},
        ],
    },
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAddOrders:
    def test_add_orders(self):
        engine = PurchaseOrderEngine()
        count = engine.add_orders(SAMPLE_POS)
        assert count == 3

    def test_get_all_orders(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        assert len(engine.get_all_orders()) == 3

    def test_add_empty_list(self):
        engine = PurchaseOrderEngine()
        assert engine.add_orders([]) == 0


class TestGetOrder:
    def test_get_existing(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        po = engine.get_order("PO-2025-0001")
        assert po is not None
        assert po["total_value"] == 250000.0

    def test_get_nonexistent(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        assert engine.get_order("PO-NONEXISTENT") is None


class TestOpenClosedOrders:
    def test_get_open_orders(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        open_pos = engine.get_open_orders()
        assert len(open_pos) == 2

    def test_get_closed_orders(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        closed_pos = engine.get_closed_orders()
        assert len(closed_pos) == 1
        assert closed_pos[0]["po_number"] == "PO-2025-0002"


class TestGetByVendor:
    def test_by_vendor(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        vendor_pos = engine.get_by_vendor("V-SAP-10001")
        assert len(vendor_pos) == 2

    def test_by_vendor_nonexistent(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        assert len(engine.get_by_vendor("V-UNKNOWN")) == 0


class TestSummarize:
    def test_summary_total_orders(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        summary = engine.summarize()
        assert summary["total_orders"] == 3

    def test_summary_open_closed(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        summary = engine.summarize()
        assert summary["open_orders"] == 2
        assert summary["closed_orders"] == 1

    def test_summary_total_value(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        summary = engine.summarize()
        assert summary["total_value"] == pytest.approx(363500.0)

    def test_summary_empty(self):
        engine = PurchaseOrderEngine()
        summary = engine.summarize()
        assert summary["total_orders"] == 0
        assert summary["total_value"] == 0.0


class TestTotalValue:
    def test_total_value(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        assert engine.get_total_value() == pytest.approx(363500.0)


class TestLineItems:
    def test_get_line_items(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        items = engine.get_line_items("PO-2025-0001")
        assert len(items) == 2

    def test_get_line_items_nonexistent(self):
        engine = PurchaseOrderEngine()
        assert engine.get_line_items("PO-NONE") == []

    def test_line_item_count(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        assert engine.get_line_item_count("PO-2025-0001") == 2
        assert engine.get_line_item_count("PO-2025-0002") == 1

    def test_calculate_line_total(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        total = engine.calculate_line_total("PO-2025-0001")
        assert total == pytest.approx(250000.0)


class TestGoodsReceipt:
    def test_record_receipt(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        result = engine.record_goods_receipt("PO-2025-0001", {"quantity": 30.0, "date": "2025-07-01"})
        assert result is True

    def test_record_receipt_nonexistent_po(self):
        engine = PurchaseOrderEngine()
        result = engine.record_goods_receipt("PO-NONE", {"quantity": 10.0})
        assert result is False

    def test_get_receipts(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        engine.record_goods_receipt("PO-2025-0001", {"quantity": 30.0})
        engine.record_goods_receipt("PO-2025-0001", {"quantity": 20.0})
        receipts = engine.get_goods_receipts("PO-2025-0001")
        assert len(receipts) == 2

    def test_is_fully_received_false(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        engine.record_goods_receipt("PO-2025-0001", {"quantity": 30.0})
        assert engine.is_fully_received("PO-2025-0001") is False

    def test_is_fully_received_true(self):
        engine = PurchaseOrderEngine()
        engine.add_orders(SAMPLE_POS)
        engine.record_goods_receipt("PO-2025-0001", {"quantity": 60.0})
        assert engine.is_fully_received("PO-2025-0001") is True

    def test_is_fully_received_nonexistent(self):
        engine = PurchaseOrderEngine()
        assert engine.is_fully_received("PO-NONE") is False
