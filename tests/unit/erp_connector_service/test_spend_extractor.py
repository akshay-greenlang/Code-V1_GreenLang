# -*- coding: utf-8 -*-
"""
Unit Tests for SpendExtractor (AGENT-DATA-003)

Tests spend data extraction, summarization, top vendors, by category,
trend analysis, filtering, and date range queries.

Coverage target: 85%+ of spend extraction logic

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline SpendExtractor
# ---------------------------------------------------------------------------


class SpendExtractor:
    """Extracts and analyzes spend data from ERP records."""

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add_records(self, records: List[Dict[str, Any]]) -> int:
        self._records.extend(records)
        return len(records)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def get_by_vendor(self, vendor_id: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if r.get("vendor_id") == vendor_id]

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if r.get("category") == category]

    def get_by_date_range(self, start: str, end: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if start <= r.get("date", "") <= end]

    def get_by_cost_center(self, cost_center: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if r.get("cost_center") == cost_center]

    def summarize(self) -> Dict[str, Any]:
        total = sum(r.get("amount", 0.0) for r in self._records)
        vendors = set(r.get("vendor_id") for r in self._records)
        categories = set(r.get("category") for r in self._records)
        return {
            "total_spend": round(total, 2),
            "record_count": len(self._records),
            "vendor_count": len(vendors),
            "category_count": len(categories),
        }

    def top_vendors(self, n: int = 5) -> List[Dict[str, Any]]:
        vendor_spend: Dict[str, float] = {}
        for r in self._records:
            vid = r.get("vendor_id", "unknown")
            vendor_spend[vid] = vendor_spend.get(vid, 0.0) + r.get("amount", 0.0)
        sorted_vendors = sorted(vendor_spend.items(), key=lambda x: x[1], reverse=True)
        return [{"vendor_id": v, "total_spend": round(s, 2)} for v, s in sorted_vendors[:n]]

    def by_category_summary(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for r in self._records:
            cat = r.get("category", "other")
            result[cat] = result.get(cat, 0.0) + r.get("amount", 0.0)
        return {k: round(v, 2) for k, v in result.items()}

    def trend_by_month(self) -> Dict[str, float]:
        monthly: Dict[str, float] = {}
        for r in self._records:
            d = r.get("date", "")
            if len(d) >= 7:
                month_key = d[:7]
                monthly[month_key] = monthly.get(month_key, 0.0) + r.get("amount", 0.0)
        return {k: round(v, 2) for k, v in sorted(monthly.items())}

    def filter_records(self, min_amount: Optional[float] = None,
                       max_amount: Optional[float] = None,
                       currency: Optional[str] = None) -> List[Dict[str, Any]]:
        result = self._records
        if min_amount is not None:
            result = [r for r in result if r.get("amount", 0.0) >= min_amount]
        if max_amount is not None:
            result = [r for r in result if r.get("amount", 0.0) <= max_amount]
        if currency is not None:
            result = [r for r in result if r.get("currency") == currency]
        return result


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

SAMPLE_SPEND = [
    {"record_id": "SPD-001", "vendor_id": "V-SAP-10001", "amount": 125000.0, "currency": "EUR", "date": "2025-06-15", "category": "raw_materials", "cost_center": "CC-MFG-001"},
    {"record_id": "SPD-002", "vendor_id": "V-SAP-10002", "amount": 45000.0, "currency": "EUR", "date": "2025-06-16", "category": "transportation", "cost_center": "CC-LOG-001"},
    {"record_id": "SPD-003", "vendor_id": "V-ORA-20001", "amount": 78500.0, "currency": "USD", "date": "2025-07-01", "category": "energy", "cost_center": "CC-OPS-001"},
    {"record_id": "SPD-004", "vendor_id": "V-ORA-20002", "amount": 12300.0, "currency": "GBP", "date": "2025-07-05", "category": "packaging", "cost_center": "CC-MFG-002"},
    {"record_id": "SPD-005", "vendor_id": "V-SAP-10001", "amount": 88000.0, "currency": "EUR", "date": "2025-07-10", "category": "raw_materials", "cost_center": "CC-MFG-001"},
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAddRecords:
    def test_add_records(self):
        ext = SpendExtractor()
        count = ext.add_records(SAMPLE_SPEND)
        assert count == 5

    def test_add_records_cumulative(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND[:3])
        ext.add_records(SAMPLE_SPEND[3:])
        assert len(ext.get_all()) == 5

    def test_add_empty_list(self):
        ext = SpendExtractor()
        assert ext.add_records([]) == 0

    def test_get_all_returns_copy(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        all_records = ext.get_all()
        assert len(all_records) == 5


class TestGetByVendor:
    def test_get_by_existing_vendor(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_vendor("V-SAP-10001")
        assert len(records) == 2

    def test_get_by_nonexistent_vendor(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_vendor("V-UNKNOWN")
        assert len(records) == 0

    def test_get_by_vendor_amounts(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_vendor("V-SAP-10001")
        total = sum(r["amount"] for r in records)
        assert total == pytest.approx(213000.0)


class TestGetByCategory:
    def test_by_category_raw_materials(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_category("raw_materials")
        assert len(records) == 2

    def test_by_category_energy(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_category("energy")
        assert len(records) == 1

    def test_by_category_nonexistent(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_category("consulting")
        assert len(records) == 0


class TestGetByDateRange:
    def test_june_only(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_date_range("2025-06-01", "2025-06-30")
        assert len(records) == 2

    def test_july_only(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_date_range("2025-07-01", "2025-07-31")
        assert len(records) == 3

    def test_full_range(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_date_range("2025-01-01", "2025-12-31")
        assert len(records) == 5

    def test_empty_range(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_date_range("2024-01-01", "2024-12-31")
        assert len(records) == 0


class TestSummarize:
    def test_summarize_total(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        summary = ext.summarize()
        assert summary["total_spend"] == pytest.approx(348800.0)

    def test_summarize_record_count(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        summary = ext.summarize()
        assert summary["record_count"] == 5

    def test_summarize_vendor_count(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        summary = ext.summarize()
        assert summary["vendor_count"] == 4

    def test_summarize_category_count(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        summary = ext.summarize()
        assert summary["category_count"] == 4

    def test_summarize_empty(self):
        ext = SpendExtractor()
        summary = ext.summarize()
        assert summary["total_spend"] == 0.0
        assert summary["record_count"] == 0


class TestTopVendors:
    def test_top_vendors_default(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        top = ext.top_vendors()
        assert len(top) == 4
        assert top[0]["vendor_id"] == "V-SAP-10001"

    def test_top_vendors_limit(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        top = ext.top_vendors(n=2)
        assert len(top) == 2

    def test_top_vendors_order(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        top = ext.top_vendors()
        for i in range(len(top) - 1):
            assert top[i]["total_spend"] >= top[i + 1]["total_spend"]


class TestByCategorySummary:
    def test_category_summary(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        summary = ext.by_category_summary()
        assert "raw_materials" in summary
        assert summary["raw_materials"] == pytest.approx(213000.0)

    def test_category_summary_all_categories(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        summary = ext.by_category_summary()
        assert len(summary) == 4


class TestTrendByMonth:
    def test_monthly_trend(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        trend = ext.trend_by_month()
        assert "2025-06" in trend
        assert "2025-07" in trend

    def test_monthly_trend_values(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        trend = ext.trend_by_month()
        assert trend["2025-06"] == pytest.approx(170000.0)

    def test_monthly_trend_ordered(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        trend = ext.trend_by_month()
        keys = list(trend.keys())
        assert keys == sorted(keys)


class TestFilterRecords:
    def test_filter_min_amount(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        filtered = ext.filter_records(min_amount=50000.0)
        assert len(filtered) == 3

    def test_filter_max_amount(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        filtered = ext.filter_records(max_amount=50000.0)
        assert len(filtered) == 2

    def test_filter_currency(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        filtered = ext.filter_records(currency="EUR")
        assert len(filtered) == 3

    def test_filter_combined(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        filtered = ext.filter_records(min_amount=50000.0, currency="EUR")
        assert len(filtered) == 2

    def test_filter_no_match(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        filtered = ext.filter_records(currency="JPY")
        assert len(filtered) == 0

    def test_filter_no_criteria(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        filtered = ext.filter_records()
        assert len(filtered) == 5


class TestGetByCostCenter:
    def test_by_cost_center(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_cost_center("CC-MFG-001")
        assert len(records) == 2

    def test_by_cost_center_nonexistent(self):
        ext = SpendExtractor()
        ext.add_records(SAMPLE_SPEND)
        records = ext.get_by_cost_center("CC-UNKNOWN")
        assert len(records) == 0
