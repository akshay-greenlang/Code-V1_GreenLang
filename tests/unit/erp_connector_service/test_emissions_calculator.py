# -*- coding: utf-8 -*-
"""
Unit Tests for EmissionsCalculator (AGENT-DATA-003)

Tests emissions calculation for all records, single records, EEIO factors,
custom factors, vendor-specific factors, methodology selection, summary,
total, by vendor, and factor lookup priority chain.

Coverage target: 85%+ of emissions calculation logic

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline EmissionsCalculator
# ---------------------------------------------------------------------------

DEFAULT_EEIO_FACTORS = {
    "cat1_purchased_goods": 0.35,
    "cat2_capital_goods": 0.45,
    "cat3_fuel_energy": 0.65,
    "cat4_upstream_transport": 0.25,
    "cat5_waste": 0.15,
    "cat6_business_travel": 0.40,
    "cat7_employee_commuting": 0.30,
    "cat8_upstream_leased": 0.20,
    "cat9_downstream_transport": 0.22,
    "cat10_processing": 0.28,
    "cat11_use_of_sold": 0.50,
    "cat12_end_of_life": 0.10,
    "cat13_downstream_leased": 0.18,
    "cat14_franchises": 0.12,
    "cat15_investments": 0.08,
    "unclassified": 0.30,
}


class EmissionsCalculator:
    """Calculates Scope 3 emissions from classified spend data."""

    def __init__(self, default_factors: Optional[Dict[str, float]] = None):
        self._factors = dict(default_factors or DEFAULT_EEIO_FACTORS)
        self._vendor_factors: Dict[str, float] = {}
        self._material_factors: Dict[str, float] = {}
        self._custom_factors: Dict[str, float] = {}
        self._results: List[Dict[str, Any]] = []

    def set_vendor_factor(self, vendor_id: str, factor: float):
        self._vendor_factors[vendor_id] = factor

    def set_material_factor(self, material: str, factor: float):
        self._material_factors[material] = factor

    def set_custom_factor(self, scope3_category: str, factor: float):
        self._custom_factors[scope3_category] = factor

    def get_factor(self, record: Dict[str, Any]) -> float:
        """Factor lookup priority: vendor > material > custom > default EEIO."""
        vendor_id = record.get("vendor_id", "")
        if vendor_id in self._vendor_factors:
            return self._vendor_factors[vendor_id]
        material = record.get("material", "")
        if material in self._material_factors:
            return self._material_factors[material]
        scope3 = record.get("scope3_category", "unclassified")
        if scope3 in self._custom_factors:
            return self._custom_factors[scope3]
        return self._factors.get(scope3, 0.30)

    def calculate_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        factor = self.get_factor(record)
        amount = record.get("amount_usd", record.get("amount", 0.0))
        emissions = round(amount * factor, 4)
        result = {
            "record_id": record.get("record_id", ""),
            "vendor_id": record.get("vendor_id", ""),
            "amount_usd": amount,
            "emission_factor": factor,
            "estimated_kgco2e": emissions,
            "scope3_category": record.get("scope3_category", "unclassified"),
            "methodology": "eeio",
        }
        self._results.append(result)
        return result

    def calculate_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.calculate_single(r) for r in records]

    def get_total_emissions(self) -> float:
        return round(sum(r["estimated_kgco2e"] for r in self._results), 4)

    def get_by_vendor(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for r in self._results:
            vid = r.get("vendor_id", "unknown")
            result[vid] = result.get(vid, 0.0) + r["estimated_kgco2e"]
        return {k: round(v, 4) for k, v in result.items()}

    def get_by_category(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for r in self._results:
            cat = r.get("scope3_category", "unclassified")
            result[cat] = result.get(cat, 0.0) + r["estimated_kgco2e"]
        return {k: round(v, 4) for k, v in result.items()}

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_emissions_kgco2e": self.get_total_emissions(),
            "total_records": len(self._results),
            "by_vendor": self.get_by_vendor(),
            "by_category": self.get_by_category(),
        }

    def get_results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    def clear_results(self):
        self._results.clear()


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

CLASSIFIED_SPEND = [
    {"record_id": "SPD-001", "vendor_id": "V-001", "amount": 125000.0, "scope3_category": "cat1_purchased_goods"},
    {"record_id": "SPD-002", "vendor_id": "V-002", "amount": 45000.0, "scope3_category": "cat4_upstream_transport"},
    {"record_id": "SPD-003", "vendor_id": "V-003", "amount": 78500.0, "scope3_category": "cat3_fuel_energy"},
    {"record_id": "SPD-004", "vendor_id": "V-004", "amount": 12300.0, "scope3_category": "cat1_purchased_goods"},
    {"record_id": "SPD-005", "vendor_id": "V-005", "amount": 34200.0, "scope3_category": "cat1_purchased_goods"},
    {"record_id": "SPD-006", "vendor_id": "V-006", "amount": 22000.0, "scope3_category": "cat5_waste"},
    {"record_id": "SPD-007", "vendor_id": "V-007", "amount": 15000.0, "scope3_category": "cat6_business_travel"},
    {"record_id": "SPD-008", "vendor_id": "V-001", "amount": 88000.0, "scope3_category": "cat1_purchased_goods"},
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCalculateSingle:
    def test_single_calculation(self):
        calc = EmissionsCalculator()
        result = calc.calculate_single(CLASSIFIED_SPEND[0])
        expected = 125000.0 * 0.35
        assert result["estimated_kgco2e"] == pytest.approx(expected)

    def test_single_has_required_fields(self):
        calc = EmissionsCalculator()
        result = calc.calculate_single(CLASSIFIED_SPEND[0])
        assert "record_id" in result
        assert "vendor_id" in result
        assert "emission_factor" in result
        assert "estimated_kgco2e" in result
        assert "methodology" in result

    def test_single_eeio_methodology(self):
        calc = EmissionsCalculator()
        result = calc.calculate_single(CLASSIFIED_SPEND[0])
        assert result["methodology"] == "eeio"


class TestCalculateBatch:
    def test_batch_all_records(self):
        calc = EmissionsCalculator()
        results = calc.calculate_batch(CLASSIFIED_SPEND)
        assert len(results) == 8

    def test_batch_all_positive_emissions(self):
        calc = EmissionsCalculator()
        results = calc.calculate_batch(CLASSIFIED_SPEND)
        for r in results:
            assert r["estimated_kgco2e"] > 0

    def test_batch_stores_results(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        assert len(calc.get_results()) == 8


class TestEEIOFactors:
    def test_cat1_factor(self):
        calc = EmissionsCalculator()
        record = {"scope3_category": "cat1_purchased_goods", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.35

    def test_cat3_factor(self):
        calc = EmissionsCalculator()
        record = {"scope3_category": "cat3_fuel_energy", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.65

    def test_cat6_factor(self):
        calc = EmissionsCalculator()
        record = {"scope3_category": "cat6_business_travel", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.40

    def test_unclassified_factor(self):
        calc = EmissionsCalculator()
        record = {"scope3_category": "unclassified", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.30

    def test_all_15_category_factors(self):
        calc = EmissionsCalculator()
        for cat, expected_factor in DEFAULT_EEIO_FACTORS.items():
            record = {"scope3_category": cat, "amount": 1000.0}
            result = calc.calculate_single(record)
            assert result["emission_factor"] == expected_factor
            calc.clear_results()


class TestCustomFactors:
    def test_custom_factor_overrides_default(self):
        calc = EmissionsCalculator()
        calc.set_custom_factor("cat1_purchased_goods", 0.50)
        record = {"scope3_category": "cat1_purchased_goods", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.50

    def test_custom_factor_only_affects_specified_category(self):
        calc = EmissionsCalculator()
        calc.set_custom_factor("cat1_purchased_goods", 0.50)
        record = {"scope3_category": "cat3_fuel_energy", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.65


class TestVendorSpecificFactors:
    def test_vendor_factor_highest_priority(self):
        calc = EmissionsCalculator()
        calc.set_vendor_factor("V-001", 0.75)
        calc.set_custom_factor("cat1_purchased_goods", 0.50)
        record = {"vendor_id": "V-001", "scope3_category": "cat1_purchased_goods", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.75

    def test_vendor_factor_only_affects_vendor(self):
        calc = EmissionsCalculator()
        calc.set_vendor_factor("V-001", 0.75)
        record = {"vendor_id": "V-002", "scope3_category": "cat1_purchased_goods", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.35


class TestMaterialFactors:
    def test_material_factor(self):
        calc = EmissionsCalculator()
        calc.set_material_factor("STEEL-HR-001", 1.85)
        record = {"material": "STEEL-HR-001", "scope3_category": "cat1_purchased_goods", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 1.85

    def test_vendor_beats_material(self):
        calc = EmissionsCalculator()
        calc.set_vendor_factor("V-001", 0.75)
        calc.set_material_factor("STEEL-HR-001", 1.85)
        record = {"vendor_id": "V-001", "material": "STEEL-HR-001", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 0.75

    def test_material_beats_custom(self):
        calc = EmissionsCalculator()
        calc.set_material_factor("STEEL-HR-001", 1.85)
        calc.set_custom_factor("cat1_purchased_goods", 0.50)
        record = {"material": "STEEL-HR-001", "scope3_category": "cat1_purchased_goods", "amount": 1000.0}
        result = calc.calculate_single(record)
        assert result["emission_factor"] == 1.85


class TestFactorPriorityChain:
    """Verify: vendor > material > custom > default EEIO."""

    def test_full_priority_chain(self):
        calc = EmissionsCalculator()
        calc.set_vendor_factor("V-001", 0.10)
        calc.set_material_factor("MAT-001", 0.20)
        calc.set_custom_factor("cat1_purchased_goods", 0.50)

        r1 = calc.calculate_single({"vendor_id": "V-001", "material": "MAT-001", "scope3_category": "cat1_purchased_goods", "amount": 1000.0})
        assert r1["emission_factor"] == 0.10

        calc.clear_results()
        r2 = calc.calculate_single({"vendor_id": "V-999", "material": "MAT-001", "scope3_category": "cat1_purchased_goods", "amount": 1000.0})
        assert r2["emission_factor"] == 0.20

        calc.clear_results()
        r3 = calc.calculate_single({"vendor_id": "V-999", "material": "MAT-999", "scope3_category": "cat1_purchased_goods", "amount": 1000.0})
        assert r3["emission_factor"] == 0.50

        calc.clear_results()
        r4 = calc.calculate_single({"vendor_id": "V-999", "material": "MAT-999", "scope3_category": "cat3_fuel_energy", "amount": 1000.0})
        assert r4["emission_factor"] == 0.65


class TestTotalEmissions:
    def test_total_emissions(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        total = calc.get_total_emissions()
        assert total > 0

    def test_total_matches_sum(self):
        calc = EmissionsCalculator()
        results = calc.calculate_batch(CLASSIFIED_SPEND)
        expected = sum(r["estimated_kgco2e"] for r in results)
        assert calc.get_total_emissions() == pytest.approx(expected)


class TestByVendor:
    def test_by_vendor_groups(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        by_vendor = calc.get_by_vendor()
        assert "V-001" in by_vendor
        assert by_vendor["V-001"] > 0

    def test_by_vendor_v001_has_two_records(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        by_vendor = calc.get_by_vendor()
        expected = (125000.0 + 88000.0) * 0.35
        assert by_vendor["V-001"] == pytest.approx(expected)


class TestByCategorySummary:
    def test_by_category(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        by_cat = calc.get_by_category()
        assert "cat1_purchased_goods" in by_cat


class TestSummary:
    def test_summary_fields(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        summary = calc.get_summary()
        assert "total_emissions_kgco2e" in summary
        assert "total_records" in summary
        assert "by_vendor" in summary
        assert "by_category" in summary
        assert summary["total_records"] == 8


class TestClearResults:
    def test_clear(self):
        calc = EmissionsCalculator()
        calc.calculate_batch(CLASSIFIED_SPEND)
        calc.clear_results()
        assert len(calc.get_results()) == 0
        assert calc.get_total_emissions() == 0.0
