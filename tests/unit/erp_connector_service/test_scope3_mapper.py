# -*- coding: utf-8 -*-
"""
Unit Tests for Scope3Mapper (AGENT-DATA-003)

Tests Scope 3 classification of spend data across all 15 GHG Protocol
categories plus unclassified, vendor mapping overrides, material mapping,
coverage analysis, and distribution summaries.

Coverage target: 85%+ of scope 3 mapping logic

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline Scope3Mapper
# ---------------------------------------------------------------------------

SPEND_TO_SCOPE3 = {
    "raw_materials": "cat1_purchased_goods",
    "energy": "cat3_fuel_energy",
    "transportation": "cat4_upstream_transport",
    "packaging": "cat1_purchased_goods",
    "waste_management": "cat5_waste",
    "it_services": "cat1_purchased_goods",
    "professional_services": "cat1_purchased_goods",
    "travel": "cat6_business_travel",
    "facilities": "cat8_upstream_leased",
    "chemicals": "cat1_purchased_goods",
    "capital_equipment": "cat2_capital_goods",
}


class Scope3Mapper:
    """Maps spend records to GHG Protocol Scope 3 categories."""

    def __init__(self):
        self._vendor_overrides: Dict[str, str] = {}
        self._material_overrides: Dict[str, str] = {}
        self._default_mapping = dict(SPEND_TO_SCOPE3)

    def classify_spend(self, record: Dict[str, Any]) -> str:
        vendor_id = record.get("vendor_id", "")
        if vendor_id in self._vendor_overrides:
            return self._vendor_overrides[vendor_id]
        material = record.get("material", "")
        if material in self._material_overrides:
            return self._material_overrides[material]
        category = record.get("category", "other")
        return self._default_mapping.get(category, "unclassified")

    def classify_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for r in records:
            scope3 = self.classify_spend(r)
            results.append({**r, "scope3_category": scope3})
        return results

    def set_vendor_override(self, vendor_id: str, scope3_category: str):
        self._vendor_overrides[vendor_id] = scope3_category

    def remove_vendor_override(self, vendor_id: str) -> bool:
        if vendor_id in self._vendor_overrides:
            del self._vendor_overrides[vendor_id]
            return True
        return False

    def set_material_override(self, material: str, scope3_category: str):
        self._material_overrides[material] = scope3_category

    def get_coverage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(records)
        if total == 0:
            return {"total": 0, "classified": 0, "unclassified": 0, "coverage_pct": 0.0}
        classified = sum(1 for r in records if self.classify_spend(r) != "unclassified")
        return {
            "total": total,
            "classified": classified,
            "unclassified": total - classified,
            "coverage_pct": round(classified / total * 100, 2),
        }

    def get_distribution(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        dist: Dict[str, Dict[str, Any]] = {}
        for r in records:
            scope3 = self.classify_spend(r)
            if scope3 not in dist:
                dist[scope3] = {"count": 0, "total_amount": 0.0}
            dist[scope3]["count"] += 1
            dist[scope3]["total_amount"] += r.get("amount", 0.0)
        for key in dist:
            dist[key]["total_amount"] = round(dist[key]["total_amount"], 2)
        return dist

    def get_vendor_overrides(self) -> Dict[str, str]:
        return dict(self._vendor_overrides)

    def get_material_overrides(self) -> Dict[str, str]:
        return dict(self._material_overrides)


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

SAMPLE_SPEND = [
    {"record_id": "SPD-001", "vendor_id": "V-001", "amount": 125000.0, "category": "raw_materials"},
    {"record_id": "SPD-002", "vendor_id": "V-002", "amount": 45000.0, "category": "transportation"},
    {"record_id": "SPD-003", "vendor_id": "V-003", "amount": 78500.0, "category": "energy"},
    {"record_id": "SPD-004", "vendor_id": "V-004", "amount": 12300.0, "category": "packaging"},
    {"record_id": "SPD-005", "vendor_id": "V-005", "amount": 34200.0, "category": "chemicals"},
    {"record_id": "SPD-006", "vendor_id": "V-006", "amount": 22000.0, "category": "waste_management"},
    {"record_id": "SPD-007", "vendor_id": "V-007", "amount": 15000.0, "category": "travel"},
    {"record_id": "SPD-008", "vendor_id": "V-008", "amount": 8500.0, "category": "it_services"},
    {"record_id": "SPD-009", "vendor_id": "V-009", "amount": 65000.0, "category": "facilities"},
    {"record_id": "SPD-010", "vendor_id": "V-010", "amount": 5000.0, "category": "other"},
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestClassifySpend:
    """Test classify_spend for all spend categories."""

    def test_raw_materials_maps_to_cat1(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "raw_materials"})
        assert result == "cat1_purchased_goods"

    def test_energy_maps_to_cat3(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "energy"})
        assert result == "cat3_fuel_energy"

    def test_transportation_maps_to_cat4(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "transportation"})
        assert result == "cat4_upstream_transport"

    def test_packaging_maps_to_cat1(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "packaging"})
        assert result == "cat1_purchased_goods"

    def test_waste_management_maps_to_cat5(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "waste_management"})
        assert result == "cat5_waste"

    def test_it_services_maps_to_cat1(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "it_services"})
        assert result == "cat1_purchased_goods"

    def test_professional_services_maps_to_cat1(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "professional_services"})
        assert result == "cat1_purchased_goods"

    def test_travel_maps_to_cat6(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "travel"})
        assert result == "cat6_business_travel"

    def test_facilities_maps_to_cat8(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "facilities"})
        assert result == "cat8_upstream_leased"

    def test_chemicals_maps_to_cat1(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "chemicals"})
        assert result == "cat1_purchased_goods"

    def test_capital_equipment_maps_to_cat2(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "capital_equipment"})
        assert result == "cat2_capital_goods"

    def test_unknown_category_returns_unclassified(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": "other"})
        assert result == "unclassified"

    def test_missing_category_returns_unclassified(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({})
        assert result == "unclassified"

    def test_empty_category_returns_unclassified(self):
        mapper = Scope3Mapper()
        result = mapper.classify_spend({"category": ""})
        assert result == "unclassified"


class TestVendorOverride:
    """Test vendor mapping overrides."""

    def test_vendor_override_takes_priority(self):
        mapper = Scope3Mapper()
        mapper.set_vendor_override("V-001", "cat2_capital_goods")
        result = mapper.classify_spend({"vendor_id": "V-001", "category": "raw_materials"})
        assert result == "cat2_capital_goods"

    def test_vendor_override_only_affects_vendor(self):
        mapper = Scope3Mapper()
        mapper.set_vendor_override("V-001", "cat2_capital_goods")
        result = mapper.classify_spend({"vendor_id": "V-002", "category": "raw_materials"})
        assert result == "cat1_purchased_goods"

    def test_remove_vendor_override(self):
        mapper = Scope3Mapper()
        mapper.set_vendor_override("V-001", "cat2_capital_goods")
        removed = mapper.remove_vendor_override("V-001")
        assert removed is True
        result = mapper.classify_spend({"vendor_id": "V-001", "category": "raw_materials"})
        assert result == "cat1_purchased_goods"

    def test_remove_nonexistent_override(self):
        mapper = Scope3Mapper()
        assert mapper.remove_vendor_override("V-UNKNOWN") is False

    def test_get_vendor_overrides(self):
        mapper = Scope3Mapper()
        mapper.set_vendor_override("V-001", "cat2_capital_goods")
        mapper.set_vendor_override("V-002", "cat6_business_travel")
        overrides = mapper.get_vendor_overrides()
        assert len(overrides) == 2

    def test_vendor_override_all_15_categories(self):
        mapper = Scope3Mapper()
        categories = [
            "cat1_purchased_goods", "cat2_capital_goods", "cat3_fuel_energy",
            "cat4_upstream_transport", "cat5_waste", "cat6_business_travel",
            "cat7_employee_commuting", "cat8_upstream_leased",
            "cat9_downstream_transport", "cat10_processing",
            "cat11_use_of_sold", "cat12_end_of_life",
            "cat13_downstream_leased", "cat14_franchises", "cat15_investments",
        ]
        for i, cat in enumerate(categories):
            mapper.set_vendor_override(f"V-{i:03d}", cat)
            result = mapper.classify_spend({"vendor_id": f"V-{i:03d}"})
            assert result == cat


class TestMaterialOverride:
    """Test material mapping overrides."""

    def test_material_override(self):
        mapper = Scope3Mapper()
        mapper.set_material_override("STEEL-HR-001", "cat2_capital_goods")
        result = mapper.classify_spend({"material": "STEEL-HR-001", "category": "raw_materials"})
        assert result == "cat2_capital_goods"

    def test_vendor_override_has_higher_priority(self):
        mapper = Scope3Mapper()
        mapper.set_vendor_override("V-001", "cat6_business_travel")
        mapper.set_material_override("STEEL-HR-001", "cat2_capital_goods")
        result = mapper.classify_spend({"vendor_id": "V-001", "material": "STEEL-HR-001", "category": "raw_materials"})
        assert result == "cat6_business_travel"

    def test_get_material_overrides(self):
        mapper = Scope3Mapper()
        mapper.set_material_override("MAT-001", "cat2_capital_goods")
        overrides = mapper.get_material_overrides()
        assert "MAT-001" in overrides


class TestClassifyBatch:
    """Test batch classification."""

    def test_batch_all_classified(self):
        mapper = Scope3Mapper()
        results = mapper.classify_batch(SAMPLE_SPEND)
        assert len(results) == 10
        for r in results:
            assert "scope3_category" in r

    def test_batch_preserves_original_fields(self):
        mapper = Scope3Mapper()
        results = mapper.classify_batch(SAMPLE_SPEND)
        assert results[0]["record_id"] == "SPD-001"
        assert results[0]["amount"] == 125000.0


class TestCoverage:
    """Test coverage analysis."""

    def test_coverage_with_all_classified(self):
        mapper = Scope3Mapper()
        records = [{"category": "raw_materials"}, {"category": "energy"}]
        coverage = mapper.get_coverage(records)
        assert coverage["classified"] == 2
        assert coverage["coverage_pct"] == 100.0

    def test_coverage_with_unclassified(self):
        mapper = Scope3Mapper()
        records = [{"category": "raw_materials"}, {"category": "other"}]
        coverage = mapper.get_coverage(records)
        assert coverage["classified"] == 1
        assert coverage["unclassified"] == 1
        assert coverage["coverage_pct"] == 50.0

    def test_coverage_empty(self):
        mapper = Scope3Mapper()
        coverage = mapper.get_coverage([])
        assert coverage["total"] == 0
        assert coverage["coverage_pct"] == 0.0

    def test_coverage_full_sample(self):
        mapper = Scope3Mapper()
        coverage = mapper.get_coverage(SAMPLE_SPEND)
        assert coverage["total"] == 10
        assert coverage["classified"] == 9
        assert coverage["unclassified"] == 1


class TestDistribution:
    """Test distribution summary."""

    def test_distribution_keys(self):
        mapper = Scope3Mapper()
        dist = mapper.get_distribution(SAMPLE_SPEND)
        assert "cat1_purchased_goods" in dist
        assert "cat4_upstream_transport" in dist

    def test_distribution_counts(self):
        mapper = Scope3Mapper()
        dist = mapper.get_distribution(SAMPLE_SPEND)
        assert dist["cat1_purchased_goods"]["count"] == 4

    def test_distribution_amounts(self):
        mapper = Scope3Mapper()
        dist = mapper.get_distribution(SAMPLE_SPEND)
        expected_cat1 = 125000.0 + 12300.0 + 34200.0 + 8500.0
        assert dist["cat1_purchased_goods"]["total_amount"] == pytest.approx(expected_cat1)

    def test_distribution_unclassified_present(self):
        mapper = Scope3Mapper()
        dist = mapper.get_distribution(SAMPLE_SPEND)
        assert "unclassified" in dist
        assert dist["unclassified"]["count"] == 1

    def test_distribution_with_override(self):
        mapper = Scope3Mapper()
        mapper.set_vendor_override("V-010", "cat15_investments")
        dist = mapper.get_distribution(SAMPLE_SPEND)
        assert "cat15_investments" in dist
        assert "unclassified" not in dist
