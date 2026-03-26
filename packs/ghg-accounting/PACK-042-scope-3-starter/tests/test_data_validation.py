# -*- coding: utf-8 -*-
"""
Unit tests for PACK-042 Fixture Data Consistency
===================================================

Validates fixture data consistency: organization profiles, spend
transaction validity, category result arithmetic, supplier data,
EEIO factor checks, DQR ranges, uncertainty bounds, provenance hash
format, and cross-fixture consistency.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.conftest import (
    SCOPE3_CATEGORIES,
    UPSTREAM_CATEGORIES,
    DOWNSTREAM_CATEGORIES,
    compute_provenance_hash,
)


# =============================================================================
# Organization Profile Validation
# =============================================================================


class TestOrganizationProfileValidation:
    """Validate organization profile fixtures."""

    @pytest.mark.parametrize("org_fixture", [
        "manufacturing_org",
        "retail_org",
        "technology_org",
        "financial_org",
        "sme_org",
    ])
    def test_org_has_required_fields(self, org_fixture, request):
        org = request.getfixturevalue(org_fixture)
        required = ["org_id", "org_name", "sector", "country", "reporting_year"]
        for field in required:
            assert field in org, f"{org_fixture} missing {field}"

    def test_manufacturing_org_sector(self, manufacturing_org):
        assert manufacturing_org["sector"] == "MANUFACTURING"

    def test_retail_org_sector(self, retail_org):
        assert retail_org["sector"] == "RETAIL"

    def test_technology_org_sector(self, technology_org):
        assert technology_org["sector"] == "TECHNOLOGY"

    def test_financial_org_sector(self, financial_org):
        assert financial_org["sector"] == "FINANCIAL"

    def test_sme_org_sector(self, sme_org):
        assert sme_org["sector"] == "SME"

    def test_all_orgs_have_positive_year(self, manufacturing_org, retail_org, technology_org):
        for org in [manufacturing_org, retail_org, technology_org]:
            assert org["reporting_year"] >= 2020

    def test_all_orgs_have_country_code(self, manufacturing_org, retail_org, sme_org):
        for org in [manufacturing_org, retail_org, sme_org]:
            assert len(org["country"]) == 2
            assert org["country"].isalpha()
            assert org["country"].isupper()


# =============================================================================
# Spend Transaction Validation
# =============================================================================


class TestSpendTransactionValidation:
    """Validate spend transaction data."""

    def test_spend_data_not_empty(self, sample_spend_data):
        assert len(sample_spend_data) >= 100

    def test_all_transactions_have_id(self, sample_spend_data):
        ids = set()
        for txn in sample_spend_data:
            assert "transaction_id" in txn
            assert txn["transaction_id"] not in ids, f"Duplicate ID: {txn['transaction_id']}"
            ids.add(txn["transaction_id"])

    def test_all_transactions_have_positive_amount(self, sample_spend_data):
        for txn in sample_spend_data:
            assert txn["amount_eur"] > 0, (
                f"Transaction {txn['transaction_id']} has non-positive amount"
            )

    def test_all_transactions_have_category(self, sample_spend_data):
        for txn in sample_spend_data:
            assert txn["scope3_category"] in SCOPE3_CATEGORIES, (
                f"Transaction {txn['transaction_id']} has invalid category {txn['scope3_category']}"
            )

    def test_all_transactions_have_eeio_sector(self, sample_spend_data):
        for txn in sample_spend_data:
            assert "eeio_sector" in txn
            assert len(txn["eeio_sector"]) > 0

    def test_all_transactions_have_date(self, sample_spend_data):
        for txn in sample_spend_data:
            assert "date" in txn
            assert len(txn["date"]) == 10  # YYYY-MM-DD format

    def test_total_spend_reasonable(self, sample_spend_data):
        total = sum(txn["amount_eur"] for txn in sample_spend_data)
        # Should be in millions for a manufacturing company
        assert total > Decimal("1000000"), "Total spend should be > 1M EUR"

    def test_cat1_has_most_spend(self, sample_spend_data):
        cat1_spend = sum(
            txn["amount_eur"] for txn in sample_spend_data
            if txn["scope3_category"] == "CAT_1"
        )
        total_spend = sum(txn["amount_eur"] for txn in sample_spend_data)
        pct = float(cat1_spend / total_spend * 100)
        assert pct > 30, f"Cat 1 should be > 30% of spend, got {pct}%"


# =============================================================================
# Category Result Arithmetic Validation
# =============================================================================


class TestCategoryResultArithmetic:
    """Validate category result arithmetic consistency."""

    def test_gas_breakdown_sums_to_total(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            gas_sum = sum(data["by_gas"].values())
            assert gas_sum == data["total_tco2e"], (
                f"{cat_id}: gas breakdown {gas_sum} != total {data['total_tco2e']}"
            )

    def test_total_scope3_equals_category_sum(self, sample_category_results):
        cats = sample_category_results["categories"]
        calculated = sum(cats[c]["total_tco2e"] for c in SCOPE3_CATEGORIES)
        assert calculated == sample_category_results["total_scope3_tco2e"]

    def test_all_emissions_non_negative(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            assert data["total_tco2e"] >= 0, f"{cat_id} has negative emissions"
            for gas, val in data["by_gas"].items():
                assert val >= 0, f"{cat_id}.{gas} has negative value"

    def test_uncertainty_pct_non_negative(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            assert data["uncertainty_pct"] >= 0

    def test_source_count_non_negative(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            assert data["source_count"] >= 0


# =============================================================================
# Supplier Data Completeness Validation
# =============================================================================


class TestSupplierDataCompleteness:
    """Validate supplier data completeness."""

    def test_supplier_data_has_20_plus(self, sample_supplier_data):
        assert len(sample_supplier_data) >= 20

    def test_all_suppliers_have_required_fields(self, sample_supplier_data):
        required = [
            "supplier_id", "name", "category", "spend_eur",
            "emissions_tco2e", "engagement_status", "data_quality_level",
        ]
        for s in sample_supplier_data:
            for field in required:
                assert field in s, f"Supplier {s.get('supplier_id', 'unknown')} missing {field}"

    def test_all_suppliers_have_positive_spend(self, sample_supplier_data):
        for s in sample_supplier_data:
            assert s["spend_eur"] > 0

    def test_all_suppliers_have_positive_emissions(self, sample_supplier_data):
        for s in sample_supplier_data:
            assert s["emissions_tco2e"] > 0

    def test_supplier_categories_valid(self, sample_supplier_data):
        for s in sample_supplier_data:
            assert s["category"] in SCOPE3_CATEGORIES

    def test_supplier_quality_levels_valid(self, sample_supplier_data):
        valid_levels = {"LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5"}
        for s in sample_supplier_data:
            assert s["data_quality_level"] in valid_levels


# =============================================================================
# EEIO Factor Sanity Checks
# =============================================================================


class TestEEIOFactorSanity:
    """Validate EEIO emission factor sanity."""

    def test_all_factors_positive(self, sample_eeio_factors):
        for sector, factor in sample_eeio_factors.items():
            assert factor > 0, f"Factor for {sector} should be positive"

    def test_all_factors_below_10(self, sample_eeio_factors):
        for sector, factor in sample_eeio_factors.items():
            assert factor < 10, f"Factor for {sector} ({factor}) seems too high"

    def test_high_intensity_sectors_above_1(self, sample_eeio_factors):
        high_sectors = ["basic_metals", "air_transport", "electricity_gas_steam"]
        for sector in high_sectors:
            if sector in sample_eeio_factors:
                assert sample_eeio_factors[sector] > 1.0

    def test_low_intensity_sectors_below_05(self, sample_eeio_factors):
        low_sectors = ["financial_services", "insurance", "it_services"]
        for sector in low_sectors:
            if sector in sample_eeio_factors:
                assert sample_eeio_factors[sector] < 0.5


# =============================================================================
# DQR Score Range Validation
# =============================================================================


class TestDQRScoreRange:
    """Validate DQR scores are within valid range."""

    def test_overall_dqr_in_range(self, sample_data_quality):
        dqr = sample_data_quality["overall_dqr"]
        assert Decimal("1.0") <= dqr <= Decimal("5.0")

    def test_per_category_dqr_in_range(self, sample_data_quality):
        for cat_id, data in sample_data_quality["categories"].items():
            assert Decimal("1.0") <= data["dqr"] <= Decimal("5.0"), (
                f"{cat_id} DQR {data['dqr']} out of range"
            )

    def test_dqi_scores_in_range(self, sample_data_quality):
        for cat_id, data in sample_data_quality["categories"].items():
            for dqi_name, score in data["dqi"].items():
                assert Decimal("1.0") <= score <= Decimal("5.0"), (
                    f"{cat_id}.{dqi_name} = {score} out of range"
                )


# =============================================================================
# Uncertainty Bounds Validation
# =============================================================================


class TestUncertaintyBoundsValidation:
    """Validate uncertainty bounds are consistent."""

    def test_lower_less_than_point(self, sample_uncertainty_results):
        total = sample_uncertainty_results["total_scope3"]
        assert total["lower_bound_tco2e"] < total["point_estimate_tco2e"]

    def test_upper_greater_than_point(self, sample_uncertainty_results):
        total = sample_uncertainty_results["total_scope3"]
        assert total["upper_bound_tco2e"] > total["point_estimate_tco2e"]

    def test_per_category_bounds(self, sample_uncertainty_results):
        for cat_id, data in sample_uncertainty_results["per_category"].items():
            assert data["lower"] < data["point"], f"{cat_id} lower >= point"
            assert data["upper"] > data["point"], f"{cat_id} upper <= point"

    def test_bounds_positive(self, sample_uncertainty_results):
        total = sample_uncertainty_results["total_scope3"]
        assert total["lower_bound_tco2e"] > 0
        assert total["upper_bound_tco2e"] > 0


# =============================================================================
# Provenance Hash Format Validation
# =============================================================================


class TestProvenanceHashFormat:
    """Validate provenance hash format across fixtures."""

    @pytest.mark.parametrize("fixture_name", [
        "sample_screening_results",
        "sample_double_counting_results",
        "sample_hotspot_analysis",
        "sample_data_quality",
        "sample_uncertainty_results",
        "sample_compliance_results",
    ])
    def test_provenance_hash_is_64_char_hex(self, fixture_name, request):
        fixture = request.getfixturevalue(fixture_name)
        h = fixture["provenance_hash"]
        assert len(h) == 64, f"{fixture_name} hash length {len(h)} != 64"
        try:
            int(h, 16)
        except ValueError:
            pytest.fail(f"{fixture_name} hash is not valid hex")

    def test_compute_hash_returns_64_char(self):
        h = compute_provenance_hash({"test": True})
        assert len(h) == 64


# =============================================================================
# Cross-Fixture Consistency
# =============================================================================


class TestCrossFixtureConsistency:
    """Validate consistency between fixtures."""

    def test_screening_categories_match_results(
        self, sample_screening_results, sample_category_results
    ):
        screening_cats = set(sample_screening_results["categories"].keys())
        result_cats = set(sample_category_results["categories"].keys())
        assert screening_cats == result_cats

    def test_supplier_categories_subset_of_all(self, sample_supplier_data):
        supplier_cats = set(s["category"] for s in sample_supplier_data)
        for cat in supplier_cats:
            assert cat in SCOPE3_CATEGORIES

    def test_inventory_total_matches_results(
        self, sample_consolidated_inventory, sample_category_results
    ):
        inv_total = sample_consolidated_inventory["total_scope3_tco2e"]
        res_total = sample_category_results["total_scope3_tco2e"]
        assert inv_total == res_total

    def test_org_scope12_matches_inventory(
        self, manufacturing_org, sample_consolidated_inventory
    ):
        org_s1 = manufacturing_org["scope1_tco2e"]
        inv_s1 = sample_consolidated_inventory["scope1_tco2e"]
        assert org_s1 == inv_s1
