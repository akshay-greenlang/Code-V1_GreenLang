# -*- coding: utf-8 -*-
"""
Unit tests for CrosswalkEngine -- ISO 14064-1 to GHG Protocol mapping.

Tests crosswalk generation, scope breakdown, gap analysis,
dual-standard compliance check, reconciliation report, comparison
table, and constant mapping correctness with 25+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    DataQualityTier,
    ISOCategory,
)
from services.models import CategoryResult
from services.crosswalk_engine import (
    CrosswalkEngine,
    _ISO_TO_SCOPE,
    _ISO_TO_GHG_CATEGORIES,
    _GHG_CAT_TO_ISO,
    _GHG_PROTOCOL_REQUIREMENTS,
    _FRAMEWORK_DIFFERENCES,
)


# ===========================================================================
# Tests
# ===========================================================================


class TestMappingConstants:
    """Test crosswalk mapping constants."""

    def test_iso_to_scope_complete(self):
        for cat in ISOCategory:
            assert cat in _ISO_TO_SCOPE, f"Missing scope mapping for {cat.value}"

    def test_cat1_maps_to_scope_1(self):
        assert _ISO_TO_SCOPE[ISOCategory.CATEGORY_1_DIRECT] == "scope_1"

    def test_cat2_maps_to_scope_2(self):
        assert _ISO_TO_SCOPE[ISOCategory.CATEGORY_2_ENERGY] == "scope_2"

    def test_cats_3_to_6_map_to_scope_3(self):
        for cat in [
            ISOCategory.CATEGORY_3_TRANSPORT,
            ISOCategory.CATEGORY_4_PRODUCTS_USED,
            ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
            ISOCategory.CATEGORY_6_OTHER,
        ]:
            assert _ISO_TO_SCOPE[cat] == "scope_3"

    def test_ghg_protocol_categories_for_cat3(self):
        cats = _ISO_TO_GHG_CATEGORIES[ISOCategory.CATEGORY_3_TRANSPORT]
        assert len(cats) == 4
        assert any("Cat 4" in c for c in cats)
        assert any("Cat 6" in c for c in cats)

    def test_ghg_protocol_categories_for_cat4(self):
        cats = _ISO_TO_GHG_CATEGORIES[ISOCategory.CATEGORY_4_PRODUCTS_USED]
        assert len(cats) == 5
        assert any("Cat 1" in c for c in cats)

    def test_ghg_protocol_categories_for_cat5(self):
        cats = _ISO_TO_GHG_CATEGORIES[ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG]
        assert len(cats) == 5
        assert any("Cat 10" in c for c in cats)

    def test_ghg_protocol_categories_for_cat6(self):
        cats = _ISO_TO_GHG_CATEGORIES[ISOCategory.CATEGORY_6_OTHER]
        assert len(cats) == 1
        assert any("Cat 15" in c for c in cats)

    def test_reverse_mapping_cat4_to_iso_cat3(self):
        assert _GHG_CAT_TO_ISO["Cat 4"] == ISOCategory.CATEGORY_3_TRANSPORT

    def test_reverse_mapping_cat1_to_iso_cat4(self):
        assert _GHG_CAT_TO_ISO["Cat 1"] == ISOCategory.CATEGORY_4_PRODUCTS_USED

    def test_reverse_mapping_cat15_to_iso_cat6(self):
        assert _GHG_CAT_TO_ISO["Cat 15"] == ISOCategory.CATEGORY_6_OTHER

    def test_ghg_protocol_requirements_scope1_mandatory(self):
        assert _GHG_PROTOCOL_REQUIREMENTS["scope_1"] == "mandatory"

    def test_framework_differences_count(self):
        assert len(_FRAMEWORK_DIFFERENCES) == 6


class TestGenerateCrosswalk:
    """Test crosswalk generation from category results."""

    def test_generate_crosswalk(self, crosswalk_engine, sample_category_results):
        result = crosswalk_engine.generate_crosswalk(
            inventory_id="inv-1",
            org_id="org-1",
            reporting_year=2025,
            category_results=sample_category_results,
        )
        assert len(result.mappings) == 6
        assert result.org_id == "org-1"
        assert result.reporting_year == 2025

    def test_scope_totals_match(self, crosswalk_engine, sample_category_results):
        result = crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        # Scope 1 = Cat1 = 5000
        assert result.ghg_protocol_totals["scope_1"] == Decimal("5000")
        # Scope 2 = Cat2 = 3000
        assert result.ghg_protocol_totals["scope_2"] == Decimal("3000")
        # Scope 3 = Cat3+Cat4+Cat5+Cat6 = 1500+2000+800+200 = 4500
        assert result.ghg_protocol_totals["scope_3"] == Decimal("4500")

    def test_iso_totals_match(self, crosswalk_engine, sample_category_results):
        result = crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        assert result.iso_totals[ISOCategory.CATEGORY_1_DIRECT.value] == Decimal("5000")

    def test_reconciliation_difference_zero(self, crosswalk_engine, sample_category_results):
        result = crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        # Both frameworks should sum to the same total when no removals treated differently
        assert result.reconciliation_difference == Decimal("0")

    def test_crosswalk_cached(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        cached = crosswalk_engine.get_crosswalk("inv-1")
        assert cached is not None

    def test_crosswalk_not_cached_for_unknown(self, crosswalk_engine):
        assert crosswalk_engine.get_crosswalk("unknown") is None


class TestScopeBreakdown:
    """Test scope breakdown retrieval."""

    def test_scope_breakdown(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        breakdown = crosswalk_engine.get_scope_breakdown("inv-1")
        assert "scope_1" in breakdown
        assert "scope_2" in breakdown
        assert "scope_3" in breakdown

    def test_scope_breakdown_empty_for_unknown(self, crosswalk_engine):
        assert crosswalk_engine.get_scope_breakdown("unknown") == {}


class TestGapAnalysis:
    """Test gap analysis between frameworks."""

    def test_gap_analysis_with_full_data(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        result = crosswalk_engine.gap_analysis("inv-1")
        assert "iso_14064_coverage" in result
        assert "ghg_protocol_coverage" in result
        assert result["iso_14064_coverage"]["populated"] == 6
        assert len(result["ghg_protocol_coverage"]["gaps"]) == 0

    def test_gap_analysis_with_missing_categories(self, crosswalk_engine):
        partial_results = {
            ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("5000"),
                data_quality_tier=DataQualityTier.TIER_3,
            ),
            ISOCategory.CATEGORY_2_ENERGY.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_2_ENERGY,
                total_tco2e=Decimal("3000"),
                data_quality_tier=DataQualityTier.TIER_2,
            ),
        }
        crosswalk_engine.generate_crosswalk(
            "inv-2", "org-1", 2025, partial_results,
        )
        result = crosswalk_engine.gap_analysis("inv-2")
        # 4 categories missing (Cat 3-6)
        assert len(result["iso_14064_coverage"]["gaps"]) == 4
        # Scope 3 missing
        scope_gaps = result["ghg_protocol_coverage"]["gaps"]
        assert any(g["scope"] == "scope_3" for g in scope_gaps)

    def test_gap_analysis_no_crosswalk_returns_error(self, crosswalk_engine):
        result = crosswalk_engine.gap_analysis("inv-none")
        assert "error" in result

    def test_gap_analysis_includes_framework_differences(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        result = crosswalk_engine.gap_analysis("inv-1")
        assert "framework_differences" in result
        assert len(result["framework_differences"]) == 6


class TestDualStandardCompliance:
    """Test dual-standard compliance check."""

    def test_dual_compliant_full_data(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        result = crosswalk_engine.dual_standard_compliance_check("inv-1")
        assert result["iso_14064_compliance"]["compliant"] is True
        assert result["ghg_protocol_compliance"]["compliant"] is True
        assert result["dual_compliant"] is True

    def test_not_compliant_without_scope_3(self, crosswalk_engine):
        # Only Cat 1 and Cat 2 -> no Scope 3
        partial = {
            ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("5000"),
                data_quality_tier=DataQualityTier.TIER_3,
            ),
            ISOCategory.CATEGORY_2_ENERGY.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_2_ENERGY,
                total_tco2e=Decimal("3000"),
                data_quality_tier=DataQualityTier.TIER_2,
            ),
        }
        crosswalk_engine.generate_crosswalk("inv-2", "org-1", 2025, partial)
        result = crosswalk_engine.dual_standard_compliance_check("inv-2")
        # ISO requires at least one indirect category -> fails
        assert result["iso_14064_compliance"]["compliant"] is False

    def test_compliance_no_crosswalk_returns_error(self, crosswalk_engine):
        result = crosswalk_engine.dual_standard_compliance_check("inv-none")
        assert "error" in result


class TestReconciliationReport:
    """Test reconciliation report generation."""

    def test_reconciliation_report_structure(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        report = crosswalk_engine.generate_reconciliation_report("inv-1")
        assert "iso_14064_total_tco2e" in report
        assert "ghg_protocol_total_tco2e" in report
        assert "reconciliation_difference_tco2e" in report
        assert "scope_summary" in report
        assert "mapping_table" in report
        assert "provenance_hash" in report

    def test_reconciliation_totals_match(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        report = crosswalk_engine.generate_reconciliation_report("inv-1")
        # 5000+3000+1500+2000+800+200 = 12500
        assert report["iso_14064_total_tco2e"] == "12500"
        assert report["ghg_protocol_total_tco2e"] == "12500"
        assert report["reconciliation_difference_tco2e"] == "0"

    def test_reconciliation_no_crosswalk_returns_error(self, crosswalk_engine):
        result = crosswalk_engine.generate_reconciliation_report("inv-none")
        assert "error" in result


class TestComparisonTable:
    """Test human-readable comparison table."""

    def test_comparison_table_rows(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        table = crosswalk_engine.generate_comparison_table("inv-1")
        assert len(table) == 6  # One row per ISO category

    def test_comparison_table_row_fields(self, crosswalk_engine, sample_category_results):
        crosswalk_engine.generate_crosswalk(
            "inv-1", "org-1", 2025, sample_category_results,
        )
        table = crosswalk_engine.generate_comparison_table("inv-1")
        row = table[0]
        assert "iso_category" in row
        assert "iso_category_name" in row
        assert "ghg_scope" in row
        assert "ghg_categories" in row
        assert "tco2e" in row
        assert "notes" in row

    def test_comparison_table_empty_for_unknown(self, crosswalk_engine):
        assert crosswalk_engine.generate_comparison_table("unknown") == []
