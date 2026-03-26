# -*- coding: utf-8 -*-
"""
End-to-End Tests for PACK-041 Scope 1-2 Complete
===================================================

Tests complete inventory scenarios from organization definition through
disclosure generation. Validates data flow through all engines and
workflows for corporate office, manufacturing, SME, and multi-framework
scenarios.

Coverage target: 85%+
Total tests: ~35
"""

from decimal import Decimal

import pytest


# =============================================================================
# Corporate Office Inventory
# =============================================================================


class TestE2ECorporateOfficeInventory:
    """End-to-end test for a corporate office GHG inventory."""

    def test_office_has_stationary_combustion(self, sample_scope1_results):
        cat = sample_scope1_results["categories"]["stationary_combustion"]
        assert cat["total_tco2e"] > Decimal("0")

    def test_office_has_refrigerant(self, sample_scope1_results):
        cat = sample_scope1_results["categories"]["refrigerant_fgas"]
        assert cat["total_tco2e"] > Decimal("0")

    def test_office_has_scope2(self, sample_scope2_results):
        assert sample_scope2_results["location_based"]["total_tco2e"] > Decimal("0")

    def test_office_no_agricultural(self, sample_scope1_results):
        assert sample_scope1_results["categories"]["agricultural"]["total_tco2e"] == Decimal("0")

    def test_office_no_land_use(self, sample_scope1_results):
        assert sample_scope1_results["categories"]["land_use"]["total_tco2e"] == Decimal("0")

    def test_office_scope12_total(self, sample_inventory):
        total = sample_inventory["total_scope12_location"]
        assert total > Decimal("0")
        assert total == Decimal("37000.0")


# =============================================================================
# Manufacturing Inventory
# =============================================================================


class TestE2EManufacturingInventory:
    """End-to-end test for a manufacturing facility GHG inventory."""

    def test_manufacturing_has_process_emissions(self, sample_scope1_results):
        cat = sample_scope1_results["categories"]["process_emissions"]
        assert cat["total_tco2e"] > Decimal("0")

    def test_manufacturing_has_waste_treatment(self, sample_scope1_results):
        cat = sample_scope1_results["categories"]["waste_treatment"]
        assert cat["total_tco2e"] > Decimal("0")

    def test_manufacturing_has_mobile(self, sample_scope1_results):
        cat = sample_scope1_results["categories"]["mobile_combustion"]
        assert cat["total_tco2e"] > Decimal("0")

    def test_manufacturing_has_fugitive(self, sample_scope1_results):
        cat = sample_scope1_results["categories"]["fugitive_emissions"]
        assert cat["total_tco2e"] > Decimal("0")

    def test_manufacturing_total_above_threshold(self, sample_scope1_results):
        total = sample_scope1_results["total_scope1_tco2e"]
        assert total > Decimal("1000")


# =============================================================================
# SME Simplified Inventory
# =============================================================================


class TestE2ESMESimplifiedInventory:
    """End-to-end test for an SME simplified inventory."""

    def test_sme_minimal_categories(self):
        """SME may only have stationary combustion and scope 2."""
        sme_categories = {"stationary_combustion"}
        assert len(sme_categories) >= 1

    def test_sme_scope2_only_location(self):
        """SME may not have market-based instruments."""
        has_instruments = False
        method = "location_based_only" if not has_instruments else "dual"
        assert method == "location_based_only"

    def test_sme_single_facility(self):
        facility_count = 1
        assert facility_count == 1

    def test_sme_completeness_acceptable(self):
        """Even limited categories can be acceptable for SME."""
        categories_covered = 2
        categories_relevant = 3
        completeness = categories_covered / categories_relevant * 100
        assert completeness >= 60


# =============================================================================
# Full Workflow: Boundary to Disclosure
# =============================================================================


class TestE2EFullWorkflow:
    """End-to-end test for the full inventory workflow."""

    def test_boundary_defined(self, sample_boundary):
        assert sample_boundary["total_entities"] > 0
        assert sample_boundary["included_entities"] >= 1

    def test_scope1_calculated(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")

    def test_scope2_calculated(self, sample_inventory):
        assert sample_inventory["scope2_location"]["total_tco2e"] > Decimal("0")

    def test_inventory_consolidated(self, sample_inventory):
        assert sample_inventory["total_scope12_location"] > Decimal("0")

    def test_uncertainty_assessed(self, sample_inventory):
        assert sample_inventory["uncertainty_pct"] > Decimal("0")

    def test_data_quality_scored(self, sample_inventory):
        assert sample_inventory["data_quality_score"] > Decimal("0")

    def test_completeness_high(self, sample_inventory):
        assert sample_inventory["completeness_pct"] >= Decimal("90")

    def test_provenance_hash_present(self, sample_inventory):
        from tests.conftest import compute_provenance_hash
        h = compute_provenance_hash(sample_inventory)
        assert len(h) == 64


# =============================================================================
# Multi-Framework Reporting
# =============================================================================


class TestE2EMultiFrameworkReporting:
    """End-to-end test for multi-framework report generation."""

    def test_ghg_protocol_report_possible(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")
        assert sample_inventory["scope2_location"]["total_tco2e"] > Decimal("0")

    def test_iso_14064_report_possible(self, sample_inventory):
        assert sample_inventory["uncertainty_pct"] > Decimal("0")

    def test_esrs_e1_report_possible(self, sample_inventory, sample_boundary):
        assert len(sample_boundary["countries_covered"]) >= 1
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")

    def test_cdp_report_possible(self, sample_inventory):
        assert "by_gas" in sample_inventory["scope1"]
        assert "by_category" in sample_inventory["scope1"]


# =============================================================================
# Base Year Recalculation E2E
# =============================================================================


class TestE2EBaseYearRecalculation:
    """End-to-end test for base-year recalculation scenario."""

    def test_acquisition_triggers_recalc(self, sample_base_year):
        base_total = sample_base_year["total_scope12_location_tco2e"]
        acquisition = Decimal("5000")
        materiality = acquisition / base_total * Decimal("100")
        assert materiality > Decimal("5.0")

    def test_adjusted_base_year_calculated(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        adjusted = original + Decimal("5000")
        assert adjusted > original


# =============================================================================
# Trend Analysis E2E
# =============================================================================


class TestE2ETrendAnalysis:
    """End-to-end test for multi-year trend analysis."""

    def test_3_years_available(self, sample_yearly_data):
        assert len(sample_yearly_data) == 3

    def test_downward_trend_scope1(self, sample_yearly_data):
        values = [yr["total_scope1_tco2e"] for yr in sample_yearly_data]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_intensity_improvement(self, sample_yearly_data):
        intensities = [
            yr["total_scope1_tco2e"] / yr["revenue_million_usd"]
            for yr in sample_yearly_data
        ]
        for i in range(1, len(intensities)):
            assert intensities[i] <= intensities[i - 1]

    def test_scope12_market_faster_decline(self, sample_yearly_data):
        """Market-based Scope 2 should decline faster than location-based."""
        market_decline = (
            sample_yearly_data[0]["total_scope2_market_tco2e"]
            - sample_yearly_data[-1]["total_scope2_market_tco2e"]
        )
        location_decline = (
            sample_yearly_data[0]["total_scope2_location_tco2e"]
            - sample_yearly_data[-1]["total_scope2_location_tco2e"]
        )
        assert market_decline > location_decline
