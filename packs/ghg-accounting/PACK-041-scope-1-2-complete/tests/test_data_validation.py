# -*- coding: utf-8 -*-
"""
Unit tests for Data Validation -- PACK-041
=============================================

Tests input data validation, unit normalization, emission factor
cross-checking, boundary arithmetic, provenance chain integrity,
and fixture data consistency. These tests ensure that the reference
data used by all engines is internally consistent.

Coverage target: 85%+
Total tests: ~110
"""

import hashlib
import json
import math
from decimal import Decimal

import pytest


# =============================================================================
# Organization Data Validation
# =============================================================================


class TestOrganizationDataValidation:
    """Validate organization fixture data consistency."""

    def test_org_has_id(self, sample_organization):
        assert sample_organization["org_id"] == "ORG-ACME-001"

    def test_org_has_name(self, sample_organization):
        assert len(sample_organization["org_name"]) > 0

    def test_org_has_3_entities(self, sample_organization):
        assert len(sample_organization["entities"]) == 3

    def test_org_has_10_facilities(self, sample_organization):
        total_facs = sum(
            len(e["facilities"]) for e in sample_organization["entities"]
        )
        assert total_facs == 10

    def test_entity_equity_range(self, sample_organization):
        for entity in sample_organization["entities"]:
            pct = entity["equity_pct"]
            assert Decimal("0") <= pct <= Decimal("100")

    def test_entity_emissions_non_negative(self, sample_organization):
        for entity in sample_organization["entities"]:
            assert entity["total_scope1_tco2e"] >= Decimal("0")
            assert entity["total_scope2_tco2e"] >= Decimal("0")

    @pytest.mark.parametrize("entity_idx,expected_type", [
        (0, "wholly_owned"),
        (1, "wholly_owned"),
        (2, "joint_venture"),
    ])
    def test_entity_types(self, entity_idx, expected_type, sample_organization):
        assert sample_organization["entities"][entity_idx]["entity_type"] == expected_type

    @pytest.mark.parametrize("entity_idx,expected_country", [
        (0, "US"),
        (1, "DE"),
        (2, "JP"),
    ])
    def test_entity_countries(self, entity_idx, expected_country, sample_organization):
        assert sample_organization["entities"][entity_idx]["country_of_incorporation"] == expected_country

    def test_all_entities_active(self, sample_organization):
        for entity in sample_organization["entities"]:
            assert entity["is_active"] is True

    def test_facility_ids_unique(self, sample_organization):
        all_ids = []
        for entity in sample_organization["entities"]:
            for fac in entity["facilities"]:
                all_ids.append(fac["facility_id"])
        assert len(all_ids) == len(set(all_ids))

    def test_facility_scope1_non_negative(self, sample_organization):
        for entity in sample_organization["entities"]:
            for fac in entity["facilities"]:
                assert fac["scope1_emissions_tco2e"] >= Decimal("0")

    def test_facility_employee_count_positive(self, sample_organization):
        for entity in sample_organization["entities"]:
            for fac in entity["facilities"]:
                assert fac["employee_count"] >= 0


# =============================================================================
# Boundary Data Validation
# =============================================================================


class TestBoundaryDataValidation:
    """Validate boundary fixture data consistency."""

    def test_boundary_approach_valid(self, sample_boundary):
        assert sample_boundary["approach"] in {
            "equity_share", "operational_control", "financial_control"
        }

    def test_boundary_totals_non_negative(self, sample_boundary):
        assert sample_boundary["total_scope1_tco2e"] >= Decimal("0")
        assert sample_boundary["total_scope2_tco2e"] >= Decimal("0")

    def test_boundary_total_equals_sum(self, sample_boundary):
        s1 = sample_boundary["total_scope1_tco2e"]
        s2 = sample_boundary["total_scope2_tco2e"]
        total = sample_boundary["total_emissions_tco2e"]
        assert s1 + s2 == total

    def test_entity_counts_consistent(self, sample_boundary):
        included = sample_boundary["included_entities"]
        excluded = sample_boundary["excluded_entities"]
        partial = sample_boundary["partial_entities"]
        total = sample_boundary["total_entities"]
        assert included + excluded + partial == total

    def test_countries_sorted(self, sample_boundary):
        countries = sample_boundary["countries_covered"]
        assert countries == sorted(countries)


# =============================================================================
# Emission Factor Data Validation
# =============================================================================


class TestEmissionFactorDataValidation:
    """Validate emission factor fixture data consistency."""

    @pytest.mark.parametrize("fuel", [
        "natural_gas", "diesel", "petrol", "lpg",
    ])
    def test_fuel_has_defra_factors(self, fuel, sample_emission_factors):
        assert "defra_2025" in sample_emission_factors["fuels"][fuel]

    @pytest.mark.parametrize("fuel", ["natural_gas", "diesel"])
    def test_fuel_has_ipcc_factors(self, fuel, sample_emission_factors):
        assert "ipcc_2006" in sample_emission_factors["fuels"][fuel]

    @pytest.mark.parametrize("fuel", ["natural_gas", "diesel", "petrol", "lpg"])
    def test_defra_co2e_positive(self, fuel, sample_emission_factors):
        defra = sample_emission_factors["fuels"][fuel]["defra_2025"]
        co2e_key = [k for k in defra.keys() if "co2e" in k][0]
        assert defra[co2e_key] > Decimal("0")

    @pytest.mark.parametrize("grid", ["DE", "US_average", "US_ERCOT", "GB", "JP", "FR"])
    def test_grid_has_location_factor(self, grid, sample_emission_factors):
        gf = sample_emission_factors["grids"][grid]
        assert "location_based_kg_per_kwh" in gf
        assert gf["location_based_kg_per_kwh"] > Decimal("0")

    @pytest.mark.parametrize("grid", ["DE", "US_average", "GB", "JP", "FR"])
    def test_grid_has_source(self, grid, sample_emission_factors):
        assert "source" in sample_emission_factors["grids"][grid]

    @pytest.mark.parametrize("ref", ["R-410A", "R-134a", "R-404A", "R-32", "SF6"])
    def test_refrigerant_has_gwp_ar6(self, ref, sample_emission_factors):
        r = sample_emission_factors["refrigerants"][ref]
        assert "gwp_ar6" in r
        assert r["gwp_ar6"] > Decimal("0")


# =============================================================================
# Scope 1 Results Validation
# =============================================================================


class TestScope1ResultsValidation:
    """Validate scope 1 results fixture data consistency."""

    def test_eight_categories_present(self, sample_scope1_results):
        assert len(sample_scope1_results["categories"]) == 8

    @pytest.mark.parametrize("category", [
        "stationary_combustion", "mobile_combustion", "process_emissions",
        "fugitive_emissions", "refrigerant_fgas", "land_use",
        "waste_treatment", "agricultural",
    ])
    def test_category_has_total(self, category, sample_scope1_results):
        assert "total_tco2e" in sample_scope1_results["categories"][category]

    @pytest.mark.parametrize("category", [
        "stationary_combustion", "mobile_combustion", "process_emissions",
        "fugitive_emissions", "refrigerant_fgas", "land_use",
        "waste_treatment", "agricultural",
    ])
    def test_category_has_uncertainty(self, category, sample_scope1_results):
        assert "uncertainty_pct" in sample_scope1_results["categories"][category]

    @pytest.mark.parametrize("category", [
        "stationary_combustion", "mobile_combustion",
    ])
    def test_active_category_has_agent(self, category, sample_scope1_results):
        cat = sample_scope1_results["categories"][category]
        assert "agent" in cat
        assert cat["agent"].startswith("MRV-")

    def test_category_sum_equals_total(self, sample_scope1_results):
        cats = sample_scope1_results["categories"]
        total = sum(c["total_tco2e"] for c in cats.values())
        assert total == sample_scope1_results["total_scope1_tco2e"]


# =============================================================================
# Scope 2 Results Validation
# =============================================================================


class TestScope2ResultsValidation:
    """Validate scope 2 results fixture data consistency."""

    def test_location_based_present(self, sample_scope2_results):
        assert "location_based" in sample_scope2_results

    def test_market_based_present(self, sample_scope2_results):
        assert "market_based" in sample_scope2_results

    def test_variance_calculated(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]["total_tco2e"]
        mb = sample_scope2_results["market_based"]["total_tco2e"]
        variance = sample_scope2_results["variance_tco2e"]
        assert variance == lb - mb

    def test_variance_pct_calculated(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]["total_tco2e"]
        variance = sample_scope2_results["variance_tco2e"]
        expected_pct = variance / lb * Decimal("100")
        assert expected_pct == sample_scope2_results["variance_pct"]


# =============================================================================
# Instrument Data Validation
# =============================================================================


class TestInstrumentDataValidation:
    """Validate contractual instrument fixture data."""

    def test_three_instruments(self, sample_instruments):
        assert len(sample_instruments) == 3

    @pytest.mark.parametrize("idx,expected_type", [
        (0, "power_purchase_agreement"),
        (1, "renewable_energy_certificate"),
        (2, "guarantee_of_origin"),
    ])
    def test_instrument_types(self, idx, expected_type, sample_instruments):
        assert sample_instruments[idx]["type"] == expected_type

    def test_all_instruments_zero_ef(self, sample_instruments):
        for inst in sample_instruments:
            assert inst["emission_factor_kg_per_kwh"] == Decimal("0")

    def test_instrument_ids_unique(self, sample_instruments):
        ids = [inst["instrument_id"] for inst in sample_instruments]
        assert len(ids) == len(set(ids))


# =============================================================================
# Inventory Data Validation
# =============================================================================


class TestInventoryDataValidation:
    """Validate consolidated inventory fixture data."""

    def test_inventory_id_present(self, sample_inventory):
        assert sample_inventory["inventory_id"] == "INV-2025-001"

    def test_scope12_location_total(self, sample_inventory):
        s1 = sample_inventory["scope1"]["total_tco2e"]
        s2 = sample_inventory["scope2_location"]["total_tco2e"]
        total = sample_inventory["total_scope12_location"]
        assert s1 + s2 == total

    def test_scope12_market_total(self, sample_inventory):
        s1 = sample_inventory["scope1"]["total_tco2e"]
        s2 = sample_inventory["scope2_market"]["total_tco2e"]
        total = sample_inventory["total_scope12_market"]
        assert s1 + s2 == total

    def test_market_less_than_location(self, sample_inventory):
        assert sample_inventory["total_scope12_market"] <= sample_inventory["total_scope12_location"]

    def test_uncertainty_in_range(self, sample_inventory):
        u = sample_inventory["uncertainty_pct"]
        assert Decimal("0") < u < Decimal("100")

    def test_quality_score_in_range(self, sample_inventory):
        q = sample_inventory["data_quality_score"]
        assert Decimal("0") <= q <= Decimal("100")

    def test_completeness_in_range(self, sample_inventory):
        c = sample_inventory["completeness_pct"]
        assert Decimal("0") <= c <= Decimal("100")


# =============================================================================
# GWP Data Validation
# =============================================================================


class TestGWPDataValidation:
    """Validate GWP values fixture data."""

    def test_gwp_has_10_gases(self, sample_gwp_values):
        assert len(sample_gwp_values) == 10

    @pytest.mark.parametrize("gas", [
        "CO2", "CH4", "N2O", "SF6", "NF3",
        "HFC-134a", "HFC-32", "R-410A", "CF4", "C2F6",
    ])
    def test_gas_present(self, gas, sample_gwp_values):
        assert gas in sample_gwp_values

    @pytest.mark.parametrize("gas", [
        "CO2", "CH4", "N2O", "SF6", "NF3",
        "HFC-134a", "HFC-32", "R-410A", "CF4", "C2F6",
    ])
    def test_gas_has_three_generations(self, gas, sample_gwp_values):
        assert len(sample_gwp_values[gas]) == 3

    def test_sf6_highest_gwp_among_simple_gases(self, sample_gwp_values):
        simple_gases = ["CO2", "CH4", "N2O", "SF6", "NF3"]
        max_gas = max(simple_gases, key=lambda g: sample_gwp_values[g]["ar6"])
        assert max_gas == "SF6"


# =============================================================================
# Yearly Trend Data Validation
# =============================================================================


class TestYearlyDataValidation:
    """Validate multi-year trend fixture data."""

    def test_three_years(self, sample_yearly_data):
        assert len(sample_yearly_data) == 3

    def test_years_sequential(self, sample_yearly_data):
        years = [yr["year"] for yr in sample_yearly_data]
        assert years == [2023, 2024, 2025]

    def test_all_fields_present(self, sample_yearly_data):
        required = [
            "year", "total_scope1_tco2e", "total_scope2_location_tco2e",
            "total_scope2_market_tco2e", "revenue_million_usd",
            "employee_count", "floor_area_m2",
        ]
        for yr in sample_yearly_data:
            for field in required:
                assert field in yr, f"Missing {field} in year {yr['year']}"

    @pytest.mark.parametrize("field", [
        "total_scope1_tco2e",
        "total_scope2_location_tco2e",
        "total_scope2_market_tco2e",
        "revenue_million_usd",
    ])
    def test_yearly_field_positive(self, field, sample_yearly_data):
        for yr in sample_yearly_data:
            assert yr[field] > Decimal("0")


# =============================================================================
# Base Year Data Validation
# =============================================================================


class TestBaseYearDataValidation:
    """Validate base year fixture data."""

    def test_base_year_is_2019(self, sample_base_year):
        assert sample_base_year["base_year"] == 2019

    def test_scope1_positive(self, sample_base_year):
        assert sample_base_year["total_scope1_tco2e"] > Decimal("0")

    def test_scope12_consistent(self, sample_base_year):
        s1 = sample_base_year["total_scope1_tco2e"]
        s2_lb = sample_base_year["total_scope2_location_tco2e"]
        total = sample_base_year["total_scope12_location_tco2e"]
        assert s1 + s2_lb == total

    def test_intensity_per_revenue(self, sample_base_year):
        expected = sample_base_year["total_scope1_tco2e"] / sample_base_year["revenue_million_usd"]
        assert expected == pytest.approx(sample_base_year["scope1_intensity_per_revenue"], abs=Decimal("0.01"))


# =============================================================================
# Provenance Hash Integrity
# =============================================================================


class TestProvenanceHashIntegrity:
    """Validate provenance hash computation integrity."""

    def test_hash_is_sha256(self, sample_inventory):
        from tests.conftest import compute_provenance_hash
        h = compute_provenance_hash(sample_inventory)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic_across_calls(self, sample_inventory):
        from tests.conftest import compute_provenance_hash
        hashes = [compute_provenance_hash(sample_inventory) for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_hash_sensitive_to_single_change(self, sample_inventory):
        from tests.conftest import compute_provenance_hash
        h1 = compute_provenance_hash(sample_inventory)
        modified = dict(sample_inventory)
        modified["reporting_year"] = 9999
        h2 = compute_provenance_hash(modified)
        assert h1 != h2

    def test_hash_order_independent(self):
        """JSON sort_keys=True ensures order independence."""
        from tests.conftest import compute_provenance_hash
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert compute_provenance_hash(d1) == compute_provenance_hash(d2)
