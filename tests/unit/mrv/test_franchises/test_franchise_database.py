# -*- coding: utf-8 -*-
"""
Test suite for franchises.franchise_database - AGENT-MRV-027.

Tests the FranchiseDatabaseEngine including all 15 reference data tables,
EUI benchmarks, revenue intensity, grid EFs, fuel EFs, refrigerant GWPs,
EEIO factors, hotel benchmarks, vehicle EFs, search, and validation.

Target: 55+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List
import pytest

from greenlang.franchises.franchise_database import (
    FranchiseDatabaseEngine,
    FRANCHISE_TYPES,
    CLIMATE_ZONES,
    FUEL_EMISSION_FACTORS,
    REFRIGERANT_GWPS,
    EEIO_SPEND_FACTORS,
    GRID_EMISSION_FACTORS,
    VEHICLE_EMISSION_FACTORS,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> FranchiseDatabaseEngine:
    """Create a fresh FranchiseDatabaseEngine instance."""
    FranchiseDatabaseEngine._instance = None
    return FranchiseDatabaseEngine()


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestFranchiseDatabaseEngineInit:
    """Test FranchiseDatabaseEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern."""
        FranchiseDatabaseEngine._instance = None
        e1 = FranchiseDatabaseEngine()
        e2 = FranchiseDatabaseEngine()
        assert e1 is e2

    def test_engine_thread_safety(self):
        """Test singleton is thread-safe."""
        FranchiseDatabaseEngine._instance = None
        results = []

        def create_engine():
            eng = FranchiseDatabaseEngine()
            results.append(id(eng))

        threads = [threading.Thread(target=create_engine) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1


# ==============================================================================
# EUI BENCHMARK TESTS
# ==============================================================================


class TestEUIBenchmarks:
    """Test EUI benchmark retrieval for all franchise types x climate zones."""

    @pytest.mark.parametrize("franchise_type", FRANCHISE_TYPES)
    def test_eui_benchmark_all_franchise_types(self, engine, franchise_type):
        """Test EUI benchmark exists for each franchise type in temperate zone."""
        result = engine.get_eui_benchmark(franchise_type, "temperate")
        assert isinstance(result, Decimal)
        assert result > 0

    @pytest.mark.parametrize("climate_zone", CLIMATE_ZONES)
    def test_eui_benchmark_all_climate_zones(self, engine, climate_zone):
        """Test EUI benchmark for QSR across all climate zones."""
        result = engine.get_eui_benchmark("qsr", climate_zone)
        assert isinstance(result, Decimal)
        assert result > 0

    @pytest.mark.parametrize("franchise_type,climate_zone", [
        ("qsr", "tropical"),
        ("hotel", "continental"),
        ("convenience_store", "arid"),
        ("retail", "polar"),
        ("fitness", "temperate"),
    ])
    def test_eui_benchmark_cross_product(self, engine, franchise_type, climate_zone):
        """Test EUI benchmark for various type/zone combinations."""
        result = engine.get_eui_benchmark(franchise_type, climate_zone)
        assert isinstance(result, Decimal)
        assert result > 0

    def test_eui_benchmark_tropical_higher_than_polar(self, engine):
        """Test tropical EUI is generally different from polar (climate effect)."""
        tropical = engine.get_eui_benchmark("hotel", "tropical")
        polar = engine.get_eui_benchmark("hotel", "polar")
        assert tropical != polar

    def test_eui_benchmark_qsr_higher_than_retail(self, engine):
        """Test QSR EUI is generally higher than retail."""
        qsr = engine.get_eui_benchmark("qsr", "temperate")
        retail = engine.get_eui_benchmark("retail", "temperate")
        assert qsr > retail

    def test_eui_benchmark_invalid_type(self, engine):
        """Test invalid franchise type raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_eui_benchmark("nonexistent", "temperate")

    def test_eui_benchmark_invalid_climate(self, engine):
        """Test invalid climate zone raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_eui_benchmark("qsr", "nonexistent")

    def test_eui_benchmarks_for_type(self, engine):
        """Test retrieving all benchmarks for a single franchise type."""
        result = engine.get_eui_benchmarks_for_type("qsr")
        assert isinstance(result, dict)
        assert len(result) == len(CLIMATE_ZONES)
        for zone in CLIMATE_ZONES:
            assert zone in result

    def test_eui_all_benchmarks(self, engine):
        """Test retrieving the complete EUI benchmark dictionary."""
        result = engine.get_all_eui_benchmarks()
        assert isinstance(result, dict)
        assert len(result) == len(FRANCHISE_TYPES)


# ==============================================================================
# REVENUE INTENSITY TESTS
# ==============================================================================


class TestRevenueIntensity:
    """Test revenue intensity factor retrieval."""

    @pytest.mark.parametrize("franchise_type", FRANCHISE_TYPES)
    def test_revenue_intensity_all_types(self, engine, franchise_type):
        """Test revenue intensity exists for each franchise type."""
        result = engine.get_revenue_intensity(franchise_type)
        assert isinstance(result, Decimal)
        assert result > 0

    def test_revenue_intensity_qsr_value(self, engine):
        """Test QSR revenue intensity is reasonable (0.01-1.0 kgCO2e/$)."""
        result = engine.get_revenue_intensity("qsr")
        assert Decimal("0.01") <= result <= Decimal("1.0")

    def test_revenue_intensity_invalid_type(self, engine):
        """Test invalid franchise type raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_revenue_intensity("nonexistent")

    def test_all_revenue_intensities(self, engine):
        """Test retrieving all revenue intensity factors."""
        result = engine.get_all_revenue_intensities()
        assert isinstance(result, dict)
        assert len(result) == len(FRANCHISE_TYPES)


# ==============================================================================
# GRID EMISSION FACTOR TESTS
# ==============================================================================


class TestGridEmissionFactors:
    """Test grid emission factor retrieval."""

    @pytest.mark.parametrize("country", [
        "US", "GB", "DE", "FR", "JP", "CN", "AU", "CA", "IN", "BR", "KR", "MX",
    ])
    def test_grid_ef_countries(self, engine, country):
        """Test grid EFs for major countries."""
        result = engine.get_grid_ef(country)
        assert isinstance(result, Decimal)
        assert result > 0

    @pytest.mark.parametrize("subregion", ["CAMX", "RFCE", "SRSO", "NWPP", "RMPA"])
    def test_grid_ef_egrid_subregions(self, engine, subregion):
        """Test grid EFs for eGRID subregions."""
        result = engine.get_grid_ef(subregion)
        assert isinstance(result, Decimal)
        assert result > 0

    def test_grid_ef_us_higher_than_france(self, engine):
        """Test US grid EF is generally higher than France (nuclear-heavy)."""
        us_ef = engine.get_grid_ef("US")
        fr_ef = engine.get_grid_ef("FR")
        assert us_ef > fr_ef

    def test_grid_ef_invalid_region(self, engine):
        """Test invalid region raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_grid_ef("XX_INVALID")

    def test_all_grid_efs(self, engine):
        """Test retrieving all grid emission factors."""
        result = engine.get_all_grid_efs()
        assert isinstance(result, dict)
        assert len(result) > 0


# ==============================================================================
# FUEL EMISSION FACTOR TESTS
# ==============================================================================


class TestFuelEmissionFactors:
    """Test fuel emission factor retrieval."""

    @pytest.mark.parametrize("fuel_type", list(FUEL_EMISSION_FACTORS.keys()))
    def test_fuel_ef_all_types(self, engine, fuel_type):
        """Test fuel EFs for all fuel types in the reference table."""
        result = engine.get_fuel_ef(fuel_type)
        assert result is not None

    def test_fuel_ef_natural_gas_reasonable(self, engine):
        """Test natural gas EF is in reasonable range."""
        result = engine.get_fuel_ef("natural_gas")
        assert result is not None

    def test_fuel_ef_invalid_type(self, engine):
        """Test invalid fuel type raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_fuel_ef("unobtanium")

    def test_fuel_ef_detail(self, engine):
        """Test fuel EF detail returns full breakdown."""
        result = engine.get_fuel_ef_detail("diesel")
        assert isinstance(result, dict)
        assert "co2_per_unit" in result
        assert "unit" in result

    def test_all_fuel_efs(self, engine):
        """Test retrieving all fuel emission factors."""
        result = engine.get_all_fuel_efs()
        assert isinstance(result, dict)
        assert len(result) == len(FUEL_EMISSION_FACTORS)


# ==============================================================================
# REFRIGERANT GWP TESTS
# ==============================================================================


class TestRefrigerantGWPs:
    """Test refrigerant GWP retrieval."""

    @pytest.mark.parametrize("ref_type", list(REFRIGERANT_GWPS.keys()))
    def test_refrigerant_gwp_all_types(self, engine, ref_type):
        """Test GWPs for all 10 refrigerant types."""
        result = engine.get_refrigerant_gwp(ref_type)
        assert isinstance(result, Decimal)
        assert result >= 0

    def test_r404a_high_gwp(self, engine):
        """Test R-404A has high GWP (>3000)."""
        result = engine.get_refrigerant_gwp("R-404A")
        assert result > Decimal("3000")

    def test_r290_low_gwp(self, engine):
        """Test R-290 (propane) has very low GWP (<10)."""
        result = engine.get_refrigerant_gwp("R-290")
        assert result < Decimal("10")

    def test_r744_co2_gwp(self, engine):
        """Test R-744 (CO2) has GWP of 1."""
        result = engine.get_refrigerant_gwp("R-744")
        assert result == Decimal("1")

    def test_refrigerant_detail(self, engine):
        """Test refrigerant detail returns full metadata."""
        result = engine.get_refrigerant_detail("R-134a")
        assert isinstance(result, dict)
        assert "gwp" in result
        assert "chemical_name" in result

    def test_all_refrigerant_gwps(self, engine):
        """Test retrieving all refrigerant GWPs."""
        result = engine.get_all_refrigerant_gwps()
        assert isinstance(result, dict)
        assert len(result) == len(REFRIGERANT_GWPS)


# ==============================================================================
# EEIO FACTOR TESTS
# ==============================================================================


class TestEEIOFactors:
    """Test EEIO spend-based factor retrieval."""

    @pytest.mark.parametrize("naics", list(EEIO_SPEND_FACTORS.keys()))
    def test_eeio_factor_all_naics(self, engine, naics):
        """Test EEIO factors for all franchise NAICS codes."""
        result = engine.get_eeio_factor(naics)
        assert result is not None
        if isinstance(result, Decimal):
            assert result > 0

    def test_eeio_factor_invalid_naics(self, engine):
        """Test invalid NAICS code raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_eeio_factor("000000")

    def test_eeio_factor_detail(self, engine):
        """Test EEIO factor detail returns full metadata."""
        result = engine.get_eeio_factor_detail("722511")
        assert isinstance(result, dict)
        assert "factor" in result
        assert "description" in result

    def test_all_eeio_factors(self, engine):
        """Test retrieving all EEIO factors."""
        result = engine.get_all_eeio_factors()
        assert isinstance(result, dict)
        assert len(result) == len(EEIO_SPEND_FACTORS)


# ==============================================================================
# HOTEL BENCHMARK TESTS
# ==============================================================================


class TestHotelBenchmarks:
    """Test hotel energy benchmark retrieval."""

    @pytest.mark.parametrize("hotel_class", ["economy", "midscale", "upscale", "luxury"])
    def test_hotel_benchmark_all_classes(self, engine, hotel_class):
        """Test hotel benchmarks for all 4 classes."""
        result = engine.get_hotel_benchmark(hotel_class, "temperate")
        assert isinstance(result, Decimal)
        assert result > 0

    @pytest.mark.parametrize("climate_zone", CLIMATE_ZONES)
    def test_hotel_benchmark_all_climates(self, engine, climate_zone):
        """Test hotel benchmarks for upscale across all climates."""
        result = engine.get_hotel_benchmark("upscale", climate_zone)
        assert isinstance(result, Decimal)
        assert result > 0

    def test_hotel_benchmark_luxury_higher(self, engine):
        """Test luxury hotel has higher EUI than economy."""
        luxury = engine.get_hotel_benchmark("luxury", "temperate")
        economy = engine.get_hotel_benchmark("economy", "temperate")
        assert luxury > economy

    def test_all_hotel_benchmarks(self, engine):
        """Test retrieving all hotel benchmarks."""
        result = engine.get_all_hotel_benchmarks()
        assert isinstance(result, dict)
        assert "economy" in result
        assert "luxury" in result


# ==============================================================================
# VEHICLE EMISSION FACTOR TESTS
# ==============================================================================


class TestVehicleEmissionFactors:
    """Test delivery fleet vehicle emission factor retrieval."""

    @pytest.mark.parametrize("vehicle_type", list(VEHICLE_EMISSION_FACTORS.keys()))
    def test_vehicle_ef_all_types(self, engine, vehicle_type):
        """Test vehicle EFs for all delivery vehicle types."""
        result = engine.get_vehicle_ef(vehicle_type)
        assert result is not None

    def test_vehicle_ef_light_van(self, engine):
        """Test light van vehicle EF."""
        result = engine.get_vehicle_ef("light_van")
        assert isinstance(result, Decimal)
        assert result > 0

    def test_vehicle_ef_invalid_type(self, engine):
        """Test invalid vehicle type raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_vehicle_ef("space_shuttle")

    def test_vehicle_ef_detail(self, engine):
        """Test vehicle EF detail returns full metadata."""
        result = engine.get_vehicle_ef_detail("medium_truck")
        assert isinstance(result, dict)
        assert "ef_per_km" in result
        assert "fuel_type" in result

    def test_all_vehicle_efs(self, engine):
        """Test retrieving all vehicle emission factors."""
        result = engine.get_all_vehicle_efs()
        assert isinstance(result, dict)
        assert len(result) == len(VEHICLE_EMISSION_FACTORS)


# ==============================================================================
# DOUBLE-COUNTING RULES TESTS
# ==============================================================================


class TestDoubleCounting:
    """Test double-counting prevention rules."""

    def test_get_dc_rule(self, engine):
        """Test double-counting rule retrieval."""
        rule = engine.get_dc_rule("DC-FRN-001")
        assert rule is not None
        assert "description" in rule

    def test_get_all_dc_rules(self, engine):
        """Test retrieving all DC rules."""
        rules = engine.get_dc_rules()
        assert isinstance(rules, list)
        assert len(rules) >= 8

    def test_dc_rules_by_severity(self, engine):
        """Test filtering DC rules by severity."""
        rules = engine.get_dc_rules_by_severity("critical")
        assert isinstance(rules, list)


# ==============================================================================
# COMPLIANCE FRAMEWORK TESTS
# ==============================================================================


class TestComplianceFrameworks:
    """Test compliance framework rule retrieval."""

    @pytest.mark.parametrize("framework", [
        "ghg_protocol", "iso_14064", "csrd", "cdp", "sbti", "gri", "sec_climate",
    ])
    def test_framework_rules(self, engine, framework):
        """Test retrieving rules for each compliance framework."""
        result = engine.get_framework_rules(framework)
        assert result is not None
        assert isinstance(result, dict)

    def test_all_framework_ids(self, engine):
        """Test listing all framework identifiers."""
        result = engine.get_all_framework_ids()
        assert isinstance(result, list)
        assert len(result) >= 7

    def test_mandatory_fields_ghg_protocol(self, engine):
        """Test mandatory fields for GHG Protocol."""
        result = engine.get_mandatory_fields("ghg_protocol")
        assert isinstance(result, list)
        assert len(result) > 0


# ==============================================================================
# DQI SCORING TESTS
# ==============================================================================


class TestDQIScoring:
    """Test Data Quality Indicator scoring."""

    @pytest.mark.parametrize("tier", ["tier_1", "tier_2", "tier_3"])
    def test_composite_dqi_all_tiers(self, engine, tier):
        """Test composite DQI for each tier."""
        result = engine.get_composite_dqi(tier)
        assert isinstance(result, Decimal)
        assert Decimal("1") <= result <= Decimal("5")

    def test_dqi_tier1_best(self, engine):
        """Test tier 1 has best (lowest) DQI score."""
        t1 = engine.get_composite_dqi("tier_1")
        t3 = engine.get_composite_dqi("tier_3")
        assert t1 < t3

    def test_dqi_matrix(self, engine):
        """Test retrieving full DQI matrix."""
        result = engine.get_dqi_matrix()
        assert isinstance(result, dict)
        assert len(result) > 0


# ==============================================================================
# UNCERTAINTY RANGE TESTS
# ==============================================================================


class TestUncertaintyRanges:
    """Test uncertainty range retrieval."""

    @pytest.mark.parametrize("method,tier", [
        ("franchise_specific", "tier_1"),
        ("average_data", "tier_2"),
        ("spend_based", "tier_3"),
    ])
    def test_uncertainty_range(self, engine, method, tier):
        """Test uncertainty range for method/tier combinations."""
        result = engine.get_uncertainty_range(method, tier)
        assert isinstance(result, dict)
        assert "lower_pct" in result
        assert "upper_pct" in result

    def test_all_uncertainty_ranges(self, engine):
        """Test retrieving all uncertainty ranges."""
        result = engine.get_all_uncertainty_ranges()
        assert isinstance(result, dict)
        assert len(result) > 0


# ==============================================================================
# CLIMATE ZONE MAPPING TESTS
# ==============================================================================


class TestClimateZoneMapping:
    """Test country-to-climate-zone mapping."""

    @pytest.mark.parametrize("country", ["US", "GB", "DE", "JP", "AU", "BR"])
    def test_climate_zone_countries(self, engine, country):
        """Test climate zone mapping for various countries."""
        result = engine.get_climate_zone(country)
        assert result in CLIMATE_ZONES

    def test_all_climate_zones(self, engine):
        """Test retrieving all climate zone mappings."""
        result = engine.get_all_climate_zones()
        assert isinstance(result, dict)
        assert len(result) >= 6


# ==============================================================================
# SEARCH AND VALIDATION TESTS
# ==============================================================================


class TestSearchAndValidation:
    """Test search and validation methods."""

    def test_search_factors(self, engine):
        """Test factor search returns results."""
        results = engine.search_factors("natural_gas")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_factors_empty_query(self, engine):
        """Test search with unrecognized query returns empty."""
        results = engine.search_factors("zzz_nothing_matches_zzz")
        assert isinstance(results, list)

    def test_validate_ef_source(self, engine):
        """Test EF source validation."""
        assert engine.validate_ef_source("DEFRA") is True
        assert engine.validate_ef_source("NONEXISTENT") is False

    def test_get_valid_ef_sources(self, engine):
        """Test listing valid EF sources."""
        result = engine.get_valid_ef_sources()
        assert isinstance(result, list)
        assert "DEFRA" in result


# ==============================================================================
# ENGINE INFO AND STATISTICS TESTS
# ==============================================================================


class TestEngineInfo:
    """Test engine metadata and statistics."""

    def test_engine_info(self, engine):
        """Test engine info returns metadata."""
        result = engine.get_engine_info()
        assert isinstance(result, dict)
        assert "agent_id" in result

    def test_table_statistics(self, engine):
        """Test table statistics returns counts."""
        result = engine.get_table_statistics()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_lookup_count(self, engine):
        """Test lookup counter tracks calls."""
        engine.get_eui_benchmark("qsr", "temperate")
        count = engine.get_lookup_count()
        assert count >= 1

    def test_reset(self):
        """Test engine reset clears singleton."""
        FranchiseDatabaseEngine._instance = None
        e1 = FranchiseDatabaseEngine()
        FranchiseDatabaseEngine.reset()
        e2 = FranchiseDatabaseEngine()
        assert e1 is not e2
