# -*- coding: utf-8 -*-
"""
Unit tests for ProcessingDatabaseEngine -- AGENT-MRV-023

Tests all lookup methods of ProcessingDatabaseEngine including product category
emission factors, processing energy intensities, grid emission factors, fuel EFs,
EEIO sector factors, processing chains, currency conversion, CPI deflation,
compatibility validation, and composite lookups.

Target: 30+ tests with exhaustive parametrize coverage.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pytest

try:
    from greenlang.agents.mrv.processing_sold_products.processing_database import (
        ProcessingDatabaseEngine,
        get_database_engine,
        calculate_provenance_hash,
        ProductCategory,
        ProcessingType,
        GridRegion,
        FuelType,
        EEIOSector,
        ProcessingChainType,
        CurrencyCode,
        PRODUCT_CATEGORY_EFS,
        PROCESSING_ENERGY_INTENSITIES,
        GRID_EMISSION_FACTORS,
        FUEL_EMISSION_FACTORS,
        EEIO_SECTOR_FACTORS,
        PROCESSING_CHAINS,
        CURRENCY_RATES,
        CPI_DEFLATORS,
        PRODUCT_PROCESSING_COMPATIBILITY,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ProcessingDatabaseEngine not available")
pytestmark = _SKIP


# ============================================================================
# HELPERS
# ============================================================================

def _is_valid_hash(h: str) -> bool:
    """Return True if h is a 64-char lowercase hex string."""
    return len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# ============================================================================
# TEST: Product Category Emission Factors
# ============================================================================


class TestProductCategoryEFs:
    """Test get_processing_ef for all 12 product categories."""

    @pytest.mark.parametrize(
        "category,expected_ef",
        [
            ("METALS_FERROUS", Decimal("280")),
            ("METALS_NON_FERROUS", Decimal("380")),
            ("PLASTICS_THERMOPLASTIC", Decimal("520")),
            ("PLASTICS_THERMOSET", Decimal("450")),
            ("CHEMICALS", Decimal("680")),
            ("FOOD_INGREDIENTS", Decimal("130")),
            ("TEXTILES", Decimal("350")),
            ("ELECTRONICS", Decimal("950")),
            ("GLASS_CERAMICS", Decimal("580")),
            ("WOOD_PAPER", Decimal("190")),
            ("MINERALS", Decimal("250")),
            ("AGRICULTURAL", Decimal("110")),
        ],
    )
    def test_get_processing_ef_all_categories(self, category, expected_ef):
        """Test emission factor lookup for each of the 12 product categories."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_processing_ef(category)
        assert result == expected_ef.quantize(Decimal("0.00000001"))

    def test_get_processing_ef_case_insensitive(self):
        """Test that category lookup is case-insensitive."""
        engine = ProcessingDatabaseEngine()
        lower = engine.get_processing_ef("metals_ferrous")
        upper = engine.get_processing_ef("METALS_FERROUS")
        assert lower == upper

    def test_get_processing_ef_unknown_category_raises(self):
        """Test that unknown category raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown product category"):
            engine.get_processing_ef("UNKNOWN_CATEGORY")

    def test_get_processing_ef_with_processing_type(self):
        """Test that providing a processing type still returns the category EF."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_processing_ef("METALS_FERROUS", "MACHINING")
        assert result == Decimal("280").quantize(Decimal("0.00000001"))

    def test_all_product_categories_returns_12(self):
        """Test get_all_product_categories returns exactly 12 categories."""
        engine = ProcessingDatabaseEngine()
        cats = engine.get_all_product_categories()
        assert len(cats) == 12
        assert "METALS_FERROUS" in cats
        assert "AGRICULTURAL" in cats


# ============================================================================
# TEST: Processing Energy Intensities
# ============================================================================


class TestProcessingEnergyIntensities:
    """Test get_energy_intensity for all 18 processing types."""

    @pytest.mark.parametrize(
        "proc_type,expected_kwh",
        [
            ("MACHINING", Decimal("280")),
            ("STAMPING", Decimal("140")),
            ("WELDING", Decimal("220")),
            ("HEAT_TREATMENT", Decimal("380")),
            ("INJECTION_MOLDING", Decimal("520")),
            ("EXTRUSION", Decimal("340")),
            ("BLOW_MOLDING", Decimal("400")),
            ("CASTING", Decimal("750")),
            ("FORGING", Decimal("580")),
            ("COATING", Decimal("120")),
            ("ASSEMBLY", Decimal("45")),
            ("CHEMICAL_REACTION", Decimal("1100")),
            ("REFINING", Decimal("900")),
            ("MILLING", Decimal("190")),
            ("DRYING", Decimal("310")),
            ("SINTERING", Decimal("1200")),
            ("FERMENTATION", Decimal("160")),
            ("TEXTILE_FINISHING", Decimal("420")),
        ],
    )
    def test_get_energy_intensity_all_types(self, proc_type, expected_kwh):
        """Test energy intensity lookup for each of the 18 processing types."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_energy_intensity(proc_type)
        assert result == expected_kwh.quantize(Decimal("0.00000001"))

    def test_get_energy_intensity_unknown_raises(self):
        """Test that unknown processing type raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown processing type"):
            engine.get_energy_intensity("UNKNOWN_TYPE")

    def test_energy_intensity_range_returns_triple(self):
        """Test that get_energy_intensity_range returns (low, mid, high) tuple."""
        engine = ProcessingDatabaseEngine()
        low, mid, high = engine.get_energy_intensity_range("MACHINING")
        assert low < mid < high
        # Mid should be 280, low = 0.75 * 280 = 210, high = 1.35 * 280 = 378
        assert mid == Decimal("280").quantize(Decimal("0.00000001"))
        assert low == Decimal("210").quantize(Decimal("0.00000001"))
        assert high == Decimal("378").quantize(Decimal("0.00000001"))

    def test_all_processing_types_returns_18(self):
        """Test get_all_processing_types returns exactly 18 types."""
        engine = ProcessingDatabaseEngine()
        types = engine.get_all_processing_types()
        assert len(types) == 18
        assert "SINTERING" in types


# ============================================================================
# TEST: Grid Emission Factors
# ============================================================================


class TestGridEmissionFactors:
    """Test get_grid_ef for all 16 grid regions."""

    @pytest.mark.parametrize(
        "region,expected_ef",
        [
            ("US", Decimal("0.417")),
            ("GB", Decimal("0.233")),
            ("DE", Decimal("0.348")),
            ("FR", Decimal("0.052")),
            ("CN", Decimal("0.555")),
            ("IN", Decimal("0.708")),
            ("JP", Decimal("0.462")),
            ("KR", Decimal("0.424")),
            ("BR", Decimal("0.075")),
            ("CA", Decimal("0.120")),
            ("AU", Decimal("0.656")),
            ("MX", Decimal("0.431")),
            ("IT", Decimal("0.256")),
            ("ES", Decimal("0.175")),
            ("PL", Decimal("0.635")),
            ("GLOBAL", Decimal("0.475")),
        ],
    )
    def test_get_grid_ef_all_regions(self, region, expected_ef):
        """Test grid EF lookup for each of the 16 regions."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_grid_ef(region)
        assert result == expected_ef.quantize(Decimal("0.00000001"))

    def test_get_grid_ef_unknown_raises(self):
        """Test that unknown region raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown grid region"):
            engine.get_grid_ef("UNKNOWN_REGION")

    def test_get_grid_ef_year_adjusted(self):
        """Test that year-adjusted grid EF applies the delta correctly."""
        engine = ProcessingDatabaseEngine()
        base_ef = engine.get_grid_ef("US")
        adjusted_ef = engine.get_grid_ef("US", year=2025)
        # US delta is -0.008 per year. 2025 is 1 year from 2024 base.
        # adjusted = 0.417 + (-0.008 * 1) = 0.409
        assert adjusted_ef == Decimal("0.409").quantize(Decimal("0.00000001"))
        assert adjusted_ef < base_ef

    def test_get_grid_ef_year_negative_floors_at_zero(self):
        """Test that a very far-future year floors grid EF at zero."""
        engine = ProcessingDatabaseEngine()
        # 200 years in the future: 0.052 + (-0.002 * 200) = 0.052 - 0.400 < 0
        result = engine.get_grid_ef("FR", year=2224)
        assert result == Decimal("0").quantize(Decimal("0.00000001"))

    def test_all_grid_regions_returns_16(self):
        """Test get_all_grid_regions returns exactly 16 regions."""
        engine = ProcessingDatabaseEngine()
        regions = engine.get_all_grid_regions()
        assert len(regions) == 16
        assert "GLOBAL" in regions


# ============================================================================
# TEST: Fuel Emission Factors
# ============================================================================


class TestFuelEmissionFactors:
    """Test get_fuel_ef for all 6 fuel types."""

    @pytest.mark.parametrize(
        "fuel_type,expected_ef",
        [
            ("NATURAL_GAS", Decimal("2.024")),
            ("DIESEL", Decimal("2.706")),
            ("HFO", Decimal("3.114")),
            ("LPG", Decimal("1.557")),
            ("COAL", Decimal("2.883")),
            ("BIOMASS", Decimal("0.015")),
        ],
    )
    def test_get_fuel_ef_all_types(self, fuel_type, expected_ef):
        """Test fuel EF lookup for each of the 6 fuel types."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_fuel_ef(fuel_type)
        assert result == expected_ef.quantize(Decimal("0.00000001"))

    def test_get_fuel_ef_unknown_raises(self):
        """Test that unknown fuel type raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown fuel type"):
            engine.get_fuel_ef("HYDROGEN")

    def test_all_fuel_types_returns_6(self):
        """Test get_all_fuel_types returns exactly 6 fuel types."""
        engine = ProcessingDatabaseEngine()
        fuels = engine.get_all_fuel_types()
        assert len(fuels) == 6


# ============================================================================
# TEST: Currency Conversion
# ============================================================================


class TestCurrencyConversion:
    """Test currency conversion methods."""

    @pytest.mark.parametrize(
        "currency,expected_rate",
        [
            ("USD", Decimal("1.0000")),
            ("EUR", Decimal("1.0850")),
            ("GBP", Decimal("1.2650")),
            ("JPY", Decimal("0.006667")),
            ("CNY", Decimal("0.1378")),
            ("INR", Decimal("0.01198")),
            ("CAD", Decimal("0.7410")),
            ("AUD", Decimal("0.6520")),
            ("KRW", Decimal("0.000752")),
            ("BRL", Decimal("0.1990")),
            ("MXN", Decimal("0.05880")),
            ("CHF", Decimal("1.1280")),
        ],
    )
    def test_get_currency_rate_all(self, currency, expected_rate):
        """Test currency rate lookup for all 12 currencies."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_currency_rate(currency)
        assert result == expected_rate.quantize(Decimal("0.00000001"))

    def test_convert_currency_eur_to_usd(self):
        """Test EUR to USD conversion: 1000 EUR * 1.085 = 1085 USD."""
        engine = ProcessingDatabaseEngine()
        result = engine.convert_currency(Decimal("1000"), "EUR", "USD")
        assert result == Decimal("1085").quantize(Decimal("0.00000001"))

    def test_convert_currency_negative_raises(self):
        """Test that negative amount raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="non-negative"):
            engine.convert_currency(Decimal("-100"), "EUR", "USD")

    def test_convert_currency_unknown_raises(self):
        """Test that unknown currency raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown currency"):
            engine.get_currency_rate("ZZZ")

    def test_all_currencies_returns_12(self):
        """Test get_all_currencies returns exactly 12 currencies."""
        engine = ProcessingDatabaseEngine()
        currencies = engine.get_all_currencies()
        assert len(currencies) == 12


# ============================================================================
# TEST: CPI Deflation
# ============================================================================


class TestCPIDeflation:
    """Test CPI deflation methods."""

    @pytest.mark.parametrize(
        "year,expected_deflator",
        [
            (2015, Decimal("0.7390")),
            (2016, Decimal("0.7483")),
            (2017, Decimal("0.7644")),
            (2018, Decimal("0.7832")),
            (2019, Decimal("0.7968")),
            (2020, Decimal("0.8071")),
            (2021, Decimal("0.8705")),
            (2022, Decimal("0.9403")),
            (2023, Decimal("0.9706")),
            (2024, Decimal("1.0000")),
            (2025, Decimal("1.0252")),
        ],
    )
    def test_get_cpi_deflator_all_years(self, year, expected_deflator):
        """Test CPI deflator lookup for all 11 years."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_cpi_deflator(year)
        assert result == expected_deflator.quantize(Decimal("0.00000001"))

    def test_get_cpi_deflator_unavailable_year_raises(self):
        """Test that unavailable year raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="CPI deflator not available"):
            engine.get_cpi_deflator(2010)

    def test_deflate_to_base_year_from_2020(self):
        """Test deflation from 2020 to 2024 base year."""
        engine = ProcessingDatabaseEngine()
        result = engine.deflate_to_base_year(Decimal("1000"), 2020)
        # 1000 * (1.0000 / 0.8071) = 1239.00...
        expected = Decimal("1000") * (Decimal("1.0000") / Decimal("0.8071"))
        assert abs(result - expected.quantize(Decimal("0.00000001"))) < Decimal("0.01")

    def test_deflate_to_base_year_negative_raises(self):
        """Test that negative amount raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="non-negative"):
            engine.deflate_to_base_year(Decimal("-100"), 2020)


# ============================================================================
# TEST: Processing Chains
# ============================================================================


class TestProcessingChains:
    """Test processing chain lookups."""

    @pytest.mark.parametrize(
        "chain_type,expected_ef",
        [
            ("STEEL_AUTOMOTIVE", Decimal("195")),
            ("ALUMINIUM_AEROSPACE", Decimal("420")),
            ("PLASTIC_PACKAGING", Decimal("385")),
            ("CHEMICAL_PHARMACEUTICAL", Decimal("820")),
            ("FOOD_BEVERAGE", Decimal("155")),
            ("TEXTILE_GARMENT", Decimal("310")),
            ("ELECTRONICS_PCB", Decimal("580")),
            ("WOOD_FURNITURE", Decimal("175")),
        ],
    )
    def test_get_chain_combined_ef_all_8(self, chain_type, expected_ef):
        """Test combined EF for all 8 processing chains."""
        engine = ProcessingDatabaseEngine()
        result = engine.get_chain_combined_ef(chain_type)
        assert result == expected_ef.quantize(Decimal("0.00000001"))

    def test_get_processing_chain_returns_full_definition(self):
        """Test that get_processing_chain returns steps, combined_ef, description."""
        engine = ProcessingDatabaseEngine()
        chain = engine.get_processing_chain("STEEL_AUTOMOTIVE")
        assert "description" in chain
        assert "steps" in chain
        assert "combined_ef" in chain
        assert "product_category" in chain
        assert len(chain["steps"]) == 5

    def test_get_processing_chain_unknown_raises(self):
        """Test that unknown chain type raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown processing chain"):
            engine.get_processing_chain("UNKNOWN_CHAIN")

    def test_all_processing_chains_returns_8(self):
        """Test get_all_processing_chains returns exactly 8 chains."""
        engine = ProcessingDatabaseEngine()
        chains = engine.get_all_processing_chains()
        assert len(chains) == 8


# ============================================================================
# TEST: Compatibility Validation
# ============================================================================


class TestCompatibilityValidation:
    """Test product-processing type compatibility validation."""

    def test_metals_ferrous_compatible_with_machining(self):
        """Test that METALS_FERROUS is compatible with MACHINING."""
        engine = ProcessingDatabaseEngine()
        assert engine.validate_product_processing_compatibility(
            "METALS_FERROUS", "MACHINING"
        ) is True

    def test_metals_ferrous_incompatible_with_fermentation(self):
        """Test that METALS_FERROUS is incompatible with FERMENTATION."""
        engine = ProcessingDatabaseEngine()
        assert engine.validate_product_processing_compatibility(
            "METALS_FERROUS", "FERMENTATION"
        ) is False

    def test_get_applicable_processing_types_metals(self):
        """Test applicable processing types for METALS_FERROUS."""
        engine = ProcessingDatabaseEngine()
        types = engine.get_applicable_processing_types("METALS_FERROUS")
        assert "MACHINING" in types
        assert "STAMPING" in types
        assert len(types) >= 8


# ============================================================================
# TEST: Composite Lookup
# ============================================================================


class TestCompositeLookup:
    """Test the lookup_ef composite method."""

    def test_lookup_ef_returns_all_fields(self):
        """Test that lookup_ef returns all expected fields."""
        engine = ProcessingDatabaseEngine()
        result = engine.lookup_ef("METALS_FERROUS", "MACHINING", "US")
        assert result["category"] == "METALS_FERROUS"
        assert result["processing_type"] == "MACHINING"
        assert result["region"] == "US"
        assert result["category_ef"] == Decimal("280").quantize(Decimal("0.00000001"))
        assert result["energy_intensity"] == Decimal("280").quantize(Decimal("0.00000001"))
        assert result["grid_ef"] == Decimal("0.417").quantize(Decimal("0.00000001"))
        # energy_based_ef = 280 * 0.417 = 116.76
        assert result["energy_based_ef"] == Decimal("116.76").quantize(Decimal("0.00000001"))
        assert result["is_compatible"] is True
        assert _is_valid_hash(result["provenance_hash"])

    def test_lookup_ef_incompatible_flag(self):
        """Test that is_compatible is False for incompatible combination."""
        engine = ProcessingDatabaseEngine()
        result = engine.lookup_ef("METALS_FERROUS", "FERMENTATION", "US")
        assert result["is_compatible"] is False


# ============================================================================
# TEST: Singleton Pattern and Engine Status
# ============================================================================


class TestSingleton:
    """Test singleton pattern and engine diagnostics."""

    def test_singleton_returns_same_instance(self):
        """Test that ProcessingDatabaseEngine is a singleton."""
        engine1 = ProcessingDatabaseEngine()
        engine2 = ProcessingDatabaseEngine()
        assert engine1 is engine2

    def test_get_database_engine_returns_singleton(self):
        """Test that get_database_engine returns the singleton."""
        engine = get_database_engine()
        assert isinstance(engine, ProcessingDatabaseEngine)

    def test_engine_status_contains_required_fields(self):
        """Test that get_engine_status returns all required diagnostic fields."""
        engine = ProcessingDatabaseEngine()
        status = engine.get_engine_status()
        assert status["agent_id"] == "GL-MRV-S3-010"
        assert status["version"] == "1.0.0"
        assert status["product_categories"] == 12
        assert status["processing_types"] == 18
        assert status["grid_regions"] == 16
        assert status["fuel_types"] == 6
        assert status["eeio_sectors"] == 12
        assert status["processing_chains"] == 8
        assert status["currencies"] == 12
        assert status["cpi_years"] == 11

    def test_lookup_count_increments(self):
        """Test that lookup_count increments on each lookup."""
        engine = ProcessingDatabaseEngine()
        initial = engine.lookup_count
        engine.get_processing_ef("METALS_FERROUS")
        assert engine.lookup_count == initial + 1


# ============================================================================
# TEST: Provenance Hash
# ============================================================================


class TestProvenanceHash:
    """Test calculate_provenance_hash helper."""

    def test_hash_is_64_char_hex(self):
        """Test that provenance hash is a 64-character hex string."""
        h = calculate_provenance_hash("test_input")
        assert _is_valid_hash(h)

    def test_hash_deterministic(self):
        """Test that same inputs produce the same hash."""
        h1 = calculate_provenance_hash("METALS_FERROUS", Decimal("280"))
        h2 = calculate_provenance_hash("METALS_FERROUS", Decimal("280"))
        assert h1 == h2

    def test_hash_changes_with_different_input(self):
        """Test that different inputs produce different hashes."""
        h1 = calculate_provenance_hash("METALS_FERROUS", Decimal("280"))
        h2 = calculate_provenance_hash("ELECTRONICS", Decimal("950"))
        assert h1 != h2


# ============================================================================
# TEST: EEIO Sector Factors
# ============================================================================


class TestEEIOSectorFactors:
    """Test EEIO sector factor lookups."""

    @pytest.mark.parametrize(
        "sector,expected_factor",
        [
            ("IRON_STEEL_MANUFACTURING", Decimal("0.820")),
            ("ALUMINIUM_MANUFACTURING", Decimal("1.150")),
            ("PLASTICS_MANUFACTURING", Decimal("0.680")),
            ("CHEMICAL_MANUFACTURING", Decimal("0.950")),
            ("FOOD_PROCESSING", Decimal("0.420")),
            ("TEXTILE_MANUFACTURING", Decimal("0.560")),
            ("ELECTRONICS_MANUFACTURING", Decimal("0.380")),
            ("GLASS_MANUFACTURING", Decimal("0.740")),
            ("PAPER_MANUFACTURING", Decimal("0.510")),
            ("MINERAL_PROCESSING", Decimal("0.890")),
            ("METALWORKING", Decimal("0.620")),
            ("AGRICULTURAL_PROCESSING", Decimal("0.350")),
        ],
    )
    def test_get_eeio_factor_all_12(self, sector, expected_factor):
        """Test EEIO factor lookup for all 12 sectors."""
        engine = ProcessingDatabaseEngine()
        factor, margin = engine.get_eeio_factor(sector)
        assert factor == expected_factor.quantize(Decimal("0.00000001"))
        assert margin > Decimal("0")

    def test_get_eeio_factor_unknown_raises(self):
        """Test that unknown sector raises ValueError."""
        engine = ProcessingDatabaseEngine()
        with pytest.raises(ValueError, match="Unknown EEIO sector"):
            engine.get_eeio_factor("UNKNOWN_SECTOR")

    def test_all_eeio_sectors_returns_12(self):
        """Test get_all_eeio_sectors returns exactly 12 sectors."""
        engine = ProcessingDatabaseEngine()
        sectors = engine.get_all_eeio_sectors()
        assert len(sectors) == 12
