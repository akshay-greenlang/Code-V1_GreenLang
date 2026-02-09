# -*- coding: utf-8 -*-
"""
Unit tests for EmissionFactorEngine - AGENT-DATA-009 Batch 3

Tests the EmissionFactorEngine with 85%+ coverage across:
- Initialization and configuration
- EPA EEIO factor lookup by sector code
- NAICS-based factor lookup (all major sectors, fallback, unknown)
- UNSPSC-based factor lookup (all segments, unknown)
- EXIOBASE regional factor lookup (US, EU, CN, JP, IN, ROW)
- DEFRA category factor lookup
- Emissions calculation (basic, zero, large)
- Batch emissions calculation
- Custom factor registration, override, retrieval
- Factor listing (all, by source, by region, pagination)
- Factor selection hierarchy (custom > supplier > regional > national > global)
- Statistics tracking (lookups by source)
- SHA-256 provenance hashes
- Thread safety (concurrent lookups)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.spend_categorizer.emission_factor import (
    EmissionCalculation,
    EmissionFactor,
    EmissionFactorEngine,
    _EPA_EEIO_FACTORS,
    _EXIOBASE_FACTORS,
    _DEFRA_FACTORS,
    _NAICS_TO_EEIO_PREFIX,
    _UNSPSC_TO_EEIO_PREFIX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> EmissionFactorEngine:
    """Create a default EmissionFactorEngine."""
    return EmissionFactorEngine()


@pytest.fixture
def engine_eu() -> EmissionFactorEngine:
    """Create an EmissionFactorEngine with EU default region."""
    return EmissionFactorEngine({"default_region": "EU"})


@pytest.fixture
def engine_no_exiobase() -> EmissionFactorEngine:
    """Create an EmissionFactorEngine with EXIOBASE disabled."""
    return EmissionFactorEngine({"enable_exiobase": False})


@pytest.fixture
def engine_no_defra() -> EmissionFactorEngine:
    """Create an EmissionFactorEngine with DEFRA disabled."""
    return EmissionFactorEngine({"enable_defra": False})


@pytest.fixture
def custom_factor_def() -> Dict[str, Any]:
    """Custom emission factor definition."""
    return {
        "code": "CUST-001",
        "factor_value": 0.55,
        "name": "Custom Supplier Factor",
        "source": "custom",
        "region": "US",
        "unit": "kgCO2e/USD",
        "year": 2025,
    }


@pytest.fixture
def batch_records() -> List[Dict[str, Any]]:
    """Batch records for emissions calculation."""
    return [
        {"record_id": "R-001", "spend_usd": 50000, "factor_value": 0.10, "factor_source": "epa_eeio", "factor_region": "US"},
        {"record_id": "R-002", "spend_usd": 100000, "factor_value": 0.85, "factor_source": "epa_eeio", "factor_region": "US"},
        {"record_id": "R-003", "spend_usd": 25000, "factor_value": 1.35, "factor_source": "exiobase", "factor_region": "EU"},
        {"record_id": "R-004", "spend_usd": 0, "factor_value": 0.50, "factor_source": "epa_eeio", "factor_region": "US"},
        {"record_id": "R-005", "spend_usd": 200000, "factor_value": 0.25, "factor_source": "defra", "factor_region": "UK"},
    ]


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test EmissionFactorEngine initialization."""

    def test_default_init(self, engine: EmissionFactorEngine):
        """Engine initializes with default configuration."""
        assert engine._default_region == "US"
        assert engine._default_year == 2024
        assert engine._global_default == 0.25
        assert engine._enable_exiobase is True
        assert engine._enable_defra is True

    def test_custom_config(self):
        """Engine respects custom configuration values."""
        cfg = {
            "default_region": "EU",
            "default_year": 2025,
            "global_default_factor": 0.50,
            "enable_exiobase": False,
            "enable_defra": False,
        }
        engine = EmissionFactorEngine(cfg)
        assert engine._default_region == "EU"
        assert engine._default_year == 2025
        assert engine._global_default == 0.50
        assert engine._enable_exiobase is False
        assert engine._enable_defra is False

    def test_empty_custom_factors(self, engine: EmissionFactorEngine):
        """Custom factors dictionary starts empty."""
        assert len(engine._custom_factors) == 0

    def test_empty_calculations(self, engine: EmissionFactorEngine):
        """Calculations dictionary starts empty."""
        assert len(engine._calculations) == 0

    def test_stats_initialized(self, engine: EmissionFactorEngine):
        """Statistics counters start at zero."""
        stats = engine.get_statistics()
        assert stats["lookups_performed"] == 0
        assert stats["calculations_performed"] == 0
        assert stats["total_emissions_kgco2e"] == 0.0
        assert stats["custom_factors_registered"] == 0

    def test_builtin_databases_loaded(self, engine: EmissionFactorEngine):
        """Built-in factor databases are accessible."""
        stats = engine.get_statistics()
        assert stats["eeio_sectors"] == len(_EPA_EEIO_FACTORS)
        assert stats["exiobase_products"] == len(_EXIOBASE_FACTORS)
        assert stats["defra_categories"] == len(_DEFRA_FACTORS)
        assert stats["eeio_sectors"] > 80
        assert stats["exiobase_products"] > 40
        assert stats["defra_categories"] > 30


# ===========================================================================
# TestGetFactor
# ===========================================================================


class TestGetFactor:
    """Test get_factor() generic lookup."""

    def test_by_taxonomy_code_naics(self, engine: EmissionFactorEngine):
        """Numeric code is treated as NAICS lookup."""
        factor = engine.get_factor("5416")
        assert isinstance(factor, EmissionFactor)
        assert factor.factor_value > 0

    def test_by_system_naics(self, engine: EmissionFactorEngine):
        """Explicit system=naics dispatches to NAICS lookup."""
        factor = engine.get_factor("4841", system="naics")
        assert factor.source == "epa_eeio"
        assert factor.factor_value == 0.92

    def test_by_system_unspsc(self, engine: EmissionFactorEngine):
        """Explicit system=unspsc dispatches to UNSPSC lookup."""
        factor = engine.get_factor("43000000", system="unspsc")
        assert isinstance(factor, EmissionFactor)

    def test_unknown_code_returns_default(self, engine: EmissionFactorEngine):
        """Unknown non-numeric code returns global default factor."""
        factor = engine.get_factor("ZZZZZ")
        assert factor.source == "global_default"
        assert factor.factor_value == 0.25

    def test_custom_factor_priority(self, engine: EmissionFactorEngine, custom_factor_def):
        """Custom factor takes priority over all other sources."""
        engine.register_custom_factor(custom_factor_def)
        factor = engine.get_factor("CUST-001")
        assert factor.source == "custom"
        assert factor.factor_value == 0.55

    def test_region_override(self, engine: EmissionFactorEngine):
        """Region parameter overrides default region."""
        factor = engine.get_factor("5416", region="EU")
        assert isinstance(factor, EmissionFactor)

    def test_with_year_parameter(self, engine: EmissionFactorEngine):
        """Year parameter is accepted without error."""
        factor = engine.get_factor("5416", year=2025)
        assert isinstance(factor, EmissionFactor)


# ===========================================================================
# TestGetFactorByNAICS
# ===========================================================================


class TestGetFactorByNAICS:
    """Test get_factor_by_naics() for all major sectors."""

    @pytest.mark.parametrize("naics,expected_source", [
        ("1111", "epa_eeio"),
        ("2111", "epa_eeio"),
        ("2211", "epa_eeio"),
        ("3251", "epa_eeio"),
        ("3311", "epa_eeio"),
        ("4841", "epa_eeio"),
        ("5112", "epa_eeio"),
        ("5221", "epa_eeio"),
        ("5416", "epa_eeio"),
    ])
    def test_known_naics_codes(self, engine: EmissionFactorEngine, naics, expected_source):
        """Known 4-digit NAICS codes return EEIO factors."""
        factor = engine.get_factor_by_naics(naics)
        assert factor.source == expected_source
        assert factor.factor_value > 0

    def test_agriculture_sector(self, engine: EmissionFactorEngine):
        """Agriculture NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("1111")
        assert factor.factor_value == 0.88
        assert factor.sector_code == "1111"

    def test_mining_sector(self, engine: EmissionFactorEngine):
        """Mining NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("2111")
        assert factor.factor_value == 1.45

    def test_utilities_sector(self, engine: EmissionFactorEngine):
        """Utilities NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("2211")
        assert factor.factor_value == 1.80

    def test_manufacturing_sector(self, engine: EmissionFactorEngine):
        """Manufacturing NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("3311")
        assert factor.factor_value == 1.35

    def test_transportation_sector(self, engine: EmissionFactorEngine):
        """Transportation NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("4841")
        assert factor.factor_value == 0.92

    def test_professional_services_sector(self, engine: EmissionFactorEngine):
        """Professional services NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("5416")
        assert factor.factor_value == 0.10

    def test_finance_sector(self, engine: EmissionFactorEngine):
        """Finance NAICS codes return expected factors."""
        factor = engine.get_factor_by_naics("5221")
        assert factor.factor_value == 0.08

    def test_six_digit_naics_prefix_match(self, engine: EmissionFactorEngine):
        """6-digit NAICS code matches via progressive prefix shortening."""
        factor = engine.get_factor_by_naics("331110")
        assert factor.source == "epa_eeio"
        assert factor.factor_value == 1.35

    def test_two_digit_naics_sector_fallback(self, engine: EmissionFactorEngine):
        """2-digit NAICS falls back to sector mapping."""
        factor = engine.get_factor_by_naics("54")
        assert factor.source == "epa_eeio"

    def test_unknown_naics_returns_default(self, engine: EmissionFactorEngine):
        """Unknown NAICS code returns global default."""
        factor = engine.get_factor_by_naics("9999")
        assert factor.source == "global_default"
        assert factor.factor_value == 0.25

    def test_non_us_region_uses_exiobase(self, engine: EmissionFactorEngine):
        """Non-US region triggers EXIOBASE lookup."""
        factor = engine.get_factor_by_naics("1111", region="EU")
        assert factor.source == "exiobase"

    def test_non_us_region_with_exiobase_disabled(self, engine_no_exiobase: EmissionFactorEngine):
        """Non-US region with EXIOBASE disabled falls back to EEIO."""
        factor = engine_no_exiobase.get_factor_by_naics("1111", region="EU")
        assert factor.source == "epa_eeio"

    def test_whitespace_stripped(self, engine: EmissionFactorEngine):
        """Leading/trailing whitespace is stripped from NAICS code."""
        factor = engine.get_factor_by_naics("  5416  ")
        assert factor.factor_value == 0.10


# ===========================================================================
# TestGetFactorByUNSPSC
# ===========================================================================


class TestGetFactorByUNSPSC:
    """Test get_factor_by_unspsc() for major segments."""

    @pytest.mark.parametrize("unspsc_segment,expected_eeio_prefix", [
        ("10", "1111"),
        ("12", "3251"),
        ("23", "3332"),
        ("25", "3361"),
        ("32", "3344"),
        ("43", "3341"),
        ("78", "4841"),
        ("80", "5416"),
        ("84", "5221"),
    ])
    def test_known_unspsc_segments(self, engine: EmissionFactorEngine, unspsc_segment, expected_eeio_prefix):
        """Known UNSPSC segments map to correct EEIO sectors."""
        factor = engine.get_factor_by_unspsc(unspsc_segment + "000000")
        assert factor.source == "epa_eeio"
        assert factor.sector_code == expected_eeio_prefix

    def test_unspsc_full_code(self, engine: EmissionFactorEngine):
        """Full 8-digit UNSPSC code uses first two digits as segment."""
        factor = engine.get_factor_by_unspsc("43211500")
        assert factor.source == "epa_eeio"

    def test_unknown_unspsc_segment(self, engine: EmissionFactorEngine):
        """Unknown UNSPSC segment returns global default."""
        factor = engine.get_factor_by_unspsc("99000000")
        assert factor.source == "global_default"
        assert factor.factor_value == 0.25

    def test_short_unspsc_code(self, engine: EmissionFactorEngine):
        """Short UNSPSC code (2 digits) is handled correctly."""
        factor = engine.get_factor_by_unspsc("43")
        assert factor.source == "epa_eeio"

    def test_single_digit_code(self, engine: EmissionFactorEngine):
        """Single digit UNSPSC code returns default (no segment match)."""
        factor = engine.get_factor_by_unspsc("4")
        assert factor.source == "global_default"

    def test_whitespace_stripped(self, engine: EmissionFactorEngine):
        """Whitespace is stripped from UNSPSC code."""
        factor = engine.get_factor_by_unspsc("  43000000  ")
        assert factor.source == "epa_eeio"


# ===========================================================================
# TestGetEEIOFactor
# ===========================================================================


class TestGetEEIOFactor:
    """Test get_eeio_factor() direct EEIO lookup."""

    @pytest.mark.parametrize("sector_code,expected_factor", [
        ("3311", 1.35),   # Iron and steel mills
        ("3241", 2.10),   # Petroleum and coal
        ("2211", 1.80),   # Electric power
        ("4841", 0.92),   # General freight trucking
        ("5416", 0.10),   # Management consulting
        ("5221", 0.08),   # Depository credit
        ("1121", 1.20),   # Cattle ranching
        ("3254", 0.42),   # Pharmaceutical
        ("4200", 0.18),   # Wholesale trade
        ("5112", 0.10),   # Software publishers
    ])
    def test_representative_sectors(self, engine: EmissionFactorEngine, sector_code, expected_factor):
        """Representative sector codes return correct factors."""
        factor = engine.get_eeio_factor(sector_code)
        assert factor.factor_value == expected_factor
        assert factor.source == "epa_eeio"

    def test_unknown_sector(self, engine: EmissionFactorEngine):
        """Unknown sector code returns global default."""
        factor = engine.get_eeio_factor("ZZZZ")
        assert factor.source == "global_default"

    def test_region_is_always_us(self, engine: EmissionFactorEngine):
        """EEIO factors always report region US."""
        factor = engine.get_eeio_factor("5416")
        assert factor.region == "US"

    def test_source_version(self, engine: EmissionFactorEngine):
        """EEIO factors report source version 2024."""
        factor = engine.get_eeio_factor("5416")
        assert factor.source_version == "2024"


# ===========================================================================
# TestGetEXIOBASEFactor
# ===========================================================================


class TestGetEXIOBASEFactor:
    """Test get_exiobase_factor() regional factor lookup."""

    @pytest.mark.parametrize("product,region,expected_factor", [
        ("agriculture_products", "US", 0.82),
        ("agriculture_products", "EU", 0.75),
        ("agriculture_products", "CN", 1.10),
        ("agriculture_products", "JP", 0.70),
        ("agriculture_products", "IN", 1.25),
        ("agriculture_products", "ROW", 0.95),
        ("chemicals", "US", 0.72),
        ("chemicals", "EU", 0.65),
        ("basic_metals", "US", 1.25),
        ("basic_metals", "CN", 1.60),
        ("petroleum_products", "US", 2.05),
        ("electricity", "US", 0.52),
        ("electricity", "EU", 0.35),
    ])
    def test_regional_factors(self, engine: EmissionFactorEngine, product, region, expected_factor):
        """Regional EXIOBASE factors match expected values."""
        factor = engine.get_exiobase_factor(product, region)
        assert factor.factor_value == expected_factor
        assert factor.source == "exiobase"

    def test_unknown_region_falls_to_row(self, engine: EmissionFactorEngine):
        """Unknown region falls back to ROW factor."""
        factor = engine.get_exiobase_factor("chemicals", "BR")
        assert factor.factor_value == _EXIOBASE_FACTORS["chemicals"]["ROW"]

    def test_unknown_product_returns_default(self, engine: EmissionFactorEngine):
        """Unknown product code returns global default."""
        factor = engine.get_exiobase_factor("nonexistent_product")
        assert factor.source == "global_default"

    def test_default_region_used(self, engine_eu: EmissionFactorEngine):
        """Engine default region is used when not specified."""
        factor = engine_eu.get_exiobase_factor("chemicals")
        assert factor.region == "EU"
        assert factor.factor_value == 0.65

    def test_sector_name_formatted(self, engine: EmissionFactorEngine):
        """Sector name is title-cased from product code."""
        factor = engine.get_exiobase_factor("agriculture_products", "US")
        assert factor.sector_name == "Agriculture Products"


# ===========================================================================
# TestGetDEFRAFactor
# ===========================================================================


class TestGetDEFRAFactor:
    """Test get_defra_factor() DEFRA category lookup."""

    @pytest.mark.parametrize("category,expected_factor,expected_unit", [
        ("electricity_kwh", 0.233, "kgCO2e/kWh"),
        ("diesel_litre", 2.556, "kgCO2e/litre"),
        ("natural_gas_kwh", 0.184, "kgCO2e/kWh"),
        ("coal_kg", 2.883, "kgCO2e/kg"),
        ("flight_long_haul_km", 0.195, "kgCO2e/passenger-km"),
        ("hotel_night", 8.000, "kgCO2e/night"),
        ("steel_kg", 1.460, "kgCO2e/kg"),
        ("aluminium_kg", 6.830, "kgCO2e/kg"),
    ])
    def test_major_categories(self, engine: EmissionFactorEngine, category, expected_factor, expected_unit):
        """Major DEFRA categories return correct factors and units."""
        factor = engine.get_defra_factor(category)
        assert factor.factor_value == expected_factor
        assert factor.unit == expected_unit
        assert factor.source == "defra"
        assert factor.region == "UK"

    def test_unknown_category_returns_default(self, engine: EmissionFactorEngine):
        """Unknown DEFRA category returns global default."""
        factor = engine.get_defra_factor("nonexistent_category")
        assert factor.source == "global_default"

    def test_defra_methodology(self, engine: EmissionFactorEngine):
        """DEFRA factors report methodology as 'defra'."""
        factor = engine.get_defra_factor("diesel_litre")
        assert factor.methodology == "defra"

    def test_defra_source_version(self, engine: EmissionFactorEngine):
        """DEFRA factors report source version '2025'."""
        factor = engine.get_defra_factor("diesel_litre")
        assert factor.source_version == "2025"


# ===========================================================================
# TestCalculateEmissions
# ===========================================================================


class TestCalculateEmissions:
    """Test calculate_emissions() for single calculations."""

    def test_basic_calculation(self, engine: EmissionFactorEngine):
        """Basic emissions = spend * factor."""
        factor = engine.get_eeio_factor("5416")  # 0.10 kgCO2e/USD
        emissions = engine.calculate_emissions(50000.0, factor)
        assert emissions == pytest.approx(5000.0, rel=1e-4)

    def test_zero_spend(self, engine: EmissionFactorEngine):
        """Zero spend produces zero emissions."""
        factor = engine.get_eeio_factor("5416")
        emissions = engine.calculate_emissions(0.0, factor)
        assert emissions == 0.0

    def test_large_spend(self, engine: EmissionFactorEngine):
        """Large spend amounts calculate correctly."""
        factor = engine.get_eeio_factor("3311")  # 1.35 kgCO2e/USD
        emissions = engine.calculate_emissions(10_000_000.0, factor)
        assert emissions == pytest.approx(13_500_000.0, rel=1e-4)

    def test_small_spend(self, engine: EmissionFactorEngine):
        """Small spend amounts calculate correctly."""
        factor = engine.get_eeio_factor("5416")  # 0.10
        emissions = engine.calculate_emissions(1.0, factor)
        assert emissions == pytest.approx(0.1, rel=1e-4)

    def test_high_factor(self, engine: EmissionFactorEngine):
        """High emission factor multiplies correctly."""
        factor = engine.get_eeio_factor("3241")  # 2.10 (petroleum)
        emissions = engine.calculate_emissions(100000.0, factor)
        assert emissions == pytest.approx(210_000.0, rel=1e-4)

    def test_calculation_updates_stats(self, engine: EmissionFactorEngine):
        """Each calculation increments statistics counters."""
        factor = engine.get_eeio_factor("5416")
        engine.calculate_emissions(1000.0, factor)
        engine.calculate_emissions(2000.0, factor)

        stats = engine.get_statistics()
        assert stats["calculations_performed"] == 2
        assert stats["total_emissions_kgco2e"] == pytest.approx(300.0, rel=1e-4)

    def test_result_is_rounded(self, engine: EmissionFactorEngine):
        """Result is rounded to 4 decimal places."""
        factor = engine.get_eeio_factor("5416")  # 0.10
        emissions = engine.calculate_emissions(33333.3333, factor)
        # 33333.3333 * 0.10 = 3333.33333 -> rounds to 3333.3333
        assert emissions == round(33333.3333 * 0.10, 4)


# ===========================================================================
# TestCalculateBatch
# ===========================================================================


class TestCalculateBatch:
    """Test calculate_batch() for multiple records."""

    def test_batch_returns_correct_count(self, engine: EmissionFactorEngine, batch_records):
        """Batch returns one result per input record."""
        results = engine.calculate_batch(batch_records)
        assert len(results) == len(batch_records)

    def test_batch_calculation_correctness(self, engine: EmissionFactorEngine, batch_records):
        """Each batch result has correct emissions = spend * factor."""
        results = engine.calculate_batch(batch_records)
        for i, result in enumerate(results):
            rec = batch_records[i]
            expected = round(float(rec["spend_usd"]) * float(rec["factor_value"]), 4)
            assert result.emissions_kgco2e == expected

    def test_batch_zero_spend_record(self, engine: EmissionFactorEngine, batch_records):
        """Zero-spend record in batch produces zero emissions."""
        results = engine.calculate_batch(batch_records)
        # Record R-004 has spend_usd=0
        r4 = [r for r in results if r.record_id == "R-004"][0]
        assert r4.emissions_kgco2e == 0.0

    def test_batch_tco2e_conversion(self, engine: EmissionFactorEngine, batch_records):
        """tCO2e equals kgCO2e / 1000."""
        results = engine.calculate_batch(batch_records)
        for r in results:
            assert r.emissions_tco2e == pytest.approx(r.emissions_kgco2e / 1000.0, abs=1e-5)

    def test_batch_stores_calculations(self, engine: EmissionFactorEngine, batch_records):
        """Batch calculations are stored in engine._calculations."""
        results = engine.calculate_batch(batch_records)
        assert len(engine._calculations) == len(batch_records)

    def test_batch_updates_stats(self, engine: EmissionFactorEngine, batch_records):
        """Batch updates calculation statistics."""
        engine.calculate_batch(batch_records)
        stats = engine.get_statistics()
        assert stats["calculations_performed"] == len(batch_records)
        assert stats["calculations_stored"] == len(batch_records)

    def test_batch_provenance_hashes(self, engine: EmissionFactorEngine, batch_records):
        """Each batch result has a unique provenance hash."""
        results = engine.calculate_batch(batch_records)
        hashes = [r.provenance_hash for r in results]
        assert all(len(h) == 64 for h in hashes)

    def test_batch_records_each_have_id(self, engine: EmissionFactorEngine, batch_records):
        """Each batch result has a calculation_id starting with 'calc-'."""
        results = engine.calculate_batch(batch_records)
        for r in results:
            assert r.calculation_id.startswith("calc-")

    def test_empty_batch(self, engine: EmissionFactorEngine):
        """Empty batch returns empty list."""
        results = engine.calculate_batch([])
        assert results == []

    def test_missing_fields_use_defaults(self, engine: EmissionFactorEngine):
        """Records with missing fields use safe defaults."""
        records = [{"record_id": "partial-1"}]
        results = engine.calculate_batch(records)
        assert len(results) == 1
        assert results[0].emissions_kgco2e == 0.0


# ===========================================================================
# TestRegisterCustomFactor
# ===========================================================================


class TestRegisterCustomFactor:
    """Test register_custom_factor() and custom factor management."""

    def test_register_returns_factor_id(self, engine: EmissionFactorEngine, custom_factor_def):
        """Registration returns a factor ID starting with 'ef-custom-'."""
        fid = engine.register_custom_factor(custom_factor_def)
        assert fid.startswith("ef-custom-")

    def test_custom_factor_retrievable(self, engine: EmissionFactorEngine, custom_factor_def):
        """Registered custom factor is retrievable via get_factor."""
        engine.register_custom_factor(custom_factor_def)
        factor = engine.get_factor("CUST-001")
        assert factor.source == "custom"
        assert factor.factor_value == 0.55

    def test_override_existing_custom(self, engine: EmissionFactorEngine, custom_factor_def):
        """Registering with same code overrides existing custom factor."""
        engine.register_custom_factor(custom_factor_def)
        custom_factor_def["factor_value"] = 0.99
        engine.register_custom_factor(custom_factor_def)

        factor = engine.get_factor("CUST-001")
        assert factor.factor_value == 0.99

    def test_negative_factor_raises_error(self, engine: EmissionFactorEngine):
        """Negative factor value raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            engine.register_custom_factor({"code": "X", "factor_value": -0.5})

    def test_empty_code_raises_error(self, engine: EmissionFactorEngine):
        """Empty code raises ValueError."""
        with pytest.raises(ValueError, match="required"):
            engine.register_custom_factor({"code": "", "factor_value": 0.5})

    def test_zero_factor_allowed(self, engine: EmissionFactorEngine):
        """Zero factor value is allowed (carbon-neutral spend category)."""
        fid = engine.register_custom_factor({"code": "ZERO-001", "factor_value": 0.0})
        assert fid.startswith("ef-custom-")
        factor = engine.get_factor("ZERO-001")
        assert factor.factor_value == 0.0

    def test_custom_factor_stats(self, engine: EmissionFactorEngine, custom_factor_def):
        """Registration increments custom factor statistics."""
        engine.register_custom_factor(custom_factor_def)
        stats = engine.get_statistics()
        assert stats["custom_factors_registered"] == 1
        assert stats["custom_factors"] == 1

    def test_custom_factor_provenance(self, engine: EmissionFactorEngine, custom_factor_def):
        """Custom factor has a valid provenance hash."""
        engine.register_custom_factor(custom_factor_def)
        factor = engine.get_factor("CUST-001")
        assert len(factor.provenance_hash) == 64

    def test_custom_factor_metadata(self, engine: EmissionFactorEngine, custom_factor_def):
        """Custom factor preserves all metadata fields."""
        engine.register_custom_factor(custom_factor_def)
        factor = engine.get_factor("CUST-001")
        assert factor.unit == "kgCO2e/USD"
        assert factor.sector_name == "Custom Supplier Factor"
        assert factor.region == "US"


# ===========================================================================
# TestListFactors
# ===========================================================================


class TestListFactors:
    """Test list_factors() with various filters."""

    def test_list_all_default_limit(self, engine: EmissionFactorEngine):
        """Default listing returns up to 50 factors."""
        factors = engine.list_factors()
        assert len(factors) <= 50
        assert len(factors) > 0

    def test_list_by_source_epa_eeio(self, engine: EmissionFactorEngine):
        """Filtering by epa_eeio returns only EEIO factors."""
        factors = engine.list_factors(source="epa_eeio", limit=200)
        assert all(f.source == "epa_eeio" for f in factors)
        assert len(factors) > 0

    def test_list_by_source_exiobase(self, engine: EmissionFactorEngine):
        """Filtering by exiobase returns only EXIOBASE factors."""
        factors = engine.list_factors(source="exiobase", limit=200)
        assert all(f.source == "exiobase" for f in factors)

    def test_list_by_source_custom(self, engine: EmissionFactorEngine, custom_factor_def):
        """Filtering by custom returns registered custom factors."""
        engine.register_custom_factor(custom_factor_def)
        factors = engine.list_factors(source="custom")
        assert len(factors) == 1
        assert factors[0].source == "custom"

    def test_list_custom_empty(self, engine: EmissionFactorEngine):
        """Custom source filter returns empty when no custom factors registered."""
        factors = engine.list_factors(source="custom")
        assert factors == []

    def test_list_with_limit(self, engine: EmissionFactorEngine):
        """Limit parameter caps the results."""
        factors = engine.list_factors(limit=5)
        assert len(factors) <= 5

    def test_list_with_region_filter(self, engine: EmissionFactorEngine, custom_factor_def):
        """Region filter works for custom factors."""
        engine.register_custom_factor(custom_factor_def)
        factors = engine.list_factors(source="custom", region="US")
        assert len(factors) == 1
        factors_other = engine.list_factors(source="custom", region="JP")
        assert len(factors_other) == 0


# ===========================================================================
# TestFactorSelectionHierarchy
# ===========================================================================


class TestFactorSelectionHierarchy:
    """Test hierarchical factor selection priority."""

    def test_custom_beats_eeio(self, engine: EmissionFactorEngine):
        """Custom factor overrides EEIO for the same code."""
        # NAICS 5416 has EEIO factor 0.10
        engine.register_custom_factor({
            "code": "5416",
            "factor_value": 0.88,
            "name": "Custom override",
        })
        factor = engine.get_factor("5416")
        assert factor.source == "custom"
        assert factor.factor_value == 0.88

    def test_exiobase_for_non_us_naics(self, engine: EmissionFactorEngine):
        """Non-US region prefers EXIOBASE over EEIO for NAICS lookup."""
        factor = engine.get_factor_by_naics("1111", region="EU")
        assert factor.source == "exiobase"

    def test_eeio_for_us_naics(self, engine: EmissionFactorEngine):
        """US region uses EEIO for NAICS lookup."""
        factor = engine.get_factor_by_naics("1111", region="US")
        assert factor.source == "epa_eeio"

    def test_global_default_when_nothing_matches(self, engine: EmissionFactorEngine):
        """Global default is the last resort."""
        factor = engine.get_factor_by_naics("0000")
        assert factor.source == "global_default"
        assert factor.factor_value == 0.25


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Test statistics tracking."""

    def test_initial_stats(self, engine: EmissionFactorEngine):
        """Initial statistics are all zero."""
        stats = engine.get_statistics()
        assert stats["lookups_performed"] == 0
        assert stats["calculations_performed"] == 0

    def test_lookup_increments_count(self, engine: EmissionFactorEngine):
        """Each lookup increments the lookups_performed counter."""
        engine.get_factor_by_naics("5416")
        engine.get_factor_by_naics("3311")
        stats = engine.get_statistics()
        assert stats["lookups_performed"] == 2

    def test_by_source_tracking(self, engine: EmissionFactorEngine):
        """Lookups are tracked by source."""
        engine.get_factor_by_naics("5416")  # EEIO
        engine.get_factor_by_naics("1111", region="EU")  # EXIOBASE
        engine.get_defra_factor("diesel_litre")  # DEFRA
        stats = engine.get_statistics()
        assert stats["by_source"].get("epa_eeio", 0) >= 1
        assert stats["by_source"].get("exiobase", 0) >= 1
        assert stats["by_source"].get("defra", 0) >= 1

    def test_by_region_tracking(self, engine: EmissionFactorEngine):
        """Lookups are tracked by region."""
        engine.get_factor_by_naics("5416", region="US")
        engine.get_factor_by_naics("1111", region="EU")
        stats = engine.get_statistics()
        assert stats["by_region"].get("US", 0) >= 1
        assert stats["by_region"].get("EU", 0) >= 1

    def test_database_size_in_stats(self, engine: EmissionFactorEngine):
        """Statistics include database sizes."""
        stats = engine.get_statistics()
        assert stats["eeio_sectors"] > 0
        assert stats["exiobase_products"] > 0
        assert stats["defra_categories"] > 0

    def test_calculations_stored_count(self, engine: EmissionFactorEngine, batch_records):
        """Statistics include stored calculation count."""
        engine.calculate_batch(batch_records)
        stats = engine.get_statistics()
        assert stats["calculations_stored"] == 5


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hash generation."""

    def test_factor_has_provenance_hash(self, engine: EmissionFactorEngine):
        """Every factor returned has a SHA-256 provenance hash."""
        factor = engine.get_factor_by_naics("5416")
        assert len(factor.provenance_hash) == 64

    def test_hash_is_hex_encoded(self, engine: EmissionFactorEngine):
        """Provenance hash is hex-encoded (only hex characters)."""
        factor = engine.get_factor_by_naics("5416")
        assert all(c in "0123456789abcdef" for c in factor.provenance_hash)

    def test_different_factors_different_hashes(self, engine: EmissionFactorEngine):
        """Different factor lookups produce different hashes (due to unique IDs)."""
        f1 = engine.get_factor_by_naics("5416")
        f2 = engine.get_factor_by_naics("3311")
        assert f1.provenance_hash != f2.provenance_hash

    def test_batch_calculation_provenance(self, engine: EmissionFactorEngine, batch_records):
        """Batch calculations each have unique provenance hashes."""
        results = engine.calculate_batch(batch_records)
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == len(hashes)

    def test_custom_factor_provenance(self, engine: EmissionFactorEngine, custom_factor_def):
        """Custom factors have provenance hashes."""
        engine.register_custom_factor(custom_factor_def)
        factor = engine.get_factor("CUST-001")
        assert len(factor.provenance_hash) == 64

    def test_defra_factor_provenance(self, engine: EmissionFactorEngine):
        """DEFRA factors have provenance hashes."""
        factor = engine.get_defra_factor("diesel_litre")
        assert len(factor.provenance_hash) == 64

    def test_exiobase_factor_provenance(self, engine: EmissionFactorEngine):
        """EXIOBASE factors have provenance hashes."""
        factor = engine.get_exiobase_factor("chemicals", "US")
        assert len(factor.provenance_hash) == 64


# ===========================================================================
# TestThreadSafety
# ===========================================================================


class TestThreadSafety:
    """Test thread-safe concurrent access."""

    def test_concurrent_lookups(self, engine: EmissionFactorEngine):
        """Concurrent factor lookups do not corrupt state."""
        errors: List[str] = []

        def lookup_task(code: str):
            try:
                for _ in range(50):
                    factor = engine.get_factor_by_naics(code)
                    assert factor.factor_value > 0
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=lookup_task, args=("5416",)),
            threading.Thread(target=lookup_task, args=("3311",)),
            threading.Thread(target=lookup_task, args=("4841",)),
            threading.Thread(target=lookup_task, args=("2211",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_calculations(self, engine: EmissionFactorEngine):
        """Concurrent emissions calculations do not corrupt statistics."""
        errors: List[str] = []
        factor = engine.get_eeio_factor("5416")

        def calc_task():
            try:
                for _ in range(100):
                    engine.calculate_emissions(1000.0, factor)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=calc_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        # 4 threads * 100 calcs + the initial get_eeio_factor lookups
        assert stats["calculations_performed"] == 400

    def test_concurrent_registration(self, engine: EmissionFactorEngine):
        """Concurrent custom factor registration is thread-safe."""
        errors: List[str] = []

        def register_task(idx: int):
            try:
                for i in range(20):
                    engine.register_custom_factor({
                        "code": f"THREAD-{idx}-{i}",
                        "factor_value": 0.5,
                        "name": f"Thread {idx} Factor {i}",
                    })
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=register_task, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(engine._custom_factors) == 80  # 4 * 20
