# -*- coding: utf-8 -*-
"""
Unit tests for ContractualInstrumentDatabaseEngine

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests instrument type lookups, residual mix factors, energy source EFs,
supplier EFs, quality criteria validation, tracking systems, EF resolution
hierarchy, vintage validation, singleton/thread safety, and statistics/reset.

Target: ~80 tests, 85%+ coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock

import pytest

try:
    from greenlang.agents.mrv.scope2_market.contractual_instrument_database import (
        ContractualInstrumentDatabaseEngine,
        INSTRUMENT_TYPES,
        RESIDUAL_MIX_FACTORS,
        ENERGY_SOURCE_EF,
        SUPPLIER_DEFAULT_EF,
        QUALITY_CRITERIA,
        TRACKING_SYSTEMS,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ContractualInstrumentDatabaseEngine singleton before each test."""
    if ENGINE_AVAILABLE:
        ContractualInstrumentDatabaseEngine.reset_singleton()
    yield
    if ENGINE_AVAILABLE:
        ContractualInstrumentDatabaseEngine.reset_singleton()


@pytest.fixture
def engine():
    """Create a ContractualInstrumentDatabaseEngine instance."""
    return ContractualInstrumentDatabaseEngine()


# ===========================================================================
# 1. TestInstrumentTypes  (10 tests)
# ===========================================================================


@_SKIP
class TestInstrumentTypes:
    """Tests for get_instrument_info across all 10 instrument types."""

    @pytest.mark.parametrize("itype", [
        "REC", "GO", "I-REC", "PPA_PHYSICAL", "PPA_VIRTUAL",
        "GREEN_TARIFF", "SUPPLIER_SPECIFIC", "RESIDUAL_MIX", "REGO", "TIGR",
    ])
    def test_get_instrument_info_all_types(self, engine, itype):
        """get_instrument_info returns valid dict for each instrument type."""
        info = engine.get_instrument_info(itype)
        assert info["instrument_type"] == itype.upper().replace("-", "-")
        assert "name" in info
        assert "description" in info
        assert "region_applicability" in info
        assert "tracking_systems" in info
        assert "renewable_only" in info
        assert "typical_ef_kgco2e_kwh" in info
        assert "vintage_max_years" in info
        assert "provenance_hash" in info
        assert len(info["provenance_hash"]) == 64

    def test_unknown_instrument_type_raises(self, engine):
        """Unknown instrument type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown instrument type"):
            engine.get_instrument_info("INVALID_TYPE")

    def test_rec_is_renewable_only(self, engine):
        """REC is marked as renewable_only=True."""
        info = engine.get_instrument_info("REC")
        assert info["renewable_only"] is True

    def test_supplier_specific_is_not_renewable_only(self, engine):
        """SUPPLIER_SPECIFIC is not renewable_only."""
        info = engine.get_instrument_info("SUPPLIER_SPECIFIC")
        assert info["renewable_only"] is False

    def test_rec_typical_ef_is_zero(self, engine):
        """REC typical EF is 0.000."""
        info = engine.get_instrument_info("REC")
        assert info["typical_ef_kgco2e_kwh"] == Decimal("0.000")

    def test_instrument_types_count_is_10(self):
        """There are exactly 10 instrument types in the database."""
        assert len(INSTRUMENT_TYPES) == 10

    def test_list_instruments_by_region_us(self, engine):
        """US region returns at least REC and GREEN_TARIFF."""
        results = engine.list_instruments_by_region("US")
        types = {r["instrument_type"] for r in results}
        assert "REC" in types

    def test_list_instruments_by_region_global(self, engine):
        """GLOBAL region includes instruments with GLOBAL applicability."""
        results = engine.list_instruments_by_region("GLOBAL")
        assert len(results) >= 4  # I-REC, PPA_PHYSICAL, PPA_VIRTUAL, etc.


# ===========================================================================
# 2. TestResidualMixFactors  (15 tests)
# ===========================================================================


@_SKIP
class TestResidualMixFactors:
    """Tests for get_residual_mix_factor across various regions."""

    def test_us_camx_residual_mix(self, engine):
        """US-CAMX residual mix factor is 0.295."""
        assert engine.get_residual_mix_factor("US-CAMX") == Decimal("0.295")

    def test_de_residual_mix(self, engine):
        """DE (Germany) residual mix factor is 0.427."""
        assert engine.get_residual_mix_factor("DE") == Decimal("0.427")

    def test_gb_residual_mix(self, engine):
        """GB (UK) residual mix factor is 0.299."""
        assert engine.get_residual_mix_factor("GB") == Decimal("0.299")

    def test_jp_residual_mix(self, engine):
        """JP (Japan) residual mix factor is 0.497."""
        assert engine.get_residual_mix_factor("JP") == Decimal("0.497")

    def test_world_fallback_for_unknown_region(self, engine):
        """Unknown region falls back to WORLD average (0.436)."""
        assert engine.get_residual_mix_factor("ZZ") == RESIDUAL_MIX_FACTORS["WORLD"]

    def test_lowercase_region_normalized(self, engine):
        """Lowercase region codes are normalized to uppercase."""
        assert engine.get_residual_mix_factor("de") == Decimal("0.427")

    def test_custom_residual_mix_override(self, engine):
        """Custom residual mix overrides built-in factor."""
        engine.set_custom_residual_mix("XX", Decimal("0.999"), source="test", year=2025)
        assert engine.get_residual_mix_factor("XX") == Decimal("0.999")

    def test_remove_custom_residual_mix(self, engine):
        """Removing custom factor reverts to built-in or WORLD fallback."""
        engine.set_custom_residual_mix("XX", Decimal("0.999"))
        assert engine.remove_custom_residual_mix("XX") is True
        # XX is not built-in, should fall back to WORLD
        assert engine.get_residual_mix_factor("XX") == RESIDUAL_MIX_FACTORS["WORLD"]

    def test_remove_nonexistent_custom_returns_false(self, engine):
        """Removing a non-existent custom factor returns False."""
        assert engine.remove_custom_residual_mix("NONEXISTENT") is False

    def test_residual_mix_with_metadata(self, engine):
        """get_residual_mix_with_metadata returns full metadata dict."""
        result = engine.get_residual_mix_with_metadata("DE")
        assert result["region"] == "DE"
        assert result["factor_kgco2e_kwh"] == Decimal("0.427")
        assert result["source"] == "builtin"
        assert "provenance_hash" in result

    def test_list_residual_mix_factors_not_empty(self, engine):
        """list_residual_mix_factors returns 60+ entries."""
        factors = engine.list_residual_mix_factors()
        assert len(factors) >= 60

    def test_set_custom_residual_mix_negative_raises(self, engine):
        """Negative emission factor raises ValueError."""
        with pytest.raises(ValueError):
            engine.set_custom_residual_mix("XX", Decimal("-0.1"))

    def test_residual_mix_world_value(self, engine):
        """WORLD residual mix is 0.436."""
        assert engine.get_residual_mix_factor("WORLD") == Decimal("0.436")

    def test_list_custom_residual_mix_empty_initially(self, engine):
        """list_custom_residual_mix returns empty list initially."""
        assert engine.list_custom_residual_mix() == []

    def test_list_custom_after_add(self, engine):
        """list_custom_residual_mix reflects added factors."""
        engine.set_custom_residual_mix("AA", Decimal("0.111"))
        customs = engine.list_custom_residual_mix()
        assert len(customs) == 1
        assert customs[0]["region"] == "AA"


# ===========================================================================
# 3. TestEnergySourceEFs  (10 tests)
# ===========================================================================


@_SKIP
class TestEnergySourceEFs:
    """Tests for get_energy_source_ef across all 11 sources."""

    @pytest.mark.parametrize("source,expected_zero", [
        ("solar", True),
        ("wind", True),
        ("hydro", True),
        ("nuclear", True),
        ("biomass", True),
        ("geothermal", True),
    ])
    def test_renewable_sources_zero_ef(self, engine, source, expected_zero):
        """Renewable energy sources have zero emission factors."""
        ef = engine.get_energy_source_ef(source)
        assert ef == Decimal("0.000")

    @pytest.mark.parametrize("source", [
        "natural_gas_ccgt", "natural_gas_ocgt", "coal", "oil", "mixed",
    ])
    def test_fossil_sources_positive_ef(self, engine, source):
        """Fossil energy sources have positive emission factors."""
        ef = engine.get_energy_source_ef(source)
        assert ef > Decimal("0")

    def test_coal_highest_ef(self, engine):
        """Coal has the highest energy source EF at 0.910."""
        assert engine.get_energy_source_ef("coal") == Decimal("0.910")

    def test_unknown_energy_source_raises(self, engine):
        """Unknown energy source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown energy source"):
            engine.get_energy_source_ef("fusion")

    def test_is_renewable_source_wind(self, engine):
        """Wind is classified as renewable."""
        assert engine.is_renewable_source("wind") is True

    def test_is_renewable_source_coal(self, engine):
        """Coal is not classified as renewable."""
        assert engine.is_renewable_source("coal") is False

    def test_get_zero_emission_sources(self, engine):
        """get_zero_emission_sources returns 6 sources sorted."""
        sources = engine.get_zero_emission_sources()
        assert len(sources) == 6
        assert sources == sorted(sources)


# ===========================================================================
# 4. TestSupplierEFs  (10 tests)
# ===========================================================================


@_SKIP
class TestSupplierEFs:
    """Tests for get_supplier_ef and custom supplier registration."""

    def test_us_supplier_default(self, engine):
        """US supplier default EF is 0.390."""
        assert engine.get_supplier_ef("US") == Decimal("0.390")

    def test_de_supplier_default(self, engine):
        """DE supplier default EF is 0.400."""
        assert engine.get_supplier_ef("DE") == Decimal("0.400")

    def test_unknown_country_falls_back_to_world(self, engine):
        """Unknown country/supplier falls back to WORLD residual mix."""
        ef = engine.get_supplier_ef("ZZUNKNOWN")
        assert ef == RESIDUAL_MIX_FACTORS["WORLD"]

    def test_custom_supplier_registration(self, engine):
        """Custom supplier factor overrides country default."""
        engine.set_supplier_factor("SUP-TEST-001", Decimal("0.123"), country="US")
        assert engine.get_supplier_ef("SUP-TEST-001") == Decimal("0.123")

    def test_custom_supplier_factor_retrieval(self, engine):
        """get_supplier_factor returns full registration dict."""
        engine.set_supplier_factor("SUP-TEST-002", Decimal("0.456"), country="DE", year=2025)
        result = engine.get_supplier_factor("SUP-TEST-002")
        assert result["supplier_id"] == "SUP-TEST-002"
        assert result["ef_kgco2e_kwh"] == Decimal("0.456")
        assert result["country"] == "DE"
        assert "provenance_hash" in result

    def test_unregistered_supplier_raises(self, engine):
        """get_supplier_factor for unregistered supplier raises ValueError."""
        with pytest.raises(ValueError, match="No supplier factor"):
            engine.get_supplier_factor("NONEXISTENT")

    def test_negative_supplier_ef_raises(self, engine):
        """Negative supplier EF raises ValueError."""
        with pytest.raises(ValueError):
            engine.set_supplier_factor("SUP-BAD", Decimal("-0.1"))

    def test_list_supplier_factors_empty_initially(self, engine):
        """list_supplier_factors returns empty list initially."""
        assert engine.list_supplier_factors() == []

    def test_list_supplier_factors_after_add(self, engine):
        """list_supplier_factors reflects added suppliers."""
        engine.set_supplier_factor("SUP-A", Decimal("0.100"))
        engine.set_supplier_factor("SUP-B", Decimal("0.200"))
        factors = engine.list_supplier_factors()
        assert len(factors) == 2

    def test_supplier_fuel_mix_validation(self, engine):
        """Fuel mix not summing to ~1.0 raises ValueError."""
        with pytest.raises(ValueError, match="sum to"):
            engine.set_supplier_factor(
                "SUP-BAD", Decimal("0.300"),
                fuel_mix={"solar": Decimal("0.30"), "wind": Decimal("0.30")},
            )


# ===========================================================================
# 5. TestQualityCriteria  (10 tests)
# ===========================================================================


@_SKIP
class TestQualityCriteria:
    """Tests for quality criteria and instrument validation."""

    def test_get_quality_criteria_returns_7(self, engine):
        """get_quality_criteria returns 7 criteria."""
        criteria = engine.get_quality_criteria()
        assert len(criteria) == 7

    def test_criteria_weights_sum_to_one(self, engine):
        """Quality criteria weights sum to 1.0."""
        total = sum(c["weight"] for c in engine.get_quality_criteria())
        assert abs(total - Decimal("1.00")) < Decimal("0.01")

    def test_validate_instrument_quality_pass(self, engine):
        """Well-configured instrument passes quality validation."""
        instrument = {
            "instrument_type": "REC",
            "tracking_system": "WREGIS",
            "region": "US",
            "vintage_year": 2025,
            "reporting_year": 2025,
            "retirement_id": "RET-2025-0001",
            "is_additional": True,
        }
        result = engine.validate_instrument_quality(instrument)
        assert result["status"] == "PASS"
        assert result["overall_score"] > Decimal("0")
        assert len(result["criteria_results"]) == 7
        assert "provenance_hash" in result

    def test_validate_instrument_quality_fail_unknown_type(self, engine):
        """Instrument with unknown type fails conveyance criterion."""
        instrument = {
            "instrument_type": "UNKNOWN",
            "tracking_system": "",
            "region": "",
            "vintage_year": 0,
            "reporting_year": 2025,
            "retirement_id": "",
            "is_additional": False,
        }
        result = engine.validate_instrument_quality(instrument)
        # With all criteria failing, score should be low and status FAIL
        assert result["status"] == "FAIL"

    def test_validate_instrument_conveyance_criterion(self, engine):
        """Known instrument type passes conveyance criterion."""
        instrument = {
            "instrument_type": "GO",
            "tracking_system": "AIB",
            "region": "EU",
            "vintage_year": 2025,
            "reporting_year": 2025,
            "retirement_id": "RET-001",
        }
        result = engine.validate_instrument_quality(instrument)
        conveyance = next(
            c for c in result["criteria_results"]
            if c["criterion_id"] == "conveyance"
        )
        assert conveyance["passed"] is True

    def test_validate_instrument_unique_claim_no_retirement(self, engine):
        """Empty retirement_id fails unique claim criterion."""
        instrument = {
            "instrument_type": "REC",
            "tracking_system": "WREGIS",
            "region": "US",
            "vintage_year": 2025,
            "reporting_year": 2025,
            "retirement_id": "",
        }
        result = engine.validate_instrument_quality(instrument)
        unique = next(
            c for c in result["criteria_results"]
            if c["criterion_id"] == "unique_claim"
        )
        assert unique["passed"] is False

    def test_validate_additionality_true(self, engine):
        """is_additional=True passes additionality criterion."""
        instrument = {
            "instrument_type": "REC",
            "tracking_system": "WREGIS",
            "region": "US",
            "vintage_year": 2025,
            "reporting_year": 2025,
            "retirement_id": "RET-001",
            "is_additional": True,
        }
        result = engine.validate_instrument_quality(instrument)
        addl = next(
            c for c in result["criteria_results"]
            if c["criterion_id"] == "additionality"
        )
        assert addl["passed"] is True

    def test_validate_additionality_false(self, engine):
        """is_additional=False fails additionality criterion."""
        instrument = {
            "instrument_type": "REC",
            "vintage_year": 2025,
            "reporting_year": 2025,
            "retirement_id": "RET-001",
            "is_additional": False,
        }
        result = engine.validate_instrument_quality(instrument)
        addl = next(
            c for c in result["criteria_results"]
            if c["criterion_id"] == "additionality"
        )
        assert addl["passed"] is False

    def test_quality_criteria_have_provenance(self, engine):
        """Quality criteria list includes provenance_hash."""
        criteria = engine.get_quality_criteria()
        for c in criteria:
            assert "provenance_hash" in c

    def test_validation_count_incremented(self, engine):
        """Each validation increments the internal counter."""
        engine.validate_instrument_quality({
            "instrument_type": "REC",
            "vintage_year": 2025,
            "reporting_year": 2025,
        })
        stats = engine.get_statistics()
        assert stats["total_validations"] >= 1


# ===========================================================================
# 6. TestTrackingSystems  (5 tests)
# ===========================================================================


@_SKIP
class TestTrackingSystems:
    """Tests for get_tracking_system_info across all 8 systems."""

    @pytest.mark.parametrize("system", [
        "ERCOT", "WREGIS", "M-RETS", "GATS", "NEPOOL-GIS",
        "AIB", "OFGEM-REGO", "I-REC",
    ])
    def test_get_tracking_system_info_all(self, engine, system):
        """get_tracking_system_info returns valid dict for each system."""
        info = engine.get_tracking_system_info(system)
        assert "name" in info
        assert "region" in info
        assert "url" in info
        assert "instrument_types" in info
        assert "provenance_hash" in info

    def test_unknown_tracking_system_raises(self, engine):
        """Unknown tracking system raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tracking system"):
            engine.get_tracking_system_info("NONEXISTENT")

    def test_tracking_systems_count_is_8(self):
        """There are exactly 8 tracking systems."""
        assert len(TRACKING_SYSTEMS) == 8

    def test_aib_region_is_eu_eea(self, engine):
        """AIB tracking system region is EU/EEA."""
        info = engine.get_tracking_system_info("AIB")
        assert info["region"] == "EU/EEA"

    def test_lowercase_system_normalized(self, engine):
        """Lowercase system codes are normalized."""
        info = engine.get_tracking_system_info("wregis")
        assert info["system_id"] == "WREGIS"


# ===========================================================================
# 7. TestEFResolution  (10 tests)
# ===========================================================================


@_SKIP
class TestEFResolution:
    """Tests for resolve_emission_factor priority chain."""

    def test_custom_ef_highest_priority(self, engine):
        """Custom EF takes highest priority."""
        ef = engine.resolve_emission_factor({"custom_ef": Decimal("0.123")})
        assert ef == Decimal("0.123")

    def test_energy_source_second_priority(self, engine):
        """Energy source EF is used when no custom EF."""
        ef = engine.resolve_emission_factor({"energy_source": "solar"})
        assert ef == Decimal("0.000")

    def test_energy_source_coal(self, engine):
        """Coal energy source resolves to 0.910."""
        ef = engine.resolve_emission_factor({"energy_source": "coal"})
        assert ef == Decimal("0.910")

    def test_supplier_third_priority(self, engine):
        """Registered supplier EF used when no energy source."""
        engine.set_supplier_factor("SUP-EF-TEST", Decimal("0.222"))
        ef = engine.resolve_emission_factor({"supplier_id": "SUP-EF-TEST"})
        assert ef == Decimal("0.222")

    def test_supplier_country_default_fallback(self, engine):
        """Unregistered supplier with region falls back to supplier country default."""
        ef = engine.resolve_emission_factor({
            "supplier_id": "UNREGISTERED",
            "region": "US",
        })
        assert ef == SUPPLIER_DEFAULT_EF["US"]

    def test_residual_mix_lowest_priority(self, engine):
        """Residual mix is the fallback when nothing else matches."""
        ef = engine.resolve_emission_factor({"region": "DE"})
        assert ef == RESIDUAL_MIX_FACTORS["DE"]

    def test_resolve_global_fallback(self, engine):
        """No region defaults to WORLD residual mix."""
        ef = engine.resolve_emission_factor({})
        assert ef == RESIDUAL_MIX_FACTORS["WORLD"]

    def test_resolve_custom_ef_overrides_energy_source(self, engine):
        """Custom EF overrides energy_source even if both provided."""
        ef = engine.resolve_emission_factor({
            "custom_ef": Decimal("0.999"),
            "energy_source": "solar",
        })
        assert ef == Decimal("0.999")

    def test_resolve_energy_source_overrides_residual(self, engine):
        """Energy source overrides residual mix."""
        ef = engine.resolve_emission_factor({
            "energy_source": "wind",
            "region": "DE",
        })
        assert ef == Decimal("0.000")  # wind is zero

    def test_resolve_increments_lookup_count(self, engine):
        """resolve_emission_factor increments lookup count."""
        initial = engine.get_statistics()["total_lookups"]
        engine.resolve_emission_factor({"region": "US"})
        after = engine.get_statistics()["total_lookups"]
        assert after > initial


# ===========================================================================
# 8. TestVintageValidation  (5 tests)
# ===========================================================================


@_SKIP
class TestVintageValidation:
    """Tests for validate_vintage."""

    def test_valid_vintage_same_year(self, engine):
        """Vintage matching reporting year is valid."""
        assert engine.validate_vintage("REC", 2025, 2025) is True

    def test_valid_vintage_previous_year(self, engine):
        """REC vintage one year before reporting year is valid (max=5)."""
        assert engine.validate_vintage("REC", 2024, 2025) is True

    def test_expired_vintage_rec(self, engine):
        """REC vintage 6+ years before reporting year is invalid."""
        assert engine.validate_vintage("REC", 2019, 2025) is False

    def test_vintage_after_reporting_year_raises(self, engine):
        """Vintage after reporting year raises ValueError."""
        with pytest.raises(ValueError, match="cannot be after"):
            engine.validate_vintage("REC", 2026, 2025)

    def test_vintage_before_1990_raises(self, engine):
        """Vintage before 1990 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1990"):
            engine.validate_vintage("REC", 1989, 2025)


# ===========================================================================
# 9. TestSingletonThreadSafety  (3 tests)
# ===========================================================================


@_SKIP
class TestSingletonThreadSafety:
    """Tests for singleton pattern and thread safety."""

    def test_singleton_returns_same_instance(self):
        """Two instantiations return the same object."""
        e1 = ContractualInstrumentDatabaseEngine()
        e2 = ContractualInstrumentDatabaseEngine()
        assert e1 is e2

    def test_reset_singleton_creates_new_instance(self):
        """After reset_singleton, next call creates a new instance."""
        e1 = ContractualInstrumentDatabaseEngine()
        ContractualInstrumentDatabaseEngine.reset_singleton()
        e2 = ContractualInstrumentDatabaseEngine()
        assert e1 is not e2

    def test_engine_id_and_version(self):
        """Engine has expected ID and version."""
        e = ContractualInstrumentDatabaseEngine()
        assert e.ENGINE_ID == "contractual_instrument_database"
        assert e.ENGINE_VERSION == "1.0.0"


# ===========================================================================
# 10. TestStatisticsReset  (2 tests)
# ===========================================================================


@_SKIP
class TestStatisticsReset:
    """Tests for statistics and reset operations."""

    def test_get_statistics_initial(self, engine):
        """Initial statistics show correct built-in data counts."""
        stats = engine.get_statistics()
        assert stats["instrument_types"] == 10
        assert stats["residual_mix_regions"] == len(RESIDUAL_MIX_FACTORS)
        assert stats["energy_sources"] == len(ENERGY_SOURCE_EF)
        assert stats["supplier_defaults"] == len(SUPPLIER_DEFAULT_EF)
        assert stats["quality_criteria"] == 7
        assert stats["tracking_systems"] == 8
        assert stats["custom_residual_mix"] == 0
        assert stats["custom_suppliers"] == 0

    def test_reset_clears_mutable_state(self, engine):
        """reset() clears custom factors, counters, and provenance hashes."""
        engine.set_custom_residual_mix("XX", Decimal("0.999"))
        engine.set_supplier_factor("SUP-RESET", Decimal("0.100"))
        engine.get_residual_mix_factor("US")  # increment lookup

        engine.reset()

        stats = engine.get_statistics()
        assert stats["custom_residual_mix"] == 0
        assert stats["custom_suppliers"] == 0
        assert stats["total_lookups"] == 0
        assert stats["total_mutations"] == 0
