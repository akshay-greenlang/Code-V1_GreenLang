# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-009 GridEmissionFactorDatabaseEngine.

Tests grid factor lookups (eGRID, IEA, EU EEA, DEFRA), steam/heat/cooling
factors, T&D loss factors, custom factor management, factor resolution
hierarchy, validation, data quality scoring, and unit conversions.

Target: 55+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

import pytest

try:
    from greenlang.scope2_location.grid_factor_database import (
        GridEmissionFactorDatabaseEngine,
        EGRID_SUBREGION_FACTORS,
        IEA_COUNTRY_FACTORS,
        EU_COUNTRY_FACTORS,
        DEFRA_UK_FACTORS,
        TD_LOSS_FACTORS,
        STEAM_EF_BY_TYPE,
        HEAT_EF_BY_TYPE,
        COOLING_EF_BY_TYPE,
        GWP_TABLE,
        MWH_TO_GJ,
        GJ_TO_MWH,
        KWH_TO_MWH,
        MWH_TO_KWH,
        MMBTU_TO_GJ,
        THERM_TO_GJ,
        TONNES_TO_KG,
        KG_TO_TONNES,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a GridEmissionFactorDatabaseEngine instance."""
    return GridEmissionFactorDatabaseEngine()


@pytest.fixture
def engine_with_metrics():
    """Create engine with a mock metrics object."""
    from unittest.mock import Mock
    metrics = Mock()
    return GridEmissionFactorDatabaseEngine(metrics=metrics)


# ===========================================================================
# TestGridFactorLookup
# ===========================================================================


@_SKIP
class TestGridFactorLookup:
    """Tests for get_grid_factor across multiple countries."""

    def test_us_grid_factor_returns_iea_source(self, engine):
        """US grid factor uses IEA source (country-level fallback)."""
        result = engine.get_grid_factor("US")
        assert result["country_code"] == "US"
        assert result["source"] == "iea"
        assert result["co2_kg_per_mwh"] > Decimal("0")

    def test_gb_grid_factor_returns_defra_source(self, engine):
        """GB grid factor uses DEFRA source."""
        result = engine.get_grid_factor("GB")
        assert result["country_code"] == "GB"
        assert result["source"] == "defra"
        assert "energy_type" in result

    def test_de_grid_factor_returns_eu_eea_source(self, engine):
        """DE (Germany) grid factor uses EU EEA source."""
        result = engine.get_grid_factor("DE")
        assert result["country_code"] == "DE"
        assert result["source"] == "eu_eea"
        assert result["co2_kg_per_mwh"] == Decimal("338.00")

    def test_fr_grid_factor_returns_eu_eea_source(self, engine):
        """FR (France) grid factor uses EU EEA source with low carbon."""
        result = engine.get_grid_factor("FR")
        assert result["source"] == "eu_eea"
        # France has low grid emissions due to nuclear
        assert result["co2_kg_per_mwh"] < Decimal("100")

    def test_cn_grid_factor_returns_iea_source(self, engine):
        """CN (China) grid factor from IEA."""
        result = engine.get_grid_factor("CN")
        assert result["country_code"] == "CN"
        assert result["source"] == "iea"
        # China has coal-heavy grid
        assert result["co2_kg_per_mwh"] > Decimal("500")

    def test_in_grid_factor_returns_iea_source(self, engine):
        """IN (India) grid factor from IEA."""
        result = engine.get_grid_factor("IN")
        assert result["country_code"] == "IN"
        assert result["source"] == "iea"
        # India has high grid intensity
        assert result["co2_kg_per_mwh"] > Decimal("600")

    def test_unknown_country_returns_ipcc_default(self, engine):
        """Unknown country code returns IPCC default (world average)."""
        result = engine.get_grid_factor("ZZ")
        assert result["country_code"] == "ZZ"
        assert result["source"] == "ipcc_default"
        assert result["data_quality_tier"] == "tier_3"

    def test_lowercase_country_code_normalized(self, engine):
        """Lowercase country codes are normalized to uppercase."""
        result = engine.get_grid_factor("us")
        assert result["country_code"] == "US"

    def test_result_contains_all_required_keys(self, engine):
        """Result dict contains co2, ch4, n2o, total, source, year, tier."""
        result = engine.get_grid_factor("US")
        required_keys = {
            "co2_kg_per_mwh", "ch4_kg_per_mwh", "n2o_kg_per_mwh",
            "total_co2e_kg_per_mwh", "source", "year", "data_quality_tier",
        }
        assert required_keys.issubset(result.keys())

    def test_lookup_count_incremented(self, engine):
        """Each lookup increments the internal counter."""
        assert engine._lookup_count == 0
        engine.get_grid_factor("US")
        assert engine._lookup_count == 1
        engine.get_grid_factor("GB")
        assert engine._lookup_count == 2


# ===========================================================================
# TestEGridFactors
# ===========================================================================


@_SKIP
class TestEGridFactors:
    """Tests for get_egrid_factor (US eGRID subregions)."""

    def test_camx_subregion_co2(self, engine):
        """CAMX CO2 factor is 225.30 kg/MWh per eGRID2022."""
        result = engine.get_egrid_factor("CAMX")
        assert result["co2_kg_per_mwh"] == Decimal("225.30")
        assert result["subregion"] == "CAMX"
        assert result["source"] == "egrid"

    def test_erct_subregion(self, engine):
        """ERCT (Texas) CO2 factor is 380.10 kg/MWh."""
        result = engine.get_egrid_factor("ERCT")
        assert result["co2_kg_per_mwh"] == Decimal("380.10")

    def test_frcc_subregion(self, engine):
        """FRCC (Florida) CO2 factor is 392.44 kg/MWh."""
        result = engine.get_egrid_factor("FRCC")
        assert result["co2_kg_per_mwh"] == Decimal("392.44")

    def test_co2e_calculation_includes_gwp(self, engine):
        """Total CO2e includes CH4*GWP_CH4 + N2O*GWP_N2O."""
        result = engine.get_egrid_factor("CAMX")
        factors = EGRID_SUBREGION_FACTORS["CAMX"]
        expected_co2e = (
            factors["co2"]
            + factors["ch4"] * GWP_TABLE["AR5"]["ch4"]
            + factors["n2o"] * GWP_TABLE["AR5"]["n2o"]
        ).quantize(Decimal("0.01"), ROUND_HALF_UP)
        assert result["total_co2e_kg_per_mwh"] == expected_co2e

    def test_invalid_subregion_raises_valueerror(self, engine):
        """Invalid eGRID subregion raises ValueError."""
        with pytest.raises(ValueError, match="Unknown eGRID subregion"):
            engine.get_egrid_factor("ZZZZ")

    def test_lowercase_subregion_normalized(self, engine):
        """Lowercase subregion is normalized to uppercase."""
        result = engine.get_egrid_factor("camx")
        assert result["subregion"] == "CAMX"

    def test_egrid_tier_is_tier_2(self, engine):
        """eGRID factors are classified as tier_2."""
        result = engine.get_egrid_factor("CAMX")
        assert result["data_quality_tier"] == "tier_2"

    def test_all_26_subregions_accessible(self, engine):
        """All 26 eGRID subregions return valid results."""
        for sr in EGRID_SUBREGION_FACTORS.keys():
            result = engine.get_egrid_factor(sr)
            assert result["co2_kg_per_mwh"] > Decimal("0")


# ===========================================================================
# TestIEAFactors
# ===========================================================================


@_SKIP
class TestIEAFactors:
    """Tests for get_iea_factor."""

    @pytest.mark.parametrize("country,expected_tco2", [
        ("US", Decimal("0.379")),
        ("CN", Decimal("0.555")),
        ("IN", Decimal("0.708")),
        ("DE", Decimal("0.338")),
        ("JP", Decimal("0.457")),
        ("BR", Decimal("0.074")),
    ])
    def test_iea_known_countries(self, engine, country, expected_tco2):
        """IEA factor values match known tCO2/MWh values."""
        result = engine.get_iea_factor(country)
        expected_kg = expected_tco2 * TONNES_TO_KG
        assert result["co2_kg_per_mwh"] == expected_kg.quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )

    def test_iea_unknown_country_raises(self, engine):
        """Unknown country in IEA raises ValueError."""
        with pytest.raises(ValueError, match="not in IEA database"):
            engine.get_iea_factor("ZZ")

    def test_iea_source_metadata(self, engine):
        """IEA factor has correct source and tier."""
        result = engine.get_iea_factor("US")
        assert result["source"] == "iea"
        assert result["data_quality_tier"] == "tier_1"

    def test_iea_factor_count(self, engine):
        """IEA database contains 130+ countries."""
        assert len(IEA_COUNTRY_FACTORS) >= 130


# ===========================================================================
# TestEUFactors
# ===========================================================================


@_SKIP
class TestEUFactors:
    """Tests for get_eu_factor (EU EEA member states)."""

    @pytest.mark.parametrize("country,expected_tco2", [
        ("DE", Decimal("0.338")),
        ("FR", Decimal("0.056")),
        ("PL", Decimal("0.635")),
        ("SE", Decimal("0.008")),
        ("ES", Decimal("0.138")),
    ])
    def test_eu_known_countries(self, engine, country, expected_tco2):
        """EU EEA factor values match reference data."""
        result = engine.get_eu_factor(country)
        expected_kg = expected_tco2 * TONNES_TO_KG
        assert result["co2_kg_per_mwh"] == expected_kg.quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )
        assert result["source"] == "eu_eea"

    def test_eu_non_member_raises(self, engine):
        """Non-EU country raises ValueError."""
        with pytest.raises(ValueError, match="not in EU EEA database"):
            engine.get_eu_factor("US")

    def test_eu_27_member_states(self, engine):
        """EU EEA database has 27 member states."""
        assert len(EU_COUNTRY_FACTORS) == 27


# ===========================================================================
# TestDEFRAFactors
# ===========================================================================


@_SKIP
class TestDEFRAFactors:
    """Tests for get_defra_factor (UK DEFRA/DESNZ)."""

    def test_defra_electricity_default(self, engine):
        """DEFRA electricity total = 0.22586 kgCO2e/kWh = 225.86 kgCO2e/MWh."""
        result = engine.get_defra_factor("electricity")
        expected_mwh = DEFRA_UK_FACTORS["electricity_total"] * MWH_TO_KWH
        assert result["co2_kg_per_mwh"] == expected_mwh.quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )
        assert result["source"] == "defra"

    def test_defra_steam_factor(self, engine):
        """DEFRA steam factor returns correct values."""
        result = engine.get_defra_factor("steam")
        expected_mwh = DEFRA_UK_FACTORS["steam"] * MWH_TO_KWH
        assert result["co2_kg_per_mwh"] == expected_mwh.quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )
        assert result["energy_type"] == "steam"

    def test_defra_heating_factor(self, engine):
        """DEFRA heating factor returns correct values."""
        result = engine.get_defra_factor("heating")
        expected_mwh = DEFRA_UK_FACTORS["heating"] * MWH_TO_KWH
        assert result["co2_kg_per_mwh"] == expected_mwh.quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )

    def test_defra_cooling_factor(self, engine):
        """DEFRA cooling factor returns correct values."""
        result = engine.get_defra_factor("cooling")
        expected_mwh = DEFRA_UK_FACTORS["cooling"] * MWH_TO_KWH
        assert result["co2_kg_per_mwh"] == expected_mwh.quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )

    def test_defra_country_always_gb(self, engine):
        """DEFRA results always show country_code = GB."""
        result = engine.get_defra_factor()
        assert result["country_code"] == "GB"


# ===========================================================================
# TestTDLossFactors
# ===========================================================================


@_SKIP
class TestTDLossFactors:
    """Tests for get_td_loss_factor."""

    def test_us_td_loss_5_percent(self, engine):
        """US T&D loss is 5% (Decimal 0.050)."""
        assert engine.get_td_loss_factor("US") == Decimal("0.050")

    def test_in_td_loss_19_4_percent(self, engine):
        """India T&D loss is 19.4% (Decimal 0.194)."""
        assert engine.get_td_loss_factor("IN") == Decimal("0.194")

    def test_world_average(self, engine):
        """World average T&D loss is 8.3% (Decimal 0.083)."""
        assert engine.get_td_loss_factor("WORLD") == Decimal("0.083")

    def test_unknown_country_returns_world_average(self, engine):
        """Unknown country falls back to WORLD average."""
        result = engine.get_td_loss_factor("ZZ")
        assert result == TD_LOSS_FACTORS["WORLD"]

    def test_gb_td_loss(self, engine):
        """GB T&D loss is 7.7%."""
        assert engine.get_td_loss_factor("GB") == Decimal("0.077")

    def test_custom_td_override(self, engine):
        """Custom T&D factor overrides built-in."""
        engine._custom_td["US"] = Decimal("0.065")
        assert engine.get_td_loss_factor("US") == Decimal("0.065")


# ===========================================================================
# TestSteamHeatCoolingFactors
# ===========================================================================


@_SKIP
class TestSteamHeatCoolingFactors:
    """Tests for get_steam_factor, get_heating_factor, get_cooling_factor."""

    @pytest.mark.parametrize("steam_type,expected", [
        ("natural_gas", Decimal("56.10")),
        ("coal", Decimal("94.60")),
        ("biomass", Decimal("0.00")),
        ("oil", Decimal("73.30")),
        ("mixed", Decimal("64.20")),
    ])
    def test_steam_factors(self, engine, steam_type, expected):
        """Steam emission factors match reference values (kgCO2e/GJ)."""
        assert engine.get_steam_factor(steam_type) == expected

    def test_unknown_steam_type_defaults_to_natural_gas(self, engine):
        """Unknown steam type defaults to natural_gas EF."""
        result = engine.get_steam_factor("unknown_fuel")
        assert result == STEAM_EF_BY_TYPE["natural_gas"]

    @pytest.mark.parametrize("heating_type,expected", [
        ("district", Decimal("43.50")),
        ("gas_boiler", Decimal("56.10")),
        ("heat_pump", Decimal("18.50")),
        ("biomass", Decimal("0.00")),
    ])
    def test_heating_factors(self, engine, heating_type, expected):
        """Heating emission factors match reference values."""
        assert engine.get_heating_factor(heating_type) == expected

    @pytest.mark.parametrize("cooling_type,expected", [
        ("absorption", Decimal("32.10")),
        ("district", Decimal("28.50")),
        ("free_cooling", Decimal("0.00")),
    ])
    def test_cooling_factors(self, engine, cooling_type, expected):
        """Cooling emission factors match reference values."""
        assert engine.get_cooling_factor(cooling_type) == expected


# ===========================================================================
# TestCustomFactors
# ===========================================================================


@_SKIP
class TestCustomFactors:
    """Tests for custom factor CRUD operations."""

    def test_add_custom_factor_returns_id(self, engine):
        """add_custom_factor returns a UUID string."""
        factor_id = engine.add_custom_factor(
            region_id="XX",
            co2_per_mwh=Decimal("400.00"),
        )
        assert isinstance(factor_id, str)
        assert len(factor_id) == 36  # UUID format

    def test_custom_factor_stored_and_retrievable(self, engine):
        """Added custom factor is retrievable via get_grid_factor."""
        engine.add_custom_factor(
            region_id="XX",
            co2_per_mwh=Decimal("400.00"),
            ch4_per_mwh=Decimal("0.05"),
            n2o_per_mwh=Decimal("0.01"),
        )
        result = engine.get_grid_factor("XX")
        assert result["source"] == "custom"
        assert result["co2_kg_per_mwh"] == Decimal("400.00")

    def test_update_custom_factor(self, engine):
        """update_custom_factor modifies existing factor."""
        factor_id = engine.add_custom_factor(
            region_id="YY",
            co2_per_mwh=Decimal("300.00"),
        )
        success = engine.update_custom_factor(
            factor_id, co2_kg_per_mwh=Decimal("350.00")
        )
        assert success is True
        result = engine.get_grid_factor("YY")
        assert result["co2_kg_per_mwh"] == Decimal("350.00")

    def test_update_nonexistent_factor_returns_false(self, engine):
        """Updating a nonexistent factor returns False."""
        result = engine.update_custom_factor("nonexistent-id")
        assert result is False

    def test_delete_custom_factor(self, engine):
        """delete_custom_factor removes the factor."""
        factor_id = engine.add_custom_factor(
            region_id="ZZ",
            co2_per_mwh=Decimal("500.00"),
        )
        assert engine.delete_custom_factor(factor_id) is True
        # After deletion, ZZ should fall back to IPCC default
        result = engine.get_grid_factor("ZZ")
        assert result["source"] == "ipcc_default"

    def test_delete_nonexistent_factor_returns_false(self, engine):
        """Deleting a nonexistent factor returns False."""
        assert engine.delete_custom_factor("no-such-id") is False

    def test_list_custom_factors_empty(self, engine):
        """list_custom_factors returns empty list initially."""
        assert engine.list_custom_factors() == []

    def test_list_custom_factors_after_add(self, engine):
        """list_custom_factors returns all added factors."""
        engine.add_custom_factor("AA", Decimal("100"))
        engine.add_custom_factor("BB", Decimal("200"))
        factors = engine.list_custom_factors()
        assert len(factors) == 2

    def test_custom_factor_co2e_includes_gwp(self, engine):
        """Custom factor total CO2e includes CH4*GWP and N2O*GWP."""
        engine.add_custom_factor(
            region_id="CC",
            co2_per_mwh=Decimal("300.00"),
            ch4_per_mwh=Decimal("0.10"),
            n2o_per_mwh=Decimal("0.02"),
        )
        result = engine.get_grid_factor("CC")
        expected_co2e = (
            Decimal("300.00")
            + Decimal("0.10") * GWP_TABLE["AR5"]["ch4"]
            + Decimal("0.02") * GWP_TABLE["AR5"]["n2o"]
        ).quantize(Decimal("0.01"), ROUND_HALF_UP)
        assert result["total_co2e_kg_per_mwh"] == expected_co2e


# ===========================================================================
# TestFactorResolution
# ===========================================================================


@_SKIP
class TestFactorResolution:
    """Tests for resolve_emission_factor hierarchy logic."""

    def test_custom_factor_has_highest_priority(self, engine):
        """Custom factor takes priority over all other sources."""
        engine.add_custom_factor("DE", Decimal("999.00"))
        result = engine.resolve_emission_factor("DE")
        assert result["source"] == "custom"
        assert result["co2_kg_per_mwh"] == Decimal("999.00")

    def test_egrid_subregion_resolves_for_us(self, engine):
        """eGRID subregion is used when provided for US."""
        result = engine.resolve_emission_factor("US", egrid_subregion="CAMX")
        assert result["source"] == "egrid"
        assert result["co2_kg_per_mwh"] == Decimal("225.30")

    def test_eu_eea_resolves_for_eu_country(self, engine):
        """EU EEA source resolves for EU member states."""
        result = engine.resolve_emission_factor("DE")
        # national comes before eu_eea in hierarchy, and DE is in IEA
        assert result["source"] in ("national", "eu_eea")

    def test_iea_fallback_for_non_eu_non_us(self, engine):
        """IEA is used as fallback for non-EU, non-US countries."""
        result = engine.resolve_emission_factor("JP")
        assert result["source"] in ("national", "iea")

    def test_resolve_best_factor_for_facility(self, engine):
        """resolve_best_factor uses facility info dict."""
        facility = {
            "country_code": "US",
            "egrid_subregion": "ERCT",
        }
        result = engine.resolve_best_factor(facility)
        assert result["source"] == "egrid"


# ===========================================================================
# TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for validate_factor and get_data_quality_score."""

    def test_valid_factor_no_errors(self, engine):
        """Valid factor produces empty error list."""
        factor = {
            "co2_kg_per_mwh": Decimal("400"),
            "ch4_kg_per_mwh": Decimal("0.03"),
            "n2o_kg_per_mwh": Decimal("0.005"),
            "year": 2023,
        }
        errors = engine.validate_factor(factor)
        assert errors == []

    def test_negative_co2_flagged(self, engine):
        """Negative CO2 factor is flagged as error."""
        factor = {"co2_kg_per_mwh": Decimal("-10")}
        errors = engine.validate_factor(factor)
        assert any("negative" in e.lower() for e in errors)

    def test_suspiciously_high_co2_flagged(self, engine):
        """CO2 factor > 2000 kg/MWh is flagged as suspicious."""
        factor = {"co2_kg_per_mwh": Decimal("2500")}
        errors = engine.validate_factor(factor)
        assert any("high" in e.lower() for e in errors)

    def test_negative_ch4_flagged(self, engine):
        """Negative CH4 factor is flagged."""
        factor = {"ch4_kg_per_mwh": Decimal("-0.01")}
        errors = engine.validate_factor(factor)
        assert len(errors) > 0

    def test_data_quality_score_egrid(self, engine):
        """eGRID source with recent year gets high quality score."""
        factor = {"source": "egrid", "year": datetime.utcnow().year}
        score = engine.get_data_quality_score(factor)
        assert score >= Decimal("0.80")

    def test_data_quality_score_ipcc_default(self, engine):
        """IPCC default source gets low quality score."""
        factor = {"source": "ipcc_default", "year": 2018}
        score = engine.get_data_quality_score(factor)
        assert score < Decimal("0.50")

    def test_data_quality_score_range(self, engine):
        """Quality score is always between 0 and 1."""
        for source in ["custom", "egrid", "iea", "ipcc_default"]:
            for year in [2020, 2023, 2025]:
                score = engine.get_data_quality_score(
                    {"source": source, "year": year}
                )
                assert Decimal("0") <= score <= Decimal("1")


# ===========================================================================
# TestUtility
# ===========================================================================


@_SKIP
class TestUtility:
    """Tests for convert_units, list_countries, list_egrid_subregions, etc."""

    def test_convert_kwh_to_mwh(self, engine):
        """1000 kWh = 1 MWh."""
        result = engine.convert_units(Decimal("1000"), "kwh", "mwh")
        assert result == Decimal("1.000000")

    def test_convert_mwh_to_gj(self, engine):
        """1 MWh = 3.6 GJ."""
        result = engine.convert_units(Decimal("1"), "mwh", "gj")
        assert result == Decimal("3.600000")

    def test_convert_gj_to_mwh(self, engine):
        """3.6 GJ ~ 1 MWh."""
        result = engine.convert_units(Decimal("3.6"), "gj", "mwh")
        assert abs(result - Decimal("1")) < Decimal("0.001")

    def test_convert_mmbtu_to_gj(self, engine):
        """1 MMBtu = 1.05506 GJ."""
        result = engine.convert_units(Decimal("1"), "mmbtu", "gj")
        assert result == MMBTU_TO_GJ.quantize(Decimal("0.000001"), ROUND_HALF_UP)

    def test_convert_same_unit_identity(self, engine):
        """Converting same unit returns identical value."""
        assert engine.convert_units(Decimal("42"), "mwh", "mwh") == Decimal("42")

    def test_convert_invalid_units_raises(self, engine):
        """Invalid unit pair raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            engine.convert_units(Decimal("1"), "kg", "mwh")

    def test_list_countries_not_empty(self, engine):
        """list_countries returns a non-empty sorted list."""
        countries = engine.list_countries()
        assert len(countries) > 100
        assert countries == sorted(countries)

    def test_list_egrid_subregions_has_26(self, engine):
        """list_egrid_subregions returns exactly 26 entries."""
        subregions = engine.list_egrid_subregions()
        assert len(subregions) == 26
        assert "CAMX" in subregions

    def test_list_eu_countries_has_27(self, engine):
        """list_eu_countries returns 27 EU member states."""
        eu = engine.list_eu_countries()
        assert len(eu) == 27

    def test_get_statistics_counts(self, engine):
        """get_statistics returns correct database counts."""
        stats = engine.get_statistics()
        assert stats["iea_countries"] == len(IEA_COUNTRY_FACTORS)
        assert stats["eu_countries"] == len(EU_COUNTRY_FACTORS)
        assert stats["egrid_subregions"] == 26
        assert stats["custom_factors"] == 0

    def test_get_statistics_after_custom_add(self, engine):
        """get_statistics reflects newly added custom factors."""
        engine.add_custom_factor("XX", Decimal("100"))
        stats = engine.get_statistics()
        assert stats["custom_factors"] == 1

    def test_search_factors_by_country_code(self, engine):
        """search_factors finds results matching a query."""
        results = engine.search_factors("US")
        assert len(results) >= 1

    def test_check_factor_freshness_recent(self, engine):
        """Recent factor year is flagged as fresh."""
        factor = {"year": datetime.utcnow().year}
        freshness = engine.check_factor_freshness(factor)
        assert freshness["is_fresh"] is True

    def test_check_factor_freshness_old(self, engine):
        """Old factor year (>5 years) includes recommendation."""
        factor = {"year": 2015}
        freshness = engine.check_factor_freshness(factor)
        assert freshness["is_fresh"] is False
        assert "outdated" in freshness["recommendation"].lower()

    def test_get_factor_metadata_us(self, engine):
        """Factor metadata for US includes has_egrid=True."""
        meta = engine.get_factor_metadata("US")
        assert meta["country_code"] == "US"
        assert meta["has_egrid"] is True
        assert "iea" in meta["available_sources"]

    def test_get_factor_history_returns_list(self, engine):
        """get_factor_history returns non-empty list for known country."""
        history = engine.get_factor_history("US", years=5)
        assert len(history) == 5
        assert all(isinstance(h, dict) for h in history)
