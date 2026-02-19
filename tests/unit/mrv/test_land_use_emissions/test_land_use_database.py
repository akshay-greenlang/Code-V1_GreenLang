# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 LandUseDatabaseEngine.

Tests get_agb_default, get_bgb_default, get_dead_wood_default/fraction,
get_litter_default, get_soc_reference, get_growth_rate, get_fire_ef,
get_peatland_ef, N2O EFs, classify_climate_zone, classify_soil_type,
get_land_subcategories, unknown/edge inputs, and Decimal precision.

Target: 110 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.land_use_emissions.land_use_database import (
    LandUseDatabaseEngine,
    LandCategory,
    ClimateZone,
    SoilType,
    CarbonPool,
    GWPSource,
    DisturbanceType,
    IPCC_AGB_DEFAULTS,
    ROOT_SHOOT_RATIOS,
    DEAD_WOOD_FRACTION,
    LITTER_STOCKS,
    SOC_REFERENCE_STOCKS,
    BIOMASS_GROWTH_RATES,
    FIRE_EMISSION_FACTORS,
    PEATLAND_EF,
    N2O_SOIL_EF,
    GWP_VALUES,
    CARBON_FRACTION,
    CONVERSION_FACTOR_CO2_C,
    LAND_SUBCATEGORIES,
    SOC_LAND_USE_FACTORS,
    SOC_MANAGEMENT_FACTORS,
    SOC_INPUT_FACTORS,
    COMBUSTION_FACTORS,
)


# ===========================================================================
# AGB Default Tests
# ===========================================================================


class TestGetAGBDefault:
    """Tests for LandUseDatabaseEngine.get_agb_default()."""

    @pytest.mark.parametrize("category,zone,expected", [
        ("FOREST_LAND", "TROPICAL_WET", Decimal("180")),
        ("FOREST_LAND", "TROPICAL_MOIST", Decimal("155")),
        ("FOREST_LAND", "TROPICAL_DRY", Decimal("65")),
        ("FOREST_LAND", "TROPICAL_MONTANE", Decimal("110")),
        ("FOREST_LAND", "TEMPERATE_OCEANIC", Decimal("120")),
        ("FOREST_LAND", "BOREAL_DRY", Decimal("20")),
        ("FOREST_LAND", "BOREAL_MOIST", Decimal("40")),
        ("FOREST_LAND", "POLAR", Decimal("3")),
    ])
    def test_forest_land_agb(self, land_use_database_engine, category, zone, expected):
        """Forest land AGB defaults match IPCC values."""
        result = land_use_database_engine.get_agb_default(category, zone)
        assert result == expected

    @pytest.mark.parametrize("category,zone,expected", [
        ("CROPLAND", "TROPICAL_WET", Decimal("10")),
        ("CROPLAND", "TROPICAL_DRY", Decimal("5")),
        ("CROPLAND", "TEMPERATE_OCEANIC", Decimal("8")),
        ("CROPLAND", "BOREAL_DRY", Decimal("2")),
        ("CROPLAND", "POLAR", Decimal("0")),
    ])
    def test_cropland_agb(self, land_use_database_engine, category, zone, expected):
        """Cropland AGB defaults match IPCC values."""
        result = land_use_database_engine.get_agb_default(category, zone)
        assert result == expected

    @pytest.mark.parametrize("category,zone,expected", [
        ("GRASSLAND", "TROPICAL_WET", Decimal("8.1")),
        ("GRASSLAND", "TROPICAL_MOIST", Decimal("6.2")),
        ("GRASSLAND", "TEMPERATE_OCEANIC", Decimal("6.8")),
        ("GRASSLAND", "POLAR", Decimal("0.2")),
    ])
    def test_grassland_agb(self, land_use_database_engine, category, zone, expected):
        """Grassland AGB defaults match IPCC values."""
        result = land_use_database_engine.get_agb_default(category, zone)
        assert result == expected

    @pytest.mark.parametrize("category,zone,expected", [
        ("WETLANDS", "TROPICAL_WET", Decimal("86")),
        ("WETLANDS", "TEMPERATE_OCEANIC", Decimal("30")),
        ("SETTLEMENTS", "TROPICAL_WET", Decimal("25")),
        ("SETTLEMENTS", "POLAR", Decimal("0")),
        ("OTHER_LAND", "TROPICAL_WET", Decimal("0")),
        ("OTHER_LAND", "POLAR", Decimal("0")),
    ])
    def test_other_categories_agb(self, land_use_database_engine, category, zone, expected):
        """Non-forest AGB defaults match IPCC values."""
        result = land_use_database_engine.get_agb_default(category, zone)
        assert result == expected

    def test_invalid_category_raises(self, land_use_database_engine):
        """Unknown land category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown land category"):
            land_use_database_engine.get_agb_default("UNKNOWN", "TROPICAL_WET")

    def test_invalid_climate_zone_raises(self, land_use_database_engine):
        """Unknown climate zone raises ValueError."""
        with pytest.raises(ValueError, match="Unknown climate zone"):
            land_use_database_engine.get_agb_default("FOREST_LAND", "UNKNOWN_ZONE")

    def test_result_is_decimal(self, land_use_database_engine):
        """Returned AGB value is a Decimal instance."""
        result = land_use_database_engine.get_agb_default("FOREST_LAND", "TROPICAL_WET")
        assert isinstance(result, Decimal)

    def test_case_insensitive_lookup(self, land_use_database_engine):
        """Lookup is case-insensitive after normalization."""
        result = land_use_database_engine.get_agb_default("forest_land", "tropical_wet")
        assert result == Decimal("180")

    def test_caching_returns_same_value(self, land_use_database_engine):
        """Second lookup returns the same cached value."""
        r1 = land_use_database_engine.get_agb_default("FOREST_LAND", "TROPICAL_WET")
        r2 = land_use_database_engine.get_agb_default("FOREST_LAND", "TROPICAL_WET")
        assert r1 == r2


# ===========================================================================
# BGB Default Tests
# ===========================================================================


class TestGetBGBDefault:
    """Tests for LandUseDatabaseEngine.get_bgb_default()."""

    def test_forest_tropical_wet_bgb(self, land_use_database_engine):
        """Forest tropical wet BGB = AGB * ratio (180 * 0.24 = 43.2)."""
        bgb = land_use_database_engine.get_bgb_default("FOREST_LAND", "TROPICAL_WET")
        expected = Decimal("180") * Decimal("0.24")
        assert bgb == expected.quantize(Decimal("0.00000001"))

    def test_forest_tropical_dry_bgb(self, land_use_database_engine):
        """Forest tropical dry BGB uses correct ratio for AGB < 75."""
        bgb = land_use_database_engine.get_bgb_default("FOREST_LAND", "TROPICAL_DRY")
        agb = Decimal("65")
        ratio = Decimal("0.56")
        expected = (agb * ratio).quantize(Decimal("0.00000001"))
        assert bgb == expected

    def test_bgb_with_agb_override(self, land_use_database_engine):
        """BGB calculation uses agb_override when provided."""
        bgb = land_use_database_engine.get_bgb_default(
            "FOREST_LAND", "TROPICAL_WET", agb_override=Decimal("50")
        )
        ratio = Decimal("0.37")
        expected = (Decimal("50") * ratio).quantize(Decimal("0.00000001"))
        assert bgb == expected

    def test_result_is_decimal(self, land_use_database_engine):
        """Returned BGB value is a Decimal instance."""
        result = land_use_database_engine.get_bgb_default("FOREST_LAND", "TROPICAL_WET")
        assert isinstance(result, Decimal)

    @pytest.mark.parametrize("zone", [
        "TROPICAL_WET", "TROPICAL_MOIST", "TROPICAL_DRY",
        "SUBTROPICAL_HUMID", "TEMPERATE_OCEANIC", "BOREAL_MOIST", "POLAR",
    ])
    def test_bgb_non_negative(self, land_use_database_engine, zone):
        """BGB is non-negative for all climate zones."""
        bgb = land_use_database_engine.get_bgb_default("FOREST_LAND", zone)
        assert bgb >= Decimal("0")


# ===========================================================================
# Root-to-Shoot Ratio Tests
# ===========================================================================


class TestGetRootShootRatio:
    """Tests for LandUseDatabaseEngine.get_root_shoot_ratio()."""

    def test_tropical_wet_high_agb(self, land_use_database_engine):
        """Tropical wet with high AGB uses 'high' ratio (0.24)."""
        ratio = land_use_database_engine.get_root_shoot_ratio(
            "TROPICAL_WET", Decimal("180")
        )
        assert ratio == Decimal("0.24")

    def test_tropical_wet_low_agb(self, land_use_database_engine):
        """Tropical wet with low AGB uses 'low' ratio (0.37)."""
        ratio = land_use_database_engine.get_root_shoot_ratio(
            "TROPICAL_WET", Decimal("50")
        )
        assert ratio == Decimal("0.37")

    def test_threshold_at_75_uses_high(self, land_use_database_engine):
        """AGB exactly at 75 tC/ha uses 'high' ratio."""
        ratio = land_use_database_engine.get_root_shoot_ratio(
            "TROPICAL_WET", Decimal("75")
        )
        assert ratio == Decimal("0.24")

    def test_threshold_below_75_uses_low(self, land_use_database_engine):
        """AGB just below 75 tC/ha uses 'low' ratio."""
        ratio = land_use_database_engine.get_root_shoot_ratio(
            "TROPICAL_WET", Decimal("74.99")
        )
        assert ratio == Decimal("0.37")

    @pytest.mark.parametrize("zone", [z.value for z in ClimateZone])
    def test_all_zones_have_ratios(self, land_use_database_engine, zone):
        """Every climate zone has root-shoot ratios for both thresholds."""
        ratio_high = land_use_database_engine.get_root_shoot_ratio(
            zone, Decimal("100")
        )
        ratio_low = land_use_database_engine.get_root_shoot_ratio(
            zone, Decimal("10")
        )
        assert isinstance(ratio_high, Decimal)
        assert isinstance(ratio_low, Decimal)

    def test_invalid_zone_raises(self, land_use_database_engine):
        """Invalid climate zone raises ValueError."""
        with pytest.raises(ValueError, match="Unknown climate zone"):
            land_use_database_engine.get_root_shoot_ratio("INVALID", Decimal("100"))


# ===========================================================================
# Dead Wood Default Tests
# ===========================================================================


class TestGetDeadWoodDefault:
    """Tests for LandUseDatabaseEngine dead wood methods."""

    def test_forest_tropical_wet_dead_wood(self, land_use_database_engine):
        """Forest tropical wet dead wood = AGB * fraction (180 * 0.08)."""
        result = land_use_database_engine.get_dead_wood_default(
            "FOREST_LAND", "TROPICAL_WET"
        )
        expected = (Decimal("180") * Decimal("0.08")).quantize(Decimal("0.00000001"))
        assert result == expected

    def test_cropland_dead_wood_is_zero(self, land_use_database_engine):
        """Cropland has zero dead wood."""
        result = land_use_database_engine.get_dead_wood_default(
            "CROPLAND", "TROPICAL_WET"
        )
        assert result == Decimal("0")

    def test_dead_wood_fraction_method(self, land_use_database_engine):
        """get_dead_wood_fraction returns the fraction, not the stock."""
        fraction = land_use_database_engine.get_dead_wood_fraction(
            "FOREST_LAND", "TROPICAL_WET"
        )
        assert fraction == Decimal("0.08")

    def test_dead_wood_turnover_tropical_wet(self, land_use_database_engine):
        """Dead wood turnover rate for tropical wet is 0.10."""
        rate = land_use_database_engine.get_dead_wood_turnover("TROPICAL_WET")
        assert rate == Decimal("0.10")

    @pytest.mark.parametrize("zone", [z.value for z in ClimateZone])
    def test_dead_wood_non_negative(self, land_use_database_engine, zone):
        """Dead wood stock is non-negative for all zones."""
        result = land_use_database_engine.get_dead_wood_default("FOREST_LAND", zone)
        assert result >= Decimal("0")


# ===========================================================================
# Litter Default Tests
# ===========================================================================


class TestGetLitterDefault:
    """Tests for LandUseDatabaseEngine.get_litter_default()."""

    def test_forest_tropical_wet(self, land_use_database_engine):
        """Forest tropical wet litter stock is 5.2 tC/ha."""
        result = land_use_database_engine.get_litter_default(
            "FOREST_LAND", "TROPICAL_WET"
        )
        assert result == Decimal("5.2")

    def test_forest_boreal_moist(self, land_use_database_engine):
        """Forest boreal moist litter stock is 30.0 tC/ha."""
        result = land_use_database_engine.get_litter_default(
            "FOREST_LAND", "BOREAL_MOIST"
        )
        assert result == Decimal("30.0")

    def test_cropland_litter_is_zero(self, land_use_database_engine):
        """Cropland has zero litter stock."""
        result = land_use_database_engine.get_litter_default(
            "CROPLAND", "TROPICAL_WET"
        )
        assert result == Decimal("0")

    def test_grassland_tropical_wet(self, land_use_database_engine):
        """Grassland tropical wet litter stock is 0.8 tC/ha."""
        result = land_use_database_engine.get_litter_default(
            "GRASSLAND", "TROPICAL_WET"
        )
        assert result == Decimal("0.8")

    @pytest.mark.parametrize("category", [c.value for c in LandCategory])
    def test_all_categories_have_litter(self, land_use_database_engine, category):
        """All land categories have litter stocks for tropical_wet."""
        result = land_use_database_engine.get_litter_default(category, "TROPICAL_WET")
        assert isinstance(result, Decimal)


# ===========================================================================
# SOC Reference Tests
# ===========================================================================


class TestGetSOCReference:
    """Tests for LandUseDatabaseEngine.get_soc_reference()."""

    @pytest.mark.parametrize("zone,soil,expected", [
        ("TROPICAL_WET", "HIGH_ACTIVITY_CLAY", Decimal("65")),
        ("TROPICAL_WET", "LOW_ACTIVITY_CLAY", Decimal("47")),
        ("TROPICAL_WET", "SANDY", Decimal("39")),
        ("TROPICAL_WET", "SPODIC", Decimal("70")),
        ("TROPICAL_WET", "VOLCANIC", Decimal("130")),
        ("TROPICAL_WET", "WETLAND", Decimal("86")),
        ("TROPICAL_WET", "ORGANIC", Decimal("200")),
    ])
    def test_tropical_wet_soc(self, land_use_database_engine, zone, soil, expected):
        """Tropical wet SOC reference stocks match IPCC Table 2.3."""
        result = land_use_database_engine.get_soc_reference(zone, soil)
        assert result == expected

    @pytest.mark.parametrize("zone,soil,expected", [
        ("TROPICAL_DRY", "HIGH_ACTIVITY_CLAY", Decimal("38")),
        ("SUBTROPICAL_HUMID", "HIGH_ACTIVITY_CLAY", Decimal("88")),
        ("TEMPERATE_OCEANIC", "HIGH_ACTIVITY_CLAY", Decimal("95")),
        ("BOREAL_MOIST", "HIGH_ACTIVITY_CLAY", Decimal("68")),
    ])
    def test_various_zones_soc(self, land_use_database_engine, zone, soil, expected):
        """SOC reference stocks for various zones match IPCC values."""
        result = land_use_database_engine.get_soc_reference(zone, soil)
        assert result == expected

    def test_invalid_zone_raises(self, land_use_database_engine):
        """Invalid climate zone raises ValueError."""
        with pytest.raises(ValueError, match="Unknown climate zone"):
            land_use_database_engine.get_soc_reference("INVALID", "SANDY")

    def test_invalid_soil_raises(self, land_use_database_engine):
        """Invalid soil type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown soil type"):
            land_use_database_engine.get_soc_reference("TROPICAL_WET", "INVALID")

    def test_organic_soil_always_200(self, land_use_database_engine):
        """Organic soil SOC reference is 200 tC/ha across all zones."""
        for zone in [z.value for z in ClimateZone]:
            if "ORGANIC" in SOC_REFERENCE_STOCKS.get(zone, {}):
                result = land_use_database_engine.get_soc_reference(zone, "ORGANIC")
                assert result == Decimal("200")

    def test_result_is_decimal(self, land_use_database_engine):
        """SOC reference stock is a Decimal instance."""
        result = land_use_database_engine.get_soc_reference(
            "TROPICAL_WET", "HIGH_ACTIVITY_CLAY"
        )
        assert isinstance(result, Decimal)


# ===========================================================================
# Growth Rate Tests
# ===========================================================================


class TestGetGrowthRate:
    """Tests for LandUseDatabaseEngine.get_growth_rate()."""

    @pytest.mark.parametrize("category,zone,expected", [
        ("FOREST_LAND", "TROPICAL_WET", Decimal("7.0")),
        ("FOREST_LAND", "TROPICAL_MOIST", Decimal("5.0")),
        ("FOREST_LAND", "TROPICAL_DRY", Decimal("2.4")),
        ("FOREST_LAND", "TEMPERATE_OCEANIC", Decimal("4.4")),
        ("FOREST_LAND", "BOREAL_DRY", Decimal("0.9")),
        ("FOREST_LAND", "POLAR", Decimal("0.1")),
    ])
    def test_forest_growth_rates(self, land_use_database_engine, category, zone, expected):
        """Forest growth rates match IPCC defaults."""
        result = land_use_database_engine.get_growth_rate(category, zone)
        assert result == expected

    @pytest.mark.parametrize("category,zone,expected", [
        ("CROPLAND", "TROPICAL_WET", Decimal("3.0")),
        ("GRASSLAND", "TROPICAL_WET", Decimal("2.0")),
        ("WETLANDS", "TROPICAL_WET", Decimal("6.0")),
        ("OTHER_LAND", "TROPICAL_WET", Decimal("0")),
    ])
    def test_other_category_growth_rates(self, land_use_database_engine, category, zone, expected):
        """Non-forest growth rates match expected values."""
        result = land_use_database_engine.get_growth_rate(category, zone)
        assert result == expected

    def test_result_is_decimal(self, land_use_database_engine):
        """Growth rate is returned as a Decimal."""
        result = land_use_database_engine.get_growth_rate("FOREST_LAND", "TROPICAL_WET")
        assert isinstance(result, Decimal)

    def test_other_land_growth_rate_is_zero(self, land_use_database_engine):
        """Other land has zero growth rate."""
        result = land_use_database_engine.get_growth_rate("OTHER_LAND", "TROPICAL_WET")
        assert result == Decimal("0")


# ===========================================================================
# Fire Emission Factor Tests
# ===========================================================================


class TestGetFireEF:
    """Tests for LandUseDatabaseEngine.get_fire_ef()."""

    def test_forest_wildfire_ef(self, land_use_database_engine):
        """Forest wildfire emission factors include expected keys."""
        result = land_use_database_engine.get_fire_ef("FOREST_LAND", "FIRE_WILDFIRE")
        assert "combustion_factor" in result
        assert "ef_co2_g_per_kg" in result
        assert "ef_ch4_g_per_kg" in result
        assert "ef_n2o_g_per_kg" in result

    def test_forest_wildfire_combustion_factor(self, land_use_database_engine):
        """Forest wildfire combustion factor is 0.45."""
        result = land_use_database_engine.get_fire_ef("FOREST_LAND", "FIRE_WILDFIRE")
        assert result["combustion_factor"] == Decimal("0.45")

    def test_forest_prescribed_combustion_factor(self, land_use_database_engine):
        """Forest prescribed fire combustion factor is 0.30."""
        result = land_use_database_engine.get_fire_ef("FOREST_LAND", "FIRE_PRESCRIBED")
        assert result["combustion_factor"] == Decimal("0.30")

    def test_grassland_wildfire_combustion(self, land_use_database_engine):
        """Grassland wildfire combustion factor is 0.74."""
        result = land_use_database_engine.get_fire_ef("GRASSLAND", "FIRE_WILDFIRE")
        assert result["combustion_factor"] == Decimal("0.74")

    @pytest.mark.parametrize("category", [
        "FOREST_LAND", "CROPLAND", "GRASSLAND", "WETLANDS",
        "SETTLEMENTS", "OTHER_LAND",
    ])
    def test_all_categories_have_wildfire_ef(self, land_use_database_engine, category):
        """All land categories have wildfire emission factors."""
        result = land_use_database_engine.get_fire_ef(category, "FIRE_WILDFIRE")
        assert result is not None


# ===========================================================================
# Peatland Emission Factor Tests
# ===========================================================================


class TestGetPeatlandEF:
    """Tests for peatland emission factors."""

    def test_drained_tropical_co2(self):
        """Drained tropical peatland has CO2 EF of 15.0."""
        ef = PEATLAND_EF["DRAINED_TROPICAL"]
        assert ef.co2_tc_ha_yr == Decimal("15.0")

    def test_drained_tropical_ch4(self):
        """Drained tropical peatland has CH4 EF of 5.0."""
        ef = PEATLAND_EF["DRAINED_TROPICAL"]
        assert ef.ch4_kg_ha_yr == Decimal("5.0")

    def test_rewetted_tropical_co2(self):
        """Rewetted tropical peatland has CO2 EF of 0.0."""
        ef = PEATLAND_EF["REWETTED_TROPICAL"]
        assert ef.co2_tc_ha_yr == Decimal("0.0")

    def test_intact_tropical_is_carbon_sink(self):
        """Intact tropical peatland CO2 is negative (carbon sink)."""
        ef = PEATLAND_EF["INTACT_TROPICAL"]
        assert ef.co2_tc_ha_yr < Decimal("0")

    def test_fire_tropical_high_co2(self):
        """Fire tropical peatland has very high CO2 EF."""
        ef = PEATLAND_EF["FIRE_TROPICAL"]
        assert ef.co2_tc_ha_yr == Decimal("200.0")

    def test_all_entries_have_source(self):
        """All peatland EF entries have a source reference."""
        for key, ef in PEATLAND_EF.items():
            assert ef.source is not None and ef.source != "", (
                f"Peatland EF '{key}' missing source"
            )


# ===========================================================================
# N2O Emission Factor Tests
# ===========================================================================


class TestN2OEF:
    """Tests for N2O soil emission factors."""

    def test_ef1_synthetic_fertilizer(self):
        """EF1 for synthetic fertilizer is 0.01."""
        assert N2O_SOIL_EF["EF1_SYNTHETIC_FERTILIZER"] == Decimal("0.01")

    def test_ef1_organic_amendment(self):
        """EF1 for organic amendment is 0.01."""
        assert N2O_SOIL_EF["EF1_ORGANIC_AMENDMENT"] == Decimal("0.01")

    def test_ef2_tropical_organic(self):
        """EF2 for tropical organic soil is 0.016."""
        assert N2O_SOIL_EF["EF2_TROPICAL_ORGANIC"] == Decimal("0.016")

    def test_frac_leach(self):
        """Leaching fraction is 0.30."""
        assert N2O_SOIL_EF["FRAC_LEACH"] == Decimal("0.30")


# ===========================================================================
# Land Subcategory Tests
# ===========================================================================


class TestGetLandSubcategories:
    """Tests for land subcategory lookup."""

    @pytest.mark.parametrize("category", [
        "FOREST_LAND", "CROPLAND", "GRASSLAND",
        "WETLANDS", "SETTLEMENTS", "OTHER_LAND",
    ])
    def test_all_six_categories_have_subcategories(self, category):
        """All 6 IPCC land categories have defined subcategories."""
        assert category in LAND_SUBCATEGORIES
        assert len(LAND_SUBCATEGORIES[category]) > 0

    def test_forest_land_has_11_subcategories(self):
        """Forest land has 11 subcategories."""
        assert len(LAND_SUBCATEGORIES["FOREST_LAND"]) == 11

    def test_cropland_has_6_subcategories(self):
        """Cropland has 6 subcategories."""
        assert len(LAND_SUBCATEGORIES["CROPLAND"]) == 6

    def test_subcategory_has_required_fields(self):
        """Each subcategory has code, name, parent_category, description, typical_agb."""
        for category, subs in LAND_SUBCATEGORIES.items():
            for sub in subs:
                assert sub.code is not None and sub.code != ""
                assert sub.name is not None and sub.name != ""
                assert sub.parent_category is not None
                assert sub.description is not None
                assert isinstance(sub.typical_agb_tc_ha, Decimal)

    def test_forest_subcategory_parent_is_forest(self):
        """All forest subcategories have FOREST_LAND as parent."""
        for sub in LAND_SUBCATEGORIES["FOREST_LAND"]:
            assert sub.parent_category == LandCategory.FOREST_LAND


# ===========================================================================
# Module-Level Constants Tests
# ===========================================================================


class TestModuleConstants:
    """Tests for module-level constants in land_use_database."""

    def test_carbon_fraction(self):
        """CARBON_FRACTION is 0.47."""
        assert CARBON_FRACTION == Decimal("0.47")

    def test_conversion_factor_co2_c(self):
        """CONVERSION_FACTOR_CO2_C is approximately 3.667."""
        assert abs(CONVERSION_FACTOR_CO2_C - Decimal("3.66667")) < Decimal("0.001")

    def test_gwp_ar6_co2_is_one(self):
        """GWP AR6 CO2 is 1."""
        assert GWP_VALUES["AR6"]["CO2"] == Decimal("1")

    def test_gwp_ar6_ch4(self):
        """GWP AR6 CH4 is 29.8."""
        assert GWP_VALUES["AR6"]["CH4"] == Decimal("29.8")

    def test_gwp_ar6_n2o(self):
        """GWP AR6 N2O is 273."""
        assert GWP_VALUES["AR6"]["N2O"] == Decimal("273")

    def test_soc_land_use_factors_has_entries(self):
        """SOC land-use factors dictionary is not empty."""
        assert len(SOC_LAND_USE_FACTORS) > 0

    def test_soc_management_factors_has_entries(self):
        """SOC management factors dictionary is not empty."""
        assert len(SOC_MANAGEMENT_FACTORS) > 0

    def test_soc_input_factors_has_entries(self):
        """SOC input factors dictionary is not empty."""
        assert len(SOC_INPUT_FACTORS) > 0

    def test_combustion_factors_has_entries(self):
        """Combustion factors dictionary is not empty."""
        assert len(COMBUSTION_FACTORS) > 0


# ===========================================================================
# Engine Initialization and Misc Tests
# ===========================================================================


class TestEngineInit:
    """Tests for LandUseDatabaseEngine initialization."""

    def test_engine_creates_successfully(self):
        """Engine initializes without error."""
        db = LandUseDatabaseEngine()
        assert db is not None

    def test_engine_has_empty_custom_factors(self):
        """Engine starts with empty custom factor registry."""
        db = LandUseDatabaseEngine()
        assert db._custom_factors == {}

    def test_lookup_counter_starts_at_zero(self):
        """Lookup counter starts at zero."""
        db = LandUseDatabaseEngine()
        assert db._total_lookups == 0

    def test_lookup_counter_increments(self):
        """Lookup counter increments on each call."""
        db = LandUseDatabaseEngine()
        db.get_agb_default("FOREST_LAND", "TROPICAL_WET")
        assert db._total_lookups >= 1
        db.get_agb_default("CROPLAND", "TROPICAL_WET")
        assert db._total_lookups >= 2
