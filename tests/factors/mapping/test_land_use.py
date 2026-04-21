# -*- coding: utf-8 -*-
"""GAP-7 Land-use / LUC commodity tests."""
from __future__ import annotations

import pytest

from greenlang.factors.mapping.base import MappingConfidence, MappingError
from greenlang.factors.mapping.land_use import (
    DeforestationRiskTier,
    LUCCommodity,
    LandUseCategory,
    PermanenceClass,
    ProductionRegion,
    eudr_is_in_scope,
    get_risk_tier,
    list_luc_commodities,
    list_regions_for,
    load_luc_commodity_data,
    map_land_use,
    map_luc_commodity,
)


class TestEnums:
    def test_land_use_categories(self):
        vals = {c.value for c in LandUseCategory}
        assert "forest" in vals
        assert "cropland" in vals
        assert "grassland" in vals
        assert "wetland" in vals
        assert "settlement" in vals
        assert "other_land" in vals

    def test_risk_tiers(self):
        vals = {r.value for r in DeforestationRiskTier}
        assert vals >= {"high", "medium", "low", "negligible", "unknown"}

    def test_permanence_classes(self):
        vals = {p.value for p in PermanenceClass}
        assert vals >= {"ephemeral", "short", "medium", "long", "unknown"}


class TestProductionRegion:
    def test_valid_iso3(self):
        r = ProductionRegion(
            "IDN", "Indonesia", DeforestationRiskTier.HIGH, 8.2, ecoregion="peat"
        )
        assert r.iso3 == "IDN"
        assert r.risk_tier == DeforestationRiskTier.HIGH
        assert r.ecoregion == "peat"

    def test_iso3_uppercased(self):
        r = ProductionRegion("idn", "Indonesia", DeforestationRiskTier.HIGH, 5.0)
        assert r.iso3 == "IDN"

    def test_invalid_iso3_length_rejected(self):
        with pytest.raises(MappingError):
            ProductionRegion("ID", "Indonesia", DeforestationRiskTier.HIGH, 5.0)

    def test_to_dict(self):
        r = ProductionRegion("BRA", "Brazil", DeforestationRiskTier.HIGH, 2.8)
        d = r.to_dict()
        assert d["iso3"] == "BRA"
        assert d["risk_tier"] == "high"
        assert d["luc_tco2e_per_t_product"] == 2.8


class TestYamlLoader:
    def test_loader_returns_dict_of_records(self):
        data = load_luc_commodity_data()
        assert isinstance(data, dict)
        assert "palm_oil" in data
        assert isinstance(data["palm_oil"], LUCCommodity)

    def test_all_commodities_have_regions(self):
        data = load_luc_commodity_data()
        for key, rec in data.items():
            assert rec.regions, f"{key} has no regions"

    def test_regions_have_valid_risk_tier(self):
        data = load_luc_commodity_data()
        for rec in data.values():
            for region in rec.regions.values():
                assert isinstance(region.risk_tier, DeforestationRiskTier)

    def test_cached(self):
        a = load_luc_commodity_data()
        b = load_luc_commodity_data()
        assert a is b


class TestListings:
    def test_list_luc_commodities(self):
        keys = list_luc_commodities()
        assert "palm_oil" in keys
        assert "beef" in keys
        assert "wood" in keys
        assert keys == sorted(keys)

    def test_list_regions_for(self):
        regions = list_regions_for("soy")
        assert "BRA" in regions
        assert "USA" in regions

    def test_list_regions_for_unknown(self):
        assert list_regions_for("unknownX") == []


class TestMapLucCommodity:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("palm", "palm_oil"),
            ("crude palm oil", "palm_oil"),
            ("soy", "soy"),
            ("soybean", "soy"),
            ("beef", "beef"),
            ("cattle", "beef"),
            ("cocoa beans", "cocoa"),
            ("coffee", "coffee"),
            ("natural rubber", "rubber"),
            ("latex", "rubber"),
            ("pulp", "pulp_paper"),
            ("timber", "wood"),
        ],
    )
    def test_known_commodities(self, text, expected):
        r = map_luc_commodity(text)
        assert r.canonical == expected

    def test_unknown_returns_none(self):
        r = map_luc_commodity("asteroid iron")
        assert r.canonical is None
        assert r.band == MappingConfidence.UNKNOWN


class TestMapLandUse:
    def test_palm_indonesia_high_risk(self):
        r = map_land_use("palm oil", "IDN")
        assert r.canonical["commodity"] == "palm_oil"
        assert r.canonical["iso3_country"] == "IDN"
        assert r.canonical["risk_tier"] == "high"
        assert r.canonical["luc_tco2e_per_t_product"] > 5.0
        assert r.canonical["eudr_covered"] is True

    def test_soy_brazil_vs_usa(self):
        brz = map_land_use("soy", "BRA")
        usa = map_land_use("soy", "USA")
        assert brz.canonical["luc_tco2e_per_t_product"] > usa.canonical["luc_tco2e_per_t_product"]
        assert brz.canonical["risk_tier"] == "high"
        assert usa.canonical["risk_tier"] == "low"

    def test_beef_fallback_to_base_when_country_unknown(self):
        r = map_land_use("beef", "ZZZ")
        assert r.canonical["region_label"] is None
        # base factor is used
        assert r.canonical["luc_tco2e_per_t_product"] == r.canonical["base_luc_tco2e_per_t_product"]

    def test_no_country_returns_base(self):
        r = map_land_use("cocoa")
        assert r.canonical["iso3_country"] is None
        assert r.canonical["luc_tco2e_per_t_product"] == r.canonical["base_luc_tco2e_per_t_product"]

    def test_unknown_commodity(self):
        r = map_land_use("asteroid iron", "USA")
        assert r.canonical is None

    def test_biogenic_and_fossil_shares_sum_to_one(self):
        r = map_land_use("palm oil", "IDN")
        bs = r.canonical["biogenic_share"]
        fs = r.canonical["fossil_share"]
        assert pytest.approx(bs + fs, rel=1e-6) == 1.0

    def test_permanence_class_present(self):
        r = map_land_use("wood", "CAN")
        assert r.canonical["permanence_class"] in {"ephemeral", "short", "medium", "long"}

    def test_rationale_informative(self):
        r = map_land_use("palm oil", "IDN")
        assert "palm_oil" in r.rationale
        assert "IDN" in r.rationale
        assert "risk=" in r.rationale

    def test_case_insensitive_commodity(self):
        a = map_land_use("PALM OIL", "IDN")
        b = map_land_use("palm oil", "IDN")
        assert a.canonical == b.canonical

    def test_iso3_case_insensitive(self):
        a = map_land_use("soy", "bra")
        b = map_land_use("soy", "BRA")
        assert a.canonical["iso3_country"] == b.canonical["iso3_country"]


class TestRiskTierHelper:
    def test_known_high_risk(self):
        assert get_risk_tier("palm", "IDN") == DeforestationRiskTier.HIGH

    def test_known_low_risk(self):
        assert get_risk_tier("soy", "USA") == DeforestationRiskTier.LOW

    def test_unknown_country(self):
        assert get_risk_tier("palm", "ZZZ") == DeforestationRiskTier.UNKNOWN

    def test_unknown_commodity(self):
        assert get_risk_tier("mystery", "BRA") == DeforestationRiskTier.UNKNOWN


class TestEudrScope:
    def test_palm_in_scope(self):
        assert eudr_is_in_scope("palm oil") is True

    def test_cocoa_in_scope(self):
        assert eudr_is_in_scope("cocoa beans") is True

    def test_unknown_commodity_out_of_scope(self):
        assert eudr_is_in_scope("mystery") is False


class TestLUCCommodityRecord:
    def test_region_for(self):
        rec = load_luc_commodity_data()["palm_oil"]
        assert rec.region_for("IDN") is not None
        assert rec.region_for("idn") is not None  # case-insensitive
        assert rec.region_for("ZZZ") is None

    def test_to_dict(self):
        rec = load_luc_commodity_data()["palm_oil"]
        d = rec.to_dict()
        assert d["key"] == "palm_oil"
        assert d["eudr_covered"] is True
        assert "IDN" in d["regions"]
