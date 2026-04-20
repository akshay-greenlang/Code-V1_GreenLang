# -*- coding: utf-8 -*-
"""Phase F4 — Mapping Layer tests (all 7 taxonomies)."""
from __future__ import annotations

import pytest

from greenlang.factors.mapping import (
    MappingConfidence,
    cross_map_classification,
    map_classification,
    map_electricity_market,
    map_fuel,
    map_material,
    map_spend,
    map_transport,
    map_waste,
    normalize_text,
)


# --------------------------------------------------------------------------
# Base helpers
# --------------------------------------------------------------------------


class TestNormalizeAndConfidence:
    def test_normalize_text_strips_and_lowers(self):
        assert normalize_text("No. 2 DIESEL!!") == "no 2 diesel"

    def test_normalize_text_handles_none(self):
        assert normalize_text(None) == ""

    def test_confidence_band_exact(self):
        assert MappingConfidence.from_score(0.97) == MappingConfidence.EXACT

    def test_confidence_band_medium(self):
        assert MappingConfidence.from_score(0.65) == MappingConfidence.MEDIUM

    def test_confidence_band_unknown(self):
        assert MappingConfidence.from_score(0.1) == MappingConfidence.UNKNOWN


# --------------------------------------------------------------------------
# Fuels
# --------------------------------------------------------------------------


class TestFuelMapping:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("diesel", "diesel"),
            ("No. 2 distillate", "diesel"),
            ("petrol", "gasoline"),
            ("Compressed natural gas", "natural_gas"),
            ("LNG", "natural_gas"),
            ("anthracite", "coal"),
            ("bioethanol", "ethanol"),
            ("Green hydrogen", "hydrogen"),
            ("grid electricity", "electricity"),
            ("wood pellets", "biomass"),
        ],
    )
    def test_known_fuels(self, text, expected):
        result = map_fuel(text)
        assert result.canonical == expected
        assert result.confidence >= 0.6

    def test_unknown_returns_none(self):
        result = map_fuel("magical unicorn dust")
        assert result.canonical is None
        assert result.band == MappingConfidence.UNKNOWN

    def test_result_has_rationale_and_raw(self):
        result = map_fuel("Diesel Fuel")
        assert result.raw_input == "Diesel Fuel"
        assert "fuel_family" in result.rationale
        assert result.to_dict()["canonical"] == "diesel"

    def test_empty_input_returns_unknown(self):
        result = map_fuel("")
        assert result.canonical is None


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


class TestTransportMapping:
    def test_heavy_truck_resolves_to_road(self):
        r = map_transport("40-tonne truck, long haul", payload_tonnes=25, distance_km=500)
        assert r.canonical["vehicle_class"] == "heavy_truck_40t"
        assert r.canonical["mode"] == "road"
        assert r.canonical["payload_tonnes"] == 25

    def test_mode_only_fallback(self):
        r = map_transport("sea")
        assert r.canonical["mode"] == "sea"
        assert r.canonical["vehicle_class"] is None

    def test_rail_electric(self):
        r = map_transport("electric rail")
        assert r.canonical["mode"] == "rail"
        # token overlap matches electric_freight_rail; mode must still be rail.

    def test_no_match(self):
        r = map_transport("underwater sled")
        assert r.canonical["mode"] is None
        assert r.band == MappingConfidence.UNKNOWN


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


class TestMaterialMapping:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hot rolled coil", "steel_hot_rolled_coil"),
            ("aluminium ingot", "aluminium_ingot_primary"),
            ("portland cement", "cement_portland"),
            ("urea", "fertilizer_urea"),
            ("HDPE", "plastic_hdpe"),
            ("PET", "plastic_pet"),
        ],
    )
    def test_known_materials(self, text, expected):
        r = map_material(text)
        assert r.canonical == expected

    def test_cbam_flag_in_rationale(self):
        r = map_material("steel rebar")
        assert "cbam_covered=True" in r.rationale

    def test_unknown_material(self):
        r = map_material("mystery meat")
        assert r.canonical is None


# --------------------------------------------------------------------------
# Waste
# --------------------------------------------------------------------------


class TestWasteMapping:
    def test_landfill(self):
        r = map_waste("sanitary landfill")
        assert r.canonical == "landfill"
        assert "generates_ch4=True" in r.rationale

    def test_anaerobic_digestion(self):
        r = map_waste("anaerobic digester")
        assert r.canonical == "anaerobic_digestion"

    def test_waste_to_energy(self):
        r = map_waste("waste to energy")
        assert r.canonical == "incineration_energy_recovery"

    def test_recycling(self):
        r = map_waste("material recycling")
        assert r.canonical == "recycling"

    def test_unknown_route(self):
        r = map_waste("space disposal")
        assert r.canonical is None


# --------------------------------------------------------------------------
# Electricity market
# --------------------------------------------------------------------------


class TestElectricityMarketMapping:
    def test_ppa(self):
        r = map_electricity_market("we signed a vPPA last quarter")
        assert r.canonical["category"] == "ppa"
        assert r.canonical["electricity_basis"] == "market_based"
        assert r.canonical["requires_certificate"] is True

    def test_grid_average_default(self):
        r = map_electricity_market("plain electricity line item")
        assert r.canonical["category"] == "grid_average"
        assert r.canonical["electricity_basis"] == "location_based"

    def test_rec(self):
        r = map_electricity_market("retired I-REC certificates")
        assert r.canonical["category"] == "rec"
        assert r.canonical["requires_certificate"] is True

    def test_residual_mix(self):
        r = map_electricity_market("EU residual mix fallback")
        assert r.canonical["category"] == "residual_mix"
        assert r.canonical["electricity_basis"] == "residual_mix"

    def test_onsite_solar(self):
        r = map_electricity_market("behind-the-meter onsite solar")
        assert r.canonical["category"] == "onsite_generation"

    def test_unknown_returns_none(self):
        r = map_electricity_market("random blob")
        assert r.canonical is None


# --------------------------------------------------------------------------
# Classifications (NAICS ↔ ISIC ↔ HS/CN ↔ GICS)
# --------------------------------------------------------------------------


class TestClassificationMapping:
    def test_naics_electric_power_generation(self):
        r = map_classification("NAICS", "221112")
        assert r.canonical["factor_family"] == "grid_intensity"
        assert r.canonical["isic"] == "D3510"

    def test_isic_lookup(self):
        r = map_classification("isic", "C2410")
        assert r.canonical["label"].startswith("Iron and steel")

    def test_hs_prefix_match(self):
        """HS code 7208 should match the '72' steel prefix."""
        r = map_classification("HS", "7208")
        assert r.canonical["factor_family"] == "material_embodied"
        assert r.canonical["label"].startswith("Iron and steel")

    def test_cn_prefix_match_aluminium(self):
        r = map_classification("CN", "7601.10")
        assert r.canonical["label"].startswith("Aluminium")

    def test_unknown_system_rejected(self):
        r = map_classification("bogus", "123")
        assert r.canonical is None
        assert "bogus" in r.rationale

    def test_cross_map_filters_target_systems(self):
        r = cross_map_classification("NAICS", "331110", target_systems=["isic", "gics"])
        assert set(r.canonical.keys()) == {"label", "factor_family", "isic", "gics"}
        assert r.canonical["isic"] == "C2410"


# --------------------------------------------------------------------------
# Spend
# --------------------------------------------------------------------------


class TestSpendMapping:
    def test_cloud_compute_routes_to_grid(self):
        r = map_spend("AWS services spend for Q2", amount_usd=42000)
        assert r.canonical["spend_category"] == "cloud_compute"
        assert r.canonical["factor_family"] == "grid_intensity"
        assert r.canonical["scope"] == "3"
        assert r.canonical["amount_usd"] == 42000

    def test_business_travel_air_routes_to_scope3_cat6(self):
        r = map_spend("Employee flights Q1")
        assert r.canonical["scope3_category"] == 6
        assert r.canonical["method_profile"] == "corporate_scope3"

    def test_professional_services_uses_finance_proxy(self):
        r = map_spend("Legal services invoice")
        assert r.canonical["factor_family"] == "finance_proxy"

    def test_unknown_spend(self):
        r = map_spend("mystery widget")
        assert r.canonical is None

    def test_rationale_includes_method_profile(self):
        r = map_spend("rail freight services")
        assert "freight_iso_14083" in r.rationale


# --------------------------------------------------------------------------
# End-to-end: mapping → resolution-engine compatibility
# --------------------------------------------------------------------------


class TestMappingToResolutionCompat:
    """Ensure mapping outputs use the same method_profile + factor_family
    string literals that the canonical-v2 enums accept."""

    def test_spend_method_profile_values_are_valid_enums(self):
        from greenlang.data.canonical_v2 import MethodProfile
        valid = {m.value for m in MethodProfile}
        for meta in [
            map_spend("electricity").canonical,
            map_spend("rail freight").canonical,
            map_spend("employee flights").canonical,
            map_spend("office supplies").canonical,
        ]:
            assert meta["method_profile"] in valid

    def test_spend_factor_family_values_are_valid_enums(self):
        from greenlang.data.canonical_v2 import FactorFamily
        valid = {f.value for f in FactorFamily}
        for meta in [
            map_spend("cloud compute").canonical,
            map_spend("trucking services").canonical,
            map_spend("waste collection").canonical,
        ]:
            assert meta["factor_family"] in valid
