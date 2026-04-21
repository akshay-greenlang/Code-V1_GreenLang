# -*- coding: utf-8 -*-
"""GAP-7 Circular economy taxonomy tests."""
from __future__ import annotations

import pytest

from greenlang.factors.mapping.base import MappingConfidence, MappingError
from greenlang.factors.mapping.circular_economy import (
    CIRCULAR_TAXONOMY,
    CircularMaterialFlow,
    CircularRoute,
    MaterialLifecycle,
    RecycledContent,
    RecycledSource,
    get_reduction_ratio,
    list_circular_materials,
    load_recycled_content_factors,
    map_circular_flow,
)


class TestRecycledContent:
    def test_valid_percentage(self):
        rc = RecycledContent(50.0, RecycledSource.POST_CONSUMER)
        assert rc.percentage == 50.0
        assert rc.as_fraction() == 0.5
        assert rc.source == RecycledSource.POST_CONSUMER

    def test_zero_percentage_allowed(self):
        rc = RecycledContent(0.0)
        assert rc.percentage == 0.0
        assert rc.source == RecycledSource.UNSPECIFIED

    def test_hundred_percentage_allowed(self):
        rc = RecycledContent(100.0)
        assert rc.as_fraction() == 1.0

    def test_negative_rejected(self):
        with pytest.raises(MappingError):
            RecycledContent(-5.0)

    def test_over_hundred_rejected(self):
        with pytest.raises(MappingError):
            RecycledContent(125.0)

    def test_to_dict_round_trip(self):
        rc = RecycledContent(33.3, RecycledSource.POST_INDUSTRIAL)
        d = rc.to_dict()
        assert d["percentage"] == 33.3
        assert d["source"] == "post_industrial"
        assert pytest.approx(d["fraction"], rel=1e-6) == 0.333


class TestEnums:
    def test_material_lifecycle_values(self):
        assert MaterialLifecycle.VIRGIN.value == "virgin"
        assert MaterialLifecycle.RECYCLED.value == "recycled"

    def test_circular_route_values(self):
        assert CircularRoute.PRIMARY.value == "primary"
        assert CircularRoute.CHEMICAL_RECYCLED.value == "chemical_recycled"

    def test_recycled_source_values(self):
        vals = {m.value for m in RecycledSource}
        assert "pre_consumer" in vals
        assert "post_consumer" in vals


class TestCircularMaterialFlow:
    def test_construction(self):
        flow = CircularMaterialFlow(
            material="steel",
            route=CircularRoute.SECONDARY,
            lifecycle=MaterialLifecycle.RECYCLED,
            recycled_content=RecycledContent(90.0, RecycledSource.POST_CONSUMER),
        )
        d = flow.to_dict()
        assert d["material"] == "steel"
        assert d["route"] == "secondary"
        assert d["lifecycle"] == "recycled"
        assert d["recycled_content"]["percentage"] == 90.0


class TestYamlLoader:
    def test_loads_data(self):
        data = load_recycled_content_factors()
        assert "materials" in data
        assert "steel" in data["materials"]
        assert "metadata" in data
        assert data["metadata"]["version"]

    def test_loader_is_cached(self):
        # Two calls return the same object (lru_cache).
        a = load_recycled_content_factors()
        b = load_recycled_content_factors()
        assert a is b

    def test_primary_gt_secondary_for_all_materials(self):
        data = load_recycled_content_factors()
        for key, body in data["materials"].items():
            p = body["primary"]["emissions_kgco2e_per_kg"]
            s = body["secondary"]["emissions_kgco2e_per_kg"]
            assert p > s, f"{key}: primary ({p}) should exceed secondary ({s})"

    def test_reduction_ratio_is_plausible(self):
        data = load_recycled_content_factors()
        for key, body in data["materials"].items():
            ratio = body.get("reduction_ratio")
            assert ratio is not None, f"{key} missing reduction_ratio"
            assert 0.0 < ratio < 1.0


class TestListing:
    def test_list_includes_steel_and_aluminium(self):
        keys = list_circular_materials()
        assert "steel" in keys
        assert "aluminium" in keys
        # Sorted
        assert keys == sorted(keys)

    def test_plastics_covered(self):
        keys = set(list_circular_materials())
        for plastic in (
            "plastic_pet", "plastic_hdpe", "plastic_ldpe",
            "plastic_pp", "plastic_ps", "plastic_pvc",
        ):
            assert plastic in keys, f"{plastic} missing"


class TestMapCircularFlow:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("steel", "steel"),
            ("HRC", "steel"),
            ("aluminum ingot", "aluminium"),
            ("aluminium", "aluminium"),
            ("PET", "plastic_pet"),
            ("HDPE", "plastic_hdpe"),
            ("rPET", "plastic_pet"),
            ("cardboard", "cardboard"),
            ("cullet", "glass"),
            ("ready-mix concrete", "concrete"),
            ("polyester", "polyester"),
            ("Li-ion battery", "battery_li_ion"),
            ("neodymium", "rare_earth_neodymium"),
        ],
    )
    def test_known_materials_match(self, text, expected):
        r = map_circular_flow(text)
        assert r.canonical["material"] == expected

    def test_unknown_material_returns_none(self):
        r = map_circular_flow("unobtainium crystal")
        assert r.canonical is None
        assert r.band == MappingConfidence.UNKNOWN

    def test_primary_route_default(self):
        r = map_circular_flow("steel", recycled_content_pct=0.0)
        assert r.canonical["route"] == "primary"
        assert r.canonical["lifecycle"] == "virgin"

    def test_secondary_route_at_high_recycled(self):
        r = map_circular_flow("steel", recycled_content_pct=95.0)
        assert r.canonical["route"] == "secondary"
        assert r.canonical["lifecycle"] == "recycled"

    def test_mid_recycled_is_mechanical(self):
        r = map_circular_flow("aluminium", recycled_content_pct=40.0)
        assert r.canonical["route"] == "mechanical_recycled"
        assert r.canonical["lifecycle"] == "recycled"

    def test_adjusted_factor_blends_correctly(self):
        # 100% recycled aluminium should equal secondary factor
        r = map_circular_flow("aluminium", recycled_content_pct=100.0)
        assert pytest.approx(r.canonical["adjusted_factor_kgco2e_per_kg"], rel=0.01) == \
            r.canonical["secondary_factor_kgco2e_per_kg"]

    def test_adjusted_factor_zero_recycled_equals_primary(self):
        r = map_circular_flow("steel", recycled_content_pct=0.0)
        assert pytest.approx(r.canonical["adjusted_factor_kgco2e_per_kg"], rel=0.01) == \
            r.canonical["primary_factor_kgco2e_per_kg"]

    def test_adjusted_factor_midpoint(self):
        r = map_circular_flow("steel", recycled_content_pct=50.0)
        primary = r.canonical["primary_factor_kgco2e_per_kg"]
        secondary = r.canonical["secondary_factor_kgco2e_per_kg"]
        expected = 0.5 * primary + 0.5 * secondary
        assert pytest.approx(r.canonical["adjusted_factor_kgco2e_per_kg"], rel=0.01) == expected

    def test_invalid_route_override_rejected(self):
        with pytest.raises(MappingError):
            map_circular_flow("steel", route_override="time-travel-route")

    def test_route_override_accepted(self):
        r = map_circular_flow("steel", recycled_content_pct=0.0, route_override="remanufactured")
        assert r.canonical["route"] == "remanufactured"
        assert r.canonical["lifecycle"] == "remanufactured"

    def test_recycled_content_pct_clamped(self):
        r = map_circular_flow("steel", recycled_content_pct=150.0)
        assert r.canonical["recycled_content_pct"] == 100.0
        r = map_circular_flow("steel", recycled_content_pct=-10.0)
        assert r.canonical["recycled_content_pct"] == 0.0

    def test_source_enum_parsed(self):
        r = map_circular_flow("steel", recycled_content_pct=80.0, source="post_consumer")
        assert r.canonical["recycled_source"] == "post_consumer"

    def test_invalid_source_falls_back_to_unspecified(self):
        r = map_circular_flow("steel", recycled_content_pct=80.0, source="rando-source")
        assert r.canonical["recycled_source"] == "unspecified"

    def test_empty_input_returns_none(self):
        r = map_circular_flow("")
        assert r.canonical is None

    def test_case_insensitive_lookup(self):
        a = map_circular_flow("STEEL")
        b = map_circular_flow("steel")
        assert a.canonical["material"] == b.canonical["material"]

    def test_rationale_includes_factors(self):
        r = map_circular_flow("paper")
        assert "primary=" in r.rationale
        assert "secondary=" in r.rationale


class TestReductionRatio:
    def test_reduction_ratio_steel(self):
        ratio = get_reduction_ratio("steel")
        assert ratio is not None
        assert 0 < ratio < 0.5

    def test_reduction_ratio_aluminium(self):
        ratio = get_reduction_ratio("aluminum ingot")
        assert ratio is not None
        assert ratio < 0.1  # secondary aluminium ~ 5% of primary

    def test_reduction_ratio_unknown_returns_none(self):
        assert get_reduction_ratio("mystery material") is None


class TestTaxonomyStaticStructure:
    def test_every_key_has_synonyms_list(self):
        for key, body in CIRCULAR_TAXONOMY.items():
            assert isinstance(body["synonyms"], list), f"{key} synonyms is not a list"
            assert len(body["synonyms"]) >= 1, f"{key} must have at least one synonym"

    def test_every_key_has_meta(self):
        for key, body in CIRCULAR_TAXONOMY.items():
            assert "meta" in body
            assert "primary_route" in body["meta"]
            assert "secondary_route" in body["meta"]
