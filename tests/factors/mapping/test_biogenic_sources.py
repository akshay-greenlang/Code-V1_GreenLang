# -*- coding: utf-8 -*-
"""GAP-7 Biogenic source taxonomy tests."""
from __future__ import annotations

import pytest

from greenlang.factors.mapping.base import MappingConfidence, MappingError
from greenlang.factors.mapping.biogenic_sources import (
    BIOGENIC_TAXONOMY,
    BiogenicCategory,
    BiogenicSource,
    CO2AccountingTreatment,
    SustainabilityCertification,
    get_certification_info,
    list_biogenic_sources,
    load_biogenic_sources,
    load_certifications,
    map_biogenic_source,
)


class TestEnums:
    def test_categories(self):
        vals = {c.value for c in BiogenicCategory}
        assert vals >= {
            "wood",
            "crop_residue",
            "manure",
            "municipal_waste_organic",
            "dedicated_energy_crop",
            "algae",
        }

    def test_co2_accounting(self):
        vals = {c.value for c in CO2AccountingTreatment}
        assert "carbon_neutral" in vals
        assert "short_rotation_sequestration" in vals

    def test_certifications(self):
        vals = {c.value for c in SustainabilityCertification}
        assert vals >= {"FSC", "PEFC", "RSPO", "Bonsucro", "RSB", "SBP"}


class TestBiogenicSourceRecord:
    def test_valid_construction(self):
        rec = BiogenicSource(
            key="test",
            category=BiogenicCategory.WOOD,
            description="test",
            moisture_content=0.10,
            lhv_gj_per_t=17.0,
            carbon_content=0.50,
            co2_accounting=CO2AccountingTreatment.CARBON_NEUTRAL,
            biogenic_share=1.0,
        )
        assert rec.moisture_content == 0.10

    def test_invalid_moisture_rejected(self):
        with pytest.raises(MappingError):
            BiogenicSource(
                key="x", category=BiogenicCategory.WOOD, description="x",
                moisture_content=1.5, lhv_gj_per_t=17.0, carbon_content=0.5,
                co2_accounting=CO2AccountingTreatment.CARBON_NEUTRAL, biogenic_share=1.0,
            )

    def test_invalid_biogenic_share_rejected(self):
        with pytest.raises(MappingError):
            BiogenicSource(
                key="x", category=BiogenicCategory.WOOD, description="x",
                moisture_content=0.1, lhv_gj_per_t=17.0, carbon_content=0.5,
                co2_accounting=CO2AccountingTreatment.CARBON_NEUTRAL, biogenic_share=2.0,
            )

    def test_invalid_carbon_content_rejected(self):
        with pytest.raises(MappingError):
            BiogenicSource(
                key="x", category=BiogenicCategory.WOOD, description="x",
                moisture_content=0.1, lhv_gj_per_t=17.0, carbon_content=1.5,
                co2_accounting=CO2AccountingTreatment.CARBON_NEUTRAL, biogenic_share=1.0,
            )

    def test_to_dict(self):
        rec = BiogenicSource(
            key="wood_pellets",
            category=BiogenicCategory.WOOD,
            description="pellets",
            moisture_content=0.08,
            lhv_gj_per_t=17.0,
            carbon_content=0.505,
            co2_accounting=CO2AccountingTreatment.CARBON_NEUTRAL,
            biogenic_share=1.0,
            typical_certs=[SustainabilityCertification.FSC, SustainabilityCertification.PEFC],
        )
        d = rec.to_dict()
        assert d["category"] == "wood"
        assert "FSC" in d["typical_certs"]
        assert "PEFC" in d["typical_certs"]


class TestYamlLoader:
    def test_loads_sources(self):
        sources = load_biogenic_sources()
        assert isinstance(sources, dict)
        assert "wood_pellets" in sources
        assert isinstance(sources["wood_pellets"], BiogenicSource)

    def test_wood_pellets_lhv_reasonable(self):
        rec = load_biogenic_sources()["wood_pellets"]
        assert 16.0 <= rec.lhv_gj_per_t <= 18.0
        assert rec.moisture_content < 0.15

    def test_manure_has_high_moisture(self):
        rec = load_biogenic_sources()["manure_cattle"]
        assert rec.moisture_content > 0.80

    def test_biomethane_is_biogas(self):
        rec = load_biogenic_sources()["biomethane"]
        assert rec.category == BiogenicCategory.BIOGAS

    def test_short_rotation_tagging(self):
        rec = load_biogenic_sources()["miscanthus"]
        assert rec.co2_accounting == CO2AccountingTreatment.SHORT_ROTATION_SEQUESTRATION

    def test_carbon_content_in_range(self):
        for key, rec in load_biogenic_sources().items():
            assert 0.30 <= rec.carbon_content <= 0.80, f"{key} carbon {rec.carbon_content}"

    def test_loader_cached(self):
        a = load_biogenic_sources()
        b = load_biogenic_sources()
        assert a is b


class TestListings:
    def test_list_biogenic_sources(self):
        keys = list_biogenic_sources()
        assert "wood_pellets" in keys
        assert "bagasse" in keys
        assert keys == sorted(keys)

    def test_list_length_reasonable(self):
        keys = list_biogenic_sources()
        assert len(keys) >= 15  # we have 20 in the YAML


class TestMapBiogenicSource:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("wood pellets", "wood_pellets"),
            ("wood chips", "wood_chips"),
            ("firewood", "wood_firewood_dry"),
            ("bagasse", "bagasse"),
            ("rice husk", "rice_husk"),
            ("dairy slurry", "manure_cattle"),
            ("swine manure", "manure_pig"),
            ("poultry litter", "manure_poultry"),
            ("food waste", "msw_organic"),
            ("seaweed", "algae_macroalgae"),
            ("biomethane", "biomethane"),
            ("miscanthus", "miscanthus"),
            ("EFB", "palm_empty_fruit_bunch"),
        ],
    )
    def test_known_sources_match(self, text, expected):
        r = map_biogenic_source(text)
        assert r.canonical["key"] == expected

    def test_case_insensitive(self):
        a = map_biogenic_source("WOOD PELLETS")
        b = map_biogenic_source("wood pellets")
        assert a.canonical["key"] == b.canonical["key"]

    def test_unknown_returns_none(self):
        r = map_biogenic_source("martian fungus")
        assert r.canonical is None
        assert r.band == MappingConfidence.UNKNOWN

    def test_rationale_includes_category(self):
        r = map_biogenic_source("wood pellets")
        assert "category=" in r.rationale

    def test_lhv_in_result(self):
        r = map_biogenic_source("wood pellets")
        assert r.canonical["lhv_gj_per_t"] == pytest.approx(17.0, abs=0.01)

    def test_certs_in_result_wood_pellets(self):
        r = map_biogenic_source("wood pellets")
        certs = r.canonical["typical_certs"]
        assert "FSC" in certs

    def test_empty_input(self):
        r = map_biogenic_source("")
        assert r.canonical is None


class TestCertifications:
    def test_load_certifications(self):
        certs = load_certifications()
        assert "FSC" in certs
        assert "RSPO" in certs

    def test_get_cert_info(self):
        info = get_certification_info("FSC")
        assert info is not None
        assert "full_name" in info
        assert "url" in info

    def test_get_cert_info_lowercase_not_found(self):
        # Only upper-case canonical names work (matching the YAML keys).
        # Upper-casing is attempted as a fallback.
        info = get_certification_info("fsc")
        assert info is not None

    def test_get_cert_info_unknown(self):
        assert get_certification_info("MadeUpCert") is None


class TestTaxonomyStaticStructure:
    def test_every_key_has_synonyms(self):
        for key, body in BIOGENIC_TAXONOMY.items():
            assert body["synonyms"], f"{key} missing synonyms"

    def test_every_key_has_category_meta(self):
        for key, body in BIOGENIC_TAXONOMY.items():
            assert body["meta"].get("category")
