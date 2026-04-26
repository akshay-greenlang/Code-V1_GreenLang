# -*- coding: utf-8 -*-
"""Wave D / WS2-T2 — V1->V0.1 Alpha Normalizer tests.

Validates that ``lift_v1_record_to_v0_1`` produces v0.1-shape records
that pass the FROZEN ``factor_record_v0_1.schema.json`` schema and the
:class:`AlphaProvenanceGate` constraints.

The tests cover:

* Happy paths per source family (combustion / electricity / refrigerant
  / cbam-default / land-use / India CEA grid).
* Category coercion mapping for every alpha-supported family.
* URN coercion is deterministic (same input -> same output).
* All 12 ``extraction.*`` fields populate.
* ``review.review_status`` defaults to ``approved``.
* AR4 records raise (AR5 are tolerated and re-tagged AR6).
* ``value > 0`` invariant — non-positive vectors raise
  :class:`NonPositiveValueError`.
* ``citations`` always has at least one entry.

CTO doc references: §6.1, §19.1.
Schema $id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict

import pytest

from greenlang.factors.etl.alpha_v0_1_normalizer import (
    NonPositiveValueError,
    NormalizerError,
    coerce_category,
    coerce_factor_id_to_urn,
    coerce_geography,
    lift_v1_record_to_v0_1,
    slugify,
)
from greenlang.factors.quality.alpha_provenance_gate import AlphaProvenanceGate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate() -> AlphaProvenanceGate:
    return AlphaProvenanceGate()


def _src_meta(**overrides: Any) -> Dict[str, Any]:
    """Default source_meta dict (mirrors a registry row)."""
    base = {
        "source_id": "ipcc_2006_nggi",
        "urn": "urn:gl:source:ipcc-2006-nggi",
        "display_name": "IPCC 2006 Guidelines + 2019 Refinement (NGGI)",
        "source_owner": "climate-methodology-lead",
        "parser_module": "greenlang.factors.ingestion.parsers.ipcc_defaults",
        "parser_function": "parse_ipcc_defaults",
        "parser_version": "0.1.0",
        "cadence": "ad_hoc",
        "license_class": "public_international",
        "source_version": "2019.1",
        "publication_url": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
    }
    base.update(overrides)
    return base


def _v1_combustion_record(factor_id: str = "EF:IPCC:stat_natural_gas:US:2024:v1") -> Dict[str, Any]:
    """Minimal v1-shape stationary combustion record (EPA/IPCC/DESNZ shape)."""
    return {
        "factor_id": factor_id,
        "fuel_type": "natural_gas",
        "unit": "mmbtu",
        "geography": "US",
        "geography_level": "country",
        "vectors": {"CO2": 53.06, "CH4": 0.001, "N2O": 0.0001},
        "gwp_100yr": {
            "gwp_set": "ipcc_ar6_100",
            "CH4_gwp": 28,
            "N2O_gwp": 273,
        },
        "scope": "scope_1",
        "boundary": "combustion",
        "valid_from": "2024-01-01",
        "valid_to": "2024-12-31",
        "license_info": {
            "license": "US-Public-Domain",
            "redistribution_allowed": True,
            "commercial_use_allowed": True,
            "attribution_required": True,
        },
        "factor_family": "combustion",
        "tags": ["epa", "stationary_combustion"],
    }


def _v1_electricity_record(factor_id: str = "EF:IN:all_india:2024-25:cea-v20.0") -> Dict[str, Any]:
    return {
        "factor_id": factor_id,
        "fuel_type": "electricity",
        "unit": "kWh",
        "geography": "IN",
        "geography_level": "country",
        "vectors": {"CO2": 0.71, "CH4": 0.0, "N2O": 0.0},
        "gwp_100yr": {"gwp_set": "ipcc_ar6_100", "CH4_gwp": 28, "N2O_gwp": 273},
        "scope": "scope_2",
        "boundary": "combustion",
        "valid_from": "2024-04-01",
        "valid_to": "2025-03-31",
        "license_info": {"license": "GoI-Public-Use"},
        "factor_family": "electricity",
        "method_profile": "corporate_scope2_location_based",
        "jurisdiction": {"country": "IN"},
    }


def _v1_egrid_record(factor_id: str = "EF:eGRID:RFCW:2022:v1") -> Dict[str, Any]:
    return {
        "factor_id": factor_id,
        "fuel_type": "electricity_grid",
        "unit": "kwh",
        "geography": "RFCW",
        "geography_level": "grid_zone",
        "vectors": {"CO2": 0.5, "CH4": 0.00001, "N2O": 0.000001},
        "gwp_100yr": {"gwp_set": "ipcc_ar6_100", "CH4_gwp": 28, "N2O_gwp": 273},
        "scope": "scope_2",
        "boundary": "combustion",
        "valid_from": "2022-01-01",
        "license_info": {"license": "US-Public-Domain"},
    }


def _v1_cbam_record(factor_id: str = "EF:CBAM:steel:CN:2024:v1") -> Dict[str, Any]:
    return {
        "factor_id": factor_id,
        "fuel_type": "cbam_steel",
        "unit": "kg_product",
        "geography": "CN",
        "geography_level": "country",
        "vectors": {"CO2": 2.8, "CH4": 0.0, "N2O": 0.0},
        "gwp_100yr": {"gwp_set": "ipcc_ar6_100", "CH4_gwp": 28, "N2O_gwp": 273},
        "scope": "scope_3",
        "boundary": "cradle_to_gate, embedded emissions per CBAM Annex IV",
        "valid_from": "2024-01-01",
        "valid_to": "9999-12-31",
        "license_info": {"license": "EU-Publication"},
        "factor_family": "material_embodied",
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_lift_combustion_record_passes_gate(gate: AlphaProvenanceGate) -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(source_id="epa_hub", urn="urn:gl:source:epa-hub", parser_module="greenlang.factors.ingestion.parsers.epa_ghg_hub", parser_function="parse_epa_ghg_hub", source_version="2024.1", publication_url="https://www.epa.gov/climateleadership/ghg-emission-factors-hub"), idx=0)
    assert gate.validate(out) == []
    assert out["category"] == "fuel"
    assert out["urn"].startswith("urn:gl:factor:epa-hub:")
    assert out["unit_urn"].startswith("urn:gl:unit:kgco2e/")
    assert out["geography_urn"] == "urn:gl:geo:country:us"


def test_registry_licence_pin_overrides_parser_label(
    gate: AlphaProvenanceGate,
) -> None:
    """Phase 1 keeps the registry as the source of truth for factor
    licence tags, even when parser records carry older upstream labels.
    """
    rec = _v1_combustion_record()
    rec["license_info"]["license"] = "US-Public-Domain"
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(
            source_id="epa_hub",
            urn="urn:gl:source:epa-hub",
            parser_module="greenlang.factors.ingestion.parsers.epa_ghg_hub",
            parser_function="parse_epa_ghg_hub",
            source_version="2024.1",
            publication_url="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            licence="public-domain-us-gov",
        ),
        idx=0,
    )
    assert gate.validate(out) == []
    assert out["licence"] == "public-domain-us-gov"


def test_lift_electricity_record_indian_grid(gate: AlphaProvenanceGate) -> None:
    rec = _v1_electricity_record()
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(
            source_id="india_cea_co2_baseline",
            urn="urn:gl:source:india-cea-co2-baseline",
            parser_module="greenlang.factors.ingestion.parsers.india_cea",
            parser_function="parse_india_cea_rows",
            source_version="20.0",
            publication_url="https://cea.nic.in/cdm-co2-baseline-database/",
        ),
        idx=0,
    )
    assert gate.validate(out) == []
    assert out["category"] == "scope2_location_based"
    assert out["geography_urn"] == "urn:gl:geo:country:in"


def test_lift_egrid_subregion_record(gate: AlphaProvenanceGate) -> None:
    rec = _v1_egrid_record()
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(
            source_id="egrid",
            urn="urn:gl:source:egrid",
            parser_module="greenlang.factors.ingestion.parsers.egrid",
            parser_function="parse_egrid",
            source_version="2022.1",
            publication_url="https://www.epa.gov/egrid/download-data",
        ),
        idx=0,
    )
    assert gate.validate(out) == []
    # No factor_family on this record, geo level is grid_zone → grid_intensity
    assert out["category"] in {"grid_intensity", "scope2_location_based"}
    assert out["geography_urn"].startswith("urn:gl:geo:grid_zone:")


def test_lift_cbam_record_uses_cbam_default_category(gate: AlphaProvenanceGate) -> None:
    rec = _v1_cbam_record()
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(
            source_id="cbam_default_values",
            urn="urn:gl:source:cbam-default-values",
            parser_module="greenlang.factors.ingestion.parsers.cbam_default_values",
            parser_function="parse_cbam_default_values",
            source_version="2024.1",
            publication_url=(
                "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R1773"
            ),
        ),
        idx=7,
    )
    assert gate.validate(out) == []
    assert out["category"] == "cbam_default"
    assert "cbam-annex-iv" in out["factor_pack_urn"]
    assert out["geography_urn"] == "urn:gl:geo:country:cn"


def test_lift_refrigerant_record(gate: AlphaProvenanceGate) -> None:
    rec = _v1_combustion_record("EF:IPCC:refrig_r404a:GLOBAL:2019:v1")
    rec["factor_family"] = "refrigerant"
    rec["fuel_type"] = "refrigerant_r404a"
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert gate.validate(out) == []
    assert out["category"] == "refrigerant"
    assert out["factor_pack_urn"].startswith("urn:gl:pack:ipcc-2006-nggi:")
    assert out["factor_pack_urn"].endswith(":v1")


# ---------------------------------------------------------------------------
# Coerce category — full mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,profile,expected",
    [
        ("combustion", None, "fuel"),
        ("stationary_combustion", None, "fuel"),
        ("mobile_combustion", None, "fuel"),
        ("electricity", "corporate_scope2_location_based", "scope2_location_based"),
        ("electricity", "corporate_scope2_market_based", "scope2_market_based"),
        ("electricity", None, "grid_intensity"),
        ("grid_intensity", None, "grid_intensity"),
        ("refrigerant", None, "refrigerant"),
        ("refrigerants", None, "refrigerant"),
        ("classification_mapping", "eu_cbam", "cbam_default"),
        ("material_embodied", None, "cbam_default"),
        ("fugitive", None, "fugitive"),
        ("process", None, "process"),
        (None, None, "fuel"),
    ],
)
def test_coerce_category_mapping(family: str, profile: str, expected: str) -> None:
    assert coerce_category(family, profile) == expected


# ---------------------------------------------------------------------------
# URN coercion — deterministic + valid pattern
# ---------------------------------------------------------------------------


# Updated 2026-04-26 (Phase 0 audit): namespace + id segments MUST be
# lowercase per CTO doc §6.1.1 / canonical URN parser at
# ``greenlang.factors.ontology.urn``. The schema regex was tightened to
# match — uppercase letters in those segments are no longer accepted.
_URN_FACTOR_RE = re.compile(
    r"^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$"
)


def test_coerce_factor_id_to_urn_is_deterministic() -> None:
    fid = "EF:IPCC:stat_natural_gas:US:2024:v1"
    a = coerce_factor_id_to_urn(fid, "ipcc-2006-nggi")
    b = coerce_factor_id_to_urn(fid, "ipcc-2006-nggi")
    assert a == b
    assert _URN_FACTOR_RE.match(a)
    assert a.startswith("urn:gl:factor:ipcc-2006-nggi:")
    assert a.endswith(":v1")


def test_coerce_factor_id_to_urn_strips_ef_prefix() -> None:
    # Phase 0 audit (2026-04-26): output is lowercased to comply with the
    # canonical URN spec. The legacy uppercase form is kept by the
    # ``factor_id_alias`` field, not the ``urn`` field.
    urn = coerce_factor_id_to_urn("EF:CBAM:steel:CN:2024:v1", "cbam-default-values")
    assert urn == "urn:gl:factor:cbam-default-values:cbam:steel:cn:2024:v1"


def test_coerce_factor_id_to_urn_handles_no_ef_prefix() -> None:
    urn = coerce_factor_id_to_urn("CBAM:steel:CN:2024", "cbam-default-values")
    assert _URN_FACTOR_RE.match(urn)


def test_coerce_factor_id_to_urn_rejects_empty() -> None:
    with pytest.raises(NormalizerError):
        coerce_factor_id_to_urn("", "src")
    with pytest.raises(NormalizerError):
        coerce_factor_id_to_urn(None, "src")  # type: ignore[arg-type]


def test_coerce_factor_id_to_urn_caps_segments_at_four() -> None:
    """Schema allows max 4 segments between source-slug and v<n>."""
    long_id = "EF:A:B:C:D:E:F:G:v1"
    urn = coerce_factor_id_to_urn(long_id, "test-src")
    assert _URN_FACTOR_RE.match(urn)


# ---------------------------------------------------------------------------
# Geography coercion
# ---------------------------------------------------------------------------


def test_coerce_geography_jurisdiction_country() -> None:
    rec = {"jurisdiction": {"country": "IN"}}
    assert coerce_geography(rec) == "urn:gl:geo:country:in"


def test_coerce_geography_jurisdiction_grid_region() -> None:
    rec = {"jurisdiction": {"country": "US", "grid_region": "RFCW"}}
    assert coerce_geography(rec) == "urn:gl:geo:grid_zone:rfcw"


def test_coerce_geography_country_xx_to_global() -> None:
    rec = {"geography": "XX"}
    assert coerce_geography(rec) == "urn:gl:geo:global:world"


def test_coerce_geography_egrid_grid_zone() -> None:
    rec = {"geography": "RFCW", "geography_level": "grid_zone"}
    out = coerce_geography(rec)
    assert out.startswith("urn:gl:geo:grid_zone:")


def test_coerce_geography_uk_country() -> None:
    rec = {"geography": "GB", "geography_level": "country"}
    assert coerce_geography(rec) == "urn:gl:geo:country:gb"


def test_coerce_geography_state_or_province() -> None:
    rec = {"geography": "US-TX", "geography_level": "state"}
    assert coerce_geography(rec) == "urn:gl:geo:state_or_province:us-tx"


# ---------------------------------------------------------------------------
# Extraction block — every field populates
# ---------------------------------------------------------------------------


_EXTRACTION_REQUIRED = (
    "source_url",
    "source_record_id",
    "source_publication",
    "source_version",
    "raw_artifact_uri",
    "raw_artifact_sha256",
    "parser_id",
    "parser_version",
    "parser_commit",
    "row_ref",
    "ingested_at",
    "operator",
)


def test_extraction_block_populates_every_field() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=42)
    extraction = out["extraction"]
    for field_name in _EXTRACTION_REQUIRED:
        assert field_name in extraction, f"missing extraction.{field_name}"
        assert extraction[field_name] not in (None, "")


def test_extraction_sha256_is_64_hex() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    sha = out["extraction"]["raw_artifact_sha256"]
    assert re.fullmatch(r"[a-f0-9]{64}", sha)


def test_extraction_parser_commit_is_hex() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    commit = out["extraction"]["parser_commit"]
    assert re.fullmatch(r"[a-f0-9]{7,40}", commit)


def test_extraction_operator_is_bot_form() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(source_id="epa_hub", urn="urn:gl:source:epa-hub"), idx=0)
    op = out["extraction"]["operator"]
    assert op.startswith("bot:parser_")


# ---------------------------------------------------------------------------
# Review block — approved by default
# ---------------------------------------------------------------------------


def test_review_block_defaults_to_approved() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    review = out["review"]
    assert review["review_status"] == "approved"
    assert review["reviewer"].startswith("human:")
    assert review["approved_by"].startswith("human:")
    assert review["reviewed_at"]
    assert review["approved_at"]


# ---------------------------------------------------------------------------
# AR4 rejection / AR5 tolerance / AR6 acceptance
# ---------------------------------------------------------------------------


def test_ar4_record_raises_normalizer_error() -> None:
    rec = _v1_combustion_record()
    rec["gwp_100yr"]["gwp_set"] = "ipcc_ar4_100"
    with pytest.raises(NormalizerError):
        lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)


def test_ar5_record_is_retagged_to_ar6() -> None:
    """DESNZ rows ship with AR5 GWPs; alpha re-tags everything AR6."""
    rec = _v1_combustion_record()
    rec["gwp_100yr"]["gwp_set"] = "ipcc_ar5_100"
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert out["gwp_basis"] == "ar6"


def test_ar6_record_passes_gwp_basis() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert out["gwp_basis"] == "ar6"
    assert out["gwp_horizon"] == 100


# ---------------------------------------------------------------------------
# value > 0 invariant
# ---------------------------------------------------------------------------


def test_zero_value_record_raises() -> None:
    rec = _v1_combustion_record()
    rec["vectors"] = {"CO2": 0.0, "CH4": 0.0, "N2O": 0.0}
    with pytest.raises(NonPositiveValueError):
        lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)


def test_negative_value_record_raises() -> None:
    """Carbon-sequestration records (negative CO2e) cannot be expressed in v0.1."""
    rec = _v1_combustion_record("EF:IPCC:lu_forest_land:GLOBAL:2019:v1")
    rec["vectors"] = {"CO2": -220.0, "CH4": 0.0, "N2O": 0.0}
    with pytest.raises(NonPositiveValueError):
        lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)


def test_value_collapse_uses_ar6_gwp_when_no_co2e() -> None:
    rec = _v1_combustion_record()
    rec["vectors"] = {"CO2": 10.0, "CH4": 1.0, "N2O": 0.5}
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    # 10 + 1*28 + 0.5*273 = 10 + 28 + 136.5 = 174.5
    assert out["value"] == pytest.approx(174.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Citations >= 1
# ---------------------------------------------------------------------------


def test_citations_always_has_at_least_one() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert isinstance(out["citations"], list)
    assert len(out["citations"]) >= 1
    for cite in out["citations"]:
        assert "type" in cite
        assert "value" in cite


def test_citations_includes_publication_url() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(publication_url="https://example.gov/factors"),
        idx=0,
    )
    urls = [c["value"] for c in out["citations"] if c["type"] == "url"]
    assert "https://example.gov/factors" in urls


# ---------------------------------------------------------------------------
# Slugify utility
# ---------------------------------------------------------------------------


def test_slugify_lowercases_and_hyphenates() -> None:
    assert slugify("Hello World") == "hello-world"
    assert slugify("Some_Snake_Case") == "some-snake-case"
    assert slugify("kgCO2e/kWh") == "kgco2e/kwh"  # keeps slash
    assert slugify("") == ""
    assert slugify(None) == ""


# ---------------------------------------------------------------------------
# Round-trip: full lift + gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,source_overrides",
    [
        (_v1_combustion_record, {"source_id": "ipcc_2006_nggi", "urn": "urn:gl:source:ipcc-2006-nggi"}),
        (_v1_electricity_record, {"source_id": "india_cea_co2_baseline", "urn": "urn:gl:source:india-cea-co2-baseline"}),
        (_v1_egrid_record, {"source_id": "egrid", "urn": "urn:gl:source:egrid"}),
        (_v1_cbam_record, {"source_id": "cbam_default_values", "urn": "urn:gl:source:cbam-default-values"}),
    ],
)
def test_lifted_records_pass_full_gate(
    factory: Any, source_overrides: Dict[str, Any], gate: AlphaProvenanceGate
) -> None:
    rec = factory()
    out = lift_v1_record_to_v0_1(rec, _src_meta(**source_overrides), idx=0)
    failures = gate.validate(out)
    assert failures == [], f"gate rejected lifted record: {failures}"


# ---------------------------------------------------------------------------
# factor_pack_urn derivation
# ---------------------------------------------------------------------------


def test_factor_pack_urn_uses_source_override() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(source_id="epa_hub", urn="urn:gl:source:epa-hub", source_version="2024.1"),
        idx=0,
    )
    # Per _SOURCE_ID_TO_PACK_ID: epa_hub -> emission-factors-hub
    assert out["factor_pack_urn"] == "urn:gl:pack:epa-hub:emission-factors-hub:v1"


def test_factor_pack_urn_unknown_source_falls_back_to_default() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(
        rec,
        _src_meta(
            source_id="unknown_source",
            urn="urn:gl:source:unknown-source",
            source_version="9.9.9",
        ),
        idx=0,
    )
    # combustion family -> tier-1-defaults pack
    assert "tier-1-defaults" in out["factor_pack_urn"]


# ---------------------------------------------------------------------------
# Schema-required string-length invariants
# ---------------------------------------------------------------------------


def test_description_is_at_least_30_chars() -> None:
    rec = _v1_combustion_record()
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert len(out["description"]) >= 30


def test_boundary_is_at_least_10_chars() -> None:
    rec = _v1_combustion_record()
    rec["boundary"] = "x"  # too short, should fall back
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert len(out["boundary"]) >= 10


# ---------------------------------------------------------------------------
# vintage_end >= vintage_start invariant
# ---------------------------------------------------------------------------


def test_vintage_end_clamped_to_vintage_start_when_inverted() -> None:
    rec = _v1_combustion_record()
    rec["valid_from"] = "2025-06-01"
    rec["valid_to"] = "2024-01-01"  # earlier than valid_from
    out = lift_v1_record_to_v0_1(rec, _src_meta(), idx=0)
    assert out["vintage_end"] >= out["vintage_start"]


def test_record_with_record_factor_id_alias_preserved() -> None:
    rec = _v1_combustion_record("EF:EPA:stat_natural_gas:US:2024:v1")
    out = lift_v1_record_to_v0_1(rec, _src_meta(source_id="epa_hub", urn="urn:gl:source:epa-hub"), idx=0)
    assert out.get("factor_id_alias") == "EF:EPA:stat_natural_gas:US:2024:v1"


def test_non_dict_input_raises() -> None:
    with pytest.raises(NormalizerError):
        lift_v1_record_to_v0_1("not a dict", _src_meta(), idx=0)  # type: ignore[arg-type]
    with pytest.raises(NormalizerError):
        lift_v1_record_to_v0_1({}, "not a dict", idx=0)  # type: ignore[arg-type]
