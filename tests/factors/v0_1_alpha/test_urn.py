# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.ontology.urn (v0.1 Alpha URN scheme).

Covers:
    * happy paths for every URN kind (incl. all CTO doc Section 6.1.1
      worked examples)
    * geographies: country / subregion / grid_zone / bidding_zone /
      balancing_authority / state / global
    * units with slash, dot, caret, percent
    * hourly factor URNs with embedded ISO-8601 timestamps
    * round-trip: build -> parse -> build (deep equality)
    * adversarial inputs: uppercase, missing version suffix, bad scheme,
      double colon, empty namespace, prefix-only, oversized input, etc.
    * legacy EF: id coercion to URN
"""

from __future__ import annotations

import pytest

from greenlang.factors.ontology.urn import (
    ALLOWED_GEO_TYPES,
    ALLOWED_KINDS,
    GLUrn,
    InvalidUrnError,
    build,
    coerce_factor_id_to_urn,
    parse,
    validate,
)


# ---------------------------------------------------------------------------
# CTO doc worked examples (must parse exactly as written)
# ---------------------------------------------------------------------------

CTO_FACTOR_EXAMPLES = [
    "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1",
    "urn:gl:factor:epa-egrid:subregion-rfcw:2023-average:v2",
    "urn:gl:factor:defra-2025:passenger-car-petrol-medium:km:v1",
    "urn:gl:factor:india-cea-baseline:national-grid:2024-annual:v1",
    "urn:gl:factor:ecoinvent-3.11:market-aluminium-ingot-in:kg:v1",
    "urn:gl:factor:entsoe-realtime:bz-de-lu:2027-01-14T13:00Z:v1",
]


@pytest.mark.parametrize("urn_str", CTO_FACTOR_EXAMPLES)
def test_cto_doc_factor_examples_parse_and_round_trip(urn_str: str) -> None:
    parsed = parse(urn_str)
    assert parsed.kind == "factor"
    assert parsed.render() == urn_str
    # build() should reconstruct it byte-for-byte from components.
    rebuilt = build(
        "factor",
        source=parsed.source,
        namespace=parsed.namespace,
        id=parsed.id,
        version=parsed.version,
    )
    assert rebuilt == urn_str


# ---------------------------------------------------------------------------
# Happy paths per kind
# ---------------------------------------------------------------------------


def test_parse_factor_basic() -> None:
    u = parse("urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1")
    assert u.kind == "factor"
    assert u.source == "ipcc-ar6"
    assert u.namespace == "stationary-combustion"
    assert u.id == "natural-gas-residential"
    assert u.version == 1


def test_parse_factor_namespace_with_underscore() -> None:
    s = "urn:gl:factor:ipcc-ar6:stationary_combustion:diesel-fuel:v3"
    u = parse(s)
    assert u.namespace == "stationary_combustion"
    assert u.version == 3
    assert u.render() == s


def test_parse_factor_high_version() -> None:
    s = "urn:gl:factor:defra-2025:road:diesel-car:v42"
    u = parse(s)
    assert u.version == 42


def test_parse_factor_hourly_timestamp_id() -> None:
    s = "urn:gl:factor:entsoe-realtime:bz-de-lu:2027-01-14T13:00Z:v1"
    u = parse(s)
    assert u.id == "2027-01-14T13:00Z"
    assert u.version == 1
    assert u.render() == s


def test_parse_factor_id_with_dot_in_source() -> None:
    s = "urn:gl:factor:ecoinvent-3.11:market-aluminium-ingot-in:kg:v1"
    u = parse(s)
    assert u.source == "ecoinvent-3.11"
    assert u.id == "kg"


def test_parse_source_simple() -> None:
    u = parse("urn:gl:source:ipcc-ar6")
    assert u.kind == "source"
    assert u.slug == "ipcc-ar6"
    assert u.render() == "urn:gl:source:ipcc-ar6"


def test_parse_source_with_dot() -> None:
    u = parse("urn:gl:source:ecoinvent-3.11")
    assert u.kind == "source"
    assert u.slug == "ecoinvent-3.11"


def test_parse_pack() -> None:
    s = "urn:gl:pack:defra-2025:transport-pack:v2"
    u = parse(s)
    assert u.kind == "pack"
    assert u.source == "defra-2025"
    assert u.pack_id == "transport-pack"
    assert u.version == 2
    assert u.render() == s


def test_parse_methodology() -> None:
    s = "urn:gl:methodology:ghgp-corporate-scope2-market"
    u = parse(s)
    assert u.kind == "methodology"
    assert u.slug == "ghgp-corporate-scope2-market"
    assert u.render() == s


def test_parse_activity() -> None:
    s = "urn:gl:activity:stationary-combustion-natural-gas"
    u = parse(s)
    assert u.kind == "activity"
    assert u.slug == "stationary-combustion-natural-gas"


def test_parse_community() -> None:
    s = "urn:gl:community:my-research-network"
    u = parse(s)
    assert u.kind == "community"
    assert u.slug == "my-research-network"


# ---------------------------------------------------------------------------
# Geographies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "geo_type,geo_id",
    [
        ("country", "in"),
        ("country", "us"),
        ("country", "de"),
        ("subregion", "rfcw"),
        ("subregion", "serc"),
        ("grid_zone", "ercot"),
        ("bidding_zone", "de-lu"),
        ("balancing_authority", "caiso"),
        ("state", "ca"),
        ("global", "world"),
    ],
)
def test_parse_geo_all_types(geo_type: str, geo_id: str) -> None:
    s = f"urn:gl:geo:{geo_type}:{geo_id}"
    u = parse(s)
    assert u.kind == "geo"
    assert u.geo_type == geo_type
    assert u.id == geo_id
    assert u.render() == s


def test_parse_geo_unknown_type_rejected() -> None:
    with pytest.raises(InvalidUrnError, match="geo type"):
        parse("urn:gl:geo:planet:earth")


def test_geo_allowed_types_includes_all_required() -> None:
    required = {
        "country",
        "subregion",
        "grid_zone",
        "bidding_zone",
        "balancing_authority",
    }
    assert required.issubset(set(ALLOWED_GEO_TYPES))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    [
        "kg",
        "kgco2e/kwh",
        "kg.km",
        "tco2e/m^3",
        "mj/kg",
        "kgco2e/passenger-km",
        "%",  # rejected because '%' alone has no leading alphanumeric
    ],
)
def test_parse_unit_various(symbol: str) -> None:
    s = f"urn:gl:unit:{symbol}"
    if symbol == "%":
        with pytest.raises(InvalidUrnError):
            parse(s)
    else:
        u = parse(s)
        assert u.kind == "unit"
        assert u.symbol == symbol
        assert u.render() == s


def test_parse_unit_slash() -> None:
    u = parse("urn:gl:unit:kgco2e/kwh")
    assert u.symbol == "kgco2e/kwh"


def test_parse_unit_dot() -> None:
    u = parse("urn:gl:unit:kg.km")
    assert u.symbol == "kg.km"


# ---------------------------------------------------------------------------
# Partner / enterprise
# ---------------------------------------------------------------------------


def test_parse_partner() -> None:
    s = "urn:gl:partner:acme-corp:supply-pack:v3"
    u = parse(s)
    assert u.kind == "partner"
    assert u.tenant == "acme-corp"
    assert u.pack_id == "supply-pack"
    assert u.version == 3


def test_parse_enterprise() -> None:
    s = "urn:gl:enterprise:bigco:internal-factors:v7"
    u = parse(s)
    assert u.kind == "enterprise"
    assert u.tenant == "bigco"
    assert u.pack_id == "internal-factors"
    assert u.version == 7


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def test_build_factor() -> None:
    s = build(
        "factor",
        source="ipcc-ar6",
        namespace="stationary-combustion",
        id="natural-gas-residential",
        version=1,
    )
    assert s == (
        "urn:gl:factor:ipcc-ar6:stationary-combustion:"
        "natural-gas-residential:v1"
    )


def test_build_unknown_kind() -> None:
    with pytest.raises(InvalidUrnError, match="Unknown URN kind"):
        build("widget", slug="foo")


def test_build_missing_required() -> None:
    with pytest.raises(InvalidUrnError, match="missing required components"):
        build("factor", source="ipcc-ar6", namespace="ns", id="thing")


def test_build_extra_components_rejected() -> None:
    with pytest.raises(InvalidUrnError, match="unexpected components"):
        build("source", slug="ipcc-ar6", version=1)


def test_build_version_must_be_int() -> None:
    with pytest.raises(InvalidUrnError, match="positive int"):
        build(
            "factor",
            source="x",
            namespace="y",
            id="z",
            version="1",  # type: ignore[arg-type]
        )


def test_build_version_zero_rejected() -> None:
    with pytest.raises(InvalidUrnError, match=">= 1"):
        build("pack", source="x", pack_id="y", version=0)


# ---------------------------------------------------------------------------
# Round-trip equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,components",
    [
        (
            "factor",
            dict(
                source="defra-2025",
                namespace="passenger-car-petrol-medium",
                id="km",
                version=1,
            ),
        ),
        ("source", dict(slug="epa-egrid")),
        (
            "pack",
            dict(source="india-cea-baseline", pack_id="grid-2024", version=4),
        ),
        ("methodology", dict(slug="ghgp-corporate-scope2-market")),
        ("geo", dict(geo_type="bidding_zone", id="de-lu")),
        ("unit", dict(symbol="kgco2e/kwh")),
        ("activity", dict(slug="electricity-purchased")),
        ("community", dict(slug="ngo-coalition")),
        (
            "partner",
            dict(tenant="my-tenant", pack_id="custom-pack", version=2),
        ),
        (
            "enterprise",
            dict(tenant="bigco", pack_id="proprietary", version=12),
        ),
    ],
)
def test_round_trip_build_parse_build(kind, components) -> None:
    s1 = build(kind, **components)
    parsed = parse(s1)
    assert parsed.kind == kind
    s2 = parsed.render()
    assert s1 == s2
    # Re-build from parsed components -> identical string.
    rebuild_kwargs = {
        f: getattr(parsed, f)
        for f in components.keys()
    }
    s3 = build(kind, **rebuild_kwargs)
    assert s3 == s1


def test_round_trip_preserves_factor_id_with_colons() -> None:
    s = "urn:gl:factor:entsoe-realtime:bz-de-lu:2027-01-14T13:00Z:v1"
    p = parse(s)
    s2 = build(
        "factor",
        source=p.source,
        namespace=p.namespace,
        id=p.id,
        version=p.version,
    )
    assert s2 == s


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_true() -> None:
    assert validate("urn:gl:source:ipcc-ar6") is True


def test_validate_false_never_raises() -> None:
    assert validate("not a urn") is False
    assert validate("") is False
    assert validate(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adversarial / malformed inputs
# ---------------------------------------------------------------------------


def test_reject_uppercase_in_source() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:IPCC-AR6:stationary-combustion:gas:v1")


def test_reject_uppercase_in_namespace() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6:Stationary-Combustion:gas:v1")


def test_reject_uppercase_in_factor_id_other_than_tz() -> None:
    # 'A' is not allowed inside a factor id (only 'T' and 'Z').
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6:ns:NaturalGas:v1")


def test_reject_missing_version_suffix() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6:stationary-combustion:gas")


def test_reject_bad_version_token() -> None:
    with pytest.raises(InvalidUrnError, match="version"):
        parse("urn:gl:factor:ipcc-ar6:ns:gas:version1")


def test_reject_v0_version() -> None:
    with pytest.raises(InvalidUrnError, match="version"):
        parse("urn:gl:factor:ipcc-ar6:ns:gas:v0")


def test_reject_negative_version() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6:ns:gas:v-1")


def test_reject_leading_zero_version() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6:ns:gas:v01")


def test_reject_bad_scheme() -> None:
    with pytest.raises(InvalidUrnError, match="urn:gl:"):
        parse("urn:gl2:factor:ipcc-ar6:ns:gas:v1")


def test_reject_completely_wrong_scheme() -> None:
    with pytest.raises(InvalidUrnError, match="urn:gl:"):
        parse("https://greenlang.io/factors/ipcc-ar6")


def test_reject_unknown_kind() -> None:
    with pytest.raises(InvalidUrnError, match="Unknown URN kind"):
        parse("urn:gl:widget:foo")


def test_reject_double_colon_factor() -> None:
    # Double colon in factor URN means an empty namespace/id segment.
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6::gas:v1")


def test_reject_double_colon_pack() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:pack:defra-2025::v1")


def test_reject_empty_namespace() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:ipcc-ar6::natural-gas:v1")


def test_reject_prefix_only() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor:")


def test_reject_kind_only() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:factor")


def test_reject_empty_string() -> None:
    with pytest.raises(InvalidUrnError, match="empty"):
        parse("")


def test_reject_non_string_input() -> None:
    with pytest.raises(InvalidUrnError, match="must be a string"):
        parse(12345)  # type: ignore[arg-type]


def test_reject_oversized_input() -> None:
    huge = "urn:gl:source:" + ("a" * 600)
    with pytest.raises(InvalidUrnError, match="max length"):
        parse(huge)


def test_reject_whitespace_in_slug() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:source:ipcc ar6")


def test_reject_trailing_separator() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:source:ipcc-ar6-")


def test_reject_leading_separator() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:source:-ipcc-ar6")


def test_reject_extra_segment_in_source_urn() -> None:
    with pytest.raises(InvalidUrnError, match="no further ':'"):
        parse("urn:gl:source:ipcc:ar6")


def test_reject_extra_segment_in_pack_urn() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:pack:defra-2025:transport-pack:extra:v1")


def test_reject_unit_with_colon() -> None:
    with pytest.raises(InvalidUrnError, match="must not contain ':'"):
        parse("urn:gl:unit:kgco2e:kwh")


def test_reject_geo_id_with_colon() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:geo:country:in:north")


# ---------------------------------------------------------------------------
# coerce_factor_id_to_urn (legacy EF: -> URN migration)
# ---------------------------------------------------------------------------


def test_coerce_factor_id_to_urn_canonical_example() -> None:
    """Exact target from task spec."""
    out = coerce_factor_id_to_urn(
        "EF:US:grid:eGRID-SERC:2024:v1",
        source="epa-egrid",
        namespace="subregion-serc",
        version=1,
    )
    assert out == "urn:gl:factor:epa-egrid:subregion-serc:2024:v1"


def test_coerce_factor_id_to_urn_drops_legacy_version() -> None:
    out = coerce_factor_id_to_urn(
        "EF:UK:transport:passenger-car:petrol:medium:v3",
        source="defra-2025",
        namespace="passenger-car-petrol-medium",
        version=2,
    )
    assert out.endswith(":v2")
    # Last non-version segment 'medium' becomes the URN id.
    assert ":medium:v2" in out


def test_coerce_factor_id_to_urn_no_version_suffix() -> None:
    out = coerce_factor_id_to_urn(
        "EF:IN:cement:portland",
        source="india-cea-baseline",
        namespace="cement",
        version=1,
    )
    assert out == "urn:gl:factor:india-cea-baseline:cement:portland:v1"


def test_coerce_factor_id_to_urn_lowercases_uppercase_leaf() -> None:
    out = coerce_factor_id_to_urn(
        "EF:US:grid:RFCW",
        source="epa-egrid",
        namespace="subregion-rfcw",
        version=1,
    )
    assert out == "urn:gl:factor:epa-egrid:subregion-rfcw:rfcw:v1"


def test_coerce_factor_id_to_urn_rejects_non_ef_prefix() -> None:
    with pytest.raises(InvalidUrnError):
        coerce_factor_id_to_urn(
            "FOO:US:grid:rfcw",
            source="epa-egrid",
            namespace="ns",
            version=1,
        )


def test_coerce_factor_id_to_urn_rejects_empty_body() -> None:
    with pytest.raises(InvalidUrnError):
        coerce_factor_id_to_urn(
            "EF:",
            source="epa-egrid",
            namespace="ns",
            version=1,
        )


def test_coerce_factor_id_to_urn_rejects_only_version_segments() -> None:
    with pytest.raises(InvalidUrnError):
        coerce_factor_id_to_urn(
            "EF:v1:v2",
            source="epa-egrid",
            namespace="ns",
            version=1,
        )


def test_coerce_factor_id_to_urn_deterministic() -> None:
    """Two calls with the same inputs must return identical output."""
    a = coerce_factor_id_to_urn(
        "EF:US:grid:eGRID-SERC:2024:v1",
        source="epa-egrid",
        namespace="subregion-serc",
        version=1,
    )
    b = coerce_factor_id_to_urn(
        "EF:US:grid:eGRID-SERC:2024:v1",
        source="epa-egrid",
        namespace="subregion-serc",
        version=1,
    )
    assert a == b


# ---------------------------------------------------------------------------
# GLUrn model behaviour
# ---------------------------------------------------------------------------


def test_glurn_is_frozen() -> None:
    u = parse("urn:gl:source:ipcc-ar6")
    with pytest.raises((TypeError, ValueError)):
        u.slug = "other"  # type: ignore[misc]


def test_glurn_str_returns_render() -> None:
    s = "urn:gl:methodology:ghgp-corporate-scope2-market"
    assert str(parse(s)) == s


def test_glurn_is_factor_property() -> None:
    assert parse(
        "urn:gl:factor:ipcc-ar6:ns:gas:v1"
    ).is_factor is True
    assert parse("urn:gl:source:ipcc-ar6").is_factor is False


def test_glurn_is_pack_property() -> None:
    assert parse(
        "urn:gl:pack:defra-2025:transport:v1"
    ).is_pack is True
    assert parse(
        "urn:gl:partner:acme:custom:v1"
    ).is_pack is True
    assert parse(
        "urn:gl:enterprise:bigco:internal:v1"
    ).is_pack is True
    assert parse("urn:gl:source:ipcc-ar6").is_pack is False


def test_allowed_kinds_complete() -> None:
    assert set(ALLOWED_KINDS) == {
        "factor",
        "source",
        "pack",
        "methodology",
        "geo",
        "unit",
        "activity",
        "community",
        "partner",
        "enterprise",
    }


# ---------------------------------------------------------------------------
# Cross-validation: parse(build(...)) and build(parse(...).fields) loops
# ---------------------------------------------------------------------------


def test_build_then_validate_true_for_all_kinds() -> None:
    examples = {
        "factor": dict(
            source="ipcc-ar6",
            namespace="ns",
            id="x",
            version=1,
        ),
        "source": dict(slug="ipcc-ar6"),
        "pack": dict(source="ipcc-ar6", pack_id="p", version=1),
        "methodology": dict(slug="ghgp"),
        "geo": dict(geo_type="country", id="us"),
        "unit": dict(symbol="kgco2e/kwh"),
        "activity": dict(slug="x"),
        "community": dict(slug="x"),
        "partner": dict(tenant="t", pack_id="p", version=1),
        "enterprise": dict(tenant="t", pack_id="p", version=1),
    }
    for kind, comps in examples.items():
        s = build(kind, **comps)
        assert validate(s), f"{kind} URN failed validate(): {s}"


def test_invalid_urn_error_is_value_error_subclass() -> None:
    """InvalidUrnError must be catchable as ValueError for ergonomic
    integration with pydantic field validators downstream.
    """
    assert issubclass(InvalidUrnError, ValueError)
