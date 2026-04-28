# -*- coding: utf-8 -*-
"""Phase 2 / WS2 — Property-based URN round-trip tests.

Per CTO Phase 2 brief Section 2.2 (URN compliance):

    "Deterministic URN builders covered by tests | 100% | Property-based
    round-trip tests"

This module uses Hypothesis to generate randomized component tuples for
every URN kind and asserts:

  * ``parse(build(**components))`` returns a :class:`GLUrn` whose
    fields equal the inputs (build -> parse identity).
  * ``parse(s).render() == s`` for every well-formed URN string
    (parse -> render identity).

Plus a curated negative corpus that exercises adversarial inputs and
asserts :class:`InvalidUrnError` is raised.

Coverage scope (per CTO brief):

  * factor / source / pack / methodology / geo / unit / community /
    partner / enterprise.
  * The ``activity`` kind is owned by the WS5 agent (extending the
    parser to support ``urn:gl:activity:<taxonomy>:<code>``); WS5 ships
    its own property tests for the extended grammar. The single-slug
    activity case is still covered by ``tests/factors/v0_1_alpha/test_urn.py``.

The default ``@given`` budget is ``max_examples=200`` per kind — high
enough to surface escape-character and slug-boundary regressions while
keeping the suite under the 5-minute CI target.
"""
from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from greenlang.factors.ontology.urn import (
    ALLOWED_GEO_TYPES,
    GLUrn,
    InvalidUrnError,
    build,
    parse,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies — slug, namespace, factor-id, geo-id, unit-symbol.
#
# Each strategy generates a string that satisfies the parser's regex
# *exactly* (including the boundary-character constraint that the first
# and last char must be alphanumeric). Strategies are deliberately
# conservative so the round-trip property is the only thing under test
# — we never generate components the parser would reject.
# ---------------------------------------------------------------------------


_SLUG_INNER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789.-"
_NAMESPACE_INNER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789._-"
_FACTOR_ID_INNER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789._:-"
_UNIT_INNER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789._/-^%"
_ALNUM_LOWER = "abcdefghijklmnopqrstuvwxyz0123456789"


def _bounded(strategy: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    """Filter to the parser's max-component length (defensive bound)."""
    return strategy.filter(lambda s: 1 <= len(s) <= 60)


@st.composite
def slug_strategy(draw: st.DrawFn) -> str:
    """Generate a parser-accepted slug.

    Slug grammar: ``[a-z0-9](?:[a-z0-9.\\-]*[a-z0-9])?`` — first and last
    char alphanumeric; interior may contain ``.`` and ``-``.
    """
    length = draw(st.integers(min_value=1, max_value=40))
    if length == 1:
        return draw(st.sampled_from(_ALNUM_LOWER))
    first = draw(st.sampled_from(_ALNUM_LOWER))
    last = draw(st.sampled_from(_ALNUM_LOWER))
    middle_len = length - 2
    middle = (
        "".join(draw(st.sampled_from(_SLUG_INNER_ALPHABET)) for _ in range(middle_len))
        if middle_len > 0
        else ""
    )
    candidate = first + middle + last
    # Reject sequences that produce empty inter-segment artefacts after split.
    if ".." in candidate or "--" in candidate:
        # Hypothesis will retry; this is rare given the alphabet but
        # keeps the property strict.
        candidate = candidate.replace("..", ".a").replace("--", "-a")
    return candidate


@st.composite
def namespace_strategy(draw: st.DrawFn) -> str:
    """Generate a parser-accepted namespace (slug + underscore)."""
    length = draw(st.integers(min_value=1, max_value=40))
    if length == 1:
        return draw(st.sampled_from(_ALNUM_LOWER))
    first = draw(st.sampled_from(_ALNUM_LOWER))
    last = draw(st.sampled_from(_ALNUM_LOWER))
    middle_len = length - 2
    middle = (
        "".join(
            draw(st.sampled_from(_NAMESPACE_INNER_ALPHABET))
            for _ in range(middle_len)
        )
        if middle_len > 0
        else ""
    )
    candidate = first + middle + last
    return candidate


_FACTOR_ID_INNER_ALPHABET_WITH_TZ = (
    _FACTOR_ID_INNER_ALPHABET + "TZ"
)


@st.composite
def factor_id_strategy(draw: st.DrawFn) -> str:
    """Generate a parser-accepted factor id.

    Factor ids permit interior ``T`` and ``Z`` for ISO-8601 markers; the
    boundary characters are still alphanumeric (lowercase since the
    factor-id rule allows but does not require uppercase, and the
    parser's lower/upper guard rejects any uppercase that is NOT ``T``
    or ``Z``).
    """
    length = draw(st.integers(min_value=1, max_value=50))
    if length == 1:
        return draw(st.sampled_from(_ALNUM_LOWER))
    first = draw(st.sampled_from(_ALNUM_LOWER))
    # Last char must be alphanumeric; allow T and Z too since the
    # boundary regex permits them.
    last = draw(st.sampled_from(_ALNUM_LOWER + "TZ"))
    middle_len = length - 2
    middle = (
        "".join(
            draw(st.sampled_from(_FACTOR_ID_INNER_ALPHABET_WITH_TZ))
            for _ in range(middle_len)
        )
        if middle_len > 0
        else ""
    )
    return first + middle + last


@st.composite
def unit_symbol_strategy(draw: st.DrawFn) -> str:
    """Generate a parser-accepted unit symbol.

    Unit grammar: ``[a-z0-9](?:[a-z0-9.\\-_/^%]*[a-z0-9%])?`` — interior
    may include ``.``, ``-``, ``_``, ``/``, ``^``, ``%``.
    """
    length = draw(st.integers(min_value=1, max_value=40))
    if length == 1:
        return draw(st.sampled_from(_ALNUM_LOWER))
    first = draw(st.sampled_from(_ALNUM_LOWER))
    last = draw(st.sampled_from(_ALNUM_LOWER + "%"))
    middle_len = length - 2
    middle = (
        "".join(draw(st.sampled_from(_UNIT_INNER_ALPHABET)) for _ in range(middle_len))
        if middle_len > 0
        else ""
    )
    candidate = first + middle + last
    return candidate


def _version_strategy() -> st.SearchStrategy[int]:
    """Positive int for ``v<n>`` segment."""
    return st.integers(min_value=1, max_value=999_999)


def _geo_type_strategy() -> st.SearchStrategy[str]:
    """One of the parser's allowed geo types."""
    return st.sampled_from(list(ALLOWED_GEO_TYPES))


# ---------------------------------------------------------------------------
# Round-trip property tests — one per kind.
#
# Settings: max_examples=200 per kind (CTO brief minimum). Health-check
# overrides skip the function-scoped fixture warning since we don't use
# any fixtures.
# ---------------------------------------------------------------------------


_CTO_PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,  # parser is fast; deadline noise is unhelpful in CI
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------- factor ---------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(
    source=slug_strategy(),
    namespace=namespace_strategy(),
    fid=factor_id_strategy(),
    version=_version_strategy(),
)
def test_factor_build_parse_roundtrip(
    source: str, namespace: str, fid: str, version: int
) -> None:
    """``parse(build(...)) == GLUrn(...)`` for every random factor URN."""
    s = build(
        "factor", source=source, namespace=namespace, id=fid, version=version
    )
    u = parse(s)
    assert u.kind == "factor"
    assert u.source == source
    assert u.namespace == namespace
    assert u.id == fid
    assert u.version == version


@_CTO_PROPERTY_SETTINGS
@given(
    source=slug_strategy(),
    namespace=namespace_strategy(),
    fid=factor_id_strategy(),
    version=_version_strategy(),
)
def test_factor_parse_render_roundtrip(
    source: str, namespace: str, fid: str, version: int
) -> None:
    """``parse(s).render() == s`` for every well-formed factor URN."""
    s = build(
        "factor", source=source, namespace=namespace, id=fid, version=version
    )
    assert parse(s).render() == s


# ---------- source ---------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(slug=slug_strategy())
def test_source_build_parse_roundtrip(slug: str) -> None:
    s = build("source", slug=slug)
    u = parse(s)
    assert u.kind == "source"
    assert u.slug == slug


@_CTO_PROPERTY_SETTINGS
@given(slug=slug_strategy())
def test_source_parse_render_roundtrip(slug: str) -> None:
    s = build("source", slug=slug)
    assert parse(s).render() == s


# ---------- pack -----------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(
    source=slug_strategy(),
    pack_id=slug_strategy(),
    version=_version_strategy(),
)
def test_pack_build_parse_roundtrip(
    source: str, pack_id: str, version: int
) -> None:
    s = build("pack", source=source, pack_id=pack_id, version=version)
    u = parse(s)
    assert u.kind == "pack"
    assert u.source == source
    assert u.pack_id == pack_id
    assert u.version == version


@_CTO_PROPERTY_SETTINGS
@given(
    source=slug_strategy(),
    pack_id=slug_strategy(),
    version=_version_strategy(),
)
def test_pack_parse_render_roundtrip(
    source: str, pack_id: str, version: int
) -> None:
    s = build("pack", source=source, pack_id=pack_id, version=version)
    assert parse(s).render() == s


# ---------- methodology ----------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(slug=slug_strategy())
def test_methodology_build_parse_roundtrip(slug: str) -> None:
    s = build("methodology", slug=slug)
    u = parse(s)
    assert u.kind == "methodology"
    assert u.slug == slug


@_CTO_PROPERTY_SETTINGS
@given(slug=slug_strategy())
def test_methodology_parse_render_roundtrip(slug: str) -> None:
    s = build("methodology", slug=slug)
    assert parse(s).render() == s


# ---------- geo ------------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(geo_type=_geo_type_strategy(), geo_id=slug_strategy())
def test_geo_build_parse_roundtrip(geo_type: str, geo_id: str) -> None:
    s = build("geo", geo_type=geo_type, id=geo_id)
    u = parse(s)
    assert u.kind == "geo"
    assert u.geo_type == geo_type
    assert u.id == geo_id


@_CTO_PROPERTY_SETTINGS
@given(geo_type=_geo_type_strategy(), geo_id=slug_strategy())
def test_geo_parse_render_roundtrip(geo_type: str, geo_id: str) -> None:
    s = build("geo", geo_type=geo_type, id=geo_id)
    assert parse(s).render() == s


# ---------- unit -----------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(symbol=unit_symbol_strategy())
def test_unit_build_parse_roundtrip(symbol: str) -> None:
    s = build("unit", symbol=symbol)
    u = parse(s)
    assert u.kind == "unit"
    assert u.symbol == symbol


@_CTO_PROPERTY_SETTINGS
@given(symbol=unit_symbol_strategy())
def test_unit_parse_render_roundtrip(symbol: str) -> None:
    s = build("unit", symbol=symbol)
    assert parse(s).render() == s


# ---------- community ------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(slug=slug_strategy())
def test_community_build_parse_roundtrip(slug: str) -> None:
    s = build("community", slug=slug)
    u = parse(s)
    assert u.kind == "community"
    assert u.slug == slug


@_CTO_PROPERTY_SETTINGS
@given(slug=slug_strategy())
def test_community_parse_render_roundtrip(slug: str) -> None:
    s = build("community", slug=slug)
    assert parse(s).render() == s


# ---------- partner --------------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(
    tenant=slug_strategy(),
    pack_id=slug_strategy(),
    version=_version_strategy(),
)
def test_partner_build_parse_roundtrip(
    tenant: str, pack_id: str, version: int
) -> None:
    s = build("partner", tenant=tenant, pack_id=pack_id, version=version)
    u = parse(s)
    assert u.kind == "partner"
    assert u.tenant == tenant
    assert u.pack_id == pack_id
    assert u.version == version


@_CTO_PROPERTY_SETTINGS
@given(
    tenant=slug_strategy(),
    pack_id=slug_strategy(),
    version=_version_strategy(),
)
def test_partner_parse_render_roundtrip(
    tenant: str, pack_id: str, version: int
) -> None:
    s = build("partner", tenant=tenant, pack_id=pack_id, version=version)
    assert parse(s).render() == s


# ---------- enterprise -----------------------------------------------------


@_CTO_PROPERTY_SETTINGS
@given(
    tenant=slug_strategy(),
    pack_id=slug_strategy(),
    version=_version_strategy(),
)
def test_enterprise_build_parse_roundtrip(
    tenant: str, pack_id: str, version: int
) -> None:
    s = build("enterprise", tenant=tenant, pack_id=pack_id, version=version)
    u = parse(s)
    assert u.kind == "enterprise"
    assert u.tenant == tenant
    assert u.pack_id == pack_id
    assert u.version == version


@_CTO_PROPERTY_SETTINGS
@given(
    tenant=slug_strategy(),
    pack_id=slug_strategy(),
    version=_version_strategy(),
)
def test_enterprise_parse_render_roundtrip(
    tenant: str, pack_id: str, version: int
) -> None:
    s = build("enterprise", tenant=tenant, pack_id=pack_id, version=version)
    assert parse(s).render() == s


# ---------------------------------------------------------------------------
# Negative corpus — curated adversarial inputs that MUST raise.
#
# The corpus exercises every documented rejection rule:
#
#   * uppercase segments outside the allowed factor-id ``T``/``Z`` slots
#   * missing version segment on factor / pack / partner / enterprise
#   * empty body / empty inter-segment colons
#   * unknown kind discriminator
#   * oversized URN (>512 chars)
#   * stray colons inside source / methodology / geo-id / unit / activity
#     / community kinds (which forbid further ``:`` after their primary
#     segment).
#
# Test framework: pytest.mark.parametrize keeps each case independently
# reportable. Hypothesis is intentionally not used here — adversarial
# inputs are easier to audit when curated.
# ---------------------------------------------------------------------------


_REJECT_CASES: list[tuple[str, str]] = [
    # --- Uppercase namespace / source / id segments --------------------
    (
        "urn:gl:factor:IPCC-AR6:stationary-combustion:natural-gas:v1",
        "uppercase source slug",
    ),
    (
        "urn:gl:factor:ipcc-ar6:STATIONARY:natural-gas:v1",
        "uppercase namespace",
    ),
    (
        "urn:gl:factor:ipcc-ar6:stationary:NATURAL-GAS:v1",
        "uppercase factor id (non-T/Z)",
    ),
    (
        "urn:gl:source:EPA-EGRID",
        "uppercase source slug",
    ),
    (
        "urn:gl:methodology:GHGP-CORPORATE",
        "uppercase methodology slug",
    ),
    (
        "urn:gl:geo:country:US",
        "uppercase geo id",
    ),
    (
        "urn:gl:unit:KGCO2E/KWH",
        "uppercase unit symbol",
    ),
    # --- Missing version segments --------------------------------------
    (
        "urn:gl:factor:ipcc-ar6:stationary:natural-gas",
        "factor missing :v<n>",
    ),
    (
        "urn:gl:factor:ipcc-ar6:stationary:natural-gas:v0",
        "factor v0 (versions must be >= 1)",
    ),
    (
        "urn:gl:pack:ipcc-ar6:tier-1-defaults",
        "pack missing :v<n>",
    ),
    (
        "urn:gl:partner:tenant-a:my-pack",
        "partner missing :v<n>",
    ),
    (
        "urn:gl:enterprise:tenant-a:my-pack",
        "enterprise missing :v<n>",
    ),
    # --- Empty body / empty segments -----------------------------------
    (
        "urn:gl:factor:",
        "factor empty body",
    ),
    (
        "urn:gl:factor:ipcc-ar6::natural-gas:v1",
        "factor empty namespace segment",
    ),
    (
        "urn:gl:factor:ipcc-ar6:stationary::v1",
        "factor empty id segment",
    ),
    (
        "urn:gl:source:",
        "source empty slug",
    ),
    (
        "urn:gl:geo::us",
        "geo empty type",
    ),
    (
        "urn:gl:geo:country:",
        "geo empty id",
    ),
    # --- Invalid kind discriminator ------------------------------------
    (
        "urn:gl:emissionfactor:ipcc-ar6:x:y:v1",
        "unknown kind discriminator",
    ),
    (
        "urn:gl::ipcc-ar6",
        "empty kind",
    ),
    # --- Wrong namespace prefix ---------------------------------------
    (
        "urn:other:factor:ipcc-ar6:x:y:v1",
        "non-gl namespace",
    ),
    (
        "factor:ipcc-ar6:x:y:v1",
        "no urn: prefix",
    ),
    # --- Oversized body (>512 chars) -----------------------------------
    (
        "urn:gl:factor:ipcc-ar6:stationary:" + ("a" * 600) + ":v1",
        "URN exceeds 512-char ceiling",
    ),
    # --- Stray ':' inside single-segment kinds ------------------------
    (
        "urn:gl:source:ipcc:ar6",
        "source has extra ':' after slug",
    ),
    (
        "urn:gl:methodology:ghgp:scope1",
        "methodology has extra ':' after slug",
    ),
    (
        "urn:gl:unit:kgco2e/kwh:extra",
        "unit symbol has extra ':' segment",
    ),
    (
        "urn:gl:community:my-cmty:extra",
        "community has extra ':' after slug",
    ),
    # --- Mismatched factor part count ----------------------------------
    (
        "urn:gl:factor:ipcc-ar6:v1",
        "factor missing namespace and id",
    ),
    # --- Disallowed characters in slug --------------------------------
    (
        "urn:gl:source:ipcc_ar6",
        "source slug contains underscore (not allowed)",
    ),
    (
        "urn:gl:source:ipcc ar6",
        "source slug contains space",
    ),
    (
        "urn:gl:source:ipcc/ar6",
        "source slug contains slash",
    ),
    # --- Boundary character failures ----------------------------------
    (
        "urn:gl:source:-ipcc",
        "source slug starts with hyphen (boundary fail)",
    ),
    (
        "urn:gl:source:ipcc-",
        "source slug ends with hyphen (boundary fail)",
    ),
]


@pytest.mark.parametrize("urn_str,why", _REJECT_CASES)
def test_negative_corpus_raises_invalid_urn_error(urn_str: str, why: str) -> None:
    """Every adversarial input MUST raise :class:`InvalidUrnError`."""
    with pytest.raises(InvalidUrnError):
        parse(urn_str)


# ---------------------------------------------------------------------------
# Bonus: confirm GLUrn.__eq__ over identical components.
# ---------------------------------------------------------------------------


def test_glurn_equality_after_roundtrip() -> None:
    """``GLUrn(**components) == parse(build(**components))`` — model
    equality is structural so the round-trip identity also holds at the
    Pydantic-model level (frozen + extra='forbid' makes this safe)."""
    components = {
        "source": "ipcc-ar6",
        "namespace": "stationary-combustion",
        "id": "natural-gas-residential",
        "version": 1,
    }
    s = build("factor", **components)
    parsed = parse(s)
    expected = GLUrn(kind="factor", **components)
    assert parsed == expected
