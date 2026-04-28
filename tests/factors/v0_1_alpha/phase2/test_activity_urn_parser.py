# -*- coding: utf-8 -*-
"""Phase 2 / WS5 — activity URN parser/builder tests.

Covers the new optional ``taxonomy`` + ``code`` segmentation introduced
for ``urn:gl:activity:<taxonomy>:<code-slug>`` while ensuring the
legacy ``urn:gl:activity:<slug>`` form parses identically. Reject any
unknown taxonomy and uppercase characters.
"""
from __future__ import annotations

import pytest

from greenlang.factors.ontology.urn import (
    ALLOWED_ACTIVITY_TAXONOMIES,
    GLUrn,
    InvalidUrnError,
    build,
    parse,
)


# ---------------------------------------------------------------------------
# One representative URN per taxonomy — round-trip parse -> render -> build.
# ---------------------------------------------------------------------------

REPRESENTATIVE_URNS = [
    ("ipcc", "1-a-1-a"),
    ("ghgp", "scope1"),
    ("hs-cn", "72"),
    ("cpc", "0"),
    ("nace", "a"),
    ("naics", "31-33"),
    ("sic", "20-39"),
    ("pact", "chemicals"),
    ("freight", "road-wtw"),
    ("cbam", "iron-steel"),
    ("pcf", "manufacturing"),
    ("refrigerants", "hfc-134a"),
    ("agriculture", "enteric-fermentation"),
    ("waste", "msw-landfill"),
    ("land-use", "deforestation"),
]


@pytest.mark.parametrize("taxonomy,code", REPRESENTATIVE_URNS)
def test_activity_taxonomy_form_parse_render_round_trip(
    taxonomy: str, code: str
) -> None:
    s = f"urn:gl:activity:{taxonomy}:{code}"
    parsed = parse(s)
    assert isinstance(parsed, GLUrn)
    assert parsed.kind == "activity"
    assert parsed.taxonomy == taxonomy
    assert parsed.code == code
    # Legacy slug field stays None on the new form.
    assert parsed.slug is None
    # Render must be byte-for-byte identical.
    assert parsed.render() == s
    # Build round-trip from parsed components.
    rebuilt = build("activity", taxonomy=parsed.taxonomy, code=parsed.code)
    assert rebuilt == s


def test_all_15_taxonomies_round_trip() -> None:
    """Every taxonomy in the CTO list survives parse -> render."""
    assert len(ALLOWED_ACTIVITY_TAXONOMIES) == 15
    seen = {t for t, _ in REPRESENTATIVE_URNS}
    assert seen == set(ALLOWED_ACTIVITY_TAXONOMIES), (
        "Test parametrisation must cover every allowed taxonomy."
    )


# ---------------------------------------------------------------------------
# Backwards compatibility: legacy single-slug form still parses.
# ---------------------------------------------------------------------------


def test_legacy_activity_slug_form_still_parses() -> None:
    s = "urn:gl:activity:stationary-combustion-natural-gas"
    parsed = parse(s)
    assert parsed.kind == "activity"
    assert parsed.slug == "stationary-combustion-natural-gas"
    assert parsed.taxonomy is None
    assert parsed.code is None
    assert parsed.render() == s


def test_legacy_activity_build_still_works() -> None:
    s = build("activity", slug="electricity-purchased")
    assert s == "urn:gl:activity:electricity-purchased"
    parsed = parse(s)
    assert parsed.slug == "electricity-purchased"
    assert parsed.taxonomy is None
    assert parsed.code is None


# ---------------------------------------------------------------------------
# Reject invalid taxonomies (form is correct but taxonomy is not in list).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_taxonomy",
    [
        "made-up",
        "iso-14064",  # plausible but not in CTO list
        "scope3",     # GHGP scopes go in code, not taxonomy
        "x",
    ],
)
def test_reject_unknown_taxonomy(bad_taxonomy: str) -> None:
    s = f"urn:gl:activity:{bad_taxonomy}:foo"
    with pytest.raises(InvalidUrnError, match="taxonomy"):
        parse(s)


# ---------------------------------------------------------------------------
# Reject uppercase anywhere in the activity URN.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_urn",
    [
        # Uppercase taxonomy
        "urn:gl:activity:IPCC:1-a-1-a",
        # Uppercase in code
        "urn:gl:activity:ipcc:1-A-1-a",
        "urn:gl:activity:nace:A",
        "urn:gl:activity:cbam:Iron-Steel",
        # Uppercase in legacy slug form
        "urn:gl:activity:Stationary-Combustion",
    ],
)
def test_reject_uppercase_in_activity_urn(bad_urn: str) -> None:
    with pytest.raises(InvalidUrnError):
        parse(bad_urn)


# ---------------------------------------------------------------------------
# Reject malformed taxonomy-form variants.
# ---------------------------------------------------------------------------


def test_reject_three_segment_activity_urn() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:activity:ipcc:1-a-1-a:extra")


def test_reject_empty_code_segment() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:activity:ipcc:")


def test_reject_empty_taxonomy_segment() -> None:
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:activity::1-a-1-a")


def test_reject_invalid_code_chars() -> None:
    # Spaces are never allowed.
    with pytest.raises(InvalidUrnError):
        parse("urn:gl:activity:ipcc:1 a 1 a")


# ---------------------------------------------------------------------------
# Build-side validation: missing or extra components on the new form.
# ---------------------------------------------------------------------------


def test_build_activity_taxonomy_form_round_trip() -> None:
    s = build("activity", taxonomy="ghgp", code="scope3-cat-15")
    assert s == "urn:gl:activity:ghgp:scope3-cat-15"
    parsed = parse(s)
    assert parsed.taxonomy == "ghgp"
    assert parsed.code == "scope3-cat-15"


def test_build_activity_with_unknown_taxonomy_rejected() -> None:
    with pytest.raises(InvalidUrnError):
        build("activity", taxonomy="iso-14064", code="anything")


def test_build_activity_with_no_components_rejected() -> None:
    with pytest.raises(InvalidUrnError, match="missing required"):
        build("activity")


def test_build_activity_extra_components_rejected_for_legacy_form() -> None:
    # Mixing slug + taxonomy is not a valid build form.
    with pytest.raises(InvalidUrnError):
        build("activity", slug="foo", taxonomy="ipcc", code="1-a")


def test_build_activity_taxonomy_form_with_empty_code_rejected() -> None:
    with pytest.raises(InvalidUrnError):
        build("activity", taxonomy="ipcc", code="")


# ---------------------------------------------------------------------------
# parse(build(...)) round-trip for every taxonomy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("taxonomy,code", REPRESENTATIVE_URNS)
def test_build_then_parse_round_trip(taxonomy: str, code: str) -> None:
    s1 = build("activity", taxonomy=taxonomy, code=code)
    parsed = parse(s1)
    s2 = parsed.render()
    assert s1 == s2
    s3 = build("activity", taxonomy=parsed.taxonomy, code=parsed.code)
    assert s3 == s1
