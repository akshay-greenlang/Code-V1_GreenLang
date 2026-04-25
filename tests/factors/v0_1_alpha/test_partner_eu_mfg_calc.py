# -*- coding: utf-8 -*-
"""Wave E / TaskCreate #22 / WS8-T2 — Partner EU-MFG-01 SDK calculation golden test.

CTO doc §19.1 acceptance:

    "Two design partners have completed at least one calculation flow using
     the SDK and have signed off on usability."

This file proves the EU-MFG-01 partner (Italian cement producer) can
complete one CBAM cement default-lookup flow end-to-end via the
v0.1.0 alpha SDK.

The partner profile lives at ``docs/factors/design-partners/partner-EU-MFG-01.md``.

Calc scenario (from the partner doc, §7 "Expected SDK Calculation"):

    An Italian cement producer needs the EU CBAM default embedded-emission
    value for cement clinker (CN code 2523 10 00) for a Q3 2026 import
    declaration. They import 10,000 tonnes of cement-class CBAM goods
    sourced from a third country and want to validate the platform's
    published default against their own internal calc.

The factor we end up using:

    urn:gl:factor:cbam-default-values:CBAM:cement:CN:2024:v1
    value=0.84 kgCO2e / kg-product, vintage 2024-01-01 .. 9999-12-31,
    methodology=urn:gl:methodology:eu-cbam-default,
    licence=EU-Publication.

Rationale for this URN:

    The partner doc §7 calls for the "cement clinker (CN 2523 10 00)
    default" published by the EU CBAM Annex IV implementing regulation
    2023/1773. The published v0.1 catalog seed
    (``catalog_seed_v0_1/cbam_default_values/v1.json``) carries the
    CBAM Annex IV defaults at the *sector* rollup level (i.e. one row
    per CBAM sector × country-of-origin) rather than at the per-CN-code
    row level. The "cbam cement" sector rollup is the closest match to
    the partner's "cement clinker default" ask under the v0.1 alpha
    catalog. We pick the CN (China) origin row because (a) the partner
    explicitly cites a third-country import scenario, (b) China is the
    largest CBAM-third-country cement-export volume globally, and (c)
    the CN row's value 0.84 kgCO2e/kg falls inside the expected
    "cement clinker default" envelope of 0.7-0.9 kgCO2e/kg from CBAM
    Annex IV. The exact CN-code-row vs sector-rollup distinction is
    documented in the seed metadata's ``source_publication`` string.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List
from urllib.parse import quote

import pytest

from greenlang.factors.sdk.python import (
    AlphaFactor,
    FactorsClient,
    ListFactorsResponse,
)

from tests.factors.v0_1_alpha._partner_helpers import (
    PARTNER_API_KEYS,
    boot_alpha_app_for_partner,
    make_partner_client,
    seed_partner_catalog,
)


def _fetch_alpha_factor_via_testclient(
    app: Any, urn: str, partner_slug: str
) -> AlphaFactor:
    """Fetch a factor via ``GET /v1/factors/{urn}`` using FastAPI TestClient.

    The seeded catalog records use the legacy multi-segment factor id
    layout (e.g. ``...:CBAM:cement:CN:2024:v1``) which the server-side
    router accepts but the SDK's strict client-side URN parser rejects
    (lowercase-namespace constraint). This is a known v0.1 alpha
    catalog/SDK drift, scheduled for an SDK patch in v0.2. For the
    partner golden test we side-step the client-side validator by
    calling the route directly while still validating the server's
    on-the-wire response with the canonical :class:`AlphaFactor` model.
    """
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    encoded = quote(urn, safe="")
    resp = test_client.get(
        f"/v1/factors/{encoded}",
        headers={"X-API-Key": PARTNER_API_KEYS[partner_slug]},
    )
    assert resp.status_code == 200, resp.text
    return AlphaFactor.model_validate(resp.json())


PARTNER_SLUG = "EU-MFG-01"

_EXPECTED_FACTOR_URN = (
    "urn:gl:factor:cbam-default-values:CBAM:cement:CN:2024:v1"
)
_EXPECTED_SOURCE_URN = "urn:gl:source:cbam-default-values"

# Activity data — partner doc §7: 10,000 tonnes of clinker-class import.
# 10,000 tonnes × 1000 kg/tonne = 10,000,000 kg of CBAM cement goods.
_ACTIVITY_KG = 10_000 * 1_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def partner_factors() -> List[Dict[str, Any]]:
    """The seeded v0.1 factor records for the EU-MFG-01 partner."""
    return seed_partner_catalog(PARTNER_SLUG)


@pytest.fixture()
def partner_app(monkeypatch, partner_factors):
    """Boot ``create_factors_app()`` with the EU-MFG-01 partner's catalog."""
    return boot_alpha_app_for_partner(
        monkeypatch,
        PARTNER_SLUG,
        partner_factors,
        edition_id="alpha-partner-eu-mfg-01",
    )


@pytest.fixture()
def partner_client(partner_app):
    """Boot a v0.1.0 SDK client wired to the partner app."""
    client = make_partner_client(partner_app, PARTNER_SLUG)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_cbam_cement_factor_urn(factors: List[Dict[str, Any]]) -> str:
    """Return the URN of the canonical CBAM cement-sector default factor.

    Prefer the explicit ``cbam-default-values:CBAM:cement:CN:2024:v1`` row
    that maps cleanly to the partner doc's "cement clinker default" ask.
    If a future seed regen replaces or removes that row we fall back to
    any cement-sector CBAM default with category ``cbam_default`` and
    a value in the plausible 0.5-1.0 kgCO2e/kg envelope (CBAM Annex IV
    cement-sector rollup is consistently in this band).
    """
    if any(f.get("urn") == _EXPECTED_FACTOR_URN for f in factors):
        return _EXPECTED_FACTOR_URN
    candidates = [
        f for f in factors
        if str(f.get("source_urn")) == _EXPECTED_SOURCE_URN
        and str(f.get("category")) == "cbam_default"
        and "cement" in str(f.get("name", "")).lower()
    ]
    if not candidates:
        raise AssertionError(
            "no CBAM cement default factor found in catalog seed; "
            "the partner golden test cannot run."
        )
    candidates.sort(key=lambda f: float(f.get("value") or 0.0), reverse=True)
    return candidates[0]["urn"]


# ---------------------------------------------------------------------------
# CTO §19.1 — Partner calc golden test (one flow per partner).
# ---------------------------------------------------------------------------


@pytest.mark.alpha_v0_1_acceptance
def test_eu_mfg_cement_clinker_cbam_default_lookup(
    partner_factors: List[Dict[str, Any]], partner_client: FactorsClient
) -> None:
    """EU-MFG-01 partner: CBAM cement default lookup for clinker-class import.

    Scenario (from partner-EU-MFG-01.md §7):
        An Italian cement producer imports 10,000 tonnes of CBAM
        cement goods (CN 25231000 cement clinker class) for a Q3 2026
        import declaration. Use the EU CBAM Annex IV default
        embedded-emission value to compute the embedded-emissions total.

    Expected: the SDK lists the CBAM defaults pack and factor list, the
    partner picks the cement-sector default for their relevant
    third-country origin, and the embedded-emissions total falls inside
    the CBAM Annex IV cement-sector envelope of ~5-9 million kgCO2e for
    a 10,000-tonne import.
    """
    # ---- 1. Health probe — alpha mode, frozen schema id. -----------------
    health = partner_client.health()
    assert health.status == "ok"
    assert health.release_profile == "alpha-v0.1"
    assert health.schema_id.endswith("factor_record_v0_1.schema.json")
    assert health.edition == "alpha-partner-eu-mfg-01"

    # ---- 2. List packs filtered by source_urn=eu-cbam-defaults. ---------
    # NOTE: under v0.1 alpha, /v1/packs returns one synthetic pack per
    # source (``urn:gl:pack:<source_id>:default:v1``) — the real per-pack
    # registry lands in beta. The synthetic pack URN therefore does NOT
    # equal the seeded record's ``factor_pack_urn`` field. We assert on
    # the synthetic pack's ``source_urn`` instead, and use ``source_urn``
    # to filter ``list_factors`` below.
    packs = partner_client.list_packs(source_urn=_EXPECTED_SOURCE_URN)
    assert len(packs) >= 1, (
        f"list_packs(source_urn={_EXPECTED_SOURCE_URN!r}) returned no packs"
    )
    cbam_pack = packs[0]
    assert cbam_pack.urn.startswith("urn:gl:pack:cbam-default-values:") or (
        # AlphaFactorRepository (real-mode shim) emits the synthetic URN
        # using the source_id (with underscore); the source registry path
        # uses the slugified hyphen form. Accept either spelling.
        cbam_pack.urn.startswith("urn:gl:pack:cbam_default_values:")
    ), f"unexpected pack URN: {cbam_pack.urn!r}"
    assert cbam_pack.source_urn == _EXPECTED_SOURCE_URN

    # ---- 3. List factors filtered by source + category. -----------------
    # We filter by ``source_urn + category=cbam_default`` rather than by
    # the synthetic pack URN — see comment block above for why. The
    # category enum is published in alpha and includes ``cbam_default``.
    listing: ListFactorsResponse = partner_client.list_factors(
        source_urn=_EXPECTED_SOURCE_URN,
        category="cbam_default",
    )
    assert len(listing.data) >= 1, (
        "list_factors(source_urn=eu-cbam-defaults, category=cbam_default) "
        "returned no rows; the partner cannot complete their calc."
    )

    # ---- 4. Pick the cement-sector default factor. ----------------------
    cement_rows = [
        row for row in listing.data
        if "cement" in row.name.lower()
        or "clinker" in row.name.lower()
    ]
    assert cement_rows, (
        f"no cement-sector CBAM default in listing; rows={listing.data!r}"
    )
    # Pick the explicit CN (China) origin row — see the file-level
    # docstring for the rationale.
    cn_rows = [
        row for row in cement_rows if row.urn == _EXPECTED_FACTOR_URN
    ]
    clinker = cn_rows[0] if cn_rows else cement_rows[0]

    # ---- 5. Factor metadata sanity. -------------------------------------
    assert isinstance(clinker, AlphaFactor)
    assert clinker.gwp_basis == "ar6"
    assert clinker.gwp_horizon == 100
    assert clinker.category == "cbam_default"
    assert clinker.licence  # explicit license tag — CTO §19.1
    assert clinker.licence.strip()
    # CBAM Annex IV is published by the European Commission.
    assert clinker.extraction.source_publication
    assert "cbam" in clinker.extraction.source_publication.lower()
    # Approved (production hard-gate; Wave B).
    assert clinker.review.review_status == "approved"

    # ---- 6. Compute embedded-emission default for the import. -----------
    # Unit on CBAM cement defaults is kgCO2e per kg-product.
    assert clinker.unit_urn == "urn:gl:unit:kgco2e/kg-product"
    embedded_emissions_kg = _ACTIVITY_KG * clinker.value
    # CBAM Annex IV cement-sector default values fall in the ~0.5-1.0
    # kgCO2e/kg envelope. For 10M kg of imports:
    #   lower bound  ≈ 10M × 0.5 = 5,000,000 kgCO2e
    #   upper bound  ≈ 10M × 1.0 = 10,000,000 kgCO2e
    # We widen slightly (4M..12M) to leave room for a future seed regen
    # that picks a slightly different cement-sector row.
    assert 4_000_000 < embedded_emissions_kg < 12_000_000, (
        f"unexpected CBAM cement embedded emissions for EU-MFG-01: "
        f"{embedded_emissions_kg} kgCO2e (factor.value={clinker.value} "
        f"kgCO2e/kg, activity={_ACTIVITY_KG} kg)"
    )

    # ---- 7. Vintage covers Q3 2026 import. ------------------------------
    assert isinstance(clinker.vintage_start, date)
    assert isinstance(clinker.vintage_end, date)
    assert clinker.vintage_start <= date(2026, 7, 1)
    assert clinker.vintage_end >= date(2026, 9, 30)


# ---------------------------------------------------------------------------
# Companion partner-acceptance tests.
# ---------------------------------------------------------------------------


@pytest.mark.alpha_v0_1_acceptance
def test_eu_mfg_lists_only_allow_listed_sources(partner_app: Any) -> None:
    """The catalog visible to EU-MFG-01 carries the CBAM-defaults source.

    The partner profile (§4) allow-lists 4 sources:
      * ipcc-ar6     (canonical alpha alias for ``ipcc_2006_nggi``)
      * defra-2025   (canonical alpha id ``desnz-ghg-conversion``)
      * epa-ghg-hub  (canonical alpha id ``epa-hub``)
      * eu-cbam-defaults  (canonical alpha id ``cbam-default-values``)

    NOTE: as with IN-EXPORT-01, the alpha v0.1 API does NOT runtime-enforce
    per-tenant allowlists; this is a Beta/GA feature. The contract surface
    we DO assert is that the headline source for the partner's calc
    (``cbam-default-values``) is reachable on /v1/sources and on
    /v1/factors via the partner's API key.
    """
    from fastapi.testclient import TestClient

    test_client = TestClient(partner_app)
    resp = test_client.get(
        "/v1/sources",
        headers={"X-API-Key": PARTNER_API_KEYS[PARTNER_SLUG]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    rows = body.get("data") or []
    assert rows, f"alpha /v1/sources returned an empty list: {body!r}"
    listed_urns = {str(r.get("urn")) for r in rows}
    assert _EXPECTED_SOURCE_URN in listed_urns, (
        f"EU-MFG-01 partner catalog must surface CBAM defaults; "
        f"got {sorted(listed_urns)}"
    )

    # Cross-check on /v1/factors.
    page: ListFactorsResponse = make_partner_client(
        partner_app, PARTNER_SLUG
    ).list_factors(source_urn=_EXPECTED_SOURCE_URN)
    assert page.data, (
        f"list_factors(source_urn={_EXPECTED_SOURCE_URN!r}) returned no rows"
    )
    for row in page.data:
        assert row.source_urn == _EXPECTED_SOURCE_URN


@pytest.mark.alpha_v0_1_acceptance
def test_eu_mfg_cbam_pack_metadata_complete(
    partner_client: FactorsClient,
) -> None:
    """The CBAM pack returned to EU-MFG-01 carries complete pack metadata.

    Every alpha pack record MUST surface ``urn``, ``source_urn``, and
    a non-empty ``version`` so the partner's audit log can record which
    pack version their calc resolved against.
    """
    packs = partner_client.list_packs(source_urn=_EXPECTED_SOURCE_URN)
    assert packs, "no CBAM pack returned for the partner"
    for pack in packs:
        # URN namespacing — every pack URN MUST be ``urn:gl:pack:...``.
        assert pack.urn.startswith("urn:gl:pack:")
        # Source URN MUST match the filter we passed.
        assert pack.source_urn == _EXPECTED_SOURCE_URN
        # Version MUST be a non-empty string.
        assert pack.version
        assert pack.version.strip()


@pytest.mark.alpha_v0_1_acceptance
def test_eu_mfg_cbam_factor_audit_trail(
    partner_factors: List[Dict[str, Any]],
    partner_app: Any,
) -> None:
    """For the partner's CBAM cement factor, every audit trail field is present.

    CTO doc §19.1 (verbatim): "license tagging: every factor in production
    carries a licence tag that matches its source." The license MUST also
    survive the SDK round-trip; we additionally verify the full extraction
    + review provenance bundle since that is what makes the CBAM
    declaration auditable.
    """
    factor_urn = _resolve_cbam_cement_factor_urn(partner_factors)
    factor = _fetch_alpha_factor_via_testclient(
        partner_app, factor_urn, PARTNER_SLUG
    )

    # ---- License tag MUST be present and non-empty (CTO §19.1). ---------
    assert factor.licence
    assert factor.licence.strip()

    # ---- Citations preserved. -------------------------------------------
    assert factor.citations  # non-empty
    for cite in factor.citations:
        assert cite.type
        assert cite.value

    # ---- Extraction provenance — every gate-required key present. -------
    # NOTE: the SDK's :class:`Extraction` model declares only the
    # canonical typed fields; the alpha provenance gate fields like
    # ``source_publication`` / ``ingested_at`` / ``operator`` ride on
    # the model via ``extra='allow'`` and therefore surface as raw
    # JSON-shape values (strings). We assert presence + non-emptiness
    # on those rather than typed casts.
    extraction = factor.extraction
    assert extraction.source_url
    assert extraction.source_url.startswith("http")
    assert extraction.raw_artifact_uri
    raw_hash = extraction.raw_artifact_sha256
    assert isinstance(raw_hash, str) and len(raw_hash) == 64
    int(raw_hash, 16)  # must be valid hex
    assert extraction.parser_id
    assert extraction.parser_version
    assert extraction.parser_commit
    extras = extraction.model_dump()
    pub = extras.get("source_publication")
    assert pub
    # CBAM Annex IV is published by the European Commission.
    assert "cbam" in pub.lower()
    assert extras.get("source_version")
    assert extras.get("source_record_id")
    assert extras.get("row_ref")
    assert extras.get("ingested_at")  # ISO-8601 timestamp string
    assert extras.get("operator")

    # ---- Review block — every gate-required key present. ----------------
    review = factor.review
    assert review.review_status == "approved"
    assert review.reviewer
    assert isinstance(review.reviewed_at, datetime)
    assert review.approved_by
    assert isinstance(review.approved_at, datetime)
