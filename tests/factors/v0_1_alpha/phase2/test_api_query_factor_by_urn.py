# -*- coding: utf-8 -*-
"""Phase 2 / WS10 - REST API URN query acceptance tests.

CTO Phase 2 brief Section 2.7, row #8:

    "test_api_query_factor_by_urn.py - REST GET /v0_1/alpha/factors/{urn}
    returns the canonical record by URN; 404 on unknown URN; alias and
    list filters return URN-primary payloads."

The alpha router (``greenlang.factors.api_v0_1_alpha_routes``) is mounted
under the ``/v1/`` prefix; the Phase 2 acceptance suite reaches it via
the bare FastAPI app pattern exercised by ``test_api_urn_primary.py``.
We bypass the broader ``factors_app.create_factors_app()`` machinery for
hermetic, fast tests.

Coverage of CTO §2.7 #8:

  1. ``GET /v1/factors/{urn}`` -> 200 with ``urn`` as the primary id in
     the body, ``factor_id_alias`` only as a sibling, and the canonical
     ontology references intact.
  2. ``GET /v1/factors/{urn}`` with an unknown URN -> 404 with the
     stable error envelope (``error`` / ``message`` / ``urn`` keys).
  3. ``GET /v1/factors/by-alias/{legacy_id}`` resolves through the
     alias table to the same canonical record (URN primary).
  4. ``GET /v1/factors?source_urn=...&geography_urn=...`` filter
     combinations narrow the list deterministically.
  5. The response NEVER carries the legacy ``factor_id`` primary slot.
"""
from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import quote

import pytest

from greenlang.factors.repositories.alpha_v0_1_repository import (
    AlphaFactorRepository,
)


pytest.importorskip("fastapi")
pytest.importorskip("starlette")


# ---------------------------------------------------------------------------
# Canonical alpha records (two so we can exercise filter narrowing).
# Both pass the AlphaProvenanceGate verbatim.
# ---------------------------------------------------------------------------


_REC_A_URN = (
    "urn:gl:factor:ipcc-2006-nggi:phase2:api-by-urn-record-a:v1"
)
_REC_B_URN = (
    "urn:gl:factor:ipcc-2006-nggi:phase2:api-by-urn-record-b:v1"
)
_REC_A_ALIAS = "EF:phase2:api-by-urn-a:v1"
_REC_B_ALIAS = "EF:phase2:api-by-urn-b:v1"


def _record(*, urn: str, alias: str, geography: str) -> Dict[str, Any]:
    return {
        "urn": urn,
        "factor_id_alias": alias,
        "source_urn": "urn:gl:source:ipcc-2006-nggi",
        "factor_pack_urn": "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1",
        "name": "Phase 2 API URN-query fixture factor",
        "description": (
            "Synthetic factor exercising the v0.1 alpha REST surface "
            "for the CTO Phase 2 §2.7 acceptance suite (row #8)."
        ),
        "category": "fuel",
        "value": 70.8,
        "unit_urn": "urn:gl:unit:kgco2e/gj",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": geography,
        "vintage_start": "2024-01-01",
        "vintage_end": "2099-12-31",
        "resolution": "annual",
        "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
        "boundary": "stationary-combustion",
        "licence": "IPCC-PUBLIC",
        "citations": [
            {
                "type": "url",
                "value": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
            }
        ],
        "published_at": "2026-04-25T07:42:30+00:00",
        "extraction": {
            "source_url": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
            "source_record_id": f"phase2-api-{alias}",
            "source_publication": "Phase 2 / WS10 API URN-query fixture",
            "source_version": "0.1",
            "raw_artifact_uri": "s3://greenlang-factors-raw/test/phase2-api.json",
            "raw_artifact_sha256": (
                "6ff38c51f0ffcb08b2057b90164c3f3e6b67a16bacffb27507526b4dab1271c6"
            ),
            "parser_id": "tests.factors.v0_1_alpha.phase2.api_query_by_urn",
            "parser_version": "0.1.0",
            "parser_commit": "0" * 40,
            "row_ref": f"phase2-api-{alias}",
            "ingested_at": "2026-04-25T07:42:30Z",
            "operator": "bot:test_api_query_factor_by_urn",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T07:42:30Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T07:42:30Z",
        },
        "tags": ["phase2", "ws10", "api-by-urn"],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_app(monkeypatch):
    """Build a hermetic FastAPI app with the alpha router and a seeded
    in-memory repository carrying two canonical records + one alias."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")

    from fastapi import FastAPI

    from greenlang.factors.api_v0_1_alpha_routes import router

    # legacy mode — Phase 1 provenance gate only; Phase 2 orchestrator covered by tests/factors/v0_1_alpha/phase2/test_publish_pipeline_e2e.py
    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:", publish_env="legacy"
    )

    rec_a = _record(
        urn=_REC_A_URN,
        alias=_REC_A_ALIAS,
        geography="urn:gl:geo:country:us",
    )
    rec_b = _record(
        urn=_REC_B_URN,
        alias=_REC_B_ALIAS,
        geography="urn:gl:geo:country:gb",
    )
    repo.publish(rec_a)
    repo.publish(rec_b)
    repo.register_alias(_REC_A_URN, _REC_A_ALIAS)
    # Note: rec_b is intentionally left without an alias entry so the
    # by-alias path 404s for it.

    app = FastAPI()
    app.state.alpha_factor_repo = repo
    app.include_router(router)
    yield app, repo
    repo.close()


@pytest.fixture()
def client(seeded_app):
    from fastapi.testclient import TestClient

    app, _repo = seeded_app
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. GET /v1/factors/{urn} -> 200 + URN-primary body
# ---------------------------------------------------------------------------


def test_get_factor_by_urn_returns_200_and_urn_primary(client) -> None:
    """The single-record endpoint returns the canonical record."""
    encoded = quote(_REC_A_URN, safe="")
    resp = client.get(f"/v1/factors/{encoded}")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    # Primary identifier MUST be ``urn``.
    assert body["urn"] == _REC_A_URN
    # The legacy primary slot must NEVER appear.
    assert "factor_id" not in body, (
        f"Response leaks the legacy primary 'factor_id': {sorted(body.keys())}"
    )
    # Optional legacy alias is the only remaining legacy slot.
    assert body.get("factor_id_alias") == _REC_A_ALIAS

    # Spot-check ontology references survive the round-trip verbatim.
    assert body["source_urn"] == "urn:gl:source:ipcc-2006-nggi"
    assert body["unit_urn"] == "urn:gl:unit:kgco2e/gj"
    assert body["geography_urn"] == "urn:gl:geo:country:us"
    assert body["methodology_urn"] == (
        "urn:gl:methodology:ipcc-tier-1-stationary-combustion"
    )
    assert body["factor_pack_urn"] == (
        "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1"
    )


def test_get_factor_by_urn_url_encoded_equals_raw(client) -> None:
    """Percent-encoded and raw URNs resolve to the same record."""
    raw = client.get(f"/v1/factors/{_REC_A_URN}")
    encoded = client.get(f"/v1/factors/{quote(_REC_A_URN, safe='')}")
    assert raw.status_code == encoded.status_code == 200
    assert raw.json()["urn"] == encoded.json()["urn"] == _REC_A_URN


# ---------------------------------------------------------------------------
# 2. GET /v1/factors/{urn} -> 404 + stable error envelope
# ---------------------------------------------------------------------------


def test_get_factor_unknown_urn_returns_404(client) -> None:
    """Unknown URN returns 404 with the stable error envelope."""
    bogus = "urn:gl:factor:nope:nope:nope-fixture:v9"
    resp = client.get(f"/v1/factors/{bogus}")
    assert resp.status_code == 404, resp.text
    body = resp.json()

    # Top-level keys, no FastAPI ``detail`` wrapper.
    assert "detail" not in body, (
        f"404 leaks FastAPI's default detail wrapper: {body!r}"
    )
    assert body["error"] == "factor_not_found"
    assert body["urn"] == bogus
    assert isinstance(body["message"], str) and body["message"]


# ---------------------------------------------------------------------------
# 3. GET /v1/factors/by-alias/{legacy_id} -> URN-primary record
# ---------------------------------------------------------------------------


def test_get_by_alias_returns_canonical_record(client) -> None:
    """Alias path resolves to the same URN-primary canonical record."""
    encoded = quote(_REC_A_ALIAS, safe="")
    resp = client.get(f"/v1/factors/by-alias/{encoded}")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["urn"] == _REC_A_URN
    assert body["factor_id_alias"] == _REC_A_ALIAS
    assert "factor_id" not in body


def test_get_by_alias_unknown_returns_404(client) -> None:
    """Alias miss returns the alias-specific 404 envelope."""
    resp = client.get("/v1/factors/by-alias/EF:phantom:does-not-exist")
    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert body["error"] == "factor_alias_not_found"
    assert body["legacy_id"] == "EF:phantom:does-not-exist"


def test_get_by_alias_and_by_urn_return_same_record(client) -> None:
    """The two GETs return identical canonical records."""
    by_urn = client.get(f"/v1/factors/{quote(_REC_A_URN, safe='')}").json()
    by_alias = client.get(
        f"/v1/factors/by-alias/{quote(_REC_A_ALIAS, safe='')}"
    ).json()
    assert by_urn["urn"] == by_alias["urn"]
    for key in (
        "source_urn",
        "factor_pack_urn",
        "value",
        "unit_urn",
        "geography_urn",
        "methodology_urn",
        "factor_id_alias",
    ):
        assert by_urn[key] == by_alias[key], key


# ---------------------------------------------------------------------------
# 4. GET /v1/factors?... filters return URN-primary lists
# ---------------------------------------------------------------------------


def test_list_factors_no_filter_returns_all(client) -> None:
    """Bare /v1/factors returns both seed records (URN-primary)."""
    resp = client.get("/v1/factors")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    urns = {item["urn"] for item in body["data"]}
    assert urns == {_REC_A_URN, _REC_B_URN}, urns
    # Every row carries `urn` as primary; none carry the legacy slot.
    for item in body["data"]:
        assert "factor_id" not in item
        assert "urn" in item


def test_list_factors_filter_by_source_urn(client) -> None:
    """source_urn filter returns rows that match (here, all rows share
    the same source - both are returned)."""
    resp = client.get(
        "/v1/factors?source_urn=urn:gl:source:ipcc-2006-nggi"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    urns = {item["urn"] for item in body["data"]}
    assert urns == {_REC_A_URN, _REC_B_URN}


def test_list_factors_filter_by_geography_urn(client) -> None:
    """geography_urn filter narrows the list to the matching record."""
    resp = client.get(
        "/v1/factors?geography_urn=urn:gl:geo:country:us"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    urns = {item["urn"] for item in body["data"]}
    assert urns == {_REC_A_URN}, urns


def test_list_factors_combined_filters(client) -> None:
    """source_urn + geography_urn AND-combine to a single record."""
    resp = client.get(
        "/v1/factors?source_urn=urn:gl:source:ipcc-2006-nggi"
        "&geography_urn=urn:gl:geo:country:gb"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    urns = {item["urn"] for item in body["data"]}
    assert urns == {_REC_B_URN}, urns


def test_list_factors_filter_no_match_returns_empty_data(client) -> None:
    """A geography filter that matches nothing returns an empty list."""
    resp = client.get(
        "/v1/factors?geography_urn=urn:gl:geo:country:zz"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"] == []


def test_list_factors_response_shape_contract(client) -> None:
    """Every list-response row carries `urn` (primary) and may carry
    `factor_id_alias` (optional). The legacy primary `factor_id` is
    structurally absent."""
    resp = client.get("/v1/factors")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("data"), list)
    assert "next_cursor" in body  # cursor key is always present
    for row in body["data"]:
        assert "urn" in row
        assert "factor_id" not in row
        # alias is allowed to be None or string but must be a recognised slot
        assert "factor_id_alias" in row
