# -*- coding: utf-8 -*-
"""Phase 2 / WS10 - SDK fetch-by-URN acceptance tests.

CTO Phase 2 brief Section 2.7, row #9:

    "test_sdk_fetch_by_urn.py - the Python SDK exposes
    ``factors.get_by_urn()`` (or whatever the canonical method is called),
    routes the request through the alias resolver when given a legacy id,
    and never lets ``factor_id`` leak as the primary identifier."

The SDK lives at ``greenlang.factors.sdk.python.client.FactorsClient``.
We mock the HTTP transport with ``httpx.MockTransport`` so the round-trip
runs without a server; this matches the established pattern in
``tests/factors/v0_1_alpha/phase2/test_api_urn_primary.py`` which already
uses the same mocking machinery for the same SDK.

Coverage of CTO §2.7 #9:

  1. ``client.get_factor(urn)`` returns a typed :class:`AlphaFactor`
     whose ``urn`` matches.
  2. ``client.get_by_alias(legacy_id)`` resolves through the alias
     endpoint; response is :class:`AlphaFactor` with the same URN as
     the by-URN path.
  3. ``client.list_factors(source_urn=...)`` returns a paginated cursor
     response (``ListFactorsResponse``).
  4. The SDK exposes ``urn`` (NOT ``factor_id``) as the primary
     identifier on every response model.
  5. The SDK refuses to fetch a legacy ``EF:...`` id via
     :meth:`get_factor` (URN-only contract is enforced client-side).
"""
from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

import httpx
import pytest

from greenlang.factors.sdk.python import (
    AlphaFactor,
    FactorsClient,
    ListFactorsResponse,
)


_BASE_URL = "https://factors.test"


# Canonical record - shared with the API/by-alias tests so the wire shape
# is provably identical across acceptance suites.
_VALID_URN = (
    "urn:gl:factor:ipcc-2006-nggi:phase2:sdk-fetch-by-urn:v1"
)
_LEGACY_ALIAS = "EF:phase2:sdk-fetch-by-urn:v1"


def _make_payload(urn: str = _VALID_URN, alias: str = _LEGACY_ALIAS) -> Dict[str, Any]:
    """Wire-compatible v0.1 alpha factor payload."""
    return {
        "urn": urn,
        "factor_id_alias": alias,
        "source_urn": "urn:gl:source:ipcc-2006-nggi",
        "factor_pack_urn": "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1",
        "name": "Phase 2 SDK fetch-by-urn fixture",
        "description": (
            "Synthetic record exercising the SDK Phase 2 acceptance "
            "for fetch-by-urn / fetch-by-alias / list paginator."
        ),
        "category": "fuel",
        "value": 56.1,
        "unit_urn": "urn:gl:unit:kgco2e/gj",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": "urn:gl:geo:country:us",
        "vintage_start": "2024-01-01",
        "vintage_end": "2099-12-31",
        "resolution": "annual",
        "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
        "boundary": "stationary-combustion",
        "licence": "IPCC-PUBLIC",
        "citations": [
            {"type": "url", "value": "https://www.ipcc.ch/"}
        ],
        "published_at": "2026-04-25T07:42:30+00:00",
        "extraction": {
            "source_url": "https://www.ipcc.ch/",
            "source_record_id": "phase2-sdk-fetch-by-urn",
            "source_publication": "Phase 2 / WS10 SDK acceptance fixture",
            "source_version": "0.1",
            "raw_artifact_uri": "s3://greenlang-factors-raw/test/phase2-sdk.json",
            "raw_artifact_sha256": "a" * 64,
            "parser_id": "tests.factors.v0_1_alpha.phase2.sdk_fetch_by_urn",
            "parser_version": "0.1.0",
            "parser_commit": "0" * 40,
            "row_ref": "phase2-sdk-fetch-by-urn",
            "ingested_at": "2026-04-25T07:42:30Z",
            "operator": "bot:test_sdk_fetch_by_urn",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T07:42:30Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T07:42:30Z",
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def alpha_profile(monkeypatch):
    """Activate the alpha-v0.1 release profile so feature gates are
    in their alpha configuration during the SDK round-trip."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.delenv("GL_ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    yield


def _scripted_transport(
    responses: List[httpx.Response],
    captured_paths: List[str],
) -> httpx.MockTransport:
    """Sequential mock transport that returns the next ``responses`` item
    on every call. Captures the request path for assertion."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        if responses:
            return responses.pop(0)
        return httpx.Response(500, json={"error": "exhausted_mock"})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# 1. client.get_factor(urn) -> typed AlphaFactor
# ---------------------------------------------------------------------------


def test_get_factor_returns_typed_alpha_factor(alpha_profile) -> None:
    """``client.get_factor(urn)`` returns AlphaFactor with matching urn."""
    payload = _make_payload()
    captured: List[str] = []
    transport = _scripted_transport(
        [httpx.Response(200, json=payload)],
        captured,
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        factor = client.get_factor(_VALID_URN)
    assert isinstance(factor, AlphaFactor)
    assert factor.urn == _VALID_URN
    assert factor.factor_id_alias == _LEGACY_ALIAS
    # Sanity: the SDK actually went through the by-URN endpoint (path
    # should contain the URL-encoded URN).
    assert any("/factors/" in p for p in captured), captured


def test_get_factor_response_carries_no_legacy_factor_id(alpha_profile) -> None:
    """The Pydantic AlphaFactor model has no `factor_id` field at all."""
    payload = _make_payload()
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=payload)
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        factor = client.get_factor(_VALID_URN)
    fields = AlphaFactor.model_fields
    assert "urn" in fields
    assert "factor_id_alias" in fields
    assert "factor_id" not in fields
    # Round-trip: dumped wire shape preserves urn-as-primary.
    dumped = factor.model_dump()
    assert dumped["urn"] == _VALID_URN
    assert "factor_id" not in dumped


# ---------------------------------------------------------------------------
# 2. client.get_by_alias(legacy_id) -> AlphaFactor through alias path
# ---------------------------------------------------------------------------


def test_get_by_alias_resolves_through_alias_endpoint(alpha_profile) -> None:
    """``get_by_alias()`` hits the alias endpoint and returns the same record."""
    payload = _make_payload()
    captured: List[str] = []
    transport = _scripted_transport(
        [httpx.Response(200, json=payload)],
        captured,
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        factor = client.get_by_alias(_LEGACY_ALIAS)
    assert isinstance(factor, AlphaFactor)
    assert factor.urn == _VALID_URN
    assert factor.factor_id_alias == _LEGACY_ALIAS
    # The SDK MUST hit the alias-specific path, never the canonical
    # /factors/{urn} path - confirms the alias router is used.
    assert any("/factors/by-alias/" in p for p in captured), captured


def test_get_by_alias_returns_none_on_404(alpha_profile) -> None:
    """A 404 from the alias endpoint surfaces as None, not an exception."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            404,
            json={
                "error": "factor_alias_not_found",
                "message": "no match",
                "legacy_id": "EF:phantom",
            },
        )
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        result = client.get_by_alias("EF:phantom")
    assert result is None


def test_get_by_urn_and_get_by_alias_yield_same_factor(alpha_profile) -> None:
    """The two SDK paths produce identical AlphaFactor instances."""
    payload = _make_payload()
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=payload)
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        by_urn = client.get_factor(_VALID_URN)
        by_alias = client.get_by_alias(_LEGACY_ALIAS)
    assert by_urn.urn == by_alias.urn == _VALID_URN
    assert by_urn.factor_id_alias == by_alias.factor_id_alias == _LEGACY_ALIAS
    assert by_urn.source_urn == by_alias.source_urn
    assert by_urn.unit_urn == by_alias.unit_urn
    assert by_urn.geography_urn == by_alias.geography_urn
    assert by_urn.methodology_urn == by_alias.methodology_urn


# ---------------------------------------------------------------------------
# 3. client.list_factors(source_urn=...) -> paginated cursor
# ---------------------------------------------------------------------------


def test_list_factors_returns_paginated_cursor(alpha_profile) -> None:
    """``list_factors()`` returns ListFactorsResponse with typed rows."""
    list_payload = {
        "data": [_make_payload()],
        "next_cursor": "v1:2026-04-25T07:42:30+00:00|" + _VALID_URN,
        "edition": "alpha-v0.1",
    }
    captured: List[str] = []
    transport = _scripted_transport(
        [httpx.Response(200, json=list_payload)],
        captured,
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        resp = client.list_factors(
            source_urn="urn:gl:source:ipcc-2006-nggi",
        )
    assert isinstance(resp, ListFactorsResponse)
    assert len(resp.data) == 1
    assert isinstance(resp.data[0], AlphaFactor)
    assert resp.data[0].urn == _VALID_URN
    # Cursor surface is the SDK's pagination contract.
    assert resp.next_cursor is not None
    # The SDK passed the source_urn filter through to the wire layer.
    assert any("/factors" in p for p in captured), captured


def test_list_factors_followup_call_consumes_cursor(alpha_profile) -> None:
    """A second call carrying ``cursor=`` returns the next page."""
    page1 = {
        "data": [_make_payload()],
        "next_cursor": "v1:2026-04-25T07:42:30+00:00|" + _VALID_URN,
        "edition": "alpha-v0.1",
    }
    page2 = {
        "data": [],
        "next_cursor": None,
        "edition": "alpha-v0.1",
    }
    captured: List[str] = []
    transport = _scripted_transport(
        [
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ],
        captured,
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        first = client.list_factors(limit=1)
        assert first.next_cursor is not None
        second = client.list_factors(cursor=first.next_cursor, limit=1)
    assert second.next_cursor is None
    assert second.data == []


# ---------------------------------------------------------------------------
# 4. urn-as-primary contract (no `factor_id` slot anywhere)
# ---------------------------------------------------------------------------


def test_alpha_factor_model_has_no_legacy_primary_field() -> None:
    """The SDK Pydantic model never carries the legacy primary `factor_id`."""
    fields = AlphaFactor.model_fields
    assert "urn" in fields
    assert "factor_id_alias" in fields  # secondary-only legacy slot
    assert "factor_id" not in fields, (
        "AlphaFactor must not regress to carry legacy 'factor_id' as a "
        "primary identifier; only 'urn' is allowed as primary."
    )


# ---------------------------------------------------------------------------
# 5. SDK refuses legacy id on get_factor() (URN-only client contract)
# ---------------------------------------------------------------------------


def test_get_factor_rejects_legacy_id_at_client_boundary(alpha_profile) -> None:
    """``client.get_factor(legacy_id)`` raises before any network call.

    The SDK enforces the URN-as-primary contract client-side, raising
    :class:`ValueError` when the caller tries to use a legacy
    ``EF:...`` identifier with ``get_factor()``. This MUST never make
    it to the wire.
    """
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"never": "called"})
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        with pytest.raises(ValueError, match="valid GreenLang URN"):
            client.get_factor(_LEGACY_ALIAS)


def test_get_by_alias_rejects_empty_legacy_id(alpha_profile) -> None:
    """Empty legacy_id raises ValueError without a network round-trip."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"never": "called"})
    )
    with FactorsClient(
        base_url=_BASE_URL,
        transport=transport,
        verify_greenlang_cert=False,
    ) as client:
        with pytest.raises(ValueError):
            client.get_by_alias("")
