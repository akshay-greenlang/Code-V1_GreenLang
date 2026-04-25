# -*- coding: utf-8 -*-
"""v0.1 Alpha — SDK public surface tests.

Validates Wave B / TaskCreate #18 + #16 (CTO doc §19.1):

  * The SDK's PUBLIC alpha surface is the FIVE read-only GETs:

      - ``client.health()``           -> GET /v1/healthz
      - ``client.list_factors(...)``  -> GET /v1/factors
      - ``client.get_factor(urn)``    -> GET /v1/factors/{urn}
      - ``client.list_sources()``     -> GET /v1/sources
      - ``client.list_packs(...)``    -> GET /v1/packs

  * The forward-development surface (``resolve``, ``explain``, ``batch``,
    ``edition`` pinning, signed-receipt verification, ``search``) STAYS in
    the codebase but raises :class:`ProfileGatedError` under the
    ``alpha-v0.1`` release profile.

  * URNs are the canonical primary id. ``get_factor("urn:gl:factor:...")``
    parses into a typed :class:`AlphaFactor`. Passing a legacy ``"EF:..."``
    string raises ``ValueError`` BEFORE any network call is made.

  * The ``beta-v0.5`` profile re-enables the forward-dev surface.
"""
from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import quote

import httpx
import pytest
import respx

from greenlang.factors.sdk.python import (
    AlphaFactor,
    AlphaPack,
    AlphaSource,
    FactorsClient,
    HealthResponse,
    ListFactorsResponse,
    ProfileGatedError,
    ResolutionRequest,
    __version__,
)


BASE_URL = "https://factors.test"
API_PREFIX = "/api/v1"

VALID_URN = (
    "urn:gl:factor:ipcc-ar6:stationary-combustion:"
    "natural-gas-residential:v1"
)
LEGACY_EF_ID = "EF:NG:RES:001"

# Sample wire payloads — match the canonical v0.1 alpha shape.
_HEALTH_PAYLOAD: Dict[str, Any] = {
    "status": "ok",
    "service": "greenlang-factors",
    "release_profile": "alpha-v0.1",
    "schema_id": "factor.v0.1",
    "edition": "2026.Q2",
    "git_commit": "0123456789abcdef0123456789abcdef01234567",
}

_FACTOR_PAYLOAD: Dict[str, Any] = {
    "urn": VALID_URN,
    "factor_id_alias": "EF:NG:RES:001",
    "source_urn": "urn:gl:source:ipcc-ar6",
    "factor_pack_urn": "urn:gl:pack:ipcc-ar6:stationary-combustion:v1",
    "name": "Natural gas, residential combustion",
    "description": "AR6 100-yr GWP for residential NG combustion.",
    "category": "scope1",
    "value": 53.06,
    "unit_urn": "urn:gl:unit:kgco2e/mmbtu",
    "gwp_basis": "ar6",
    "gwp_horizon": 100,
    "geography_urn": "urn:gl:geo:country:us",
    "vintage_start": "2024-01-01",
    "vintage_end": "2024-12-31",
    "resolution": "annual",
    "methodology_urn": "urn:gl:methodology:ipcc-2006-tier-1",
    "boundary": "stationary-combustion",
    "licence": "CC-BY-4.0",
    "citations": [
        {"type": "url", "value": "https://www.ipcc.ch/report/ar6/wg1/"},
    ],
    "published_at": "2026-01-15T00:00:00+00:00",
    "extraction": {
        "source_url": "https://www.ipcc.ch/report.pdf",
        "raw_artifact_sha256": "a" * 64,
        "parser_id": "ipcc-pdf-v3",
        "parser_version": "3.2.1",
        "parser_commit": "f" * 40,
    },
    "review": {
        "review_status": "approved",
        "reviewer": "scientist@greenlang.io",
        "reviewed_at": "2026-01-10T00:00:00+00:00",
        "approved_by": "cto@greenlang.io",
        "approved_at": "2026-01-12T00:00:00+00:00",
    },
}

_LIST_FACTORS_PAYLOAD: Dict[str, Any] = {
    "data": [_FACTOR_PAYLOAD],
    "next_cursor": "opaque-cursor-page-2",
}

_SOURCES_PAYLOAD: Dict[str, Any] = {
    "data": [
        {
            "urn": "urn:gl:source:ipcc-ar6",
            "name": "IPCC AR6",
            "organization": "IPCC",
            "year": 2023,
        },
        {
            "urn": "urn:gl:source:epa-egrid",
            "name": "EPA eGRID",
            "organization": "US EPA",
            "year": 2024,
        },
    ]
}

_PACKS_PAYLOAD: Dict[str, Any] = {
    "data": [
        {
            "urn": "urn:gl:pack:ipcc-ar6:stationary-combustion:v1",
            "source_urn": "urn:gl:source:ipcc-ar6",
            "version": "v1",
            "name": "IPCC AR6 - Stationary Combustion",
        }
    ]
}


# ---------------------------------------------------------------------------
# Profile-gating fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def alpha_profile(monkeypatch):
    """Activate the ``alpha-v0.1`` release profile for the test."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.delenv("GL_ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    yield


@pytest.fixture()
def beta_profile(monkeypatch):
    """Activate the ``beta-v0.5`` release profile (forward-dev unlocked)."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "beta-v0.5")
    monkeypatch.delenv("GL_ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    yield


# ---------------------------------------------------------------------------
# Profile gate — beta-only methods raise under alpha-v0.1
# ---------------------------------------------------------------------------


def test_alpha_profile_gates_resolve(alpha_profile):
    """``resolve()`` raises :class:`ProfileGatedError` under alpha-v0.1."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"never": "called"})
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        req = ResolutionRequest(
            activity="diesel", method_profile="corporate_scope1"
        )
        with pytest.raises(ProfileGatedError, match=r"resolve"):
            client.resolve(req)


def test_alpha_profile_gates_resolve_explain(alpha_profile):
    """``resolve_explain()`` is gated under alpha-v0.1."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        with pytest.raises(ProfileGatedError):
            client.resolve_explain("EF:foo")


def test_alpha_profile_gates_search(alpha_profile):
    """``search()`` is gated under alpha-v0.1."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"factors": []})
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        with pytest.raises(ProfileGatedError):
            client.search("diesel")


def test_alpha_profile_gates_batch_resolve(alpha_profile):
    """``resolve_batch()`` and friends are gated under alpha-v0.1."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        with pytest.raises(ProfileGatedError):
            client.resolve_batch([])
        with pytest.raises(ProfileGatedError):
            client.get_batch_job("job-1")


def test_alpha_profile_gates_edition_pinning(alpha_profile):
    """``pin_edition()`` is gated — alpha is read-only at HEAD edition."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        with pytest.raises(ProfileGatedError):
            client.pin_edition("2027.Q1")


def test_alpha_profile_gates_verify_receipt(alpha_profile):
    """Signed-receipt verification is gated under alpha-v0.1."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        with pytest.raises(ProfileGatedError):
            client.verify_receipt({"signed_receipt": {}}, secret="x")


# ---------------------------------------------------------------------------
# Beta profile — forward-dev surface is unlocked
# ---------------------------------------------------------------------------


def test_beta_profile_unlocks_resolve(beta_profile):
    """Under ``beta-v0.5`` the forward-dev resolve does NOT raise the gate.

    We mock the HTTP layer with a respx route returning a minimal valid
    payload; the test only asserts the gate path is opened, not the full
    cascade behaviour (covered in the v1.2 envelope tests).
    """
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "chosen_factor_id": "ef:diesel",
                "factor_id": "ef:diesel",
                "edition_id": "2026.Q1",
            },
        )
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        req = ResolutionRequest(
            activity="diesel", method_profile="corporate_scope1"
        )
        result = client.resolve(req)
        assert result.factor_id == "ef:diesel"


# ---------------------------------------------------------------------------
# Five always-on alpha-allowed GETs
# ---------------------------------------------------------------------------


@respx.mock(assert_all_called=False, base_url=BASE_URL)
def test_health_parses_response_into_typed_model(respx_mock, alpha_profile):
    """``client.health()`` returns a typed :class:`HealthResponse`."""
    route = respx_mock.get(f"{API_PREFIX}/healthz").mock(
        return_value=httpx.Response(200, json=_HEALTH_PAYLOAD)
    )
    with FactorsClient(base_url=BASE_URL, verify_greenlang_cert=False) as c:
        result = c.health()
    assert route.called
    assert isinstance(result, HealthResponse)
    assert result.status == "ok"
    assert result.release_profile == "alpha-v0.1"
    assert result.schema_id == "factor.v0.1"
    assert result.edition == "2026.Q2"


@respx.mock(assert_all_called=False, base_url=BASE_URL)
def test_get_factor_returns_typed_alpha_factor(respx_mock, alpha_profile):
    """``client.get_factor(urn)`` returns a typed :class:`AlphaFactor`.

    The URN is URL-encoded on the wire (colons travel as ``%3A``).
    """
    encoded = quote(VALID_URN, safe="")
    route = respx_mock.get(f"{API_PREFIX}/factors/{encoded}").mock(
        return_value=httpx.Response(200, json=_FACTOR_PAYLOAD)
    )
    with FactorsClient(base_url=BASE_URL, verify_greenlang_cert=False) as c:
        factor = c.get_factor(VALID_URN)
    assert route.called
    assert isinstance(factor, AlphaFactor)
    # ``urn`` is the primary id — assert the model surfaces it as such.
    assert factor.urn == VALID_URN
    assert factor.category == "scope1"
    assert factor.gwp_basis == "ar6"
    assert factor.value == pytest.approx(53.06)
    assert factor.review.review_status == "approved"
    assert factor.extraction.parser_id == "ipcc-pdf-v3"
    # Legacy alias is preserved as a non-canonical hint.
    assert factor.factor_id_alias == "EF:NG:RES:001"


def test_get_factor_rejects_non_urn_input(alpha_profile):
    """Passing a legacy ``EF:`` id raises ``ValueError`` PRE-network."""
    transport = httpx.MockTransport(
        lambda request: pytest.fail(
            "client.get_factor() should not hit the network on a malformed urn"
        )
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as c:
        with pytest.raises(ValueError, match=r"valid GreenLang URN"):
            c.get_factor(LEGACY_EF_ID)


@respx.mock(assert_all_called=False, base_url=BASE_URL)
def test_list_factors_issues_get_with_query_params(respx_mock, alpha_profile):
    """``list_factors(...)`` issues GET /v1/factors with the right params.

    Asserts:
      * URNs are validated client-side (good ones pass, bad ones raise).
      * cursor / limit / category / vintage filters are forwarded as
        querystring parameters.
      * The response is parsed into :class:`ListFactorsResponse`.
    """
    route = respx_mock.get(f"{API_PREFIX}/factors").mock(
        return_value=httpx.Response(200, json=_LIST_FACTORS_PAYLOAD)
    )
    with FactorsClient(base_url=BASE_URL, verify_greenlang_cert=False) as c:
        result = c.list_factors(
            geography_urn="urn:gl:geo:country:us",
            source_urn="urn:gl:source:ipcc-ar6",
            pack_urn="urn:gl:pack:ipcc-ar6:stationary-combustion:v1",
            category="scope1",
            vintage_start_after="2024-01-01",
            vintage_end_before="2025-12-31",
            cursor="opaque-cursor-page-1",
            limit=25,
        )
    assert route.called
    sent = route.calls.last.request
    qp = dict(sent.url.params)
    assert qp["geography_urn"] == "urn:gl:geo:country:us"
    assert qp["source_urn"] == "urn:gl:source:ipcc-ar6"
    assert (
        qp["pack_urn"] == "urn:gl:pack:ipcc-ar6:stationary-combustion:v1"
    )
    assert qp["category"] == "scope1"
    assert qp["vintage_start_after"] == "2024-01-01"
    assert qp["vintage_end_before"] == "2025-12-31"
    assert qp["cursor"] == "opaque-cursor-page-1"
    assert qp["limit"] == "25"

    assert isinstance(result, ListFactorsResponse)
    assert len(result.data) == 1
    assert isinstance(result.data[0], AlphaFactor)
    assert result.data[0].urn == VALID_URN
    assert result.next_cursor == "opaque-cursor-page-2"


def test_list_factors_rejects_malformed_urn_filter(alpha_profile):
    """A malformed URN filter raises ``ValueError`` BEFORE the network call."""
    transport = httpx.MockTransport(
        lambda request: pytest.fail(
            "list_factors() should never hit the network on a bad urn"
        )
    )
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as c:
        with pytest.raises(ValueError):
            c.list_factors(geography_urn="not-a-urn-at-all")


@respx.mock(assert_all_called=False, base_url=BASE_URL)
def test_list_sources_returns_typed_alpha_sources(respx_mock, alpha_profile):
    """``client.list_sources()`` returns a list of :class:`AlphaSource`."""
    route = respx_mock.get(f"{API_PREFIX}/sources").mock(
        return_value=httpx.Response(200, json=_SOURCES_PAYLOAD)
    )
    with FactorsClient(base_url=BASE_URL, verify_greenlang_cert=False) as c:
        sources = c.list_sources()
    assert route.called
    assert len(sources) == 2
    assert all(isinstance(s, AlphaSource) for s in sources)
    assert sources[0].urn == "urn:gl:source:ipcc-ar6"
    assert sources[0].name == "IPCC AR6"


@respx.mock(assert_all_called=False, base_url=BASE_URL)
def test_list_packs_returns_typed_alpha_packs(respx_mock, alpha_profile):
    """``client.list_packs(source_urn=...)`` parses :class:`AlphaPack` rows."""
    route = respx_mock.get(f"{API_PREFIX}/packs").mock(
        return_value=httpx.Response(200, json=_PACKS_PAYLOAD)
    )
    with FactorsClient(base_url=BASE_URL, verify_greenlang_cert=False) as c:
        packs = c.list_packs(source_urn="urn:gl:source:ipcc-ar6")
    assert route.called
    sent = route.calls.last.request
    assert dict(sent.url.params)["source_urn"] == "urn:gl:source:ipcc-ar6"
    assert len(packs) == 1
    assert isinstance(packs[0], AlphaPack)
    assert (
        packs[0].urn
        == "urn:gl:pack:ipcc-ar6:stationary-combustion:v1"
    )
    assert packs[0].source_urn == "urn:gl:source:ipcc-ar6"


# ---------------------------------------------------------------------------
# User-Agent / SDK identity
# ---------------------------------------------------------------------------


@respx.mock(assert_all_called=False, base_url=BASE_URL)
def test_alpha_methods_emit_alpha_user_agent(respx_mock, alpha_profile):
    """The five alpha methods send the v0.1 alpha User-Agent header."""
    route = respx_mock.get(f"{API_PREFIX}/healthz").mock(
        return_value=httpx.Response(200, json=_HEALTH_PAYLOAD)
    )
    with FactorsClient(base_url=BASE_URL, verify_greenlang_cert=False) as c:
        c.health()
    sent = route.calls.last.request
    assert (
        sent.headers["user-agent"]
        == "greenlang-factors-sdk/0.1.0 (python)"
    )


# ---------------------------------------------------------------------------
# Version sanity
# ---------------------------------------------------------------------------


def test_sdk_version_is_0_1_0() -> None:
    """SDK has been renumbered to v0.1.0 (CTO doc §19.1 Wave A)."""
    assert __version__ == "0.1.0"
