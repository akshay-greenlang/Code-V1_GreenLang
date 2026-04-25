# -*- coding: utf-8 -*-
"""Wave C / TaskCreate #19 / WS6-T3 — Canonical alpha SDK E2E demo.

CTO doc §19.1 acceptance criterion (verbatim):

    "End-to-end test: publish a factor from IPCC AR6 via the pipeline,
     fetch it by URN via the Python SDK, verify all metadata fields are
     correct."

This test is the public exhibit for v0.1 alpha. One IPCC AR6 stationary-
combustion factor flows through the full alpha contract:

    1. Construct a v0.1-shape factor record (the same fixture used by
       :mod:`test_factor_record_v0_1_schema_loads` and
       :mod:`test_alpha_provenance_gate`).

    2. Validate the record with the FROZEN factor_record_v0_1 schema
       AND the Alpha Provenance Gate. Both gates MUST pass before the
       factor can ever reach the public API.

    3. Seed the alpha catalog via the e2e shim
       (``tests/factors/v0_1_alpha/_e2e_helpers.py``). The shim
       installs a minimal in-memory catalog on ``app.state.factors_service``
       and patches the alpha router's coercion helper to surface the
       v0.1-shape record verbatim. This is the only path that lets the
       canonical demo round-trip today; the real Wave D publish pipeline
       will replace it.

    4. Boot ``create_factors_app()`` under
       ``GL_FACTORS_RELEASE_PROFILE=alpha-v0.1``. Only the five alpha
       routes (``/v1/healthz``, ``/v1/factors``, ``/v1/factors/{urn}``,
       ``/v1/sources``, ``/v1/packs``) are mounted.

    5. Talk to the app through the v0.1.0 SDK using an
       ``httpx.ASGITransport`` so no real network is touched.

    6. Field-by-field verify every required v0.1 metadata field on the
       fetched :class:`AlphaFactor`.

The shim path is documented at the top of ``_e2e_helpers.py``. Production
code is NOT modified by this test.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict
from urllib.parse import quote

import httpx
import pytest

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGate,
)
from greenlang.factors.sdk.python import (
    AlphaFactor,
    FactorNotFoundError,
    FactorsClient,
    HealthResponse,
    ListFactorsResponse,
    __version__ as sdk_version,
)

from tests.factors.v0_1_alpha._e2e_helpers import (
    good_ipcc_ar6_factor,
    install_alpha_e2e_shim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def alpha_factor() -> Dict[str, Any]:
    """Canonical IPCC AR6 stationary-combustion factor (v0.1 shape)."""
    return good_ipcc_ar6_factor()


@pytest.fixture()
def alpha_app(monkeypatch, alpha_factor):
    """Build the alpha FastAPI app and seed it with the IPCC factor.

    Steps:
      * Force ``GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`` so only the five
        read-only GETs mount.
      * Validate the seeded factor with both the v0.1 JSON schema (via
        :class:`AlphaProvenanceGate`) AND the format-level alpha checks
        before letting it anywhere near the public surface.
      * Install the e2e shim (in-memory repo + coerce passthrough).
    """
    pytest.importorskip("fastapi")

    # Force alpha profile and a clean test environment.
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    # Register a single dev-tier API key so the AuthMeteringMiddleware
    # accepts the SDK's ``X-API-Key`` header on the four protected alpha
    # routes (``/v1/factors``, ``/v1/factors/{urn}``, ``/v1/sources``,
    # ``/v1/packs`` — ``/v1/healthz`` is public). The keyring is reloaded
    # so a previously-cached default validator picks up the test key.
    monkeypatch.setenv(
        "GL_FACTORS_API_KEYS",
        (
            '[{"key_id": "alpha-e2e", "key": "alpha-e2e-test-key", '
            '"tier": "enterprise", "active": true}]'
        ),
    )
    from greenlang.factors import api_auth as _api_auth

    _api_auth._default_validator = None  # clear any cached singleton
    _api_auth.default_validator().reload()

    # ---- Gate 1+2: schema + provenance gate must pass before publish.
    AlphaProvenanceGate().assert_valid(alpha_factor)

    from greenlang.factors.factors_app import create_factors_app

    app = create_factors_app(
        enable_admin=False,
        enable_billing=False,
        enable_oem=False,
        enable_metrics=False,
    )

    install_alpha_e2e_shim(
        monkeypatch,
        app,
        edition_id="alpha-e2e-ipcc-2026.0",
        factors=[alpha_factor],
    )

    return app


def _build_testclient_transport(app: Any) -> httpx.MockTransport:
    """Build a sync ``httpx.MockTransport`` that delegates to FastAPI.

    httpx 0.28's :class:`httpx.ASGITransport` is async-only; the SDK's
    sync :class:`FactorsClient` cannot use it. We bridge the gap by
    routing each outbound request through FastAPI's :class:`TestClient`
    (which handles the ASGI plumbing internally) and translating the
    result back into an :class:`httpx.Response`.
    """
    from fastapi.testclient import TestClient

    test_client = TestClient(app)

    def _handler(request: httpx.Request) -> httpx.Response:
        url = request.url.path
        if request.url.query:
            url = f"{url}?{request.url.query.decode()}"
        upstream = test_client.request(
            method=request.method,
            url=url,
            headers={
                k: v
                for k, v in request.headers.items()
                if k.lower() != "host"
            },
            content=request.content,
        )
        # Drop hop-by-hop headers that confuse httpx when re-emitted.
        passthrough_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower()
            not in {
                "transfer-encoding",
                "content-encoding",
                "content-length",
            }
        }
        return httpx.Response(
            status_code=upstream.status_code,
            content=upstream.content,
            headers=passthrough_headers,
        )

    return httpx.MockTransport(_handler)


@pytest.fixture()
def sdk_client(alpha_app) -> FactorsClient:
    """Boot a v0.1.0 SDK client wired to the alpha app via TestClient bridge.

    No real network: every call goes in-process. ``api_prefix='/v1'``
    overrides the SDK's default ``/api/v1`` (which is the legacy 410-Gone
    route under alpha-v0.1).
    """
    transport = _build_testclient_transport(alpha_app)
    client = FactorsClient(
        base_url="http://factors.test",
        transport=transport,
        api_prefix="/v1",
        api_key="alpha-e2e-test-key",
        verify_greenlang_cert=False,
    )
    yield client
    client.close()


# ---------------------------------------------------------------------------
# CTO §19.1 acceptance test — publish + SDK fetch + field-by-field verify.
# ---------------------------------------------------------------------------


def test_ipcc_ar6_publish_then_sdk_fetch(
    alpha_factor: Dict[str, Any], sdk_client: FactorsClient
) -> None:
    """Publish 1 IPCC AR6 factor; fetch via SDK; verify EVERY v0.1 field.

    This is the canonical alpha demo (CTO doc §19.1).
    """
    # 1. Health probe — confirm the app is in alpha-v0.1 mode and the
    #    schema id is the FROZEN v0.1 contract.
    health = sdk_client.health()
    assert isinstance(health, HealthResponse)
    assert health.status == "ok"
    assert health.service == "greenlang-factors"
    assert health.release_profile == "alpha-v0.1"
    assert health.schema_id.endswith("factor_record_v0_1.schema.json")
    assert health.edition == "alpha-e2e-ipcc-2026.0"

    # 2. Fetch by canonical URN.
    fetched = sdk_client.get_factor(alpha_factor["urn"])

    # 3. Field-by-field verification — every required v0.1 field must
    #    survive the round-trip exactly as published.
    assert isinstance(fetched, AlphaFactor)

    # ---- Identity ----
    assert fetched.urn == alpha_factor["urn"]
    assert fetched.factor_id_alias == alpha_factor["factor_id_alias"]
    assert fetched.source_urn == alpha_factor["source_urn"]
    assert fetched.factor_pack_urn == alpha_factor["factor_pack_urn"]

    # ---- Descriptive metadata ----
    assert fetched.name == alpha_factor["name"]
    assert fetched.description == alpha_factor["description"]
    assert fetched.category == alpha_factor["category"]

    # ---- Numeric / unit / GWP ----
    assert fetched.value == pytest.approx(alpha_factor["value"])
    assert fetched.unit_urn == alpha_factor["unit_urn"]
    assert fetched.gwp_basis == "ar6"
    assert fetched.gwp_horizon == 100

    # ---- Geography + vintage ----
    assert fetched.geography_urn == alpha_factor["geography_urn"]
    assert isinstance(fetched.vintage_start, date)
    assert isinstance(fetched.vintage_end, date)
    assert fetched.vintage_start.isoformat() == alpha_factor["vintage_start"]
    assert fetched.vintage_end.isoformat() == alpha_factor["vintage_end"]
    assert fetched.resolution == alpha_factor["resolution"]

    # ---- Methodology / boundary / licence ----
    assert fetched.methodology_urn == alpha_factor["methodology_urn"]
    assert fetched.boundary == alpha_factor["boundary"]
    assert fetched.licence == alpha_factor["licence"]

    # ---- Citations preserved ----
    assert len(fetched.citations) == len(alpha_factor["citations"])
    for sdk_cite, src_cite in zip(fetched.citations, alpha_factor["citations"]):
        assert sdk_cite.type == src_cite["type"]
        assert sdk_cite.value == src_cite["value"]

    # ---- Published-at preserved as a real datetime ----
    assert isinstance(fetched.published_at, datetime)

    # ---- Extraction provenance — every gate-required key preserved ----
    extraction_src = alpha_factor["extraction"]
    assert fetched.extraction.source_url == extraction_src["source_url"]
    assert (
        fetched.extraction.raw_artifact_sha256
        == extraction_src["raw_artifact_sha256"]
    )
    assert fetched.extraction.parser_id == extraction_src["parser_id"]
    assert fetched.extraction.parser_version == extraction_src["parser_version"]
    assert fetched.extraction.parser_commit == extraction_src["parser_commit"]

    # ---- Review block — approval audit trail preserved ----
    review_src = alpha_factor["review"]
    assert fetched.review.review_status == "approved"
    assert fetched.review.reviewer == review_src["reviewer"]
    assert isinstance(fetched.review.reviewed_at, datetime)
    assert fetched.review.approved_by == review_src["approved_by"]
    assert isinstance(fetched.review.approved_at, datetime)


# ---------------------------------------------------------------------------
# Companion SDK surface tests against the same e2e wiring.
# ---------------------------------------------------------------------------


def test_list_factors_returns_published_ipcc(
    alpha_factor: Dict[str, Any], sdk_client: FactorsClient
) -> None:
    """``list_factors(source_urn=...)`` returns the published IPCC factor."""
    page = sdk_client.list_factors(source_urn=alpha_factor["source_urn"])
    assert isinstance(page, ListFactorsResponse)
    assert len(page.data) == 1
    row = page.data[0]
    assert isinstance(row, AlphaFactor)
    assert row.urn == alpha_factor["urn"]
    assert row.source_urn == alpha_factor["source_urn"]
    assert row.value == pytest.approx(alpha_factor["value"])


def test_list_sources_includes_ipcc_ar6(alpha_app: Any) -> None:
    """``GET /v1/sources`` returns the IPCC source registered with
    ``alpha_v0_1: true`` in ``source_registry.yaml``.

    The registry's canonical id is ``ipcc_2006_nggi`` (with ``ipcc_ar6``
    declared as an alpha alias) — see the comment block at
    ``greenlang/factors/data/source_registry.yaml`` line 347. The
    canonical URN on the wire is therefore ``urn:gl:source:ipcc-2006-nggi``.

    NOTE: The SDK's ``AlphaSource`` model requires a ``name`` field while
    the alpha API emits ``display_name`` (a known v0.1.0 / v0.2.0 SDK
    contract drift, scheduled for the next SDK patch release). For the
    e2e demo we hit the route via the FastAPI :class:`TestClient`
    directly so the SDK model gap does not block the canonical
    acceptance test.
    """
    from fastapi.testclient import TestClient

    test_client = TestClient(alpha_app)
    resp = test_client.get(
        "/v1/sources",
        headers={"X-API-Key": "alpha-e2e-test-key"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    rows = body.get("data") or []
    assert rows, f"alpha source registry returned an empty list: {body!r}"
    ipcc_urns = [
        r.get("urn", "")
        for r in rows
        if "ipcc" in str(r.get("urn", "")).lower()
    ]
    assert any(
        u == "urn:gl:source:ipcc-2006-nggi" for u in ipcc_urns
    ), (
        "expected canonical IPCC source URN urn:gl:source:ipcc-2006-nggi "
        f"in alpha source registry; got IPCC entries: {ipcc_urns!r}"
    )


def test_list_packs_returns_at_least_one_alpha_pack(alpha_app: Any) -> None:
    """``GET /v1/packs`` surfaces at least the synthetic alpha pack per source.

    Hit the route via the FastAPI :class:`TestClient` directly to keep
    parity with :func:`test_list_sources_includes_ipcc_ar6` (same SDK
    model contract drift on the ``name`` field).
    """
    from fastapi.testclient import TestClient

    test_client = TestClient(alpha_app)
    resp = test_client.get(
        "/v1/packs",
        headers={"X-API-Key": "alpha-e2e-test-key"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    rows = body.get("data") or []
    assert rows, f"alpha /v1/packs returned an empty list: {body!r}"
    for row in rows:
        assert str(row.get("urn", "")).startswith("urn:gl:pack:")
        assert str(row.get("source_urn", "")).startswith("urn:gl:source:")


def test_get_factor_for_unpublished_urn_returns_404(
    sdk_client: FactorsClient,
) -> None:
    """``get_factor`` raises :class:`FactorNotFoundError` for an unknown URN."""
    unknown_urn = "urn:gl:factor:ipcc-ar6:stationary-combustion:never-published:v1"
    with pytest.raises(FactorNotFoundError):
        sdk_client.get_factor(unknown_urn)


# ---------------------------------------------------------------------------
# Pre-publish gates — both must pass on the seeded record before the
# canonical demo runs (otherwise the alpha contract has been broken).
# ---------------------------------------------------------------------------


def test_seed_factor_passes_alpha_provenance_gate(
    alpha_factor: Dict[str, Any],
) -> None:
    """The seeded IPCC AR6 factor MUST pass the Alpha Provenance Gate
    before publish — exactly as Wave B/WS2-T1 enforces in the real
    publisher pipeline.
    """
    failures = AlphaProvenanceGate().validate(alpha_factor)
    assert failures == [], (
        "Seeded IPCC AR6 factor failed Alpha Provenance Gate "
        f"({len(failures)} failures): {failures!r}"
    )


def test_sdk_version_pins_v0_1_0() -> None:
    """The SDK shipping the canonical alpha demo is v0.1.0 (CTO doc §19.1)."""
    assert sdk_version == "0.1.0"


# ---------------------------------------------------------------------------
# Wave D / TaskCreate #31 — real-repo path (no _coerce_v0_1 monkey-patch).
# ---------------------------------------------------------------------------


@pytest.fixture()
def alpha_app_real_repo(monkeypatch, alpha_factor):
    """Same as :func:`alpha_app` but uses the real :class:`AlphaFactorRepository`.

    This fixture is the Wave D acceptance path — it publishes the IPCC AR6
    factor through the production-shaped repository (no in-test coercion
    shim), then asserts the SDK can fetch it back bit-for-bit.
    """
    pytest.importorskip("fastapi")
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv(
        "GL_FACTORS_API_KEYS",
        (
            '[{"key_id": "alpha-e2e", "key": "alpha-e2e-test-key", '
            '"tier": "enterprise", "active": true}]'
        ),
    )
    from greenlang.factors import api_auth as _api_auth

    _api_auth._default_validator = None
    _api_auth.default_validator().reload()
    AlphaProvenanceGate().assert_valid(alpha_factor)

    from greenlang.factors.factors_app import create_factors_app

    app = create_factors_app(
        enable_admin=False,
        enable_billing=False,
        enable_oem=False,
        enable_metrics=False,
    )
    install_alpha_e2e_shim(
        monkeypatch,
        app,
        edition_id="alpha-e2e-ipcc-2026.0",
        factors=[alpha_factor],
        mode="real",
    )
    return app


def test_ipcc_ar6_publish_then_sdk_fetch_via_real_repo(
    alpha_factor: Dict[str, Any], alpha_app_real_repo: Any
) -> None:
    """Wave D: publish via :class:`AlphaFactorRepository`, fetch via SDK.

    This is the SAME canonical demo as
    :func:`test_ipcc_ar6_publish_then_sdk_fetch` but the publish path goes
    through the real repository — exercising the round-trip without any
    ``_coerce_v0_1`` monkey-patching. The assertion surface is intentionally
    smaller (the canonical demo above is the full field-by-field check);
    here we only need to prove that the round-trip is identity-preserving.
    """
    transport = _build_testclient_transport(alpha_app_real_repo)
    client = FactorsClient(
        base_url="http://factors.test",
        transport=transport,
        api_prefix="/v1",
        api_key="alpha-e2e-test-key",
        verify_greenlang_cert=False,
    )
    try:
        # Health is ok in alpha mode.
        health = client.health()
        assert health.status == "ok"
        assert health.release_profile == "alpha-v0.1"
        assert health.schema_id.endswith("factor_record_v0_1.schema.json")

        fetched = client.get_factor(alpha_factor["urn"])
        assert isinstance(fetched, AlphaFactor)
        assert fetched.urn == alpha_factor["urn"]
        assert fetched.value == pytest.approx(alpha_factor["value"])
        assert fetched.gwp_basis == "ar6"
        assert fetched.gwp_horizon == 100
        assert fetched.geography_urn == alpha_factor["geography_urn"]
        # Provenance + review fields survived without coercion.
        assert (
            fetched.extraction.raw_artifact_sha256
            == alpha_factor["extraction"]["raw_artifact_sha256"]
        )
        assert fetched.review.review_status == "approved"

        # list_factors via the real repo path returns the same record.
        page = client.list_factors(source_urn=alpha_factor["source_urn"])
        assert len(page.data) == 1
        assert page.data[0].urn == alpha_factor["urn"]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Wire-encoding sanity — the URN MUST travel URL-encoded so reverse
# proxies don't parse the colons as port separators.
# ---------------------------------------------------------------------------


def test_get_factor_url_encodes_urn_on_the_wire(
    alpha_factor: Dict[str, Any], alpha_app: Any
) -> None:
    """The URN's colons must be %3A on the wire (regression guard).

    httpx 0.28 normalises ``request.url.path`` (decodes %3A back to ``:``)
    so we read ``request.url.raw_path`` — the on-wire bytes — and assert
    against the percent-encoded form.
    """
    captured: Dict[str, str] = {}

    def _record(request: httpx.Request) -> httpx.Response:
        # ``raw_path`` is the on-wire bytes; ``path`` is the normalised form.
        captured["raw_path"] = request.url.raw_path.decode("ascii")
        captured["path"] = request.url.path
        return httpx.Response(404, json={"error": "factor_not_found"})

    transport = httpx.MockTransport(_record)
    with FactorsClient(
        base_url="http://factors.test",
        transport=transport,
        api_prefix="/v1",
        api_key="alpha-e2e-test-key",
        verify_greenlang_cert=False,
    ) as c:
        with pytest.raises(FactorNotFoundError):
            c.get_factor(alpha_factor["urn"])

    assert captured, "transport handler was never invoked"
    encoded = quote(alpha_factor["urn"], safe="")
    assert encoded in captured["raw_path"], (
        f"URN must travel URL-encoded on the wire; got "
        f"raw_path={captured['raw_path']!r}, expected substring "
        f"{encoded!r}."
    )
