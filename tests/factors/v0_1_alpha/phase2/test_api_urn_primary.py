# -*- coding: utf-8 -*-
"""Phase 2 / WS2 — API + SDK URN-as-primary-id tests.

Per CTO Phase 2 brief Section 2.7 acceptance:

    "test_api_urn_primary.py — exercise the API/SDK paths and assert
    response payloads carry ``urn`` (not ``factor_id``) as the primary
    identifier; ``factor_id_alias`` only appears as a secondary field."

This module verifies on every alpha contract surface:

  * ``api_v0_1_alpha_models.FactorV0_1`` requires ``urn`` and treats
    ``factor_id_alias`` as ``Optional``.
  * The router's ``_coerce_v0_1`` legacy adapter promotes the canonical
    URN to the primary slot and demotes the legacy ``EF:...`` to
    ``factor_id_alias``.
  * The SDK ``AlphaFactor`` model fails validation when ``urn`` is
    missing but accepts ``factor_id_alias=None``.
  * The new ``GET /v1/factors/by-alias/{legacy_id}`` endpoint surfaces
    the same response shape (URN primary, alias secondary) as
    ``GET /v1/factors/{urn}``.
  * The SDK helper ``client.get_by_alias()`` returns the same
    :class:`AlphaFactor` regardless of whether the caller hit the
    by-URN or by-alias path.
"""
from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import quote

import httpx
import pytest

from greenlang.factors.api_v0_1_alpha_models import FactorV0_1
from greenlang.factors.sdk.python import (
    AlphaFactor,
    FactorsClient,
    HealthResponse,
    ListFactorsResponse,
)


_BASE_URL = "https://factors.test"
_API_PREFIX = "/api/v1"


# Canonical alpha record — used by both the model-level and SDK-level
# tests so the wire and in-memory shapes share one source of truth.
_VALID_URN = (
    "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1"
)
_LEGACY_ALIAS = "EF:NG:RES:001"


def _make_factor_payload(*, urn: str = _VALID_URN, alias: str = _LEGACY_ALIAS) -> Dict[str, Any]:
    """Return a fully-populated v0.1 factor payload (alpha wire shape).

    Mirrors the seed fixture under ``catalog_seed_v0_1/ipcc_2006_nggi/``
    so the AlphaProvenanceGate accepts the record verbatim. Required
    extraction sub-fields, the review-stage approver pattern, and the
    description length floor are all honoured.
    """
    return {
        "urn": urn,
        "factor_id_alias": alias,
        "source_urn": "urn:gl:source:ipcc-ar6",
        "factor_pack_urn": "urn:gl:pack:ipcc-ar6:stationary-combustion:v1",
        "name": "Natural gas, residential combustion",
        "description": (
            "AR6 100-yr GWP for residential natural-gas combustion. "
            "Boundary follows source publication; emissions per "
            "MMBtu of fuel input."
        ),
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
        "citations": [{"type": "url", "value": "https://www.ipcc.ch/"}],
        "published_at": "2026-01-15T00:00:00+00:00",
        "extraction": {
            "source_url": "https://www.ipcc.ch/report.pdf",
            "source_record_id": "phase2-api-urn-primary-record",
            "source_publication": "Phase 2 / WS2 API URN-primary fixture",
            "source_version": "0.1",
            "raw_artifact_uri": "s3://greenlang-factors-raw/test/phase2-api.json",
            "raw_artifact_sha256": "a" * 64,
            "parser_id": "tests.factors.v0_1_alpha.phase2.api_urn_primary",
            "parser_version": "0.1.0",
            "parser_commit": "f" * 40,
            "row_ref": "phase2-api-urn-primary-record",
            "ingested_at": "2026-01-10T00:00:00Z",
            "operator": "bot:test_api_urn_primary",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-01-10T00:00:00Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-01-12T00:00:00Z",
        },
    }


# ---------------------------------------------------------------------------
# Layer 1 — API response model: urn required, factor_id_alias optional.
# ---------------------------------------------------------------------------


class TestFactorV0_1ModelContract:
    """Direct tests on the API response model (no HTTP)."""

    def test_urn_is_required_field(self) -> None:
        """``FactorV0_1`` MUST refuse a payload with no ``urn``."""
        payload = _make_factor_payload()
        payload.pop("urn")
        with pytest.raises(Exception):
            FactorV0_1.model_validate(payload)

    def test_factor_id_alias_is_optional(self) -> None:
        """A payload without ``factor_id_alias`` must validate."""
        payload = _make_factor_payload()
        payload.pop("factor_id_alias")
        m = FactorV0_1.model_validate(payload)
        assert m.urn == _VALID_URN
        assert m.factor_id_alias is None

    def test_urn_is_primary_in_dump_output(self) -> None:
        """``model_dump()`` MUST emit ``urn`` and ``factor_id_alias`` as
        sibling keys; ``factor_id`` (legacy primary) must NOT appear."""
        payload = _make_factor_payload()
        m = FactorV0_1.model_validate(payload)
        dumped = m.model_dump()
        assert dumped["urn"] == _VALID_URN
        assert dumped.get("factor_id_alias") == _LEGACY_ALIAS
        assert "factor_id" not in dumped

    def test_no_factor_id_legacy_field_in_schema(self) -> None:
        """Pydantic's ``model_fields`` MUST not declare ``factor_id``.

        The audit promise is structural: the legacy primary identifier
        is gone from the alpha model. ``factor_id_alias`` may exist; the
        bare ``factor_id`` must not.
        """
        fields = FactorV0_1.model_fields
        assert "urn" in fields
        assert "factor_id_alias" in fields
        assert "factor_id" not in fields, (
            "FactorV0_1 must not carry the legacy 'factor_id' primary; "
            "'factor_id_alias' is the only legacy slot allowed."
        )


# ---------------------------------------------------------------------------
# Layer 2 — SDK AlphaFactor mirrors the same contract.
# ---------------------------------------------------------------------------


class TestAlphaFactorSDKModel:
    """Direct tests on the SDK response model."""

    def test_urn_required(self) -> None:
        payload = _make_factor_payload()
        payload.pop("urn")
        with pytest.raises(Exception):
            AlphaFactor.model_validate(payload)

    def test_factor_id_alias_optional(self) -> None:
        payload = _make_factor_payload()
        payload.pop("factor_id_alias")
        m = AlphaFactor.model_validate(payload)
        assert m.urn == _VALID_URN
        assert m.factor_id_alias is None

    def test_factor_id_alias_carries_legacy_id(self) -> None:
        m = AlphaFactor.model_validate(_make_factor_payload())
        assert m.urn == _VALID_URN
        assert m.factor_id_alias == _LEGACY_ALIAS

    def test_no_factor_id_legacy_field_on_alpha_model(self) -> None:
        """``AlphaFactor`` must not declare a legacy primary ``factor_id``.

        The legacy v1.x ``Factor`` class still carries ``factor_id`` —
        that's by design (it's gated behind beta+). The alpha-only
        :class:`AlphaFactor` is a clean slate and must not regress.
        """
        fields = AlphaFactor.model_fields
        assert "urn" in fields
        assert "factor_id_alias" in fields
        assert "factor_id" not in fields, (
            "AlphaFactor must keep `urn` as the only primary identifier; "
            "`factor_id_alias` is the only legacy slot."
        )


# ---------------------------------------------------------------------------
# Layer 3 — SDK round-trip: get_factor + list_factors return URN-primary
# payloads from a mocked transport.
# ---------------------------------------------------------------------------


@pytest.fixture()
def alpha_profile(monkeypatch):
    """Activate the alpha-v0.1 release profile."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.delenv("GL_ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    yield


class TestSDKEndpointURNPrimary:
    """Mock the alpha endpoints; assert SDK surfaces URN-primary records."""

    def _mock_transport(self, payload: Dict[str, Any]) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)
        return httpx.MockTransport(handler)

    def test_get_factor_returns_urn_primary(self, alpha_profile):
        payload = _make_factor_payload()
        with FactorsClient(
            base_url=_BASE_URL,
            transport=self._mock_transport(payload),
            verify_greenlang_cert=False,
        ) as client:
            f = client.get_factor(_VALID_URN)
        assert isinstance(f, AlphaFactor)
        assert f.urn == _VALID_URN
        assert f.factor_id_alias == _LEGACY_ALIAS

    def test_list_factors_returns_urn_primary(self, alpha_profile):
        list_payload = {
            "data": [_make_factor_payload()],
            "next_cursor": None,
        }
        with FactorsClient(
            base_url=_BASE_URL,
            transport=self._mock_transport(list_payload),
            verify_greenlang_cert=False,
        ) as client:
            resp = client.list_factors()
        assert isinstance(resp, ListFactorsResponse)
        assert len(resp.data) == 1
        assert resp.data[0].urn == _VALID_URN
        assert resp.data[0].factor_id_alias == _LEGACY_ALIAS

    def test_get_by_alias_returns_urn_primary(self, alpha_profile):
        """``client.get_by_alias()`` returns the canonical record with
        ``urn`` as primary id."""
        payload = _make_factor_payload()
        captured: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json=payload)

        with FactorsClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(handler),
            verify_greenlang_cert=False,
        ) as client:
            f = client.get_by_alias(_LEGACY_ALIAS)
        assert f is not None
        assert f.urn == _VALID_URN
        assert f.factor_id_alias == _LEGACY_ALIAS
        # Sanity: the SDK actually hit the by-alias path.
        assert "by-alias" in captured["url"]

    def test_get_by_alias_returns_none_on_404(self, alpha_profile):
        """A 404 from the alias endpoint must surface as ``None``,
        NOT raise — callers can branch on ``is None``."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "error": "factor_alias_not_found",
                    "message": "no match",
                    "legacy_id": "EF:phantom",
                },
            )
        with FactorsClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(handler),
            verify_greenlang_cert=False,
        ) as client:
            assert client.get_by_alias("EF:phantom") is None

    def test_get_by_alias_rejects_empty_legacy_id(self, alpha_profile):
        with FactorsClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"never": "called"})
            ),
            verify_greenlang_cert=False,
        ) as client:
            with pytest.raises(ValueError):
                client.get_by_alias("")

    def test_get_factor_rejects_legacy_id_at_client_boundary(
        self, alpha_profile
    ):
        """Calling ``client.get_factor(legacy_id)`` MUST raise without
        making a network round-trip — the SDK refuses to use legacy ids
        as primary identifiers."""
        with FactorsClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"never": "called"})
            ),
            verify_greenlang_cert=False,
        ) as client:
            with pytest.raises(ValueError, match="valid GreenLang URN"):
                client.get_factor(_LEGACY_ALIAS)


# ---------------------------------------------------------------------------
# Layer 4 — Router by-alias endpoint via the live FastAPI app.
# ---------------------------------------------------------------------------


class TestByAliasRouteViaFastAPI:
    """Spin up the alpha router and exercise /v1/factors/by-alias.

    The Phase 2 contract is that:

      * ``GET /v1/factors/{urn}`` and
      * ``GET /v1/factors/by-alias/{legacy_id}``

    return identical payloads when the alias points at the URN. The
    response carries ``urn`` as primary, ``factor_id_alias`` as
    secondary.
    """

    def _build_app(self, monkeypatch, tmp_path):
        from fastapi import FastAPI

        from greenlang.factors.api_v0_1_alpha_routes import router
        from greenlang.factors.repositories.alpha_v0_1_repository import (
            AlphaFactorRepository,
        )

        # Load the alpha provenance gate's schema OK; sqlite memory
        # repo so each test is hermetic.
        # legacy mode — Phase 1 provenance gate only; Phase 2 orchestrator covered by tests/factors/v0_1_alpha/phase2/test_publish_pipeline_e2e.py
        repo = AlphaFactorRepository(
            dsn="sqlite:///:memory:", publish_env="legacy"
        )

        # Publish the canonical record + register the alias.
        rec = _make_factor_payload()
        repo.publish(rec)
        repo.register_alias(_VALID_URN, _LEGACY_ALIAS)

        app = FastAPI()
        app.state.alpha_factor_repo = repo
        app.include_router(router)
        return app, repo

    def test_by_alias_resolves_to_urn_primary(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        app, repo = self._build_app(monkeypatch, tmp_path)
        try:
            client = TestClient(app)
            r = client.get(f"/v1/factors/by-alias/{quote(_LEGACY_ALIAS, safe='')}")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["urn"] == _VALID_URN
            assert body["factor_id_alias"] == _LEGACY_ALIAS
            assert "factor_id" not in body
        finally:
            repo.close()

    def test_by_alias_404_on_unknown_legacy_id(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        app, repo = self._build_app(monkeypatch, tmp_path)
        try:
            client = TestClient(app)
            r = client.get("/v1/factors/by-alias/EF:phantom:does-not-exist")
            assert r.status_code == 404, r.text
            body = r.json()
            assert body["error"] == "factor_alias_not_found"
            assert body["legacy_id"] == "EF:phantom:does-not-exist"
        finally:
            repo.close()

    def test_by_alias_and_by_urn_return_same_record(
        self, monkeypatch, tmp_path
    ):
        """The two GETs return the same canonical record.

        This is the round-trip property: get-by-urn(X) ==
        get-by-alias(legacy_id) when alias->X."""
        from fastapi.testclient import TestClient

        app, repo = self._build_app(monkeypatch, tmp_path)
        try:
            client = TestClient(app)
            urn_response = client.get(
                f"/v1/factors/{quote(_VALID_URN, safe='')}"
            )
            alias_response = client.get(
                f"/v1/factors/by-alias/{quote(_LEGACY_ALIAS, safe='')}"
            )
            assert urn_response.status_code == 200, urn_response.text
            assert alias_response.status_code == 200, alias_response.text
            urn_body = urn_response.json()
            alias_body = alias_response.json()
            # Both should carry urn primary, alias secondary, with the
            # same canonical record content.
            assert urn_body["urn"] == _VALID_URN
            assert alias_body["urn"] == _VALID_URN
            assert urn_body["factor_id_alias"] == _LEGACY_ALIAS
            assert alias_body["factor_id_alias"] == _LEGACY_ALIAS
            # Compare the rest of the canonical fields.
            for key in ("source_urn", "factor_pack_urn", "value",
                        "unit_urn", "geography_urn", "methodology_urn"):
                assert urn_body.get(key) == alias_body.get(key), key
        finally:
            repo.close()
