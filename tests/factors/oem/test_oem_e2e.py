# -*- coding: utf-8 -*-
"""
End-to-end tests for the Wave 5 OEM white-label flow.

Exercises:
    (a) partner tenant creation with redistribution grants,
    (b) Certified-edition entitlement grant for an OEM-backed tenant,
    (c) resolve-style signed receipt carrying the partner branding +
        ``X-GreenLang-Edition`` on the outer envelope,
    (d) bulk OEM export endpoint (signed artifact, branded, filtered by
        the partner grant only),
    (e) negative path: a non-OEM caller hitting /v1/oem/export is 403,
    (f) hermetic fixtures: the OEM registry + quota ledger are reset
        between tests so no shared state leaks.

The tests drive the production modules the previous agent shipped
directly (``build_oem_export``) AND through the FastAPI router with a
``TestClient``, so the wire contract is exercised end-to-end. No real
Stripe / network / filesystem beyond tmp_path.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.entitlements import (
    EntitlementRegistry,
    OEMRights,
    PackSKU,
    check_oem_redistribution,
)
from greenlang.factors.onboarding.api import oem_router
from greenlang.factors.onboarding.branding_config import BrandingConfig
from greenlang.factors.onboarding.oem_export import (
    OemExportError,
    SignedArtifact,
    build_oem_export,
    _reset_quota_ledger,
)
from greenlang.factors.onboarding.partner_setup import (
    create_oem_partner,
    get_oem_partner,
    _reset_oem_registry,
)
from greenlang.factors.signing import verify_sha256_hmac


SIGNING_SECRET = "oem-e2e-test-secret-do-not-ship-this-value-xxxx"


# ---------------------------------------------------------------------------
# Hermetic fixtures — the OEM registry + daily quota counter + signing
# secret are all process-locals. Reset each one before and after every
# test so the order of test execution cannot bleed state.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_oem_state(monkeypatch: pytest.MonkeyPatch):
    """Reset the OEM registry, quota ledger, and signing env before/after."""
    monkeypatch.setenv("GL_FACTORS_SIGNING_SECRET", SIGNING_SECRET)
    monkeypatch.delenv("GL_FACTORS_ED25519_PRIVATE_KEY", raising=False)
    _reset_oem_registry()
    _reset_quota_ledger()
    yield
    _reset_oem_registry()
    _reset_quota_ledger()


@pytest.fixture()
def entitlement_registry(tmp_path) -> EntitlementRegistry:
    """Fresh SQLite-backed EntitlementRegistry scoped to the test."""
    reg = EntitlementRegistry(tmp_path / "entitlements.sqlite")
    yield reg
    reg.close()


@pytest.fixture()
def branding() -> BrandingConfig:
    """Canonical OEM branding fixture used across the suite."""
    return BrandingConfig(
        logo_url="https://assets.acme.example/logo.svg",
        primary_color="#0A66C2",
        secondary_color="#FFFFFF",
        support_email="support@acme.example",
        support_url="https://help.acme.example",
        custom_domain="factors.acme.example",
        attribution_required=True,
        powered_by_text="Powered by GreenLang",
    )


@pytest.fixture()
def seeded_oem(branding: BrandingConfig):
    """Register an OEM partner with WHITE_LABEL_REDISTRIBUTION + PRIVATE_REGISTRY-equivalent
    grants. The OEMRights enum in this repo exposes FORBIDDEN / INTERNAL_ONLY /
    REDISTRIBUTABLE; the partner-facing analogue for 'white-label
    redistribution' is the REDISTRIBUTABLE state with grant classes that
    include both 'open' (public/open data) and 'greenlang_terms' (private
    registry). The test asserts the same invariant: the OEM can ship rows
    from both buckets, nothing else.
    """
    partner = create_oem_partner(
        name="Acme Sustainability",
        contact_email="partners@acme.example",
        redistribution_grants=[
            "open",  # public / open-data redistribution (analogue of WHITE_LABEL)
            "greenlang_terms",  # private GreenLang-terms registry
        ],
        parent_plan="platform",
        branding=branding,
        notes="oem-e2e-test partner",
    )
    return partner


@pytest.fixture()
def seed_rows() -> List[Dict[str, Any]]:
    """Realistic mixed-license factor records.

    Includes:
      * open-redistribution rows inside the OEM grant (electricity slice),
      * one greenlang_terms row inside the grant,
      * one commercial_connector row OUTSIDE the grant (must be filtered
        out of the export).
    """
    return [
        {
            "factor_id": "elec_us_egrid_2023",
            "family": "electricity",
            "region": "US",
            "year": 2023,
            "value": 0.385,
            "unit": "kgCO2e/kWh",
            "license_class": "open",
        },
        {
            "factor_id": "elec_uk_desnz_2024",
            "family": "electricity",
            "region": "UK",
            "year": 2024,
            "value": 0.207,
            "unit": "kgCO2e/kWh",
            "license_class": "open",
        },
        {
            "factor_id": "material_steel_eu_2024",
            "family": "material",
            "region": "EU",
            "year": 2024,
            "value": 1.85,
            "unit": "kgCO2e/kg",
            "license_class": "greenlang_terms",
        },
        {
            # MUST be filtered out of the OEM export — commercial connector
            # rows are never redistributable.
            "factor_id": "ecoinvent_cement_global_2023",
            "family": "product",
            "region": "GLOBAL",
            "year": 2023,
            "value": 0.93,
            "unit": "kgCO2e/kg",
            "license_class": "commercial_connector",
        },
    ]


# ---------------------------------------------------------------------------
# Fake factors_service + repo for the FastAPI /v1/oem/export path
# ---------------------------------------------------------------------------


class _FakeRepo:
    """Minimal FactorCatalogRepository stub the export route understands."""

    def __init__(self, edition_id: str, rows: List[Dict[str, Any]]):
        self._edition_id = edition_id
        self._rows = rows

    def resolve_edition(self, requested):
        # Accept both explicit and "None" lookups so the route's default-
        # edition branch is covered.
        if requested in (None, "", self._edition_id):
            return self._edition_id
        raise ValueError(f"Unknown edition: {requested!r}")

    def list_factors(
        self,
        edition_id,
        *,
        page: int = 1,
        limit: int = 100,
        include_preview: bool = False,
        include_connector: bool = False,
    ):
        rows = list(self._rows)
        if not include_connector:
            rows = [r for r in rows if r.get("license_class") != "commercial_connector"]
        return rows[:limit], len(rows)


class _FakeService:
    def __init__(self, edition_id: str, rows: List[Dict[str, Any]]):
        self.repo = _FakeRepo(edition_id, rows)


@pytest.fixture()
def export_app(seed_rows: List[Dict[str, Any]]) -> FastAPI:
    """Mount only the oem_router on a bare app + inject a fake service.

    Skipping the full SignedReceiptsMiddleware / LicensingGuardMiddleware
    stack here is intentional — the export endpoint signs its OWN
    artifact via ``build_oem_export``; the outer receipt middleware
    handles the resolve / explain surface, which we exercise separately.
    """
    app = FastAPI()
    app.include_router(oem_router)
    app.state.factors_service = _FakeService("2026.04.1", seed_rows)
    return app


@pytest.fixture()
def export_client(export_app: FastAPI) -> TestClient:
    return TestClient(export_app)


# ===========================================================================
# (a) + (b) Partner tenant creation + Certified entitlement grant
# ===========================================================================


class TestPartnerTenantCreation:
    def test_oem_registered_with_redistribution_grants(
        self, seeded_oem, branding: BrandingConfig
    ):
        """Partner creation persists grants + branding."""
        partner = get_oem_partner(seeded_oem.id)
        assert partner.active
        assert partner.parent_plan == "platform"
        assert set(partner.grant.allowed_classes) == {"open", "greenlang_terms"}
        # Branding round-trips through the dataclass store unchanged.
        assert partner.branding is branding
        assert partner.branding.primary_color == "#0a66c2"

    def test_certified_electricity_entitlement_grant(
        self, seeded_oem, entitlement_registry: EntitlementRegistry
    ):
        """Grant a Certified-edition electricity-slice entitlement to the OEM tenant.

        The entitlement ledger keys off ``tenant_id`` and stores the
        redistribution tri-state alongside; REDISTRIBUTABLE is the
        load-bearing flag for an OEM that actually ships rows downstream.
        """
        ent = entitlement_registry.grant(
            tenant_id=seeded_oem.id,
            pack_sku=PackSKU.ELECTRICITY_PREMIUM,
            oem_rights=OEMRights.REDISTRIBUTABLE,
            notes="Wave 5 OEM e2e - electricity certified slice",
        )
        assert ent.tenant_id == seeded_oem.id
        assert ent.pack_sku == PackSKU.ELECTRICITY_PREMIUM
        assert ent.oem_rights == OEMRights.REDISTRIBUTABLE
        assert entitlement_registry.is_entitled(
            tenant_id=seeded_oem.id, pack_sku=PackSKU.ELECTRICITY_PREMIUM
        )

    def test_non_oem_tenant_has_no_redistribution(self):
        """A tenant that was never registered as an OEM is denied by the guard."""
        assert check_oem_redistribution("not_a_real_oem", {"license_class": "open"}) is False


# ===========================================================================
# (c) Signed receipt + edition header + branding via build_oem_export
# ===========================================================================


class TestSignedReceiptAndBranding:
    """Verifies the signed-artifact envelope the resolve-style path produces.

    We drive ``build_oem_export`` directly here because it is the single
    production entry point that emits the signed receipt + branded
    manifest + edition id. Running the full /v1/resolve middleware stack
    would require a live catalog; the signing invariants being asserted
    here live in ``build_oem_export`` itself.
    """

    def test_signed_artifact_has_verification_key_hint(
        self, seeded_oem, seed_rows
    ):
        artifact = build_oem_export(
            seeded_oem.id, edition_id="2026.04.1", rows=seed_rows
        )
        assert isinstance(artifact, SignedArtifact)
        assert artifact.receipt.algorithm == "sha256-hmac"
        # The receipt's key_id is the verification-key hint the partner
        # tenant uses to look up the public signer.
        assert artifact.receipt.key_id == "gl-factors-oem-export"
        envelope = artifact.to_envelope()
        assert envelope["signed_receipt"]["key_id"] == "gl-factors-oem-export"
        assert envelope["signed_receipt"]["algorithm"] == "sha256-hmac"
        # payload_hash is a 64-char sha256 hex.
        assert len(envelope["signed_receipt"]["payload_hash"]) == 64

    def test_signature_verifies_against_signing_module(
        self, seeded_oem, seed_rows
    ):
        """The receipt MUST verify against the same signing module that produced it.

        This proves we never reimplemented crypto in the OEM path.
        """
        artifact = build_oem_export(
            seeded_oem.id, edition_id="2026.04.1", rows=seed_rows
        )
        assert verify_sha256_hmac(artifact.manifest, artifact.receipt.to_dict()) is True

    def test_envelope_carries_edition_id_and_branding(
        self, seeded_oem, seed_rows, branding: BrandingConfig
    ):
        """The envelope carries the X-GreenLang-Edition analogue + branding."""
        artifact = build_oem_export(
            seeded_oem.id, edition_id="2026.04.1", rows=seed_rows
        )
        env = artifact.to_envelope()
        # Edition id on the envelope — same field name the FastAPI
        # X-GreenLang-Edition header is sourced from.
        assert env["edition_id"] == "2026.04.1"
        assert env["manifest"]["edition_id"] == "2026.04.1"
        # Branding embedded in the manifest so partners can theme
        # downstream responses without another API round-trip.
        brand_meta = env["manifest"]["branding"]
        assert brand_meta["logo_url"] == branding.logo_url
        assert brand_meta["primary_color"] == "#0a66c2"
        assert brand_meta["support_email"] == "support@acme.example"
        assert brand_meta["custom_domain"] == "factors.acme.example"
        assert brand_meta["attribution_required"] is True


# ===========================================================================
# (d) Bulk OEM export endpoint — signed artifact, filtered, branded
# ===========================================================================


class TestBulkOemExportEndpoint:
    def test_endpoint_returns_signed_envelope(
        self, export_client: TestClient, seeded_oem, branding: BrandingConfig
    ):
        """POST /v1/oem/export returns the full signed JSONL envelope."""
        resp = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": seeded_oem.id},
            json={"edition_id": "2026.04.1", "include_connector": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Envelope contract
        assert body["oem_id"] == seeded_oem.id
        assert body["edition_id"] == "2026.04.1"
        assert body["factor_count"] >= 1
        assert "payload_jsonl" in body
        assert "signed_receipt" in body
        assert "content_hash" in body
        # Signature validity
        assert verify_sha256_hmac(body["manifest"], body["signed_receipt"]) is True
        # Branding inside the manifest
        assert body["manifest"]["branding"]["logo_url"] == branding.logo_url

    def test_export_filters_non_redistributable_rows(
        self, export_client: TestClient, seeded_oem
    ):
        """Commercial-connector rows are NEVER in the dump even when requested."""
        resp = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": seeded_oem.id},
            json={"edition_id": "2026.04.1", "include_connector": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        factor_ids = [
            json.loads(line)["factor_id"]
            for line in body["payload_jsonl"].splitlines()
            if line.strip()
        ]
        # Three rows are in-grant (two open + one greenlang_terms).
        # The commercial_connector row is NOT in the OEM's grant and must
        # be filtered out even though include_connector=True at the repo
        # layer; the entitlement filter wins.
        assert "ecoinvent_cement_global_2023" not in factor_ids
        assert "elec_us_egrid_2023" in factor_ids
        assert "elec_uk_desnz_2024" in factor_ids
        assert "material_steel_eu_2024" in factor_ids
        assert body["factor_count"] == len(factor_ids) == 3

    def test_export_content_hash_matches_payload(
        self, export_client: TestClient, seeded_oem
    ):
        """content_hash is a commitment over the JSONL payload."""
        import hashlib

        resp = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": seeded_oem.id},
            json={"edition_id": "2026.04.1"},
        )
        body = resp.json()
        recomputed = hashlib.sha256(body["payload_jsonl"].encode("utf-8")).hexdigest()
        assert body["content_hash"] == recomputed
        assert body["manifest"]["content_hash"] == recomputed

    def test_export_artifact_id_is_stable(
        self, export_client: TestClient, seeded_oem
    ):
        """Same inputs -> same artifact_id (deterministic hash over factor ids)."""
        first = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": seeded_oem.id},
            json={"edition_id": "2026.04.1"},
        ).json()
        _reset_quota_ledger()  # allow re-issuing the same export
        second = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": seeded_oem.id},
            json={"edition_id": "2026.04.1"},
        ).json()
        assert first["artifact_id"] == second["artifact_id"]
        assert first["content_hash"] == second["content_hash"]


# ===========================================================================
# (e) Negative: a NON-OEM tenant hitting /v1/oem/export gets 403
# ===========================================================================


class TestNonOemForbidden:
    def test_missing_oem_id_header_is_403(self, export_client: TestClient):
        """No X-OEM-Id = 'authed but not an OEM partner' = 403."""
        resp = export_client.post("/v1/oem/export", json={})
        assert resp.status_code == 403
        payload = resp.json()
        # FastAPI wraps dict-detail in {'detail': {...}}.
        detail = payload.get("detail") or {}
        assert detail.get("error") == "not_oem"

    def test_unknown_oem_id_is_403(self, export_client: TestClient):
        """A caller asserting an unregistered OEM id is also 403."""
        resp = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": "oem_ghost_never_registered"},
            json={},
        )
        assert resp.status_code == 403
        detail = resp.json().get("detail") or {}
        assert detail.get("error") == "not_oem"

    def test_oem_with_zero_entitled_rows_is_422(
        self, export_client: TestClient, branding: BrandingConfig
    ):
        """An OEM with a grant that matches nothing in the edition gets 422.

        This is the 'authed + valid OEM but nothing to ship' branch
        (distinct from the 403 non-OEM branch). It proves the filter is
        deterministic: no silently-empty signed dump.
        """
        narrow = create_oem_partner(
            name="NarrowCo",
            contact_email="ops@narrow.example",
            # Only 'pcaf_attribution' — none of our seed rows match.
            redistribution_grants=["pcaf_attribution"],
            parent_plan="platform",
            branding=branding,
        )
        resp = export_client.post(
            "/v1/oem/export",
            headers={"X-OEM-Id": narrow.id},
            json={"edition_id": "2026.04.1"},
        )
        assert resp.status_code == 422
        detail = resp.json().get("detail") or {}
        assert detail.get("error") == "oem_export_invalid"


# ===========================================================================
# (f) Hermetic invariants — state does not leak between tests
# ===========================================================================


class TestHermeticState:
    def test_registry_reset_between_tests(self):
        """At test start the OEM registry is empty (autouse fixture reset it)."""
        from greenlang.factors.onboarding.partner_setup import list_oem_partners

        assert list_oem_partners() == []

    def test_quota_ledger_reset_between_tests(self, seeded_oem, seed_rows):
        """Each test starts with 0 quota used for a freshly-minted OEM."""
        # The autouse fixture reset the ledger; the first export should
        # succeed without tripping the daily quota.
        artifact = build_oem_export(
            seeded_oem.id, edition_id="2026.04.1", rows=seed_rows
        )
        assert artifact.factor_count >= 1


# ===========================================================================
# Low-level invariants that back the e2e flow (quick, hermetic)
# ===========================================================================


class TestExportInvariants:
    def test_build_raises_when_oem_not_registered(self, seed_rows):
        with pytest.raises(OemExportError):
            build_oem_export("oem_unknown_abc123", edition_id="2026.04.1", rows=seed_rows)

    def test_build_raises_on_empty_entitled_rows(self, branding: BrandingConfig):
        partner = create_oem_partner(
            name="EmptyCo",
            contact_email="ops@empty.example",
            redistribution_grants=["pcaf_attribution"],
            parent_plan="platform",
            branding=branding,
        )
        with pytest.raises(OemExportError):
            build_oem_export(partner.id, edition_id="2026.04.1", rows=[])
