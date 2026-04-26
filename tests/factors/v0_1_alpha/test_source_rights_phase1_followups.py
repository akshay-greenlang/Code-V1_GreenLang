# -*- coding: utf-8 -*-
"""Phase 1 follow-up regression tests (CTO audit corrections).

Plugs the gaps the CTO flagged in the Phase 1 follow-up review:

  P1.a — `GET /v1/factors` list endpoint MUST drop denied sources.
  P1.b — `GET /v1/factors` list endpoint MUST emit audit events for
         non-`community_open` sources (allow + deny + metadata).
  P1.c — Rights-service runtime errors MUST fail CLOSED in production
         (publish path raises; route returns 503).
  P2.a — `alpha_publisher.publish_to_staging` MUST reject records whose
         `licence` field does not match the registry pin.
  P3   — v0.1 sources MUST have non-null `latest_source_version` and
         `latest_ingestion_timestamp` (placeholders for v0.5+ may be
         null per Phase 1 schema policy).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.factors.api_v0_1_alpha_routes import (
    _phase1_rights_filter_list,
    _phase1_rights_filter_one,
)
from greenlang.factors.rights import (
    EntitlementRecord,
    EntitlementStatus,
    EntitlementStore,
    EntitlementType,
    SourceRightsService,
    get_audit_log,
)
from greenlang.factors.rights.audit import AuditDecision


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _src(
    urn: str,
    licence_class: str,
    redistribution_class: str = "tenant_entitled_only",
    legal_status: str = "approved",
    licence: str = None,
) -> Dict[str, Any]:
    s = {
        "source_id": urn.rsplit(":", 1)[-1].replace("-", "_"),
        "urn": urn,
        "display_name": urn,
        "authority": "Test",
        "publisher": "Test",
        "licence_class": licence_class,
        "redistribution_class": redistribution_class,
        "cadence": "annual",
        "source_owner": "test",
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": legal_status,
            "reviewed_by": "test" if legal_status == "approved" else None,
            "reviewed_at": "2026-04-26T00:00:00Z" if legal_status == "approved" else None,
            "evidence_uri": "test://evidence" if legal_status == "approved" else None,
        },
        "publication_url": "https://example.org/test",
        "citation_text": "Test.",
        "entitlement_rules": {
            "model": "tenant_entitlement_required" if licence_class == "commercial_licensed" else "public_no_entitlement",
            "metadata_visibility": "public",
        },
        "release_milestone": "v0.1",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
    }
    if licence is not None:
        s["licence"] = licence
    return s


@pytest.fixture()
def fake_request_with_tenant():
    """Build a minimal Request shim with .state.user populated."""
    class _State:
        user = {"tenant_id": "T1", "api_key_id": "ak-T1"}

    class _Headers(dict):
        def get(self, k, default=None):
            return super().get(k, default)

    class _Req:
        state = _State()
        headers = _Headers({"X-Request-Id": "req-test"})

    return _Req()


@pytest.fixture()
def fake_request_anonymous():
    class _State:
        user = None

    class _Headers(dict):
        def get(self, k, default=None):
            return super().get(k, default)

    class _Req:
        state = _State()
        headers = _Headers({})

    return _Req()


@pytest.fixture()
def patched_default_service(monkeypatch):
    """Replace `default_service()` with a synthetic 4-source service."""
    registry = {
        "urn:gl:source:test-open": _src(
            "urn:gl:source:test-open",
            licence_class="community_open",
            redistribution_class="redistribution_allowed",
        ),
        "urn:gl:source:test-commercial": _src(
            "urn:gl:source:test-commercial",
            licence_class="commercial_licensed",
            redistribution_class="tenant_entitled_only",
        ),
        "urn:gl:source:test-private": _src(
            "urn:gl:source:test-private",
            licence_class="private_tenant_scoped",
            redistribution_class="tenant_entitled_only",
        ),
        "urn:gl:source:test-pending": _src(
            "urn:gl:source:test-pending",
            licence_class="community_open",
            redistribution_class="redistribution_allowed",
            legal_status="pending_legal_review",
        ),
    }
    entitlements = EntitlementStore.from_records([
        EntitlementRecord(
            tenant_id="T-ENTITLED",
            source_urn="urn:gl:source:test-commercial",
            pack_urn=None,
            entitlement_type=EntitlementType.SOURCE_ACCESS,
            status=EntitlementStatus.ACTIVE,
            valid_from="2026-01-01T00:00:00Z",
            valid_until=None,
            approved_by="test",
        ),
    ])
    svc = SourceRightsService(
        registry_index=registry,
        entitlements=entitlements,
        release_profile="alpha-v0.1",
    )
    # Patch the import the route layer uses.
    import greenlang.factors.rights as rights_mod
    monkeypatch.setattr(rights_mod, "default_service", lambda: svc)
    monkeypatch.setattr(
        "greenlang.factors.rights.service.default_service", lambda: svc
    )
    get_audit_log(clear=True)
    return svc


# ---------------------------------------------------------------------------
# P1.a — list endpoint drops denied sources
# ---------------------------------------------------------------------------


def test_list_query_filters_denied_commercial_when_unauth(
    patched_default_service, fake_request_with_tenant
) -> None:
    """T1 (no entitlement) MUST NOT receive commercial records via list."""
    rows = [
        {"urn": "urn:gl:factor:test-open:n:r:v1", "source_urn": "urn:gl:source:test-open"},
        {"urn": "urn:gl:factor:test-commercial:n:r:v1", "source_urn": "urn:gl:source:test-commercial"},
        {"urn": "urn:gl:factor:test-pending:n:r:v1", "source_urn": "urn:gl:source:test-pending"},
        {"urn": "urn:gl:factor:test-private:n:r:v1", "source_urn": "urn:gl:source:test-private"},
    ]
    out = _phase1_rights_filter_list(fake_request_with_tenant, rows)
    out_urns = [r["source_urn"] for r in out]
    assert "urn:gl:source:test-open" in out_urns
    assert "urn:gl:source:test-commercial" not in out_urns
    assert "urn:gl:source:test-pending" not in out_urns
    assert "urn:gl:source:test-private" not in out_urns


def test_list_query_passes_entitled_commercial_records(
    patched_default_service, monkeypatch
) -> None:
    """An entitled tenant DOES receive commercial records via list."""
    class _State:
        user = {"tenant_id": "T-ENTITLED", "api_key_id": "ak-ent"}

    class _Headers(dict):
        def get(self, k, default=None):
            return super().get(k, default)

    class _Req:
        state = _State()
        headers = _Headers({})

    rows = [
        {"urn": "urn:gl:factor:test-open:n:r:v1", "source_urn": "urn:gl:source:test-open"},
        {"urn": "urn:gl:factor:test-commercial:n:r:v1", "source_urn": "urn:gl:source:test-commercial"},
    ]
    out = _phase1_rights_filter_list(_Req(), rows)
    assert len(out) == 2  # both kept


# ---------------------------------------------------------------------------
# P1.b — list endpoint emits audit per non-community_open source
# ---------------------------------------------------------------------------


def test_list_query_audits_licensed_access(
    patched_default_service, fake_request_with_tenant
) -> None:
    """Every non-community_open record in a list query MUST emit one
    audit event (allow OR deny). community_open records emit none.
    """
    get_audit_log(clear=True)
    rows = [
        {"urn": "urn:gl:factor:test-open:n:r:v1", "source_urn": "urn:gl:source:test-open"},
        {"urn": "urn:gl:factor:test-commercial:n:r:v1", "source_urn": "urn:gl:source:test-commercial"},
        {"urn": "urn:gl:factor:test-private:n:r:v1", "source_urn": "urn:gl:source:test-private"},
    ]
    _phase1_rights_filter_list(fake_request_with_tenant, rows)
    log = get_audit_log()
    # 2 non-community_open rows → 2 audit events. The community_open
    # row MUST NOT emit (it would dominate the audit log).
    assert len(log) == 2
    licence_classes = sorted(e.licence_class for e in log)
    assert licence_classes == ["commercial_licensed", "private_tenant_scoped"]
    # Both should be DENY for tenant T1 (no entitlements).
    assert all(e.decision == AuditDecision.DENY for e in log)
    assert all(e.action == "list" for e in log)
    assert all(e.tenant_id == "T1" for e in log)


def test_list_query_does_not_audit_community_open(
    patched_default_service, fake_request_with_tenant
) -> None:
    """community_open reads MUST NOT spam the audit log."""
    get_audit_log(clear=True)
    rows = [
        {"urn": f"urn:gl:factor:test-open:n:r{i}:v1", "source_urn": "urn:gl:source:test-open"}
        for i in range(50)
    ]
    _phase1_rights_filter_list(fake_request_with_tenant, rows)
    log = get_audit_log()
    assert len(log) == 0


# ---------------------------------------------------------------------------
# P1.c — rights-service runtime errors fail CLOSED
# ---------------------------------------------------------------------------


def test_route_layer_fails_closed_on_rights_runtime_error(
    monkeypatch, fake_request_with_tenant
) -> None:
    """If the rights service raises at runtime, single-read returns 503."""
    def _broken_service():
        class _Broken:
            def check_factor_read_allowed(self, *args, **kwargs):
                raise RuntimeError("synthetic rights failure")
        return _Broken()
    monkeypatch.setattr(
        "greenlang.factors.rights.service.default_service", _broken_service
    )
    monkeypatch.setattr(
        "greenlang.factors.rights.default_service", _broken_service
    )
    rec = {
        "urn": "urn:gl:factor:any:n:r:v1",
        "source_urn": "urn:gl:source:any-source",
    }
    resp = _phase1_rights_filter_one(fake_request_with_tenant, rec)
    assert resp is not None
    assert resp.status_code == 503
    body = resp.body.decode("utf-8")
    assert "rights_unavailable" in body


def test_list_filter_drops_records_on_rights_runtime_error(
    monkeypatch, fake_request_with_tenant
) -> None:
    """If the rights service raises, list filter drops the record (fail-closed)."""
    def _broken_service():
        class _Broken:
            def check_factor_read_allowed(self, *args, **kwargs):
                raise RuntimeError("synthetic rights failure")
        return _Broken()
    monkeypatch.setattr(
        "greenlang.factors.rights.service.default_service", _broken_service
    )
    monkeypatch.setattr(
        "greenlang.factors.rights.default_service", _broken_service
    )
    rows = [
        {"urn": "urn:gl:factor:any:n:r:v1", "source_urn": "urn:gl:source:any-source"},
    ]
    out = _phase1_rights_filter_list(fake_request_with_tenant, rows)
    assert out == []  # dropped on rights failure


# ---------------------------------------------------------------------------
# P2.a — publish_to_staging rejects record with mismatched licence tag
# ---------------------------------------------------------------------------


def test_publish_rejects_record_with_mismatched_licence(monkeypatch, tmp_path) -> None:
    """When the registry pins a `licence` value, a record carrying a
    different `licence` must be REJECTED via LicenceMismatch.
    """
    monkeypatch.setenv("GL_FACTORS_RIGHTS_FAIL_OPEN", "0")  # production semantic

    registry = {
        "urn:gl:source:test-pinned": {
            **_src(
                "urn:gl:source:test-pinned",
                licence_class="community_open",
                redistribution_class="attribution_required",
            ),
            "licence": "OGL-UK-3.0",  # registry pins a specific tag
        },
    }
    svc = SourceRightsService(
        registry_index=registry,
        entitlements=EntitlementStore(),
        release_profile="alpha-v0.1",
    )
    monkeypatch.setattr(
        "greenlang.factors.rights.service.default_service", lambda: svc
    )
    monkeypatch.setattr(
        "greenlang.factors.rights.default_service", lambda: svc
    )

    from greenlang.factors.repositories.alpha_v0_1_repository import (
        AlphaFactorRepository,
    )
    from greenlang.factors.release.alpha_publisher import (
        AlphaPublisher,
    )
    from greenlang.factors.rights import LicenceMismatch

    db_path = tmp_path / "publish.sqlite"
    repo = AlphaFactorRepository(dsn=f"sqlite:///{db_path}")
    publisher = AlphaPublisher(repo=repo)

    # Build a minimally valid v0.1 record (the publisher's provenance
    # gate runs first; provide a record that will pass the pre-rights
    # checks). The rights gate rejects on the licence mismatch.
    record = {
        "urn": "urn:gl:factor:test-pinned:ns:rec:v1",
        "source_urn": "urn:gl:source:test-pinned",
        "factor_pack_urn": "urn:gl:pack:test-pinned:p:v1",
        "name": "Test record",
        "description": "A test record long enough to satisfy the description floor.",
        "category": "scope1",
        "value": 1.0,
        "unit_urn": "urn:gl:unit:kgco2e/kwh",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": "urn:gl:geo:country:gb",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "resolution": "annual",
        "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
        "boundary": "combustion-only test record",
        "licence": "EU-COMMISSION-CBAM-DEFAULTS",  # MISMATCH vs registry pin
        "citations": [{"type": "url", "value": "https://example.org/cite"}],
        "published_at": "2026-04-26T00:00:00Z",
        "extraction": {
            "source_url": "https://example.org/x",
            "source_record_id": "rec-1",
            "source_publication": "Test Pub",
            "source_version": "1.0",
            "raw_artifact_uri": "s3://x/y",
            "raw_artifact_sha256": "0" * 64,
            "parser_id": "tbd.placeholder",
            "parser_version": "0.0.0",
            "parser_commit": "0000000",
            "row_ref": "row-1",
            "ingested_at": "2026-04-26T00:00:00Z",
            "operator": "human:test",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:test",
            "reviewed_at": "2026-04-26T00:00:00Z",
            "approved_by": "human:test",
            "approved_at": "2026-04-26T00:00:00Z",
        },
    }

    with pytest.raises((LicenceMismatch, Exception)) as excinfo:
        publisher.publish_to_staging(record)
    # Either LicenceMismatch or AlphaPublisherError wrapping it.
    msg = str(excinfo.value).lower()
    assert "licence" in msg or "license" in msg


# ---------------------------------------------------------------------------
# P3 — v0.1 sources MUST have populated latest_source_version + latest_ingestion_timestamp
# ---------------------------------------------------------------------------


def test_v0_1_sources_have_freshness_metadata() -> None:
    """v0.1 sources MUST have non-null `latest_source_version` AND
    `latest_ingestion_timestamp` so the freshness view is honest.

    Placeholders for v0.5+ MAY have these as null — the policy is
    documented in the schema description.
    """
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    offenders: List[tuple[str, str, Any]] = []
    for urn, src in registry.items():
        if src.get("release_milestone") != "v0.1":
            continue
        if not src.get("latest_source_version"):
            offenders.append((urn, "latest_source_version", src.get("latest_source_version")))
        if not src.get("latest_ingestion_timestamp"):
            offenders.append((urn, "latest_ingestion_timestamp", src.get("latest_ingestion_timestamp")))
    assert not offenders, (
        f"v0.1 sources missing freshness metadata: {offenders}"
    )


# ---------------------------------------------------------------------------
# Bonus: publish path fails CLOSED on rights-service runtime error
# (CTO P1.c, publish-side counterpart of test_route_layer_fails_closed_*)
# ---------------------------------------------------------------------------


def test_publish_fails_closed_on_rights_runtime_error(monkeypatch, tmp_path) -> None:
    """In production (GL_FACTORS_RIGHTS_FAIL_OPEN=0), a rights-service
    runtime error MUST raise AlphaPublisherError; the record MUST
    NOT be published.
    """
    monkeypatch.setenv("GL_FACTORS_RIGHTS_FAIL_OPEN", "0")

    def _broken_service():
        class _Broken:
            def check_ingestion_allowed(self, *args, **kwargs):
                raise RuntimeError("synthetic rights failure")
            def check_record_licence_matches_registry(self, *args, **kwargs):
                raise RuntimeError("synthetic rights failure")
        return _Broken()

    monkeypatch.setattr(
        "greenlang.factors.rights.service.default_service", _broken_service
    )
    monkeypatch.setattr(
        "greenlang.factors.rights.default_service", _broken_service
    )

    from greenlang.factors.repositories.alpha_v0_1_repository import (
        AlphaFactorRepository,
    )
    from greenlang.factors.release.alpha_publisher import (
        AlphaPublisher,
        AlphaPublisherError,
    )

    repo = AlphaFactorRepository(dsn=f"sqlite:///{tmp_path / 'fail-closed.sqlite'}")
    publisher = AlphaPublisher(repo=repo)

    record = {
        "urn": "urn:gl:factor:any:ns:rec:v1",
        "source_urn": "urn:gl:source:any-source",
    }
    with pytest.raises(AlphaPublisherError) as excinfo:
        publisher.publish_to_staging(record)
    assert "rights" in str(excinfo.value).lower()
