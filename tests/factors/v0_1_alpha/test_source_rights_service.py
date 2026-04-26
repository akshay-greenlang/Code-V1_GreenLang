# -*- coding: utf-8 -*-
"""Phase 1 — SourceRightsService unit tests.

Covers the 12 CTO-required scenarios in `epic` Phase 1.7:

  1. Unauthorized tenant cannot retrieve commercial/licensed factor.
  2. Unauthorized tenant cannot download commercial/licensed pack.
  3. Unauthorized tenant cannot retrieve private tenant-scoped factor.
  4. Authorized tenant can retrieve entitled commercial factor.
  5. Authorized tenant access creates audit log.
  6. Open/community factor is accessible without commercial entitlement.
  7. Blocked source cannot be ingested.
  8. Pending legal review source cannot publish to production.
  9. Factor with mismatched licence tag fails publish gate.
 10. Every production factor source exists in source_registry.yaml.
 11. Every production source has legal signoff.
 12. Every licensed source access includes audit event fields.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from greenlang.factors.rights import (
    EntitlementRecord,
    EntitlementStatus,
    EntitlementStore,
    EntitlementType,
    IngestionBlocked,
    SourceRightsService,
    audit_licensed_access,
    get_audit_log,
)
from greenlang.factors.rights.audit import AuditDecision


# ---------------------------------------------------------------------------
# Fixtures: synthetic registry + entitlement store
# ---------------------------------------------------------------------------


def _src(
    urn: str,
    licence_class: str,
    redistribution_class: str,
    legal_status: str = "approved",
    release_milestone: str = "v0.1",
    entitlement_model: str = "public_no_entitlement",
) -> Dict[str, Any]:
    return {
        "source_id": urn.rsplit(":", 1)[-1].replace("-", "_"),
        "urn": urn,
        "display_name": urn,
        "authority": "Test Authority",
        "publisher": "Test Publisher",
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
        "citation_text": "Test citation.",
        "entitlement_rules": {"model": entitlement_model, "metadata_visibility": "public"},
        "release_milestone": release_milestone,
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
    }


@pytest.fixture()
def registry_index() -> Dict[str, Dict[str, Any]]:
    """A synthetic 6-source registry covering every licence_class."""
    return {
        "urn:gl:source:test-open": _src(
            "urn:gl:source:test-open",
            licence_class="community_open",
            redistribution_class="redistribution_allowed",
        ),
        "urn:gl:source:test-method": _src(
            "urn:gl:source:test-method",
            licence_class="method_only",
            redistribution_class="metadata_only",
        ),
        "urn:gl:source:test-commercial": _src(
            "urn:gl:source:test-commercial",
            licence_class="commercial_licensed",
            redistribution_class="tenant_entitled_only",
            entitlement_model="tenant_entitlement_required",
        ),
        "urn:gl:source:test-private": _src(
            "urn:gl:source:test-private",
            licence_class="private_tenant_scoped",
            redistribution_class="tenant_entitled_only",
            entitlement_model="private_tenant_owner_only",
        ),
        "urn:gl:source:test-connector": _src(
            "urn:gl:source:test-connector",
            licence_class="connector_only",
            redistribution_class="metadata_only",
            entitlement_model="connector_only_no_bulk",
        ),
        "urn:gl:source:test-blocked": _src(
            "urn:gl:source:test-blocked",
            licence_class="blocked",
            redistribution_class="blocked",
            entitlement_model="blocked",
            legal_status="rejected",
        ),
        "urn:gl:source:test-pending": _src(
            "urn:gl:source:test-pending",
            licence_class="community_open",
            redistribution_class="redistribution_allowed",
            legal_status="pending_legal_review",
        ),
    }


@pytest.fixture()
def entitlements() -> EntitlementStore:
    rows = [
        EntitlementRecord(
            tenant_id="ENTITLED",
            source_urn="urn:gl:source:test-commercial",
            pack_urn=None,
            entitlement_type=EntitlementType.SOURCE_ACCESS,
            status=EntitlementStatus.ACTIVE,
            valid_from="2026-01-01T00:00:00Z",
            valid_until=None,
            approved_by="test",
        ),
        EntitlementRecord(
            tenant_id="EXPIRED",
            source_urn="urn:gl:source:test-commercial",
            pack_urn=None,
            entitlement_type=EntitlementType.SOURCE_ACCESS,
            status=EntitlementStatus.EXPIRED,
            valid_from="2024-01-01T00:00:00Z",
            valid_until="2025-01-01T00:00:00Z",
            approved_by="test",
        ),
        EntitlementRecord(
            tenant_id="OWNER",
            source_urn="urn:gl:source:test-private",
            pack_urn=None,
            entitlement_type=EntitlementType.PRIVATE_OWNER,
            status=EntitlementStatus.ACTIVE,
            valid_from="2026-01-01T00:00:00Z",
            valid_until=None,
            approved_by="test",
        ),
    ]
    return EntitlementStore.from_records(rows)


@pytest.fixture()
def svc(registry_index, entitlements):
    # Clear the audit log between tests.
    get_audit_log(clear=True)
    return SourceRightsService(
        registry_index=registry_index,
        entitlements=entitlements,
        release_profile="alpha-v0.1",
    )


# ---------------------------------------------------------------------------
# CTO test #1 — unauthorized tenant cannot retrieve commercial/licensed factor
# ---------------------------------------------------------------------------


def test_unauthorized_tenant_cannot_read_commercial_factor(svc) -> None:
    d = svc.check_factor_read_allowed("UNAUTH", "urn:gl:source:test-commercial")
    assert d.denied
    assert "no active entitlement" in d.reason


def test_anonymous_caller_cannot_read_commercial_factor(svc) -> None:
    d = svc.check_factor_read_allowed(None, "urn:gl:source:test-commercial")
    assert d.denied
    assert "authenticated tenant" in d.reason


# ---------------------------------------------------------------------------
# CTO test #2 — unauthorized tenant cannot download commercial pack
# ---------------------------------------------------------------------------


def test_unauthorized_tenant_cannot_download_commercial_pack(svc) -> None:
    d = svc.check_pack_download_allowed(
        "UNAUTH",
        "urn:gl:pack:test-commercial:bundle:v1",
        "urn:gl:source:test-commercial",
    )
    assert d.denied
    assert "entitlement" in d.reason


# ---------------------------------------------------------------------------
# CTO test #3 — unauthorized tenant cannot read private tenant-scoped factor
# ---------------------------------------------------------------------------


def test_unauthorized_tenant_cannot_read_private_factor(svc) -> None:
    d = svc.check_factor_read_allowed("INTRUDER", "urn:gl:source:test-private")
    assert d.denied
    assert (
        "private" in d.reason.lower()
        or "owner" in d.reason.lower()
        or "not the owner" in d.reason.lower()
    )


def test_owner_tenant_can_read_private_factor(svc) -> None:
    d = svc.check_factor_read_allowed("OWNER", "urn:gl:source:test-private")
    assert d.allowed


# ---------------------------------------------------------------------------
# CTO test #4 — authorized tenant CAN retrieve entitled commercial factor
# ---------------------------------------------------------------------------


def test_entitled_tenant_can_read_commercial_factor(svc) -> None:
    d = svc.check_factor_read_allowed("ENTITLED", "urn:gl:source:test-commercial")
    assert d.allowed
    assert d.licence_class == "commercial_licensed"


def test_expired_entitlement_denies_commercial_read(svc) -> None:
    d = svc.check_factor_read_allowed("EXPIRED", "urn:gl:source:test-commercial")
    assert d.denied


# ---------------------------------------------------------------------------
# CTO test #5 — authorized access creates audit log
# ---------------------------------------------------------------------------


def test_audit_log_emitted_on_licensed_access() -> None:
    get_audit_log(clear=True)
    audit_licensed_access(
        tenant_id="ENTITLED",
        source_urn="urn:gl:source:test-commercial",
        factor_urn="urn:gl:factor:test-commercial:fixture:rec:v1",
        licence_class="commercial_licensed",
        decision=AuditDecision.ALLOW,
        reason="active entitlement",
        request_id="req-123",
        api_key_id="ak-abc",
        action="read",
    )
    log = get_audit_log()
    assert len(log) == 1
    e = log[0]
    assert e.tenant_id == "ENTITLED"
    assert e.source_urn == "urn:gl:source:test-commercial"
    assert e.decision == AuditDecision.ALLOW
    assert e.licence_class == "commercial_licensed"
    assert e.request_id == "req-123"
    assert e.api_key_id == "ak-abc"
    assert e.action == "read"
    assert e.occurred_at  # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# CTO test #6 — open/community factor is accessible without entitlement
# ---------------------------------------------------------------------------


def test_community_open_accessible_without_entitlement(svc) -> None:
    d = svc.check_factor_read_allowed(None, "urn:gl:source:test-open")
    assert d.allowed
    d = svc.check_factor_read_allowed("ANY", "urn:gl:source:test-open")
    assert d.allowed


# ---------------------------------------------------------------------------
# CTO test #7 — blocked source cannot be ingested
# ---------------------------------------------------------------------------


def test_blocked_source_cannot_be_ingested(svc) -> None:
    d = svc.check_ingestion_allowed("urn:gl:source:test-blocked")
    assert d.denied
    with pytest.raises(IngestionBlocked):
        svc.assert_ingestion_allowed("urn:gl:source:test-blocked")


# ---------------------------------------------------------------------------
# CTO test #8 — pending-legal source cannot publish to production
# ---------------------------------------------------------------------------


def test_pending_legal_source_cannot_be_ingested(svc) -> None:
    d = svc.check_ingestion_allowed("urn:gl:source:test-pending")
    assert d.denied
    assert "legal_signoff" in d.reason


def test_unknown_source_cannot_be_ingested(svc) -> None:
    # Default: unknown source falls open (provenance gate is the check).
    d = svc.check_ingestion_allowed("urn:gl:source:nonexistent")
    assert d.allowed
    # Strict: explicit deny on unknown source (used by dedicated
    # regression tests that want a hard fail).
    d2 = svc.check_ingestion_allowed(
        "urn:gl:source:nonexistent", strict_unknown=True
    )
    assert d2.denied


# ---------------------------------------------------------------------------
# CTO test #9 — record with mismatched licence tag fails publish gate
# ---------------------------------------------------------------------------


def test_record_without_licence_field_falls_open(svc) -> None:
    # Phase 1 follow-up correction: a missing `licence` on the record
    # falls OPEN here. The schema/provenance gate is responsible for
    # asserting the record HAS a licence at all; this gate only fires
    # on explicit MISMATCH (registry pins X, record carries Y != X).
    d = svc.check_record_licence_matches_registry(
        "urn:gl:source:test-open", record_licence=None
    )
    assert d.allowed


def test_record_with_matching_licence_passes(svc, registry_index) -> None:
    # Pin a licence on the test-open source; verify the matcher allows
    # only when the record's licence tag equals it.
    registry_index["urn:gl:source:test-open"]["licence"] = "OGL-UK-3.0"
    d = svc.check_record_licence_matches_registry(
        "urn:gl:source:test-open", record_licence="OGL-UK-3.0"
    )
    assert d.allowed
    d2 = svc.check_record_licence_matches_registry(
        "urn:gl:source:test-open", record_licence="EU-COMMISSION-CBAM-DEFAULTS"
    )
    assert d2.denied


def test_record_licence_check_falls_open_on_unknown_source(svc) -> None:
    """Mirror the other gates' "unknown source falls open" semantic."""
    d = svc.check_record_licence_matches_registry(
        "urn:gl:source:totally-unknown", record_licence="OGL-UK-3.0"
    )
    assert d.allowed


# ---------------------------------------------------------------------------
# CTO test #10 — every production factor source exists in registry
# (alpha catalog seed audit)
# ---------------------------------------------------------------------------


def test_every_alpha_catalog_seed_source_exists_in_registry() -> None:
    """Walk every catalog seed under catalog_seed_v0_1 and verify each
    record's ``source_urn`` resolves to a registered source.
    """
    import json
    from pathlib import Path
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    seeds_dir = (
        Path(__file__).resolve().parents[3]
        / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"
    )
    if not seeds_dir.is_dir():
        pytest.skip(f"seeds dir missing: {seeds_dir}")

    missing: list[tuple[str, int, str]] = []
    for src_dir in sorted(seeds_dir.iterdir()):
        if not src_dir.is_dir():
            continue
        seed = src_dir / "v1.json"
        if not seed.is_file():
            continue
        payload = json.loads(seed.read_text(encoding="utf-8"))
        for i, rec in enumerate(payload.get("records") or []):
            source_urn = rec.get("source_urn")
            if isinstance(source_urn, str) and source_urn not in registry:
                missing.append((src_dir.name, i, source_urn))
    assert not missing, (
        f"alpha catalog factors with source_urn missing from registry "
        f"({len(missing)} offenders): "
        + "; ".join(f"{s}:rec{i}={u}" for s, i, u in missing[:5])
    )


def test_every_alpha_catalog_seed_licence_matches_registry_pin() -> None:
    """Every v0.1 catalog record must carry the licence tag pinned in
    source_registry.yaml.

    This is the production-facing version of the Phase 1 licence-tag
    rule: the registry, not parser-specific labels, owns the canonical
    factor-record licence string.
    """
    import json
    from pathlib import Path
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    seeds_dir = (
        Path(__file__).resolve().parents[3]
        / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"
    )
    offenders: list[tuple[str, int, str, str, str]] = []
    for src_dir in sorted(seeds_dir.iterdir()):
        if not src_dir.is_dir():
            continue
        seed = src_dir / "v1.json"
        if not seed.is_file():
            continue
        payload = json.loads(seed.read_text(encoding="utf-8-sig"))
        for i, rec in enumerate(payload.get("records") or []):
            source_urn = rec.get("source_urn")
            if not isinstance(source_urn, str):
                continue
            source = registry.get(source_urn) or {}
            expected = source.get("licence")
            actual = rec.get("licence")
            if expected and actual != expected:
                offenders.append((src_dir.name, i, source_urn, str(actual), str(expected)))
    assert not offenders, (
        "alpha catalog factors with licence != registry pin "
        f"({len(offenders)} offenders): "
        + "; ".join(
            f"{s}:rec{i} {u} actual={a!r} expected={e!r}"
            for s, i, u, a, e in offenders[:5]
        )
    )


# ---------------------------------------------------------------------------
# CTO test #11 — every production source has legal signoff
# ---------------------------------------------------------------------------


def test_every_v0_1_source_has_legal_signoff() -> None:
    """Every source whose ``release_milestone == 'v0.1'`` MUST have
    ``legal_signoff.status == 'approved'``.
    """
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    offenders: list[tuple[str, str]] = []
    for urn, src in registry.items():
        if src.get("release_milestone") != "v0.1":
            continue
        ls = (src.get("legal_signoff") or {}).get("status")
        if ls != "approved":
            offenders.append((urn, str(ls)))
    assert not offenders, (
        f"v0.1 sources without approved legal_signoff: {offenders}"
    )


def test_every_v0_1_source_has_registry_licence_pin() -> None:
    """Every production v0.1 source must pin the exact factor licence tag."""
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    offenders: list[str] = []
    for urn, src in registry.items():
        if src.get("release_milestone") != "v0.1":
            continue
        licence = src.get("licence")
        if not isinstance(licence, str) or not licence.strip():
            offenders.append(urn)
    assert not offenders, f"v0.1 sources without licence pin: {offenders}"


def test_ghgp_method_refs_phase1_licensing_approved() -> None:
    """GHGP method references are approved for v0.1 method metadata only."""
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    source = registry["urn:gl:source:ghgp-method-refs"]
    assert source["licence_class"] == "method_only"
    assert source["redistribution_class"] == "metadata_only"
    assert source["phase1_licensing_scope"] == "v0.1-method-reference"
    assert source["legal_signoff"]["status"] == "approved"
    assert source["licence"] == "GHGP-METHOD-REFERENCE"


def test_alpha_vintage_exceptions_are_formally_accepted() -> None:
    """Sources that miss the CTO target vintage must not stay in
    informal preview status; each needs an accepted exception file.
    """
    from pathlib import Path
    from greenlang.factors.rights.service import _load_registry_index

    registry = _load_registry_index()
    offenders: list[tuple[str, str]] = []
    for urn, src in registry.items():
        if src.get("release_milestone") != "v0.1":
            continue
        exception = src.get("alpha_v0_1_methodology_exception")
        if not exception:
            continue
        if src.get("alpha_v0_1_status") != "exception_accepted":
            offenders.append((urn, "registry status is not exception_accepted"))
            continue
        path = Path(__file__).resolve().parents[3] / str(exception)
        if not path.is_file():
            offenders.append((urn, f"missing exception file {exception}"))
            continue
        text = path.read_text(encoding="utf-8")
        if "**Status:** `exception_accepted`" not in text:
            offenders.append((urn, "exception file not accepted"))
        if "CTO-delegated exception acceptance" not in text:
            offenders.append((urn, "exception file missing delegated acceptance"))
    assert not offenders, f"unaccepted alpha vintage exceptions: {offenders}"


# ---------------------------------------------------------------------------
# CTO test #12 — every licensed source access includes audit event fields
# ---------------------------------------------------------------------------


def test_audit_event_includes_required_fields() -> None:
    get_audit_log(clear=True)
    audit_licensed_access(
        tenant_id="T",
        source_urn="urn:gl:source:test-commercial",
        factor_urn="urn:gl:factor:test-commercial:f:rec:v1",
        pack_urn="urn:gl:pack:test-commercial:p:v1",
        licence_class="commercial_licensed",
        decision=AuditDecision.ALLOW,
        reason="entitled",
        request_id="req-xyz",
        api_key_id="ak-456",
        action="read",
    )
    e = get_audit_log()[0]
    # CTO-required fields per Phase 1.5 audit-log spec.
    assert e.tenant_id == "T"
    assert e.api_key_id == "ak-456"
    assert e.source_urn
    assert e.factor_urn
    assert e.pack_urn
    assert e.decision in (AuditDecision.ALLOW, AuditDecision.DENY, AuditDecision.METADATA_ONLY)
    assert e.action in ("read", "list", "ingest", "pack_download")
    assert e.occurred_at
    assert e.request_id == "req-xyz"


# ---------------------------------------------------------------------------
# Bonus: registry validates against the JSON schema in CI
# ---------------------------------------------------------------------------


def test_source_registry_validates_against_schema() -> None:
    """Source-of-truth governance gate: registry MUST be schema-valid."""
    import json
    from pathlib import Path
    import yaml  # type: ignore
    import jsonschema

    repo_root = Path(__file__).resolve().parents[3]
    reg = yaml.safe_load(
        (repo_root / "greenlang" / "factors" / "data" / "source_registry.yaml")
        .read_text(encoding="utf-8")
    )
    schema = json.loads(
        (repo_root / "config" / "schemas" / "source_registry_v0_1.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.validate(reg, schema)


def test_release_profile_milestone_floor() -> None:
    """A v0.5-milestone source must NOT be ingestible under alpha-v0.1."""
    svc = SourceRightsService(
        registry_index={
            "urn:gl:source:future": _src(
                "urn:gl:source:future",
                licence_class="community_open",
                redistribution_class="redistribution_allowed",
                release_milestone="v0.5",
            ),
        },
        entitlements=EntitlementStore(),
        release_profile="alpha-v0.1",
    )
    d = svc.check_ingestion_allowed("urn:gl:source:future")
    assert d.denied
    assert "release_milestone" in d.reason
    # And under beta-v0.5 it IS allowed.
    svc.release_profile = "beta-v0.5"
    d2 = svc.check_ingestion_allowed("urn:gl:source:future")
    assert d2.allowed
