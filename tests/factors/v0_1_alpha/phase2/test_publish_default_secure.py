# -*- coding: utf-8 -*-
"""Phase 2 / WS8 — secure-by-default publish-gate regression test (CTO P0/P1).

The CTO verified locally that, prior to this fix, ``AlphaFactorRepository``
would accept records with invalid ``unit_urn`` and licence-mismatched
records under default construction. Those bypasses were the result of the
seven-gate orchestrator being opt-IN rather than opt-OUT.

This regression locks the new default in place:

  * Test 1 — default constructor + seeded ontology + bad ``unit_urn``
            -> :class:`OntologyReferenceError`.
  * Test 2 — default constructor + seeded ontology + licence mismatch
            -> :class:`LicenceMismatchError`.
  * Test 3 — default constructor + UN-seeded ontology
            -> :class:`OntologyReferenceError` (fail-CLOSED).
  * Test 4 — explicit ``publish_env='legacy'`` accepts the bad record
            (proves the bypass is explicit, not implicit).
  * Test 5 — passing ``publish_orchestrator=False`` triggers a
            :class:`DeprecationWarning`.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict

import pytest

from greenlang.factors.quality.publish_gates import (
    LicenceMismatchError,
    OntologyReferenceError,
    PublishGateOrchestrator,
)
from greenlang.factors.repositories import AlphaFactorRepository

from tests.factors.v0_1_alpha.phase2.conftest import (
    SEEDED_GEOGRAPHY_URN,
    SEEDED_LICENCE,
    SEEDED_METHODOLOGY_URN,
    SEEDED_PACK_URN,
    SEEDED_SOURCE_URN,
    SEEDED_UNIT_URN,
    SEEDED_VALID_SHA256,
)


# ---------------------------------------------------------------------------
# Canonical fully-valid record used by all five tests.
# ---------------------------------------------------------------------------


def _good_record(urn_leaf: str = "default-secure") -> Dict[str, Any]:
    """A v0.1 record that passes every gate against the seeded ontology."""
    return {
        "urn": f"urn:gl:factor:phase2-alpha:default:{urn_leaf}:v1",
        "factor_id_alias": f"EF:PHASE2:default:{urn_leaf}:v1",
        "source_urn": SEEDED_SOURCE_URN,
        "factor_pack_urn": SEEDED_PACK_URN,
        "name": "Phase 2 default-secure regression record",
        "description": (
            "Synthetic emission factor used by the Phase 2 / WS8 default-"
            "secure regression. Boundary excludes upstream extraction."
        ),
        "category": "fuel",
        "value": 1.234,
        "unit_urn": SEEDED_UNIT_URN,
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": SEEDED_GEOGRAPHY_URN,
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "resolution": "annual",
        "methodology_urn": SEEDED_METHODOLOGY_URN,
        "boundary": (
            "Boundary excludes upstream extraction and distribution losses."
        ),
        "licence": SEEDED_LICENCE,
        "citations": [{"type": "url", "value": "https://example.test/source"}],
        "published_at": "2026-04-27T12:00:00Z",
        "extraction": {
            "source_url": "https://example.test/source",
            "source_record_id": "row=1",
            "source_publication": "Phase 2 default-secure publication",
            "source_version": "2024.1",
            "raw_artifact_uri": "s3://phase2-fixture/2024.1/file.pdf",
            "raw_artifact_sha256": SEEDED_VALID_SHA256,
            "parser_id": "greenlang.factors.ingestion.parsers.test",
            "parser_version": "0.1.0",
            "parser_commit": "deadbeefcafe",
            "row_ref": "Sheet=A;Row=1",
            "ingested_at": "2026-04-27T11:00:00Z",
            "operator": "bot:phase2_default_secure",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:phase2-lead@greenlang.io",
            "reviewed_at": "2026-04-27T11:30:00Z",
            "approved_by": "human:phase2-lead@greenlang.io",
            "approved_at": "2026-04-27T11:31:00Z",
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — default constructor + seeded ontology + bad unit_urn rejects.
# ---------------------------------------------------------------------------


def test_default_construction_rejects_invalid_unit_urn(seeded_repo) -> None:
    """A record with a bogus ``unit_urn`` MUST be rejected under defaults.

    This is the CTO's headline P0: prior to the fix, the same record
    would be persisted because the orchestrator was opt-in. Now the
    seven-gate orchestrator runs by default and gate 3 (ontology FK)
    blocks the bad URN.
    """
    rec = _good_record(urn_leaf="bad-unit")
    rec["unit_urn"] = "urn:gl:unit:does-not-exist"
    with pytest.raises(OntologyReferenceError) as exc_info:
        seeded_repo.publish(rec)
    # Assert the failure message names the offending field.
    assert "unit_urn" in exc_info.value.reason
    # Assert NO row was written.
    assert seeded_repo.get_by_urn(rec["urn"]) is None


# ---------------------------------------------------------------------------
# Test 2 — default constructor + licence mismatch rejects.
# ---------------------------------------------------------------------------


def test_default_construction_rejects_licence_mismatch(seeded_repo) -> None:
    """A record whose ``licence`` differs from the registry pin MUST be
    rejected under default construction (gate 5 — licence match).
    """
    rec = _good_record(urn_leaf="bad-licence")
    rec["licence"] = "GPL-3.0"  # registry pins SEEDED_LICENCE (CC-BY-4.0)
    with pytest.raises(LicenceMismatchError) as exc_info:
        seeded_repo.publish(rec)
    assert "GPL-3.0" in exc_info.value.reason
    assert SEEDED_LICENCE in exc_info.value.reason
    assert seeded_repo.get_by_urn(rec["urn"]) is None


# ---------------------------------------------------------------------------
# Test 3 — default constructor + UN-seeded ontology fails closed.
# ---------------------------------------------------------------------------


def test_default_construction_fails_closed_on_missing_ontology() -> None:
    """A bare ``AlphaFactorRepository(dsn=':memory:')`` with NO ontology
    seed MUST raise :class:`OntologyReferenceError` — production publish
    requires a seeded ontology (Phase 2 CTO P0 fail-closed fix).

    Construction is intentionally minimal: ``AlphaFactorRepository(...)``
    with no ``publish_env`` override + no fixture seeding. This is the
    secure-by-default failure mode the CTO requested.
    """
    repo = AlphaFactorRepository(dsn="sqlite:///:memory:")
    try:
        rec = _good_record(urn_leaf="no-ontology")
        # No ontology tables seeded -> SELECT 1 FROM unit fails ->
        # production env raises OntologyReferenceError immediately.
        with pytest.raises(OntologyReferenceError) as exc_info:
            repo.publish(rec)
        # Reason should mention "not present" / production env to make
        # the failure forensically obvious.
        reason = exc_info.value.reason.lower()
        assert "not present" in reason or "missing" in reason or "production" in reason
        # And NO row was written.
        assert repo.get_by_urn(rec["urn"]) is None
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Test 4 — explicit publish_env='legacy' accepts the bad record.
# ---------------------------------------------------------------------------


def test_explicit_legacy_mode_accepts_bad_unit_urn() -> None:
    """``publish_env='legacy'`` is the EXPLICIT bypass — the orchestrator
    is skipped, the legacy :class:`AlphaProvenanceGate` is used, and the
    bad ``unit_urn`` is persisted (because the legacy gate doesn't probe
    the ontology table).

    This proves the bypass is INTENTIONAL and visible in the call site;
    the CTO's mandate is that secure behaviour is the DEFAULT, not that
    bypasses are impossible.
    """
    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:", publish_env="legacy"
    )
    try:
        rec = _good_record(urn_leaf="legacy-bypass")
        # The legacy AlphaProvenanceGate runs the v0.1 JSON Schema, which
        # constrains URN format but does NOT probe the ontology table.
        # A unit_urn that satisfies the schema regex but doesn't exist in
        # the (un-seeded) unit table goes through.
        rec["unit_urn"] = "urn:gl:unit:legacy-bypass-target"
        urn = repo.publish(rec)
        assert urn == rec["urn"]
        fetched = repo.get_by_urn(urn)
        assert fetched is not None
        assert fetched["unit_urn"] == "urn:gl:unit:legacy-bypass-target"
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Test 5 — DeprecationWarning on publish_orchestrator=False.
# ---------------------------------------------------------------------------


def test_publish_orchestrator_false_emits_deprecation_warning() -> None:
    """The legacy ``publish_orchestrator=False`` kwarg is now a deprecated
    alias for ``publish_env='legacy'``. Constructing the repository with
    that kwarg must emit a :class:`DeprecationWarning`. The repository
    still functions in legacy mode (backwards-compat).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repo = AlphaFactorRepository(
            dsn="sqlite:///:memory:", publish_orchestrator=False
        )
        try:
            # Find the deprecation warning we emitted.
            dep_warnings = [
                w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "publish_orchestrator" in str(w.message)
            ]
            assert dep_warnings, (
                "Expected DeprecationWarning when "
                "publish_orchestrator=False is passed; got "
                f"{[str(w.message) for w in caught]}"
            )
            # And the resolved env IS legacy (not production) — the
            # bypass landed.
            assert repo._publish_env == "legacy"  # type: ignore[attr-defined]
            # Sanity: a record can be published under the legacy gate.
            rec = _good_record(urn_leaf="deprecated-kwarg")
            rec["unit_urn"] = "urn:gl:unit:does-not-exist"
            urn = repo.publish(rec)
            assert urn == rec["urn"]
        finally:
            repo.close()


# ---------------------------------------------------------------------------
# Bonus: positive case — defaults + fully-valid record persists cleanly.
# ---------------------------------------------------------------------------


def test_default_construction_accepts_valid_record(seeded_repo) -> None:
    """The same fixture that rejects bad records MUST accept a fully-valid
    one — otherwise the regression is too aggressive.
    """
    rec = _good_record(urn_leaf="happy-path")
    urn = seeded_repo.publish(rec)
    assert urn == rec["urn"]
    fetched = seeded_repo.get_by_urn(urn)
    assert fetched is not None
    assert fetched == rec


# ---------------------------------------------------------------------------
# Bonus: invalid publish_env raises ValueError.
# ---------------------------------------------------------------------------


def test_invalid_publish_env_raises_value_error() -> None:
    """A typo in ``publish_env`` MUST be caught at construction time."""
    with pytest.raises(ValueError, match="publish_env must be one of"):
        AlphaFactorRepository(
            dsn="sqlite:///:memory:", publish_env="prodution"
        )


# ---------------------------------------------------------------------------
# Bonus: legacy-mode warning is emitted exactly once.
# ---------------------------------------------------------------------------


def test_legacy_mode_emits_one_time_warning(caplog) -> None:
    """Legacy mode must log a one-time WARNING on first publish so any
    bypass is forensically traceable.
    """
    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:", publish_env="legacy"
    )
    try:
        rec_a = _good_record(urn_leaf="warning-first")
        rec_b = _good_record(urn_leaf="warning-second")
        with caplog.at_level("WARNING"):
            repo.publish(rec_a)
            repo.publish(rec_b)
        # Find the legacy-mode warnings.
        legacy_warnings = [
            r for r in caplog.records
            if "publish_env='legacy'" in r.getMessage()
        ]
        assert len(legacy_warnings) == 1, (
            "Expected exactly one legacy-mode warning per repo instance; "
            f"got {len(legacy_warnings)} -> "
            f"{[r.getMessage() for r in legacy_warnings]}"
        )
        # And the warning carries the FIRST URN (audit anchor).
        msg = legacy_warnings[0].getMessage()
        assert rec_a["urn"] in msg
    finally:
        repo.close()
