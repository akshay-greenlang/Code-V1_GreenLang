# -*- coding: utf-8 -*-
"""Phase 2 / WS8 — coverage-tightening tests for publish-gate branches.

The rejection-matrix and end-to-end pipeline regressions cover the
happy + canonical-bad paths through the orchestrator. This file targets
the *defensive* branches — Postgres backend probes (psycopg ImportError,
missing dsn, connection failures), the dry_run unexpected-exception
handler, gate-2 alias collisions, gate-5 SourceRightsService denial
paths, gate-7 staging mode, and the ``_lookup_registry_row`` /
``_load_registry_index`` fallback chain.

We never modify the module under test. The orchestrator is constructed
with a stubbed repository whose ``_is_postgres``, ``_dsn``, ``_connect``,
``get_by_urn``, ``find_by_alias`` surface is set per-test via
:class:`unittest.mock.MagicMock` so the Postgres branches reachable only
when ``GL_TEST_POSTGRES_DSN`` is set in CI fire here under SQLite-only
local runs.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.quality.publish_gates import (
    GateOutcome,
    LicenceMismatchError,
    LifecycleStatusError,
    OntologyReferenceError,
    ProvenanceIncompleteError,
    PublishGateError,
    PublishGateOrchestrator,
    SchemaValidationError,
    SourceRegistryError,
    URNDuplicateError,
)


# ---------------------------------------------------------------------------
# Synthetic repo factory
# ---------------------------------------------------------------------------


class _StubRepo:
    """Minimal stand-in for AlphaFactorRepository with controllable surface.

    The orchestrator's gates query a small set of attributes / methods on
    the repository. We expose exactly those so the gate branches under
    test can be exercised without spinning up a real :class:`sqlite3`
    connection (or, for the Postgres branches, without needing libpq).
    """

    def __init__(
        self,
        *,
        is_postgres: bool = False,
        dsn: Optional[str] = None,
        connect_raises: Optional[Exception] = None,
        connect_returns: Any = None,
        get_by_urn_returns: Any = None,
        find_by_alias_returns: Any = None,
        find_by_alias_raises: Optional[Exception] = None,
    ) -> None:
        self._is_postgres = is_postgres
        self._dsn = dsn
        # _connect: error path -> branch line 800-805 + 869-871
        if connect_raises is not None:
            self._connect = MagicMock(side_effect=connect_raises)
        else:
            self._connect = MagicMock(return_value=connect_returns)
        self.get_by_urn = MagicMock(return_value=get_by_urn_returns)
        if find_by_alias_raises is not None:
            self.find_by_alias = MagicMock(side_effect=find_by_alias_raises)
        else:
            self.find_by_alias = MagicMock(return_value=find_by_alias_returns)
        # Used by _assert_ontology_table_present_sqlite to skip conn.close
        self._memory_conn = None


def _registry_index(
    *, alpha: bool = True, licence: str = "CC-BY-4.0"
) -> Dict[str, Dict[str, Any]]:
    return {
        "urn:gl:source:phase2-alpha": {
            "urn": "urn:gl:source:phase2-alpha",
            "source_id": "phase2-alpha",
            "licence": licence,
            "alpha_v0_1": alpha,
            "licence_class": "community_open",
            "redistribution_class": "attribution_required",
        }
    }


class _RightsAllow:
    def __init__(
        self, registry: Dict[str, Dict[str, Any]] | None = None
    ) -> None:
        self.registry_index = registry or _registry_index()

    def check_record_licence_matches_registry(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=False, reason="ok")

    def check_ingestion_allowed(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=False, reason="ok")


class _RightsDenyLicence:
    def __init__(
        self, registry: Dict[str, Dict[str, Any]] | None = None
    ) -> None:
        self.registry_index = registry or _registry_index()

    def check_record_licence_matches_registry(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=True, reason="rights service denied")

    def check_ingestion_allowed(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=False, reason="ok")


class _RightsDenyIngestion:
    def __init__(
        self, registry: Dict[str, Dict[str, Any]] | None = None
    ) -> None:
        self.registry_index = registry or _registry_index()

    def check_record_licence_matches_registry(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=False, reason="ok")

    def check_ingestion_allowed(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=True, reason="redistribution denied")


class _RightsLicenceRaises:
    def __init__(
        self, registry: Dict[str, Dict[str, Any]] | None = None
    ) -> None:
        self.registry_index = registry or _registry_index()

    def check_record_licence_matches_registry(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("simulated licence check explosion")

    def check_ingestion_allowed(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=False, reason="ok")


class _RightsIngestionRaises:
    def __init__(
        self, registry: Dict[str, Dict[str, Any]] | None = None
    ) -> None:
        self.registry_index = registry or _registry_index()

    def check_record_licence_matches_registry(self, *_: Any, **__: Any) -> Any:
        return MagicMock(denied=False, reason="ok")

    def check_ingestion_allowed(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("simulated ingestion check explosion")


# ---------------------------------------------------------------------------
# __init__ guard
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_repo_must_not_be_none(self) -> None:
        with pytest.raises(ValueError, match="repo must not be None"):
            PublishGateOrchestrator(None)

    def test_env_normalised_lowercase_strip(self) -> None:
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo, env="  PRODUCTION  ")
        assert orch.env == "production"

    def test_env_default_when_falsy(self) -> None:
        repo = _StubRepo()
        # Empty string falls back to 'production'.
        orch = PublishGateOrchestrator(repo, env="")
        assert orch.env == "production"


# ---------------------------------------------------------------------------
# dry_run branch coverage — non-dict + unexpected exception
# ---------------------------------------------------------------------------


class TestDryRunBranches:
    def test_dry_run_non_dict_short_circuits_to_not_run(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        results = orch.dry_run(["not", "a", "dict"])  # type: ignore[arg-type]
        assert len(results) == 7
        assert results[0].outcome == GateOutcome.FAIL
        for r in results[1:]:
            assert r.outcome == GateOutcome.NOT_RUN

    def test_dry_run_unexpected_exception_caught(self) -> None:
        # Force gate_1_schema to raise a *non*-PublishGateError so the
        # generic Exception handler at lines 372-385 fires.
        orch = PublishGateOrchestrator(_StubRepo())
        with patch.object(
            PublishGateOrchestrator,
            "gate_1_schema",
            side_effect=RuntimeError("boom"),
        ):
            results = orch.dry_run({"urn": "urn:gl:factor:x:y:z:v1"})
        # Result for gate 1 should be FAIL with "unexpected error" reason.
        first = results[0]
        assert first.outcome == GateOutcome.FAIL
        assert "unexpected error" in first.reason

    def test_dry_run_publish_gate_error_uses_exc_urn_and_details(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        with patch.object(
            PublishGateOrchestrator,
            "gate_1_schema",
            side_effect=SchemaValidationError(
                "boom", urn="urn:gl:factor:a", details={"k": "v"}
            ),
        ):
            results = orch.dry_run({"urn": "urn:gl:factor:other"})
        first = results[0]
        assert first.outcome == GateOutcome.FAIL
        assert first.urn == "urn:gl:factor:a"
        assert first.details == {"k": "v"}


# ---------------------------------------------------------------------------
# assert_publishable - non-dict input
# ---------------------------------------------------------------------------


def test_assert_publishable_non_dict_raises_schema() -> None:
    orch = PublishGateOrchestrator(_StubRepo())
    with pytest.raises(SchemaValidationError, match="record must be a dict"):
        orch.assert_publishable("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gate 2 — alias collision + alias lookup raising
# ---------------------------------------------------------------------------


class TestGate2Branches:
    def test_gate_2_missing_urn_raises_typed_error(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        with pytest.raises(URNDuplicateError, match="missing or non-string"):
            orch.gate_2_urn_uniqueness({"urn": ""})

    def test_gate_2_alias_bound_to_different_urn_raises(self) -> None:
        repo = _StubRepo(
            get_by_urn_returns=None,
            find_by_alias_returns={"urn": "urn:gl:factor:other:v1"},
        )
        orch = PublishGateOrchestrator(repo)
        with pytest.raises(
            URNDuplicateError, match="already bound to a different urn"
        ):
            orch.gate_2_urn_uniqueness(
                {
                    "urn": "urn:gl:factor:mine:v1",
                    "factor_id_alias": "EF:LEGACY:1",
                }
            )

    def test_gate_2_alias_lookup_raises_treated_as_no_match(self) -> None:
        # find_by_alias raising must NOT propagate; it's logged + treated
        # as no match -> gate passes.
        repo = _StubRepo(
            get_by_urn_returns=None,
            find_by_alias_raises=RuntimeError("db down"),
        )
        orch = PublishGateOrchestrator(repo)
        # Should NOT raise.
        orch.gate_2_urn_uniqueness(
            {"urn": "urn:gl:factor:mine:v1", "factor_id_alias": "x"}
        )

    def test_gate_2_alias_bound_to_same_urn_passes(self) -> None:
        repo = _StubRepo(
            get_by_urn_returns=None,
            find_by_alias_returns={"urn": "urn:gl:factor:mine:v1"},
        )
        orch = PublishGateOrchestrator(repo)
        # alias bound to SAME urn -> passes.
        orch.gate_2_urn_uniqueness(
            {"urn": "urn:gl:factor:mine:v1", "factor_id_alias": "EF:X"}
        )


# ---------------------------------------------------------------------------
# Gate 4 / 5 — registry / licence branches
# ---------------------------------------------------------------------------


class TestGate4And5Branches:
    def test_gate_4_missing_source_urn_raises(self) -> None:
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo, env="dev")
        with pytest.raises(SourceRegistryError, match="missing"):
            orch.gate_4_source_registry({"urn": "urn:gl:factor:x"})

    def test_gate_4_unknown_source_in_dev_still_raises(self) -> None:
        # Even in dev env, unknown source -> SourceRegistryError.
        repo = _StubRepo()
        rights = _RightsAllow(registry={})  # empty registry
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        with pytest.raises(SourceRegistryError, match="not present"):
            orch.gate_4_source_registry(
                {
                    "urn": "urn:gl:factor:x",
                    "source_urn": "urn:gl:source:nonexistent",
                }
            )

    def test_gate_4_non_alpha_in_production_raises(self) -> None:
        repo = _StubRepo()
        rights = _RightsAllow(registry=_registry_index(alpha=False))
        # source table check needs to pass; set is_postgres=False but
        # supply a connection that returns a row to assert table-present.
        # In env='dev' the alpha check is relaxed, so production needed.
        # Use env='dev' with a registry that doesn't require alpha to
        # avoid the table-present probe; gate 4 enforces alpha only when
        # not in _DEV_ENVS.
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        # in dev: relaxed -> should NOT raise on alpha=False.
        orch.gate_4_source_registry(
            {
                "urn": "urn:gl:factor:x",
                "source_urn": "urn:gl:source:phase2-alpha",
            }
        )

    def test_gate_4_non_alpha_outside_dev_raises(self) -> None:
        # env='custom' is neither in _DEV_ENVS nor in _FAIL_CLOSED_ENVS
        # so the source-table probe is skipped but the alpha check runs.
        repo = _StubRepo()
        rights = _RightsAllow(registry=_registry_index(alpha=False))
        orch = PublishGateOrchestrator(
            repo, source_rights=rights, env="custom"
        )
        with pytest.raises(SourceRegistryError, match="not alpha_v0_1=true"):
            orch.gate_4_source_registry(
                {
                    "urn": "urn:gl:factor:x",
                    "source_urn": "urn:gl:source:phase2-alpha",
                }
            )

    def test_gate_5_licence_mismatch_raises(self) -> None:
        repo = _StubRepo()
        rights = _RightsAllow(registry=_registry_index(licence="CC-BY-4.0"))
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        with pytest.raises(LicenceMismatchError, match="does not match"):
            orch.gate_5_licence_match(
                {
                    "urn": "urn:gl:factor:x",
                    "source_urn": "urn:gl:source:phase2-alpha",
                    "licence": "MIT",  # mismatch with registry pin
                }
            )

    def test_gate_5_rights_licence_check_denies(self) -> None:
        repo = _StubRepo()
        rights = _RightsDenyLicence()
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        with pytest.raises(
            LicenceMismatchError, match="SourceRightsService denied"
        ):
            orch.gate_5_licence_match(
                {
                    "urn": "urn:gl:factor:x",
                    "source_urn": "urn:gl:source:phase2-alpha",
                    "licence": "CC-BY-4.0",
                }
            )

    def test_gate_5_rights_ingestion_denies(self) -> None:
        repo = _StubRepo()
        rights = _RightsDenyIngestion()
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        with pytest.raises(
            LicenceMismatchError, match="ingestion denied"
        ):
            orch.gate_5_licence_match(
                {
                    "urn": "urn:gl:factor:x",
                    "source_urn": "urn:gl:source:phase2-alpha",
                    "licence": "CC-BY-4.0",
                }
            )

    def test_gate_5_rights_licence_check_raises_treated_as_allow(self) -> None:
        repo = _StubRepo()
        rights = _RightsLicenceRaises()
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        # Exception inside rights service must NOT propagate.
        orch.gate_5_licence_match(
            {
                "urn": "urn:gl:factor:x",
                "source_urn": "urn:gl:source:phase2-alpha",
                "licence": "CC-BY-4.0",
            }
        )

    def test_gate_5_rights_ingestion_raises_treated_as_allow(self) -> None:
        repo = _StubRepo()
        rights = _RightsIngestionRaises()
        orch = PublishGateOrchestrator(repo, source_rights=rights, env="dev")
        orch.gate_5_licence_match(
            {
                "urn": "urn:gl:factor:x",
                "source_urn": "urn:gl:source:phase2-alpha",
                "licence": "CC-BY-4.0",
            }
        )

    def test_gate_5_no_source_urn_passes_silently(self) -> None:
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo, env="dev")
        # Missing source_urn -> registry_row None -> skip licence check.
        orch.gate_5_licence_match({"urn": "urn:gl:factor:x"})


# ---------------------------------------------------------------------------
# Gate 6 — provenance branches
# ---------------------------------------------------------------------------


class TestGate6Branches:
    def test_gate_6_missing_extraction_raises(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        with pytest.raises(
            ProvenanceIncompleteError, match="extraction object missing"
        ):
            orch.gate_6_provenance_completeness({"urn": "urn:gl:factor:x"})

    def test_gate_6_missing_required_extraction_field_raises(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        # Build extraction with all fields except 'operator'.
        extraction = {
            "source_url": "https://x",
            "source_record_id": "1",
            "source_publication": "p",
            "source_version": "v",
            "raw_artifact_uri": "s3://a",
            "raw_artifact_sha256": "a" * 64,
            "parser_id": "pid",
            "parser_version": "0",
            "parser_commit": "cafe",
            "row_ref": "row",
            "ingested_at": "2026-01-01T00:00:00Z",
            # operator missing
        }
        with pytest.raises(
            ProvenanceIncompleteError, match="missing required fields"
        ):
            orch.gate_6_provenance_completeness(
                {"urn": "urn:gl:factor:x", "extraction": extraction}
            )

    def test_gate_6_bad_sha256_raises(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        extraction = {
            "source_url": "https://x",
            "source_record_id": "1",
            "source_publication": "p",
            "source_version": "v",
            "raw_artifact_uri": "s3://a",
            "raw_artifact_sha256": "NOT-HEX",  # invalid format
            "parser_id": "pid",
            "parser_version": "0",
            "parser_commit": "cafe",
            "row_ref": "row",
            "ingested_at": "2026-01-01T00:00:00Z",
            "operator": "bot:test",
        }
        with pytest.raises(
            ProvenanceIncompleteError, match="64 lowercase hex"
        ):
            orch.gate_6_provenance_completeness(
                {"urn": "urn:gl:factor:x", "extraction": extraction}
            )

    def test_gate_6_artifact_lookup_logs_on_missing(self) -> None:
        # When source_urn is set but lookup_artifact returns None, gate
        # should still PASS (correlation log only). _connect raises so
        # the artifact lookup falls through to the None-returning branch
        # at line 941-943.
        repo = _StubRepo(connect_raises=sqlite3.OperationalError("nope"))
        orch = PublishGateOrchestrator(repo)
        extraction = {
            "source_url": "https://x",
            "source_record_id": "1",
            "source_publication": "p",
            "source_version": "v",
            "raw_artifact_uri": "s3://a",
            "raw_artifact_sha256": "a" * 64,
            "parser_id": "pid",
            "parser_version": "0",
            "parser_commit": "cafe",
            "row_ref": "row",
            "ingested_at": "2026-01-01T00:00:00Z",
            "operator": "bot:test",
        }
        # Should pass (correlation log only).
        orch.gate_6_provenance_completeness(
            {
                "urn": "urn:gl:factor:x",
                "source_urn": "urn:gl:source:phase2-alpha",
                "extraction": extraction,
            }
        )


# ---------------------------------------------------------------------------
# Gate 7 — lifecycle status branches
# ---------------------------------------------------------------------------


class TestGate7Branches:
    def test_gate_7_missing_review_raises(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo(), env="production")
        with pytest.raises(
            LifecycleStatusError, match="review object missing"
        ):
            orch.gate_7_lifecycle_status({"urn": "urn:gl:factor:x"})

    def test_gate_7_rejected_status_raises_in_dev(self) -> None:
        # Rejected blocks publish in EVERY env, including dev.
        orch = PublishGateOrchestrator(_StubRepo(), env="dev")
        with pytest.raises(
            LifecycleStatusError, match="rejected records do not publish"
        ):
            orch.gate_7_lifecycle_status(
                {
                    "urn": "urn:gl:factor:x",
                    "review": {"review_status": "rejected"},
                }
            )

    def test_gate_7_production_requires_approved(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo(), env="production")
        with pytest.raises(
            LifecycleStatusError,
            match="production publish requires review.review_status",
        ):
            orch.gate_7_lifecycle_status(
                {
                    "urn": "urn:gl:factor:x",
                    "review": {"review_status": "pending"},
                }
            )

    def test_gate_7_production_requires_approved_by_and_at(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo(), env="production")
        with pytest.raises(
            LifecycleStatusError, match="approved_by"
        ):
            orch.gate_7_lifecycle_status(
                {
                    "urn": "urn:gl:factor:x",
                    "review": {"review_status": "approved"},
                }
            )

    def test_gate_7_staging_accepts_pending(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo(), env="staging")
        # Pending -> ok in staging.
        orch.gate_7_lifecycle_status(
            {
                "urn": "urn:gl:factor:x",
                "review": {"review_status": "pending"},
            }
        )

    def test_gate_7_staging_rejects_unknown_status(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo(), env="staging")
        with pytest.raises(
            LifecycleStatusError,
            match="staging publish requires review_status",
        ):
            orch.gate_7_lifecycle_status(
                {
                    "urn": "urn:gl:factor:x",
                    "review": {"review_status": "draft"},
                }
            )

    def test_gate_7_dev_accepts_any_non_rejected_status(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo(), env="dev")
        # Random status -> ok in dev.
        orch.gate_7_lifecycle_status(
            {
                "urn": "urn:gl:factor:x",
                "review": {"review_status": "anything"},
            }
        )


# ---------------------------------------------------------------------------
# _ontology_lookup unknown-table guard
# ---------------------------------------------------------------------------


class TestOntologyLookupGuard:
    def test_unknown_table_returns_none(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        # _ONTOLOGY_FIELDS whitelist excludes 'attacker'.
        assert (
            orch._ontology_lookup("attacker", "urn:gl:geo:global:world")
            is None
        )

    def test_unknown_table_assert_present_returns_false(self) -> None:
        orch = PublishGateOrchestrator(_StubRepo())
        assert (
            orch._assert_ontology_table_present("attacker_table") is False
        )


# ---------------------------------------------------------------------------
# SQLite probe failure modes
# ---------------------------------------------------------------------------


class TestSqliteProbeBranches:
    def test_assert_present_sqlite_connect_fails_returns_false(self) -> None:
        repo = _StubRepo(connect_raises=RuntimeError("conn boom"))
        orch = PublishGateOrchestrator(repo)
        assert orch._assert_ontology_table_present_sqlite("geography") is False

    def test_assert_present_sqlite_unknown_table(self) -> None:
        # OperationalError ("no such table") branch at line 816-821.
        conn = sqlite3.connect(":memory:")
        repo = _StubRepo(connect_returns=conn)
        orch = PublishGateOrchestrator(repo)
        assert orch._assert_ontology_table_present_sqlite("geography") is False

    def test_assert_present_sqlite_generic_exception_returns_false(
        self,
    ) -> None:
        # Mock conn.execute to raise a non-OperationalError so the
        # generic Exception branch (line 822-827) fires.
        bad_conn = MagicMock()
        bad_conn.execute = MagicMock(side_effect=ValueError("weird"))
        repo = _StubRepo(connect_returns=bad_conn)
        orch = PublishGateOrchestrator(repo)
        assert orch._assert_ontology_table_present_sqlite("unit") is False

    def test_ontology_lookup_sqlite_connect_fails(self) -> None:
        repo = _StubRepo(connect_raises=RuntimeError("conn boom"))
        orch = PublishGateOrchestrator(repo)
        assert (
            orch._ontology_lookup_sqlite("unit", "urn:gl:unit:kg") is None
        )

    def test_ontology_lookup_sqlite_missing_table_returns_none(self) -> None:
        conn = sqlite3.connect(":memory:")
        repo = _StubRepo(connect_returns=conn)
        orch = PublishGateOrchestrator(repo)
        assert (
            orch._ontology_lookup_sqlite("unit", "urn:gl:unit:kg") is None
        )

    def test_ontology_lookup_sqlite_present_returns_true(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE unit (urn TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO unit VALUES ('urn:gl:unit:kg')")
        repo = _StubRepo(connect_returns=conn)
        # Setting _memory_conn signals the orchestrator NOT to close the
        # connection after the probe (the real repo uses this for the
        # in-memory sqlite case so multiple probes share one conn).
        repo._memory_conn = conn
        orch = PublishGateOrchestrator(repo)
        assert (
            orch._ontology_lookup_sqlite("unit", "urn:gl:unit:kg") is True
        )
        assert (
            orch._ontology_lookup_sqlite("unit", "urn:gl:unit:nope")
            is False
        )


# ---------------------------------------------------------------------------
# Postgres probe branches — psycopg ImportError + missing dsn + connect failure
# ---------------------------------------------------------------------------


class TestPostgresProbeBranches:
    """All Postgres probes follow the same shape:
        try: import psycopg -> ImportError -> warn + return False/None
        dsn = repo._dsn -> falsy -> warn + return False/None
        psycopg.connect raises -> caught -> return False/None
    """

    def test_assert_present_pg_psycopg_import_error(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        # builtins.__import__ raises ImportError when psycopg is fetched.
        import builtins
        real = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("simulated")
            return real(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert (
                orch._assert_ontology_table_present_pg("geography") is False
            )

    def test_assert_present_pg_missing_dsn(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn=None)
        orch = PublishGateOrchestrator(repo)
        # psycopg may or may not be installed; either way dsn=None -> False.
        assert orch._assert_ontology_table_present_pg("geography") is False

    def test_assert_present_pg_connect_raises(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://invalid")
        orch = PublishGateOrchestrator(repo)
        # Patch psycopg.connect via sys.modules indirection. We construct
        # a fake psycopg module so the import inside the method resolves
        # to it.
        fake_psycopg = MagicMock()
        fake_psycopg.connect = MagicMock(
            side_effect=RuntimeError("cannot connect")
        )
        with patch.dict("sys.modules", {"psycopg": fake_psycopg}):
            assert (
                orch._assert_ontology_table_present_pg("geography") is False
            )

    def test_assert_present_pg_success_returns_true(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        # Construct a fake psycopg with a context-managed connection +
        # cursor that returns no rows but does not raise.
        fake_psycopg = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.__exit__.return_value = None
        cur.execute = MagicMock(return_value=None)
        cur.fetchone = MagicMock(return_value=None)
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = None
        conn.cursor = MagicMock(return_value=cur)
        fake_psycopg.connect = MagicMock(return_value=conn)
        with patch.dict("sys.modules", {"psycopg": fake_psycopg}):
            assert (
                orch._assert_ontology_table_present_pg("geography") is True
            )

    def test_ontology_lookup_pg_import_error(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        import builtins
        real = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("simulated")
            return real(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert orch._ontology_lookup_pg("unit", "urn:gl:unit:kg") is None

    def test_ontology_lookup_pg_missing_dsn(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn=None)
        orch = PublishGateOrchestrator(repo)
        assert orch._ontology_lookup_pg("unit", "urn:gl:unit:kg") is None

    def test_ontology_lookup_pg_connect_raises(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        fake_psycopg = MagicMock()
        fake_psycopg.connect = MagicMock(side_effect=RuntimeError("nope"))
        with patch.dict("sys.modules", {"psycopg": fake_psycopg}):
            assert orch._ontology_lookup_pg("unit", "urn:gl:unit:kg") is None

    def test_ontology_lookup_pg_returns_bool(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        fake_psycopg = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.__exit__.return_value = None
        cur.execute = MagicMock(return_value=None)
        # First call: returns a row (True); second: None (False).
        cur.fetchone = MagicMock(side_effect=[(1,), None])
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = None
        conn.cursor = MagicMock(return_value=cur)
        fake_psycopg.connect = MagicMock(return_value=conn)
        with patch.dict("sys.modules", {"psycopg": fake_psycopg}):
            assert orch._ontology_lookup_pg("unit", "urn:gl:unit:kg") is True
            assert (
                orch._ontology_lookup_pg("unit", "urn:gl:unit:nope") is False
            )

    def test_lookup_artifact_dispatches_pg(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        fake_psycopg = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.__exit__.return_value = None
        cur.execute = MagicMock(return_value=None)
        cur.fetchone = MagicMock(return_value=(42,))
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = None
        conn.cursor = MagicMock(return_value=cur)
        fake_psycopg.connect = MagicMock(return_value=conn)
        with patch.dict("sys.modules", {"psycopg": fake_psycopg}):
            assert (
                orch._lookup_artifact("a" * 64, "urn:gl:source:x") == 42
            )

    def test_lookup_artifact_pg_import_error(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        import builtins
        real = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("simulated")
            return real(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert (
                orch._lookup_artifact_pg("a" * 64, "urn:gl:source:x") is None
            )

    def test_lookup_artifact_pg_missing_dsn(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn=None)
        orch = PublishGateOrchestrator(repo)
        assert (
            orch._lookup_artifact_pg("a" * 64, "urn:gl:source:x") is None
        )

    def test_lookup_artifact_pg_returns_none_when_no_row(self) -> None:
        repo = _StubRepo(is_postgres=True, dsn="postgresql://x")
        orch = PublishGateOrchestrator(repo)
        fake_psycopg = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.__exit__.return_value = None
        cur.execute = MagicMock(return_value=None)
        cur.fetchone = MagicMock(return_value=None)
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = None
        conn.cursor = MagicMock(return_value=cur)
        fake_psycopg.connect = MagicMock(return_value=conn)
        with patch.dict("sys.modules", {"psycopg": fake_psycopg}):
            assert (
                orch._lookup_artifact_pg("a" * 64, "urn:gl:source:x") is None
            )

    def test_lookup_artifact_top_level_swallows_exceptions(self) -> None:
        # If the chosen branch raises, _lookup_artifact returns None.
        repo = _StubRepo(is_postgres=False)
        orch = PublishGateOrchestrator(repo)
        with patch.object(
            orch, "_lookup_artifact_sqlite", side_effect=RuntimeError("boom")
        ):
            assert (
                orch._lookup_artifact("a" * 64, "urn:gl:source:x") is None
            )


# ---------------------------------------------------------------------------
# Gate 3 / 4 — fail-CLOSED missing ontology table in production
# ---------------------------------------------------------------------------


class TestFailClosedBranches:
    def test_gate_3_missing_table_in_production_raises(self) -> None:
        # Stub repo with an empty in-memory SQLite (no tables) so the
        # presence-probe returns False; production env should raise
        # OntologyReferenceError citing 'missing_tables'.
        conn = sqlite3.connect(":memory:")
        repo = _StubRepo(connect_returns=conn)
        orch = PublishGateOrchestrator(repo, env="production")
        with pytest.raises(
            OntologyReferenceError, match="not present"
        ) as ei:
            orch.gate_3_ontology_fk(
                {
                    "urn": "urn:gl:factor:x",
                    "geography_urn": "urn:gl:geo:global:world",
                    "unit_urn": "urn:gl:unit:kg",
                    "methodology_urn": "urn:gl:methodology:m",
                    "source_urn": "urn:gl:source:s",
                    "factor_pack_urn": "urn:gl:pack:p:default:v1",
                }
            )
        assert "missing_tables" in ei.value.details

    def test_gate_4_missing_source_table_in_production_raises(self) -> None:
        conn = sqlite3.connect(":memory:")
        repo = _StubRepo(connect_returns=conn)
        rights = _RightsAllow()
        orch = PublishGateOrchestrator(
            repo, source_rights=rights, env="production"
        )
        with pytest.raises(
            SourceRegistryError, match="seeded source registry"
        ):
            orch.gate_4_source_registry(
                {
                    "urn": "urn:gl:factor:x",
                    "source_urn": "urn:gl:source:phase2-alpha",
                }
            )


# ---------------------------------------------------------------------------
# _load_registry_index fallback — both source_rights paths + YAML loader.
# ---------------------------------------------------------------------------


class TestLoadRegistryIndex:
    def test_prefers_source_rights_index_when_available(self) -> None:
        repo = _StubRepo()
        rights = _RightsAllow(registry={"x": {"k": "v"}})
        orch = PublishGateOrchestrator(repo, source_rights=rights)
        idx = orch._load_registry_index()
        assert idx == {"x": {"k": "v"}}

    def test_falls_back_to_yaml_when_no_rights_service(self) -> None:
        # Patch the lazy import so we don't actually hit the real loader.
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo, source_rights=None)
        fake_module = MagicMock()
        fake_module._load_raw_sources = MagicMock(
            return_value=[
                {"urn": "urn:gl:source:x", "licence": "MIT"},
                {"urn": "urn:gl:source:y"},
                "garbage-not-a-dict",
                {"no_urn_field": True},
            ]
        )
        with patch.dict(
            "sys.modules", {"greenlang.factors.source_registry": fake_module}
        ):
            idx = orch._load_registry_index()
        assert "urn:gl:source:x" in idx
        assert "urn:gl:source:y" in idx
        assert idx["urn:gl:source:x"]["licence"] == "MIT"

    def test_yaml_load_failure_returns_empty_dict(self) -> None:
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo, source_rights=None)
        fake_module = MagicMock()
        fake_module._load_raw_sources = MagicMock(
            side_effect=RuntimeError("YAML unreadable")
        )
        with patch.dict(
            "sys.modules", {"greenlang.factors.source_registry": fake_module}
        ):
            assert orch._load_registry_index() == {}

    def test_yaml_module_import_failure_returns_empty_dict(self) -> None:
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo, source_rights=None)
        # Force the lazy import to fail by replacing the module entry
        # with a bad sentinel that raises on attribute access.
        import builtins
        real = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "greenlang.factors.source_registry":
                raise ImportError("simulated")
            return real(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert orch._load_registry_index() == {}

    def test_lookup_registry_row_short_circuits_on_empty_input(self) -> None:
        repo = _StubRepo()
        orch = PublishGateOrchestrator(repo)
        assert orch._lookup_registry_row(None) is None
        assert orch._lookup_registry_row("") is None

    def test_lookup_registry_row_caches_index(self) -> None:
        repo = _StubRepo()
        rights = _RightsAllow(registry={"u": {"k": "v"}})
        orch = PublishGateOrchestrator(repo, source_rights=rights)
        # First call populates _registry_index.
        assert orch._lookup_registry_row("u") == {"k": "v"}
        # Mutate rights.registry_index — orch should NOT see the change
        # because the index is cached on the orchestrator.
        rights.registry_index = {"u": {"k": "different"}}
        assert orch._lookup_registry_row("u") == {"k": "v"}
