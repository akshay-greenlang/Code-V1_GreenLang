# -*- coding: utf-8 -*-
"""Tests for the production candidate-source loader in
:mod:`greenlang.factors.resolution.engine`.

Replaces the old empty placeholder with a 3-tier loader (Postgres ➜
file-backed ➜ ConfigurationError).  These tests verify each tier is
selected under the right conditions and that misconfigured pods fail
loudly instead of silently returning ``ResolutionError``.
"""

from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.exceptions import ConfigurationError
from greenlang.factors.resolution import ResolutionEngine, ResolutionRequest
from greenlang.factors.resolution import engine as engine_mod


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _request() -> ResolutionRequest:
    return ResolutionRequest(
        activity="diesel combustion stationary",
        method_profile=MethodProfile.CORPORATE_SCOPE1,
        jurisdiction="US",
    )


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Strip the DB-related env vars so tests start from a clean slate."""
    for var in ("DATABASE_URL", "GL_FACTORS_PG_DSN"):
        monkeypatch.delenv(var, raising=False)
    yield


# ----------------------------------------------------------------------------
# Tier 3: ConfigurationError when neither backend is available
# ----------------------------------------------------------------------------


class TestNoSourceConfigured:
    """When no backend is available, .resolve() must raise ConfigurationError."""

    def test_unconfigured_source_raises_on_first_call(self):
        # Call the raw helper — it must raise immediately.
        with pytest.raises(ConfigurationError) as exc_info:
            engine_mod._unconfigured_candidate_source(_request(), "global_default")
        assert "no candidate source configured" in str(exc_info.value).lower()

    def test_engine_raises_configuration_error_when_no_source(self, monkeypatch):
        """When both pg + file loaders return None, the engine should
        wire :func:`_unconfigured_candidate_source` and raise on resolve()."""
        monkeypatch.setattr(engine_mod, "_build_pg_candidate_source", lambda: None)
        monkeypatch.setattr(engine_mod, "_build_file_candidate_source", lambda: None)

        engine = ResolutionEngine()
        with pytest.raises(ConfigurationError):
            engine.resolve(_request())


# ----------------------------------------------------------------------------
# Tier 2: file-backed source when DATABASE_URL is unset
# ----------------------------------------------------------------------------


class TestFileBackedSourceWhenNoDsn:
    """Without DATABASE_URL, the engine falls back to the built-in DB."""

    def test_build_default_source_uses_file_backed_loader(self, monkeypatch):
        # No DATABASE_URL set (autouse fixture clears it).  pg loader
        # should be skipped; file loader should win.
        sentinel = object()

        def _fake_pg():
            return None  # simulates "no DATABASE_URL"

        def _fake_file():
            return sentinel

        monkeypatch.setattr(engine_mod, "_build_pg_candidate_source", _fake_pg)
        monkeypatch.setattr(engine_mod, "_build_file_candidate_source", _fake_file)

        source = engine_mod.build_default_candidate_source()
        assert source is sentinel

    def test_engine_resolves_via_file_backed_source(self, monkeypatch):
        """End-to-end: with no DATABASE_URL, the engine picks a US diesel
        factor from the built-in DB and returns a ResolvedFactor."""
        # Force pg path off so we know we're going through file backend.
        monkeypatch.setattr(engine_mod, "_build_pg_candidate_source", lambda: None)

        engine = ResolutionEngine()
        resolved = engine.resolve(_request())

        assert resolved.chosen_factor_id
        assert resolved.method_profile == "corporate_scope1"
        assert resolved.fallback_rank in {1, 2, 3, 4, 5, 6, 7}


# ----------------------------------------------------------------------------
# Tier 1: pg-backed source when DATABASE_URL is set
# ----------------------------------------------------------------------------


class TestPgBackedSource:
    """With DATABASE_URL set, the pg loader is preferred (mocked)."""

    def test_build_default_source_prefers_pg_when_dsn_set(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://fake/whatever")
        pg_sentinel = object()
        file_sentinel = object()

        called = {"file": False}

        def _fake_file():
            called["file"] = True
            return file_sentinel

        monkeypatch.setattr(
            engine_mod, "_build_pg_candidate_source", lambda: pg_sentinel,
        )
        monkeypatch.setattr(
            engine_mod, "_build_file_candidate_source", _fake_file,
        )

        source = engine_mod.build_default_candidate_source()
        assert source is pg_sentinel
        # File loader must NOT be consulted when pg succeeded.
        assert called["file"] is False

    def test_pg_loader_returns_none_when_repo_init_fails(self, monkeypatch):
        """If psycopg cannot connect, the pg loader returns None and we
        fall back to file (does not crash)."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://invalid/host:9999")

        # Patch the pg repo so its init blows up cleanly.
        class _Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("simulated psycopg failure")

        import greenlang.factors.catalog_repository_pg as pg_mod

        monkeypatch.setattr(pg_mod, "PostgresFactorCatalogRepository", _Boom)

        result = engine_mod._build_pg_candidate_source()
        assert result is None  # silent fallback, with WARNING log

    def test_pg_loader_returns_none_when_no_default_edition(self, monkeypatch):
        """If the schema is empty (no stable edition), fall back to file."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")

        class _EmptyRepo:
            def __init__(self, *a, **kw):
                pass

            def get_default_edition_id(self):
                return ""

        import greenlang.factors.catalog_repository_pg as pg_mod

        monkeypatch.setattr(pg_mod, "PostgresFactorCatalogRepository", _EmptyRepo)

        result = engine_mod._build_pg_candidate_source()
        assert result is None

    def test_engine_uses_pg_source_when_dsn_set(self, monkeypatch):
        """End-to-end: a mocked pg repo is wrapped + invoked by the engine."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")

        # Build a fake repo whose list_factors yields a single canonical
        # diesel record from the built-in DB so the cascade can succeed.
        from greenlang.data.emission_factor_database import EmissionFactorDatabase

        db = EmissionFactorDatabase(enable_cache=False)
        diesel = next(
            f for f in db.factors.values()
            if "diesel" in (getattr(f, "fuel_type", None) or "").lower()
        )

        seen_calls = {"count": 0}

        class _FakePgRepo:
            def __init__(self, *a, **kw):
                pass

            def get_default_edition_id(self):
                return "pg-edition-test"

            def list_factors(self, edition_id, **kwargs):
                seen_calls["count"] += 1
                return [diesel], 1

        import greenlang.factors.catalog_repository_pg as pg_mod

        monkeypatch.setattr(pg_mod, "PostgresFactorCatalogRepository", _FakePgRepo)
        # Force the file loader off so we can assert pg path is used.
        monkeypatch.setattr(engine_mod, "_build_file_candidate_source", lambda: None)

        engine = ResolutionEngine()
        resolved = engine.resolve(_request())

        assert resolved.chosen_factor_id == diesel.factor_id
        # The wrapped pg source must have been called at least once.
        assert seen_calls["count"] >= 1


# ----------------------------------------------------------------------------
# Explicit override always wins
# ----------------------------------------------------------------------------


class TestExplicitCandidateSourceOverride:
    """Passing candidate_source= must bypass the loader entirely."""

    def test_explicit_source_overrides_default_loader(self, monkeypatch):
        # Prove the loader is NOT consulted when a candidate_source is
        # passed.
        called = {"loader": False}

        def _spy_loader():
            called["loader"] = True
            return engine_mod._unconfigured_candidate_source

        monkeypatch.setattr(
            engine_mod, "build_default_candidate_source", _spy_loader,
        )

        def _stub_source(_req, _label):
            return []

        ResolutionEngine(candidate_source=_stub_source)
        assert called["loader"] is False
