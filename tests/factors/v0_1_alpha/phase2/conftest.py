# -*- coding: utf-8 -*-
"""Phase 2 / WS8 — shared fixtures for publish-gate orchestrator tests.

Provides:
  * ``seeded_repo`` — an :class:`AlphaFactorRepository` constructed with
    secure-by-default settings (``publish_env='production'``) AND with
    every ontology / source-registry table the orchestrator probes
    pre-seeded. Tests that exercise Phase 2 publish enforcement should
    request this fixture; their constructor calls remain default
    (no ``publish_env`` override) so the orchestrator is the real
    secure-by-default surface under test.
  * ``seeded_repo_factory`` — a callable that returns a freshly seeded
    :class:`AlphaFactorRepository` with the orchestrator wired to a
    synthetic SourceRightsService. Useful for tests that need multiple
    repos in the same test (e.g. cross-database parity).

CTO P0/P1 fix (2026-04-27): Phase 2 publish gates are mandatory by
default. ``publish_env='legacy'`` is the explicit opt-out for tests
that exercise the legacy :class:`AlphaProvenanceGate` in isolation.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Callable, Dict, Generator

import pytest


# ---------------------------------------------------------------------------
# Synthetic ontology + source registry — shared canonical URNs the seeded
# fixtures use. Tests can reference these constants when building factor
# records so the records line up with what the seeder wrote.
# ---------------------------------------------------------------------------


SEEDED_GEOGRAPHY_URN = "urn:gl:geo:global:world"
SEEDED_UNIT_URN = "urn:gl:unit:kgco2e/kwh"
SEEDED_METHODOLOGY_URN = "urn:gl:methodology:phase2-default"
SEEDED_ACTIVITY_URN = "urn:gl:activity:phase2:electricity-grid"
SEEDED_SOURCE_URN = "urn:gl:source:phase2-alpha"
SEEDED_PACK_URN = "urn:gl:pack:phase2-alpha:default:v1"
SEEDED_LICENCE = "CC-BY-4.0"
SEEDED_VALID_SHA256 = "abcdef01" * 8  # 64 lowercase hex chars


def _seed_ontology_tables(conn: sqlite3.Connection) -> None:
    """Create + populate every ontology table the orchestrator probes.

    Tables created (mirrors the V502 / V506 Postgres DDL on SQLite):

      * geography
      * unit
      * methodology
      * activity (V506 additive)
      * source (gate 4 fail-closed probe)
      * factor_pack
    """
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS geography ("
        " urn TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS unit ("
        " urn TEXT PRIMARY KEY, symbol TEXT NOT NULL, dimension TEXT"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS methodology ("
        " urn TEXT PRIMARY KEY, name TEXT NOT NULL, framework TEXT"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS activity ("
        " urn TEXT PRIMARY KEY, name TEXT NOT NULL"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS source ("
        " urn TEXT PRIMARY KEY, source_id TEXT NOT NULL,"
        " licence TEXT, alpha_v0_1 INTEGER DEFAULT 0"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS factor_pack ("
        " urn TEXT PRIMARY KEY, name TEXT"
        ")"
    )
    # Seed the canonical rows referenced by the SEEDED_* URN constants
    # above. Idempotent — INSERT OR IGNORE means re-running the seeder
    # is safe.
    cur.execute(
        "INSERT OR IGNORE INTO geography (urn, type, name) VALUES (?, ?, ?)",
        (SEEDED_GEOGRAPHY_URN, "global", "World"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO unit (urn, symbol, dimension) VALUES (?, ?, ?)",
        (SEEDED_UNIT_URN, "kgCO2e/kWh", "composite_climate"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO methodology (urn, name, framework) "
        "VALUES (?, ?, ?)",
        (SEEDED_METHODOLOGY_URN, "Phase 2 default methodology", "test"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO activity (urn, name) VALUES (?, ?)",
        (SEEDED_ACTIVITY_URN, "Electricity grid"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO source (urn, source_id, licence, alpha_v0_1) "
        "VALUES (?, ?, ?, ?)",
        (SEEDED_SOURCE_URN, "phase2-alpha", SEEDED_LICENCE, 1),
    )
    cur.execute(
        "INSERT OR IGNORE INTO factor_pack (urn, name) VALUES (?, ?)",
        (SEEDED_PACK_URN, "Phase 2 default pack"),
    )


# ---------------------------------------------------------------------------
# Synthetic SourceRightsService surface
# ---------------------------------------------------------------------------


class _PhaseTwoFakeRights:
    """Minimal SourceRightsService surface used by phase2 fixtures.

    The orchestrator's gate 4 / gate 5 prefer ``self.registry_index``
    over the YAML loader when the rights service is wired, so injecting
    this fake gives tests a hermetic registry with the SEEDED_SOURCE_URN
    pin already in place.
    """

    def __init__(self, registry_index: Dict[str, Dict[str, Any]]) -> None:
        self.registry_index = dict(registry_index)

    def check_record_licence_matches_registry(
        self, source_urn: str, record_licence: Any
    ):
        class _D:
            denied = False
            reason = "phase2 fixture: licence ok"

        return _D()

    def check_ingestion_allowed(self, source_urn: str):
        class _D:
            denied = False
            reason = "phase2 fixture: ingestion allowed"

        return _D()


def _build_phase2_fake_rights() -> _PhaseTwoFakeRights:
    """Return a SourceRightsService with the SEEDED_SOURCE_URN pinned."""
    return _PhaseTwoFakeRights(
        {
            SEEDED_SOURCE_URN: {
                "urn": SEEDED_SOURCE_URN,
                "source_id": "phase2-alpha",
                "licence": SEEDED_LICENCE,
                "alpha_v0_1": True,
                "licence_class": "community_open",
                "redistribution_class": "attribution_required",
            }
        }
    )


# ---------------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def phase2_fake_rights() -> _PhaseTwoFakeRights:
    """Synthetic SourceRightsService with the SEEDED_SOURCE_URN pinned."""
    return _build_phase2_fake_rights()


@pytest.fixture()
def seeded_repo(
    phase2_fake_rights: _PhaseTwoFakeRights,
) -> Generator[Any, None, None]:
    """An :class:`AlphaFactorRepository` with secure defaults + seeded ontology.

    * ``publish_env='production'`` (the default; spelled out for clarity).
    * Every ontology table the orchestrator's gate 3 probes is created
      AND populated with the SEEDED_* URN constants.
    * The orchestrator is pre-built with ``phase2_fake_rights`` so gate 4
      / gate 5 use the synthetic registry instead of the YAML loader.
    """
    # Lazy import — the orchestrator class lives in greenlang.factors.quality
    # and pulling it at module-import time slows pytest collection.
    from greenlang.factors.quality.publish_gates import PublishGateOrchestrator
    from greenlang.factors.repositories import AlphaFactorRepository

    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:", publish_env="production"
    )
    conn = repo._connect()  # type: ignore[attr-defined]
    _seed_ontology_tables(conn)

    # Pre-build the orchestrator wired to the fake rights service so
    # the registry pin matches the SEEDED_SOURCE_URN.
    orchestrator = PublishGateOrchestrator(
        repo, source_rights=phase2_fake_rights, env="production"
    )
    repo._publish_orchestrator = orchestrator  # type: ignore[attr-defined]

    # Pre-register the artifact so gate 6's correlation log fires "found"
    # for any record using SEEDED_VALID_SHA256.
    repo.register_artifact(
        sha256=SEEDED_VALID_SHA256,
        source_urn=SEEDED_SOURCE_URN,
        version="2024.1",
        uri="s3://phase2-fixture/2024.1/file.pdf",
    )

    yield repo
    repo.close()


@pytest.fixture()
def seeded_repo_factory(
    phase2_fake_rights: _PhaseTwoFakeRights,
) -> Callable[..., Any]:
    """Factory that returns a freshly seeded :class:`AlphaFactorRepository`.

    Useful for tests that need multiple repos in one test (e.g. cross-
    backend parity). Caller is responsible for ``.close()``.
    """
    from greenlang.factors.quality.publish_gates import PublishGateOrchestrator
    from greenlang.factors.repositories import AlphaFactorRepository

    built: list = []

    def _factory(*, env: str = "production", dsn: str = "sqlite:///:memory:") -> Any:
        repo = AlphaFactorRepository(dsn=dsn, publish_env=env)
        conn = repo._connect()  # type: ignore[attr-defined]
        _seed_ontology_tables(conn)
        if env != "legacy":
            orchestrator = PublishGateOrchestrator(
                repo, source_rights=phase2_fake_rights, env=env,
            )
            repo._publish_orchestrator = orchestrator  # type: ignore[attr-defined]
        repo.register_artifact(
            sha256=SEEDED_VALID_SHA256,
            source_urn=SEEDED_SOURCE_URN,
            version="2024.1",
            uri="s3://phase2-fixture/2024.1/file.pdf",
        )
        built.append(repo)
        return repo

    yield _factory

    for repo in built:
        try:
            repo.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "SEEDED_GEOGRAPHY_URN",
    "SEEDED_UNIT_URN",
    "SEEDED_METHODOLOGY_URN",
    "SEEDED_ACTIVITY_URN",
    "SEEDED_SOURCE_URN",
    "SEEDED_PACK_URN",
    "SEEDED_LICENCE",
    "SEEDED_VALID_SHA256",
    "phase2_fake_rights",
    "seeded_repo",
    "seeded_repo_factory",
]
