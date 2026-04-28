# -*- coding: utf-8 -*-
"""Phase 2 / WS8 acceptance test — end-to-end publish pipeline.

Builds a fresh in-memory sqlite repository, seeds the ontology + source
registry + source_artifacts row, publishes a fully-valid factor through
the seven-gate orchestrator, and confirms:

  * publish() succeeds.
  * find_by_urn / find_by_alias / find_by_methodology all resolve the row.
  * a ``factor_publish`` row is appended to ``changelog_events``.

This is the convergence test the CTO Phase 2 brief calls out as the
"all gates pass + audit fires + queries work" smoke that signals WS8 is
done.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pytest

from greenlang.factors.repositories import AlphaFactorRepository


_VALID_SHA256 = "feedface" * 8  # 64 lowercase hex chars
_TEST_SOURCE_URN = "urn:gl:source:e2e-alpha"
_TEST_LICENCE = "CC-BY-4.0"


# ---------------------------------------------------------------------------
# Fakes — a SourceRightsService surface narrow enough for the gates.
# ---------------------------------------------------------------------------


class _FakeRights:
    def __init__(self, registry_index: Dict[str, Dict[str, Any]]) -> None:
        self.registry_index = registry_index

    def check_record_licence_matches_registry(self, source_urn: str, record_licence: Any):
        class _D:
            denied = False
            reason = "stub: licence ok"
        return _D()

    def check_ingestion_allowed(self, source_urn: str):
        class _D:
            denied = False
            reason = "stub: ingestion allowed"
        return _D()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_ontology(conn: sqlite3.Connection) -> None:
    """Seed every ontology table the seven-gate orchestrator probes.

    Phase 2 CTO P0 (2026-04-27): production / staging fail CLOSED on
    missing ontology tables, including ``source`` and ``factor_pack``.
    The fixture seeds ALL tables the orchestrator probes so the e2e
    happy-path runs end-to-end without a table-missing short-circuit.
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
    cur.execute(
        "INSERT OR IGNORE INTO geography (urn, type, name) VALUES (?, ?, ?)",
        ("urn:gl:geo:country:in", "country", "India"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO unit (urn, symbol, dimension) VALUES (?, ?, ?)",
        ("urn:gl:unit:kgco2e/kwh", "kgCO2e/kWh", "composite_climate"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO methodology (urn, name, framework) VALUES (?, ?, ?)",
        ("urn:gl:methodology:e2e-tier-1", "E2E Tier 1", "test"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO source (urn, source_id, licence, alpha_v0_1) "
        "VALUES (?, ?, ?, ?)",
        (_TEST_SOURCE_URN, "e2e-alpha", _TEST_LICENCE, 1),
    )
    cur.execute(
        "INSERT OR IGNORE INTO factor_pack (urn, name) VALUES (?, ?)",
        ("urn:gl:pack:e2e-alpha:default:v1", "E2E Alpha Default Pack"),
    )


def _build_record(*, urn: str, alias: str, methodology: str) -> Dict[str, Any]:
    return {
        "urn": urn,
        "factor_id_alias": alias,
        "source_urn": _TEST_SOURCE_URN,
        "factor_pack_urn": "urn:gl:pack:e2e-alpha:default:v1",
        "name": "E2E test factor",
        "description": (
            "Synthetic e2e factor used by the Phase 2 / WS8 acceptance "
            "test. Boundary excludes upstream extraction."
        ),
        "category": "grid_intensity",
        "value": 0.708,
        "unit_urn": "urn:gl:unit:kgco2e/kwh",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": "urn:gl:geo:country:in",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "resolution": "annual",
        "methodology_urn": methodology,
        "boundary": "Boundary excludes upstream extraction and distribution losses.",
        "licence": _TEST_LICENCE,
        "citations": [{"type": "url", "value": "https://example.test/e2e"}],
        "published_at": "2026-04-27T12:00:00Z",
        "extraction": {
            "source_url": "https://example.test/e2e",
            "source_record_id": "row=1",
            "source_publication": "E2E test publication",
            "source_version": "2024.1",
            "raw_artifact_uri": "s3://e2e-bucket/2024.1/file.pdf",
            "raw_artifact_sha256": _VALID_SHA256,
            "parser_id": "greenlang.factors.ingestion.parsers.e2e",
            "parser_version": "0.1.0",
            "parser_commit": "feedfacecafe",
            "row_ref": "Sheet=A;Row=1",
            "ingested_at": "2026-04-27T11:00:00Z",
            "operator": "bot:parser_e2e",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:e2e@greenlang.io",
            "reviewed_at": "2026-04-27T11:30:00Z",
            "approved_by": "human:e2e@greenlang.io",
            "approved_at": "2026-04-27T11:31:00Z",
        },
    }


@pytest.fixture()
def fake_rights() -> _FakeRights:
    return _FakeRights({
        _TEST_SOURCE_URN: {
            "urn": _TEST_SOURCE_URN,
            "source_id": "e2e-alpha",
            "licence": _TEST_LICENCE,
            "alpha_v0_1": True,
            "licence_class": "community_open",
            "redistribution_class": "attribution_required",
        }
    })


# ---------------------------------------------------------------------------
# E2E acceptance test
# ---------------------------------------------------------------------------


def test_publish_pipeline_e2e_full_round_trip(
    fake_rights: _FakeRights,
) -> None:
    """End-to-end Phase 2 publish: gates pass -> row stored -> queries OK."""
    # 1. Build a fresh repo with the orchestrator opted in.
    from greenlang.factors.quality.publish_gates import PublishGateOrchestrator

    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:",
    )
    # Inject a fully-built orchestrator that uses our fake_rights so the
    # registry pin is the ``e2e-alpha`` row, not the YAML-backed registry.
    orchestrator = PublishGateOrchestrator(
        repo, source_rights=fake_rights, env="production"
    )
    repo._publish_orchestrator = orchestrator

    # 2. Seed the ontology tables that gate 3 probes.
    conn = repo._connect()
    _seed_ontology(conn)

    # 3. Pre-register the source artifact (matching sha256 + source_urn).
    artifact_pk = repo.register_artifact(
        sha256=_VALID_SHA256,
        source_urn=_TEST_SOURCE_URN,
        version="2024.1",
        uri="s3://e2e-bucket/2024.1/file.pdf",
    )
    assert artifact_pk > 0

    # 4. Publish a fully-valid factor.
    record = _build_record(
        urn="urn:gl:factor:e2e-alpha:grid:in-2024:v1",
        alias="EF:E2E:in-grid-2024:v1",
        methodology="urn:gl:methodology:e2e-tier-1",
    )
    published_urn = repo.publish(record)
    assert published_urn == record["urn"]

    # 5. Confirm round-trip queries.
    fetched = repo.get_by_urn(published_urn)
    assert fetched is not None
    assert fetched["urn"] == published_urn

    # The repo still requires explicit alias registration. The
    # orchestrator does NOT auto-register the alias (that's the parser's
    # job per Phase 2 §2.2). Register and confirm.
    repo.register_alias(published_urn, record["factor_id_alias"])
    aliased = repo.find_by_alias(record["factor_id_alias"])
    assert aliased is not None
    assert aliased["urn"] == published_urn

    by_methodology = repo.find_by_methodology(
        "urn:gl:methodology:e2e-tier-1"
    )
    assert any(r["urn"] == published_urn for r in by_methodology)

    # 6. Confirm the ``factor_publish`` changelog event was emitted.
    cur = conn.cursor()
    cur.execute(
        "SELECT event_type, subject_urn, change_class, actor "
        "FROM alpha_changelog_events_v0_1 "
        "WHERE event_type = ? AND subject_urn = ?",
        ("factor_publish", published_urn),
    )
    rows = cur.fetchall()
    assert len(rows) == 1, (
        f"Expected exactly one factor_publish changelog row for "
        f"{published_urn!r}, got {len(rows)}"
    )
    row = rows[0]
    assert row[0] == "factor_publish"
    assert row[1] == published_urn
    assert row[2] == "additive"
    # actor pulled from extraction.operator
    assert row[3] == "bot:parser_e2e"

    repo.close()


def test_publish_pipeline_orchestrator_blocks_bad_record(
    fake_rights: _FakeRights,
) -> None:
    """Negative-path acceptance: a bad record is blocked BEFORE the DB write."""
    from greenlang.factors.quality.publish_gates import (
        PublishGateOrchestrator,
        OntologyReferenceError,
    )

    repo = AlphaFactorRepository(dsn="sqlite:///:memory:")
    orchestrator = PublishGateOrchestrator(
        repo, source_rights=fake_rights, env="production"
    )
    repo._publish_orchestrator = orchestrator

    # Seed ontology.
    conn = repo._connect()
    _seed_ontology(conn)

    # Bad record: methodology_urn not seeded.
    record = _build_record(
        urn="urn:gl:factor:e2e-alpha:grid:in-bad:v1",
        alias="EF:E2E:in-bad:v1",
        methodology="urn:gl:methodology:phantom-tier",
    )
    with pytest.raises(OntologyReferenceError):
        repo.publish(record)

    # Confirm NO row was written and NO changelog event fired.
    assert repo.get_by_urn(record["urn"]) is None
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM alpha_changelog_events_v0_1 WHERE subject_urn = ?",
        (record["urn"],),
    )
    assert cur.fetchone()[0] == 0
    repo.close()
