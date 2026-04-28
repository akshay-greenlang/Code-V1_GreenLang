# -*- coding: utf-8 -*-
"""Phase 2 / WS10 - Ontology FK enforcement tests.

CTO Phase 2 brief Section 2.7, row #4:

    "test_ontology_fk_enforcement.py - invalid geography / unit /
    methodology / source / pack URN refs are rejected by the database
    foreign keys (Postgres path) and by the alpha provenance gate
    (SQLite path)."

Two layers of enforcement are exercised:

  1. **Postgres path** (skipped without ``GL_TEST_POSTGRES_DSN``) - the
     V500 DDL declares ``REFERENCES source(urn)`` /
     ``REFERENCES factor_pack(urn)`` / ``REFERENCES unit(urn)`` /
     ``REFERENCES geography(urn)`` / ``REFERENCES methodology(urn)`` on
     ``factors_v0_1.factor``. A direct INSERT carrying any non-existent
     ontology URN must raise ``psycopg.errors.ForeignKeyViolation`` (an
     :class:`IntegrityError` subclass). Deletion of a referenced
     geography row likewise fails because the V500 DDL omits an
     ``ON DELETE`` clause - the default ``NO ACTION`` is the safety
     guarantee Phase 2 relies on.

  2. **SQLite path** (always runs) - the alpha-mirrored repository
     does NOT carry pure SQL FK constraints (the catalog joins are kept
     loose for fast in-memory tests), so the equivalent enforcement
     happens at *publish* time through the
     :class:`AlphaProvenanceGate`. The schema validation step rejects
     malformed URN references before they ever reach the DB. We assert
     this happens for each of the five FK-bound URN columns.

Together, these tests close the ontology-FK acceptance gate from CTO
§2.7 #4.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytest

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGate,
    AlphaProvenanceGateError,
)
from greenlang.factors.repositories.alpha_v0_1_repository import (
    AlphaFactorRepository,
)


# ---------------------------------------------------------------------------
# Postgres opt-in marker
# ---------------------------------------------------------------------------

_PG_DSN: Optional[str] = os.environ.get("GL_TEST_POSTGRES_DSN")
_HAS_PG = bool(_PG_DSN)


# ---------------------------------------------------------------------------
# Canonical valid record (passes both schema + gate). Per-test overrides
# swap one URN at a time to a non-existent value so we can attribute the
# rejection to a single FK violation.
# ---------------------------------------------------------------------------


def _valid_record(urn_suffix: str = "fk-base") -> Dict[str, Any]:
    """Return a fully-populated v0.1 factor record with a unique URN."""
    return {
        "urn": f"urn:gl:factor:ipcc-2006-nggi:phase2:fk-{urn_suffix}:v1",
        "factor_id_alias": None,
        "source_urn": "urn:gl:source:ipcc-2006-nggi",
        "factor_pack_urn": "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1",
        "name": "Phase 2 ontology-FK fixture factor",
        "description": (
            "Synthetic factor that exercises the V500 ontology-FK "
            "constraints on geography_urn / unit_urn / methodology_urn / "
            "source_urn / factor_pack_urn."
        ),
        "category": "fuel",
        "value": 100.0,
        "unit_urn": "urn:gl:unit:kgco2e/gj",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": "urn:gl:geo:global:world",
        "vintage_start": "2024-01-01",
        "vintage_end": "2099-12-31",
        "resolution": "annual",
        "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
        "boundary": "stationary-combustion",
        "licence": "IPCC-PUBLIC",
        "citations": [
            {
                "type": "url",
                "value": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
            }
        ],
        "published_at": "2026-04-25T07:42:30+00:00",
        "extraction": {
            "source_url": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
            "source_record_id": "phase2-fk-fixture",
            "source_publication": "Phase 2 / WS10 ontology-FK acceptance test",
            "source_version": "0.1",
            "raw_artifact_uri": "s3://greenlang-factors-raw/test/phase2-fk.json",
            "raw_artifact_sha256": (
                "6ff38c51f0ffcb08b2057b90164c3f3e6b67a16bacffb27507526b4dab1271c6"
            ),
            "parser_id": "tests.factors.v0_1_alpha.phase2.fk_enforcement",
            "parser_version": "0.1.0",
            "parser_commit": "0" * 40,
            "row_ref": "phase2-fk-fixture",
            "ingested_at": "2026-04-25T07:42:30Z",
            "operator": "bot:test_ontology_fk_enforcement",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T07:42:30Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T07:42:30Z",
        },
        "tags": ["phase2", "ws10", "fk-enforcement"],
    }


# ---------------------------------------------------------------------------
# SQLite path: Schema/Gate-level enforcement
#
# The alpha repository's SQLite mirror omits cross-table FK constraints
# (the alpha catalog is small and keeps the integration test surface
# light). The equivalent guarantee is provided by the
# AlphaProvenanceGate which validates every URN matches the v0.1 JSON
# Schema's URN regexes. We validate that every FK-bound column rejects
# malformed values at gate time.
# ---------------------------------------------------------------------------


@pytest.fixture()
def gate() -> AlphaProvenanceGate:
    return AlphaProvenanceGate()


@pytest.fixture()
def sqlite_repo() -> AlphaFactorRepository:
    """In-memory SQLite repository for the always-on SQLite path.

    legacy mode — this suite tests the URN-pattern surface enforced by the
    Phase 1 :class:`AlphaProvenanceGate`; the Phase 2 ontology FK probe
    (gate_3) is covered by ``test_publish_pipeline_e2e.py`` and
    ``test_publish_default_secure.py``.
    """
    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:", publish_env="legacy"
    )
    yield repo
    repo.close()


@pytest.mark.parametrize(
    "field,bad_value",
    [
        # Each value is structurally invalid (uppercase namespace, bad
        # kind, missing version) - the schema regexes reject them.
        ("source_urn", "urn:gl:SOURCE:not-real-source"),
        ("factor_pack_urn", "urn:gl:pack:not-real-source:NOT-REAL:v1"),
        ("unit_urn", "urn:gl:UNIT:not-a-real-unit"),
        ("geography_urn", "urn:gl:geo:invalid_kind:nowhere"),
        ("methodology_urn", "urn:gl:METHODOLOGY:not-real"),
    ],
)
def test_sqlite_publish_rejects_malformed_ontology_urn(
    sqlite_repo: AlphaFactorRepository,
    field: str,
    bad_value: str,
) -> None:
    """SQLite path: gate rejects malformed FK-bound URN values.

    The AlphaProvenanceGate is invoked before the INSERT; it runs the
    JSON-Schema validator which carries pattern constraints for every
    URN-shaped field. Asserting a structurally-malformed URN is the
    closest SQLite equivalent of a Postgres FK violation - both block
    the publish.
    """
    rec = _valid_record(urn_suffix=f"sqlite-{field}")
    rec[field] = bad_value
    with pytest.raises(AlphaProvenanceGateError) as exc_info:
        sqlite_repo.publish(rec)
    # Sanity: failure list mentions either the offending field or the
    # schema-validator path that captured it.
    failures_blob = "\n".join(exc_info.value.failures).lower()
    assert (
        field in failures_blob
        or "schema" in failures_blob
        or "pattern" in failures_blob
    ), (
        f"Gate failure list did not surface offending field {field!r}. "
        f"Failures: {exc_info.value.failures}"
    )


def test_sqlite_publish_accepts_valid_ontology_urns(
    sqlite_repo: AlphaFactorRepository,
) -> None:
    """SQLite path: a fully-populated valid record passes the gate."""
    rec = _valid_record(urn_suffix="sqlite-happy")
    urn = sqlite_repo.publish(rec)
    assert urn == rec["urn"]
    fetched = sqlite_repo.get_by_urn(urn)
    assert fetched is not None
    # Round-trip the FK-bound columns to confirm the wire shape is preserved.
    for fk_col in (
        "source_urn",
        "factor_pack_urn",
        "unit_urn",
        "geography_urn",
        "methodology_urn",
    ):
        assert fetched[fk_col] == rec[fk_col], (
            f"FK-bound column {fk_col} drifted on round-trip: "
            f"got={fetched[fk_col]!r} expected={rec[fk_col]!r}"
        )


# ---------------------------------------------------------------------------
# Postgres path: REAL FK enforcement at the DB level.
#
# Skipped unless GL_TEST_POSTGRES_DSN points at a Postgres instance with
# the V500 schema applied.
# ---------------------------------------------------------------------------


pytestmark_pg = pytest.mark.skipif(
    not _HAS_PG,
    reason=(
        "Postgres FK tests skipped: GL_TEST_POSTGRES_DSN not set. "
        "Set it to a DSN with the V500 migration applied to enable."
    ),
)


def _pg_connect():
    """Lazy-import psycopg and return a connection. Skip on import failure."""
    try:
        import psycopg  # type: ignore
    except ImportError:
        pytest.skip("psycopg not installed; cannot run Postgres FK tests.")
    return psycopg.connect(_PG_DSN)


@pytest.mark.requires_postgres
@pytestmark_pg
@pytest.mark.parametrize(
    "fk_column,bad_value",
    [
        # Each bogus URN matches the column-pattern CHECK so we get a
        # *real* FK violation instead of a CHECK violation.
        ("source_urn", "urn:gl:source:does-not-exist-fixture"),
        ("factor_pack_urn",
         "urn:gl:pack:does-not-exist:default-fixture:v1"),
        ("unit_urn", "urn:gl:unit:non-existent-unit"),
        ("geography_urn", "urn:gl:geo:country:zz"),
        ("methodology_urn", "urn:gl:methodology:does-not-exist-fixture"),
    ],
)
def test_postgres_insert_violates_fk_for_each_ontology_column(
    fk_column: str, bad_value: str,
) -> None:
    """Postgres path: a direct INSERT with a non-existent ontology URN
    on any of the five FK columns raises an IntegrityError.

    The V500 DDL (deployment/database/migrations/sql/V500__factors_v0_1_canonical.sql)
    declares::

        source_urn         TEXT NOT NULL REFERENCES source(urn),
        factor_pack_urn    TEXT NOT NULL REFERENCES factor_pack(urn),
        unit_urn           TEXT NOT NULL REFERENCES unit(urn),
        geography_urn      TEXT NOT NULL REFERENCES geography(urn),
        methodology_urn    TEXT NOT NULL REFERENCES methodology(urn),

    We bypass the repository's gate path and emit a raw INSERT so we
    directly observe the FK enforcement.
    """
    import psycopg  # type: ignore

    rec = _valid_record(urn_suffix=f"pg-{fk_column}")
    rec[fk_column] = bad_value

    with _pg_connect() as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO factors_v0_1.factor "
                    " (urn, source_urn, factor_pack_urn, name, description, "
                    "  category, value, unit_urn, gwp_basis, gwp_horizon, "
                    "  geography_urn, vintage_start, vintage_end, resolution, "
                    "  methodology_urn, boundary, licence, citations, "
                    "  extraction, review, published_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)",
                    (
                        rec["urn"],
                        rec["source_urn"],
                        rec["factor_pack_urn"],
                        rec["name"],
                        rec["description"],
                        rec["category"],
                        rec["value"],
                        rec["unit_urn"],
                        rec["gwp_basis"],
                        rec["gwp_horizon"],
                        rec["geography_urn"],
                        rec["vintage_start"],
                        rec["vintage_end"],
                        rec["resolution"],
                        rec["methodology_urn"],
                        rec["boundary"],
                        rec["licence"],
                        '[{"type":"url","value":"https://example.com"}]',
                        '{}',
                        '{}',
                        rec["published_at"],
                    ),
                )
        conn.rollback()


@pytest.mark.requires_postgres
@pytestmark_pg
def test_postgres_delete_referenced_geography_blocked() -> None:
    """Postgres: DELETE on a geography that is referenced by any factor
    must fail with a foreign-key violation.

    V500 declares ``geography_urn TEXT NOT NULL REFERENCES geography(urn)``
    with no ``ON DELETE`` clause; the default is ``NO ACTION`` (the same
    behaviour as ``RESTRICT`` for committed FK checks). We pick an
    existing referenced row, attempt a DELETE, and expect the
    ForeignKeyViolation. Always rolled back so the test is non-destructive.
    """
    import psycopg  # type: ignore

    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT geography_urn FROM factors_v0_1.factor "
                "LIMIT 1"
            )
            row = cur.fetchone()
        if row is None:
            pytest.skip(
                "Postgres factor table empty; cannot exercise geography "
                "FK delete path."
            )
        geography_urn = row[0]
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "DELETE FROM factors_v0_1.geography WHERE urn = %s",
                    (geography_urn,),
                )
        conn.rollback()


@pytest.mark.requires_postgres
@pytestmark_pg
def test_postgres_insert_with_all_valid_fks_succeeds() -> None:
    """Sanity: valid record inserts cleanly when every FK resolves."""
    import psycopg  # type: ignore

    rec = _valid_record(urn_suffix="pg-happy")
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            # Quick precondition: every referenced URN must already exist
            # (seeded by V500 / WS3-WS6). If the seed has not landed in
            # this DB, skip rather than fail spuriously.
            for table, urn_val in (
                ("source", rec["source_urn"]),
                ("factor_pack", rec["factor_pack_urn"]),
                ("unit", rec["unit_urn"]),
                ("geography", rec["geography_urn"]),
                ("methodology", rec["methodology_urn"]),
            ):
                cur.execute(
                    f"SELECT 1 FROM factors_v0_1.{table} WHERE urn = %s",
                    (urn_val,),
                )
                if cur.fetchone() is None:
                    pytest.skip(
                        f"Ontology row missing from {table}: {urn_val!r}; "
                        "seed Phase 2 ontology before running this test."
                    )
            try:
                cur.execute(
                    "INSERT INTO factors_v0_1.factor "
                    " (urn, source_urn, factor_pack_urn, name, description, "
                    "  category, value, unit_urn, gwp_basis, gwp_horizon, "
                    "  geography_urn, vintage_start, vintage_end, resolution, "
                    "  methodology_urn, boundary, licence, citations, "
                    "  extraction, review, published_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)",
                    (
                        rec["urn"],
                        rec["source_urn"],
                        rec["factor_pack_urn"],
                        rec["name"],
                        rec["description"],
                        rec["category"],
                        rec["value"],
                        rec["unit_urn"],
                        rec["gwp_basis"],
                        rec["gwp_horizon"],
                        rec["geography_urn"],
                        rec["vintage_start"],
                        rec["vintage_end"],
                        rec["resolution"],
                        rec["methodology_urn"],
                        rec["boundary"],
                        rec["licence"],
                        '[{"type":"url","value":"https://example.com"}]',
                        '{"source_url":"https://example.com",'
                        '"source_record_id":"r","source_publication":"p",'
                        '"source_version":"0.1","raw_artifact_uri":"s3://x",'
                        '"raw_artifact_sha256":'
                        '"' + ("a" * 64) + '",'
                        '"parser_id":"p","parser_version":"0.1",'
                        '"parser_commit":"' + ("0" * 40) + '",'
                        '"row_ref":"r","ingested_at":"2026-01-01T00:00:00Z",'
                        '"operator":"bot:test"}',
                        '{"review_status":"approved",'
                        '"reviewer":"x","reviewed_at":"2026-01-01T00:00:00Z",'
                        '"approved_by":"x","approved_at":"2026-01-01T00:00:00Z"}',
                        rec["published_at"],
                    ),
                )
            finally:
                # Always roll back so the test is non-destructive.
                conn.rollback()
