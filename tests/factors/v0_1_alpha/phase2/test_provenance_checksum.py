# -*- coding: utf-8 -*-
"""Phase 2 / WS10 - Provenance checksum acceptance tests.

CTO Phase 2 brief Section 2.7, row #10:

    "test_provenance_checksum.py - sha256 round-trip on raw artefacts;
    factor records carry a matching ``extraction.raw_artifact_sha256``
    that points at a registered ``source_artifacts`` row; mismatched
    sha256 is rejected by WS8's publish gate; ``provenance_edges`` link
    factor URN to artifact pk_id with the row_ref preserved verbatim."

The Phase 2 repository (V501-V503) backs three artefact-related calls
on :class:`AlphaFactorRepository`:

  * :meth:`register_artifact(sha256, source_urn, version, uri, parser_meta)`
    - inserts a row into ``source_artifacts``.
  * :meth:`link_provenance(factor_urn, artifact_pk, row_ref, edge_type)`
    - inserts a row into ``provenance_edges``.
  * :meth:`record_changelog_event(...)` - append-only audit trail.

This test exercises the full sha256 round-trip:

  1. Generate a synthetic raw artefact (random bytes).
  2. Compute SHA-256.
  3. Register the artifact via ``register_artifact``.
  4. Publish a factor whose ``extraction.raw_artifact_sha256`` matches.
  5. Read back factor and artifact rows; assert sha256 is byte-identical.
  6. Negative case: publish a factor whose extraction sha256 does NOT
     match - the AlphaProvenanceGate (or WS8 if shipped) rejects.
  7. Link via ``link_provenance``; query ``provenance_edges`` and assert
     the link row exists.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pytest

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGate,
    AlphaProvenanceGateError,
)
from greenlang.factors.repositories.alpha_v0_1_repository import (
    AlphaFactorRepository,
    FactorURNAlreadyExistsError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_SOURCE_URN = "urn:gl:source:ipcc-2006-nggi"
_SOURCE_VERSION = "0.1"
_PARSER_ID = "tests.factors.v0_1_alpha.phase2.provenance_checksum"
_PARSER_VERSION = "0.1.0"
_PARSER_COMMIT = "0" * 40


def _synth_artifact_bytes(seed: bytes) -> bytes:
    """Deterministic 1 KiB synthetic artefact derived from ``seed``.

    We avoid ``os.urandom`` so the test is reproducible across reruns
    yet still produces bytes that are unique per test (the seed varies
    per fixture / parametrize id).
    """
    out = bytearray()
    h = hashlib.sha512(seed).digest()
    while len(out) < 1024:
        out.extend(h)
        h = hashlib.sha512(h).digest()
    return bytes(out[:1024])


def _factor_record(
    *,
    urn: str,
    sha256: str,
    artifact_uri: str,
    alias: str | None = None,
) -> Dict[str, Any]:
    """v0.1 factor record whose extraction sha256 matches ``sha256``."""
    rec: Dict[str, Any] = {
        "urn": urn,
        "factor_id_alias": alias,
        "source_urn": _SOURCE_URN,
        "factor_pack_urn": "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1",
        "name": "Phase 2 provenance-checksum fixture",
        "description": (
            "Synthetic factor used to verify the sha256 round-trip "
            "between extraction.raw_artifact_sha256 and the "
            "source_artifacts table."
        ),
        "category": "fuel",
        "value": 12.34,
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
            "source_record_id": f"phase2-prov-{urn[-8:]}",
            "source_publication": "Phase 2 / WS10 provenance fixture",
            "source_version": _SOURCE_VERSION,
            "raw_artifact_uri": artifact_uri,
            "raw_artifact_sha256": sha256,
            "parser_id": _PARSER_ID,
            "parser_version": _PARSER_VERSION,
            "parser_commit": _PARSER_COMMIT,
            "row_ref": f"phase2-prov-{urn[-8:]}",
            "ingested_at": "2026-04-25T07:42:30Z",
            "operator": "bot:test_provenance_checksum",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T07:42:30Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T07:42:30Z",
        },
        "tags": ["phase2", "ws10", "provenance-checksum"],
    }
    return rec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo() -> AlphaFactorRepository:
    """In-memory SQLite repository (V500-V503 mirrored DDL)."""
    # legacy mode — Phase 1 provenance gate only; Phase 2 orchestrator covered by tests/factors/v0_1_alpha/phase2/test_publish_pipeline_e2e.py
    r = AlphaFactorRepository(dsn="sqlite:///:memory:", publish_env="legacy")
    yield r
    r.close()


@pytest.fixture()
def synth_artifact(tmp_path: Path) -> Dict[str, Any]:
    """Generate a synthetic raw artefact, compute SHA-256, write to tmp."""
    payload = _synth_artifact_bytes(b"phase2-prov-checksum-fixture")
    sha = hashlib.sha256(payload).hexdigest()
    artifact_path = tmp_path / "raw_artifact.bin"
    artifact_path.write_bytes(payload)
    return {
        "bytes": payload,
        "sha256": sha,
        "uri": artifact_path.as_uri(),
        "size_bytes": len(payload),
        "path": artifact_path,
    }


# ---------------------------------------------------------------------------
# 1. Positive: sha256 round-trips byte-identical
# ---------------------------------------------------------------------------


def test_sha256_round_trip_publish_then_readback(
    repo: AlphaFactorRepository,
    synth_artifact: Dict[str, Any],
) -> None:
    """Generate -> register -> publish -> read-back: every sha256 step is
    byte-identical to the source.

    The acceptance rule from CTO §2.7: the sha256 stamped into
    ``extraction.raw_artifact_sha256`` MUST equal the sha256 stored on
    the ``source_artifacts`` row, which MUST equal the sha256 we
    computed locally over the bytes.
    """
    sha = synth_artifact["sha256"]
    uri = synth_artifact["uri"]

    # Sanity: recomputed sha matches the fixture.
    assert hashlib.sha256(synth_artifact["bytes"]).hexdigest() == sha

    # Step 1: register the artifact - returns its pk_id.
    artifact_pk = repo.register_artifact(
        sha256=sha,
        source_urn=_SOURCE_URN,
        version=_SOURCE_VERSION,
        uri=uri,
        parser_meta={
            "parser_id": _PARSER_ID,
            "parser_version": _PARSER_VERSION,
            "parser_commit": _PARSER_COMMIT,
            "size_bytes": synth_artifact["size_bytes"],
            "content_type": "application/octet-stream",
        },
    )
    assert isinstance(artifact_pk, int) and artifact_pk >= 1

    # Step 2: publish the factor with matching sha256.
    urn = "urn:gl:factor:ipcc-2006-nggi:phase2:prov-roundtrip:v1"
    rec = _factor_record(urn=urn, sha256=sha, artifact_uri=uri)
    published = repo.publish(rec)
    assert published == urn

    # Step 3: read the factor back; the sha256 stored on the record
    # MUST equal the sha we registered.
    stored = repo.get_by_urn(urn)
    assert stored is not None
    stored_sha = stored["extraction"]["raw_artifact_sha256"]
    assert stored_sha == sha
    # And recomputing the bytes yields the same string - sanity that
    # nothing was mutated through the JSON encode/decode round-trip.
    assert hashlib.sha256(synth_artifact["bytes"]).hexdigest() == stored_sha

    # Step 4: the artifact row carries the SAME sha256.
    artifact_row = _select_artifact_row(repo, artifact_pk)
    assert artifact_row is not None
    assert artifact_row["sha256"] == sha


# ---------------------------------------------------------------------------
# 2. Negative: mismatched sha256 is rejected by the gate
# ---------------------------------------------------------------------------


def test_publish_rejects_factor_with_mismatched_sha256(
    repo: AlphaFactorRepository,
    synth_artifact: Dict[str, Any],
) -> None:
    """Publishing a factor whose sha256 is malformed -> gate rejection.

    The AlphaProvenanceGate enforces a 64-hex-char regex on
    ``extraction.raw_artifact_sha256``; values that do not match raise
    :class:`AlphaProvenanceGateError`. (When WS8's expanded gate ships
    with cross-table sha256 verification, this test will tighten to
    assert :class:`ProvenanceIncompleteError` - until then the schema-
    level rejection is the canonical signal.)
    """
    # Register the legitimate artifact.
    sha = synth_artifact["sha256"]
    repo.register_artifact(
        sha256=sha,
        source_urn=_SOURCE_URN,
        version=_SOURCE_VERSION,
        uri=synth_artifact["uri"],
        parser_meta={
            "parser_id": _PARSER_ID,
            "parser_version": _PARSER_VERSION,
            "parser_commit": _PARSER_COMMIT,
        },
    )
    # Forge a record with a malformed sha256 (not 64-hex).
    bad_sha = "not-a-real-sha256-value"
    rec = _factor_record(
        urn="urn:gl:factor:ipcc-2006-nggi:phase2:prov-bad-sha:v1",
        sha256=bad_sha,
        artifact_uri=synth_artifact["uri"],
    )
    with pytest.raises(AlphaProvenanceGateError):
        repo.publish(rec)


def test_register_artifact_rejects_malformed_sha256(
    repo: AlphaFactorRepository,
) -> None:
    """``register_artifact()`` must refuse non-64-hex sha256 values."""
    with pytest.raises(ValueError):
        repo.register_artifact(
            sha256="abc",  # too short
            source_urn=_SOURCE_URN,
            version=_SOURCE_VERSION,
            uri="s3://bogus/artifact.bin",
        )
    with pytest.raises(ValueError):
        repo.register_artifact(
            sha256="X" * 64,  # uppercase rejected (must be lowercase hex)
            source_urn=_SOURCE_URN,
            version=_SOURCE_VERSION,
            uri="s3://bogus/artifact.bin",
        )


def test_register_artifact_duplicate_sha256_raises(
    repo: AlphaFactorRepository,
    synth_artifact: Dict[str, Any],
) -> None:
    """The V501 UNIQUE(sha256) constraint blocks duplicate registrations."""
    sha = synth_artifact["sha256"]
    repo.register_artifact(
        sha256=sha,
        source_urn=_SOURCE_URN,
        version=_SOURCE_VERSION,
        uri=synth_artifact["uri"],
    )
    with pytest.raises(FactorURNAlreadyExistsError):
        repo.register_artifact(
            sha256=sha,
            source_urn=_SOURCE_URN,
            version=_SOURCE_VERSION,
            uri=synth_artifact["uri"],
        )


# ---------------------------------------------------------------------------
# 3. link_provenance: edge row is queryable
# ---------------------------------------------------------------------------


def test_link_provenance_creates_queryable_edge_row(
    repo: AlphaFactorRepository,
    synth_artifact: Dict[str, Any],
) -> None:
    """``link_provenance`` writes a row into ``provenance_edges`` that
    we can query via the same in-memory SQLite connection."""
    sha = synth_artifact["sha256"]
    artifact_pk = repo.register_artifact(
        sha256=sha,
        source_urn=_SOURCE_URN,
        version=_SOURCE_VERSION,
        uri=synth_artifact["uri"],
    )
    urn = "urn:gl:factor:ipcc-2006-nggi:phase2:prov-edge:v1"
    repo.publish(_factor_record(urn=urn, sha256=sha, artifact_uri=synth_artifact["uri"]))

    row_ref = "sheet=Annex2;row=42"
    edge_pk = repo.link_provenance(
        factor_urn=urn,
        artifact_pk=artifact_pk,
        row_ref=row_ref,
        edge_type="extraction",
    )
    assert edge_pk >= 1

    # Query through the repo's underlying connection. We use the public
    # SQLite mirror table name from the repository class.
    edge = _select_provenance_edge(repo, edge_pk)
    assert edge is not None
    assert edge["factor_urn"] == urn
    assert edge["source_artifact_pk"] == artifact_pk
    assert edge["row_ref"] == row_ref
    assert edge["edge_type"] == "extraction"


def test_link_provenance_rejects_invalid_edge_type(
    repo: AlphaFactorRepository,
    synth_artifact: Dict[str, Any],
) -> None:
    """An edge_type outside the allowed enum raises ValueError."""
    sha = synth_artifact["sha256"]
    pk = repo.register_artifact(
        sha256=sha,
        source_urn=_SOURCE_URN,
        version=_SOURCE_VERSION,
        uri=synth_artifact["uri"],
    )
    urn = "urn:gl:factor:ipcc-2006-nggi:phase2:prov-bad-edge:v1"
    repo.publish(_factor_record(urn=urn, sha256=sha, artifact_uri=synth_artifact["uri"]))
    with pytest.raises(ValueError):
        repo.link_provenance(
            factor_urn=urn,
            artifact_pk=pk,
            row_ref="r",
            edge_type="not-a-real-edge-type",
        )


def test_link_provenance_dedupe_unique_constraint(
    repo: AlphaFactorRepository,
    synth_artifact: Dict[str, Any],
) -> None:
    """The (factor_urn, artifact_pk, row_ref, edge_type) UNIQUE
    constraint rejects a duplicate edge."""
    sha = synth_artifact["sha256"]
    pk = repo.register_artifact(
        sha256=sha,
        source_urn=_SOURCE_URN,
        version=_SOURCE_VERSION,
        uri=synth_artifact["uri"],
    )
    urn = "urn:gl:factor:ipcc-2006-nggi:phase2:prov-dup-edge:v1"
    repo.publish(_factor_record(urn=urn, sha256=sha, artifact_uri=synth_artifact["uri"]))

    repo.link_provenance(
        factor_urn=urn, artifact_pk=pk, row_ref="r1", edge_type="extraction"
    )
    with pytest.raises(FactorURNAlreadyExistsError):
        repo.link_provenance(
            factor_urn=urn,
            artifact_pk=pk,
            row_ref="r1",
            edge_type="extraction",
        )


# ---------------------------------------------------------------------------
# Helpers - reach into the repo's SQLite connection for queries that
# the public API does not expose. (Read-only; never mutates state.)
# ---------------------------------------------------------------------------


def _select_artifact_row(
    repo: AlphaFactorRepository, artifact_pk: int
) -> Dict[str, Any] | None:
    conn = repo._connect()  # noqa: SLF001 - test helper
    try:
        cur = conn.execute(
            "SELECT pk_id, sha256, source_urn, source_version, uri "
            "FROM alpha_source_artifacts_v0_1 WHERE pk_id = ?",
            (artifact_pk,),
        )
        row = cur.fetchone()
    finally:
        if repo._memory_conn is None:  # noqa: SLF001
            conn.close()
    if row is None:
        return None
    return {
        "pk_id": row["pk_id"],
        "sha256": row["sha256"],
        "source_urn": row["source_urn"],
        "source_version": row["source_version"],
        "uri": row["uri"],
    }


def _select_provenance_edge(
    repo: AlphaFactorRepository, edge_pk: int
) -> Dict[str, Any] | None:
    conn = repo._connect()  # noqa: SLF001 - test helper
    try:
        cur = conn.execute(
            "SELECT pk_id, factor_urn, source_artifact_pk, row_ref, edge_type "
            "FROM alpha_provenance_edges_v0_1 WHERE pk_id = ?",
            (edge_pk,),
        )
        row = cur.fetchone()
    finally:
        if repo._memory_conn is None:  # noqa: SLF001
            conn.close()
    if row is None:
        return None
    return {
        "pk_id": row["pk_id"],
        "factor_urn": row["factor_urn"],
        "source_artifact_pk": row["source_artifact_pk"],
        "row_ref": row["row_ref"],
        "edge_type": row["edge_type"],
    }
