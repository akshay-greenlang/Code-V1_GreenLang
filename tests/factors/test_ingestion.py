# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.ingestion (normalizer, parser_harness, artifacts, sqlite_metadata)."""

from __future__ import annotations

from pathlib import Path

from greenlang.factors.ingestion.normalizer import CanonicalNormalizer
from greenlang.factors.ingestion.parser_harness import ParserContext, ParserResult, run_parser
from greenlang.factors.ingestion.artifacts import LocalArtifactStore


def _make_minimal_record_dict():
    return {
        "factor_id": "EF:TEST:fuel:2024:v1",
        "fuel_type": "test_fuel",
        "unit": "kg",
        "geography": "US",
        "geography_level": "country",
        "vectors": {"CO2": 1.5, "CH4": 0.001, "N2O": 0.0001},
        "gwp_100yr": {"gwp_set": "IPCC_AR6_100", "CH4_gwp": 28, "N2O_gwp": 273},
        "scope": "1",
        "boundary": "combustion",
        "provenance": {
            "source_org": "Test",
            "source_publication": "Test Pub",
            "source_year": 2024,
            "methodology": "IPCC_Tier_1",
            "source_url": "",
            "version": "v1",
        },
        "valid_from": "2024-01-01",
        "uncertainty_95ci": 0.1,
        "dqs": {
            "temporal": 4,
            "geographical": 4,
            "technological": 3,
            "representativeness": 3,
            "methodological": 4,
        },
        "license_info": {
            "license": "test-license",
            "redistribution_allowed": True,
            "commercial_use_allowed": True,
            "attribution_required": False,
        },
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "created_by": "test",
        "tags": ["test"],
    }


def test_canonical_normalizer():
    norm = CanonicalNormalizer()
    row = _make_minimal_record_dict()
    rec, lineage = norm.normalize(row, {"raw": "fragment"})
    assert rec.factor_id == "EF:TEST:fuel:2024:v1"


def test_normalizer_preserves_lineage():
    norm = CanonicalNormalizer()
    row = _make_minimal_record_dict()
    rec, lineage = norm.normalize(row, {"source": "test_raw"})
    assert lineage["raw"]["source"] == "test_raw"
    assert lineage["normalized_factor_id"] == rec.factor_id


def test_parser_harness_success():
    ctx = ParserContext(artifact_id="a1", source_id="test", parser_id="dummy")

    def dummy_parser(ctx, raw):
        return ParserResult(status="ok", rows=[{"x": 1}])

    result = run_parser(ctx, b"data", dummy_parser)
    assert result.status == "ok"
    assert len(result.rows) == 1


def test_parser_harness_failure():
    ctx = ParserContext(artifact_id="a2", source_id="test", parser_id="bad_parser")

    def bad_parser(ctx, raw):
        raise ValueError("parse error")

    result = run_parser(ctx, b"data", bad_parser)
    assert result.status == "failed"
    assert "parse error" in result.error


def test_artifacts_store_and_retrieve(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    data = b"test emission factor raw data"
    artifact = store.put_bytes(data, "test_source")
    assert artifact.sha256
    assert artifact.bytes_size == len(data)
    assert artifact.storage_uri.startswith("file://")
    # Verify the file exists
    from urllib.parse import urlparse
    parsed = urlparse(artifact.storage_uri)
    stored_path = Path(parsed.path.lstrip("/"))
    # On Windows, strip leading slash from file:///C:/...
    if not stored_path.exists():
        stored_path = Path(artifact.storage_uri.replace("file:///", ""))
    assert stored_path.read_bytes() == data


def test_sqlite_metadata_roundtrip(tmp_path):
    import sqlite3
    from greenlang.factors.catalog_repository import _init_sqlite_schema, _apply_sqlite_migrations
    from greenlang.factors.ingestion.sqlite_metadata import (
        insert_raw_artifact,
        insert_ingest_run,
        upsert_factor_lineage,
    )

    dbfile = tmp_path / "meta.sqlite"
    conn = sqlite3.connect(str(dbfile))
    _init_sqlite_schema(conn)
    _apply_sqlite_migrations(conn)
    conn.commit()

    aid = insert_raw_artifact(
        conn,
        source_id="test_src",
        sha256="abc123",
        storage_uri="file:///tmp/test.bin",
        bytes_size=100,
    )
    assert aid

    rid = insert_ingest_run(
        conn,
        artifact_id=aid,
        edition_id="ed1",
        parser_id="test_parser",
        status="ok",
        row_counts={"total": 10, "valid": 8},
    )
    assert rid

    upsert_factor_lineage(conn, "ed1", "EF:TEST:1", aid, rid, {"step": "normalize"})
    conn.commit()

    # Verify lineage was written
    cur = conn.execute(
        "SELECT lineage_json FROM factor_lineage WHERE edition_id=? AND factor_id=?",
        ("ed1", "EF:TEST:1"),
    )
    row = cur.fetchone()
    assert row is not None
    conn.close()
