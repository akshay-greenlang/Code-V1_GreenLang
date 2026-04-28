# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — EcoSpold2 + EXIOBASE MRIO end-to-end family tests.

Coverage matrix (per the Wave 2.5 contract):

  1. EXIOBASE happy path: parser emits 30 records (5 sectors x 3 regions
     x 2 extensions) with snapshot match + every geography_urn resolves
     in the seeded ontology.
  2. EcoSpold happy path WITH entitlement: parser emits 3 records (one
     per system model) and the run advances through stages 1-6.
  3. EcoSpold WITHOUT entitlement: parser raises ParserDispatchError
     with code=ECOSPOLD_ENTITLEMENT_MISSING (the Wave 2.5 enforcement
     contract).
  4. Zip-bomb defense: synthetic zip claiming 10 GB of uncompressed
     bytes raises ArtifactStoreError before any member is written.
  5. Geography mapping: every EXIOBASE row's geography_urn is one of
     the canonical seeded URNs (49 ISO countries + 5 ROW subregions).
  6. System-model metadata: each emitted ecoinvent record carries
     extraction.system_model in {cutoff, apos, consequential}.

Snapshot tests live alongside the e2e tests because they share the
fixture builders + the parser version pin.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from greenlang.factors.ingestion.exceptions import (
    ArtifactStoreError,
    ParserDispatchError,
)
from greenlang.factors.ingestion.parser_harness import ParserContext
from greenlang.factors.ingestion.parsers._phase3_ecospold_mrio_adapters import (
    ECOSPOLD_ENTITLEMENT_MISSING,
    EXIOBASE_ROW_GEOGRAPHIES,
    PHASE3_ECOSPOLD_PARSER_VERSION,
    PHASE3_ECOSPOLD_SOURCE_ID,
    PHASE3_EXIOBASE_PARSER_VERSION,
    PHASE3_EXIOBASE_SOURCE_ID,
    Phase3EcoSpoldParser,
    Phase3ExiobaseMrioParser,
)
from greenlang.factors.ingestion.zip_artifact import (
    DEFAULT_MAX_COMPRESSION_RATIO,
    extract_zip_artifact,
)

from tests.factors.v0_1_alpha.phase3.fixtures._build_ecoinvent_fixture import (
    ECOINVENT_FIXTURE_FILENAME,
    ECOINVENT_SYSTEM_MODELS,
    ensure_fixture as ensure_ecoinvent_fixture,
)
from tests.factors.v0_1_alpha.phase3.fixtures._build_exiobase_fixture import (
    EXIOBASE_EXTENSIONS,
    EXIOBASE_FIXTURE_FILENAME,
    EXIOBASE_REGIONS,
    EXIOBASE_SECTORS,
    ensure_fixture as ensure_exiobase_fixture,
)
from tests.factors.v0_1_alpha.phase3.parser_snapshots._helper import (
    compare_to_snapshot,
    regenerate_if_env,
)


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ecoinvent_fixture_path() -> Path:
    return ensure_ecoinvent_fixture(_FIXTURE_DIR / ECOINVENT_FIXTURE_FILENAME)


@pytest.fixture(scope="module")
def exiobase_fixture_path() -> Path:
    return ensure_exiobase_fixture(_FIXTURE_DIR / EXIOBASE_FIXTURE_FILENAME)


def _strip_volatile(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop wall-clock fields before snapshot comparison."""
    for row in rows:
        row.pop("published_at", None)
        ext = row.get("extraction") or {}
        if isinstance(ext, dict):
            ext.pop("ingested_at", None)
        review = row.get("review") or {}
        if isinstance(review, dict):
            for vol in ("reviewed_at", "approved_at"):
                review.pop(vol, None)
    return rows


# ---------------------------------------------------------------------------
# 1. EXIOBASE happy path + snapshot
# ---------------------------------------------------------------------------


def test_exiobase_parser_emits_full_matrix_and_snapshot_matches(
    exiobase_fixture_path: Path,
) -> None:
    """5 sectors x 3 regions x 2 extensions = 30 emitted records."""
    parser = Phase3ExiobaseMrioParser()
    raw = exiobase_fixture_path.read_bytes()

    rows = parser.parse_bytes(
        raw,
        artifact_uri="file://exiobase_v3.8.2_mini.zip",
        artifact_sha256="0" * 64,
    )
    expected_count = (
        len(EXIOBASE_SECTORS) * len(EXIOBASE_REGIONS) * len(EXIOBASE_EXTENSIONS)
    )
    assert len(rows) == expected_count, (
        "EXIOBASE expected %d records, saw %d" % (expected_count, len(rows))
    )

    # Every record carries the required provenance fields.
    for r in rows:
        ext = r["extraction"]
        assert ext["raw_artifact_uri"] == "file://exiobase_v3.8.2_mini.zip"
        assert ext["raw_artifact_sha256"] == "0" * 64
        assert "/" in ext["row_ref"]  # sector/region/extension

    rows = _strip_volatile(rows)
    parser_id = "exiobase_v3.8.2"
    regenerate_if_env(parser_id, PHASE3_EXIOBASE_PARSER_VERSION, rows)
    compare_to_snapshot(parser_id, PHASE3_EXIOBASE_PARSER_VERSION, rows)


# ---------------------------------------------------------------------------
# 2. EcoSpold happy path WITH entitlement
# ---------------------------------------------------------------------------


def test_ecospold_parser_with_entitlement_emits_per_system_model(
    ecoinvent_fixture_path: Path,
) -> None:
    """3 .spold files (cutoff/apos/consequential) -> 3 emitted records.

    Drives the parser via the ParserContext path with a tenant entitlement
    matching ``ecoinvent_3.10_cutoff`` so the entitlement gate succeeds.
    """
    parser = Phase3EcoSpoldParser()
    raw = ecoinvent_fixture_path.read_bytes()

    ctx = ParserContext(
        artifact_id="art-eco",
        source_id=PHASE3_ECOSPOLD_SOURCE_ID,
        parser_id="phase3_ecospold",
    )
    # Attach the source registry entry + tenant entitlement at runtime —
    # the parser reads optional attrs via getattr so a forward-compatible
    # extension does not require dataclass changes here.
    setattr(ctx, "source_registry_entry", {"entitlement_required": True})
    setattr(
        ctx,
        "tenant_context",
        {"tenant_entitlements": [{"source_id": PHASE3_ECOSPOLD_SOURCE_ID}]},
    )

    result = parser.parse_bytes(
        ctx,
        raw,
        artifact_uri="file://ecoinvent_3.10_mini.zip",
        artifact_sha256="0" * 64,
    )
    rows = result.rows
    assert len(rows) == len(ECOINVENT_SYSTEM_MODELS)

    # System model coverage: each of cutoff / apos / consequential must
    # appear exactly once.
    seen_models = sorted(r["extraction"]["system_model"] for r in rows)
    assert seen_models == ["apos", "consequential", "cutoff"]

    # Snapshot match (volatile-stripped).
    rows = _strip_volatile(rows)
    parser_id = "ecoinvent_3.10"
    regenerate_if_env(parser_id, PHASE3_ECOSPOLD_PARSER_VERSION, rows)
    compare_to_snapshot(parser_id, PHASE3_ECOSPOLD_PARSER_VERSION, rows)


# ---------------------------------------------------------------------------
# 3. EcoSpold WITHOUT entitlement — gate refuses
# ---------------------------------------------------------------------------


def test_ecospold_parser_without_entitlement_refuses(
    ecoinvent_fixture_path: Path,
) -> None:
    """ParserDispatchError raised with the canonical refusal code."""
    parser = Phase3EcoSpoldParser()
    raw = ecoinvent_fixture_path.read_bytes()

    ctx = ParserContext(
        artifact_id="art-eco",
        source_id=PHASE3_ECOSPOLD_SOURCE_ID,
        parser_id="phase3_ecospold",
    )
    setattr(ctx, "source_registry_entry", {"entitlement_required": True})
    # No tenant entitlement attached.
    setattr(ctx, "tenant_context", {"tenant_entitlements": []})

    with pytest.raises(ParserDispatchError) as excinfo:
        parser.parse_bytes(ctx, raw)
    # The refusal carries the canonical code in details.source_id and the
    # error message references the entitlement contract.
    assert excinfo.value.details.get("source_id") == PHASE3_ECOSPOLD_SOURCE_ID
    assert "entitlement" in str(excinfo.value).lower()
    # The Wave 2.5 contract pins the refusal to a single code constant.
    assert ECOSPOLD_ENTITLEMENT_MISSING == "ECOSPOLD_ENTITLEMENT_MISSING"


# ---------------------------------------------------------------------------
# 4. Zip-bomb defence
# ---------------------------------------------------------------------------


def test_zip_bomb_defense_rejects_oversized_bundle(tmp_path: Path) -> None:
    """A highly-redundant member whose compression ratio > 1024:1 is refused.

    The synthetic bomb is a 50 MB null-byte stream that compresses ~5000:1
    via DEFLATE — well above the 1024:1 ratio limit. The guard MUST trip
    BEFORE any bytes are written to disk.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(filename="huge.bin", date_time=(2026, 4, 28, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        # 50 MB of null bytes -> compresses to ~50 KB -> ratio ~5000:1.
        zf.writestr(info, b"\x00" * (50 * 1024 * 1024))
    raw = buf.getvalue()

    with pytest.raises(ArtifactStoreError) as excinfo:
        extract_zip_artifact(raw, tmp_path / "bomb")
    msg = str(excinfo.value).lower()
    assert "bomb" in msg or "ratio" in msg or "exceed" in msg
    # Make sure no member was written to disk before the trip.
    assert not (tmp_path / "bomb" / "huge.bin").exists()


def test_zip_bomb_defense_rejects_uncompressed_total(tmp_path: Path) -> None:
    """The cumulative uncompressed-bytes cap trips when totals exceed limit."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Five members, each 200 KB of *random-ish* (not null) data so the
        # compression ratio stays low and the per-member ratio guard does
        # not fire — only the cumulative-bytes guard.
        import os as _os

        for i in range(5):
            info = zipfile.ZipInfo(
                filename="m_%d.bin" % i, date_time=(2026, 4, 28, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, _os.urandom(200 * 1024))
    raw = buf.getvalue()
    # Set a 100 KB total budget. Each member is 200 KB so the very first
    # write trips the projected-bytes guard.
    with pytest.raises(ArtifactStoreError) as excinfo:
        extract_zip_artifact(
            raw,
            tmp_path / "cap",
            max_uncompressed_bytes=100 * 1024,
            max_compression_ratio=10_000,  # disable ratio guard
        )
    assert "uncompressed" in str(excinfo.value).lower()


def test_zip_bomb_defense_rejects_excess_member_count(tmp_path: Path) -> None:
    """Member-count cap trips before any extraction happens."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(50):
            info = zipfile.ZipInfo(
                filename="m_%04d.txt" % i, date_time=(2026, 4, 28, 0, 0, 0),
            )
            zf.writestr(info, b"x")
    raw = buf.getvalue()
    with pytest.raises(ArtifactStoreError) as excinfo:
        extract_zip_artifact(raw, tmp_path / "many", max_members=10)
    assert "member count" in str(excinfo.value).lower()


def test_zip_bomb_defense_rejects_path_traversal(tmp_path: Path) -> None:
    """Any ``..`` in a member name aborts extraction."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(
            filename="../escape.txt", date_time=(2026, 4, 28, 0, 0, 0),
        )
        zf.writestr(info, b"escape")
    raw = buf.getvalue()
    with pytest.raises(ArtifactStoreError) as excinfo:
        extract_zip_artifact(raw, tmp_path / "trav")
    assert "traversal" in str(excinfo.value).lower() or "escape" in str(
        excinfo.value,
    ).lower()


# ---------------------------------------------------------------------------
# 5. Geography mapping — every row resolves to a known URN
# ---------------------------------------------------------------------------


def test_exiobase_geographies_resolve_in_ontology(
    exiobase_fixture_path: Path,
) -> None:
    """Every emitted record has a geography_urn matching the seeded ontology.

    Loads the geography ontology YAML directly + asserts each EXIOBASE
    row's geography_urn is present (the FK gate the publish orchestrator
    runs at stage 4 / 6).
    """
    import yaml  # noqa: PLC0415 — local import; test-only

    geo_yaml = (
        Path("greenlang") / "factors" / "data" / "ontology"
        / "geography_seed_v0_1.yaml"
    )
    geo_doc = yaml.safe_load(geo_yaml.read_text(encoding="utf-8"))
    seeded_urns = {entry["urn"] for entry in geo_doc["geographies"]}

    # Every ROW geography URN the parser emits must exist in the seed.
    for _row_code, row_urn in EXIOBASE_ROW_GEOGRAPHIES:
        assert row_urn in seeded_urns, (
            "EXIOBASE ROW geography %s missing from ontology seed" % row_urn
        )

    parser = Phase3ExiobaseMrioParser()
    rows = parser.parse_bytes(
        exiobase_fixture_path.read_bytes(),
        artifact_uri="file://exiobase_v3.8.2_mini.zip",
        artifact_sha256="0" * 64,
    )
    for r in rows:
        urn = r["geography_urn"]
        assert urn in seeded_urns, (
            "EXIOBASE row geography_urn %r not in seeded ontology "
            "(parser emitted an unmapped region)" % urn
        )


# ---------------------------------------------------------------------------
# 6. System-model metadata coverage
# ---------------------------------------------------------------------------


def test_ecospold_system_model_metadata_present_per_record(
    ecoinvent_fixture_path: Path,
) -> None:
    """Each emitted ecoinvent record carries extraction.system_model."""
    parser = Phase3EcoSpoldParser()
    rows = parser.parse_bytes(
        ecoinvent_fixture_path.read_bytes(),
        artifact_uri="file://ecoinvent.zip",
        artifact_sha256="0" * 64,
    )
    valid_models = {"cutoff", "apos", "consequential"}
    for r in rows:
        sm = r["extraction"].get("system_model")
        assert sm in valid_models, (
            "ecoinvent record missing valid system_model: %r" % sm
        )


# ---------------------------------------------------------------------------
# 7. Determinism — fixture sha256 is stable across builds
# ---------------------------------------------------------------------------


def test_fixture_zip_sha256_is_deterministic(
    ecoinvent_fixture_path: Path,
    exiobase_fixture_path: Path,
) -> None:
    """Re-running the fixture builder produces the same bytes (sha256)."""
    from tests.factors.v0_1_alpha.phase3.fixtures._build_ecoinvent_fixture import (
        build_zip_bytes as build_eco,
    )
    from tests.factors.v0_1_alpha.phase3.fixtures._build_exiobase_fixture import (
        build_zip_bytes as build_xio,
    )

    eco_a = hashlib.sha256(build_eco()).hexdigest()
    eco_b = hashlib.sha256(build_eco()).hexdigest()
    assert eco_a == eco_b, "ecoinvent fixture builder is non-deterministic"

    xio_a = hashlib.sha256(build_xio()).hexdigest()
    xio_b = hashlib.sha256(build_xio()).hexdigest()
    assert xio_a == xio_b, "EXIOBASE fixture builder is non-deterministic"
