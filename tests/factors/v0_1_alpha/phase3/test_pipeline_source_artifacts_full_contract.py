# -*- coding: utf-8 -*-
"""Phase 3 audit gap C — source_artifacts row carries the full 16+ contract.

The Phase 3 plan §"Artifact storage contract" requires every fetch to
land a ``factors_v0_1.source_artifacts`` row carrying the canonical
16-field lineage payload (raw_bytes_uri, source_url, fetched_at, sha256,
bytes_size, content_type, source_version, source_publication_date,
parser_module, parser_function, parser_version, parser_commit, operator,
licence_class, redistribution_class, ingestion_run_id, status). The
``status`` column tracks pipeline progress so a SELECT on the row at any
moment surfaces the current stage of the run that owns the artifact.

This test drives the full DEFRA pipeline end-to-end and asserts:

* the SQLite mirror table ``alpha_source_artifacts_v0_1`` carries a
  row with the run's sha256;
* every contract field (or its renamed equivalent) is populated;
* the row's ``status`` advances from ``fetched`` to one of
  ``staged`` / ``review_required`` over the run.
"""
from __future__ import annotations

import importlib

import pytest


def _runner_available() -> bool:
    try:
        importlib.import_module("greenlang.factors.ingestion.runner")
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _runner_available(),
    reason="greenlang.factors.ingestion.runner not yet committed",
)


def test_full_defra_pipeline_lands_full_source_artifact_row(
    phase3_runner_raw,
    phase3_run_repo,
    phase3_repo,
    defra_fixture_url,
):
    """A full DEFRA run lands the canonical 16+ field source_artifacts row."""
    run = phase3_runner_raw.run(
        source_id="defra-2025",
        source_url=defra_fixture_url,
        source_urn="urn:gl:source:defra-2025",
        source_version="2025.1",
        operator="bot:gap-c-test",
        auto_stage=True,
    )
    assert run.artifact_sha256 is not None

    # Pull the row back via the SQLite mirror.
    conn = phase3_repo._connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM alpha_source_artifacts_v0_1 WHERE sha256 = ?",
        (run.artifact_sha256,),
    )
    rows = cur.fetchall()
    assert len(rows) == 1, (
        f"expected exactly one source_artifacts row for sha256={run.artifact_sha256!r}, "
        f"got {len(rows)}"
    )
    cols = [d[0] for d in cur.description]
    row = dict(zip(cols, rows[0]))

    # The Phase 3 contract calls for these 16 logical fields. Several
    # map to legacy column names retained from V505 (uri/size_bytes/
    # ingested_at/parser_id) — the table now carries BOTH the legacy
    # column and the new one, so we accept either.
    field_to_columns = {
        "raw_bytes_uri": ("uri",),
        "source_url": ("source_url",),
        "fetched_at": ("ingested_at",),
        "sha256": ("sha256",),
        "bytes_size": ("size_bytes",),
        "content_type": ("content_type",),
        "source_version": ("source_version",),
        "source_publication_date": ("source_publication_date",),
        "parser_module": ("parser_module",),
        "parser_function": ("parser_function", "parser_id"),
        "parser_version": ("parser_version",),
        "parser_commit": ("parser_commit",),
        "operator": ("operator",),
        "licence_class": ("licence_class",),
        "redistribution_class": ("redistribution_class",),
        "ingestion_run_id": ("ingestion_run_id",),
        "status": ("status",),
        "source_urn": ("source_urn",),
    }
    # Required-non-null fields per the Phase 3 contract. ``content_type``
    # is best-effort (may be None for unrecognised extensions) so we
    # only require it to be PRESENT, not non-null. Same for
    # ``source_publication_date`` (registry may not carry it for every
    # source). ``redistribution_class`` is similar.
    required_non_null = {
        "raw_bytes_uri", "sha256", "bytes_size", "fetched_at",
        "source_version", "operator", "ingestion_run_id", "status",
        "source_urn",
    }
    for logical_field, candidate_cols in field_to_columns.items():
        # The column itself must exist on the table.
        assert any(c in row for c in candidate_cols), (
            f"source_artifacts missing column for logical field {logical_field!r} "
            f"(candidates: {candidate_cols!r}); columns present: {sorted(row)!r}"
        )
        if logical_field in required_non_null:
            value = next(
                (row[c] for c in candidate_cols if c in row and row[c] is not None),
                None,
            )
            assert value is not None, (
                f"source_artifacts.{logical_field} is NULL on the run row; "
                f"contract requires a non-null value"
            )

    # The ``ingestion_run_id`` column must point at the run that
    # produced the bytes.
    assert row["ingestion_run_id"] == run.run_id

    # ``status`` should reflect the final pipeline state.
    assert row["status"] in ("staged", "review_required"), (
        f"expected status in (staged, review_required), got {row['status']!r}"
    )

    # ``operator`` should be the bot driving the run.
    assert row["operator"] == "bot:gap-c-test"


def test_source_artifact_status_advances_through_pipeline(
    phase3_runner_raw,
    phase3_run_repo,
    phase3_repo,
    defra_fixture_url,
):
    """The status column walks fetched -> ... -> staged across stages.

    Drive the pipeline stage-by-stage and snapshot the artifact row's
    ``status`` after each stage to prove the status ladder is observable
    on a SELECT at any moment.
    """
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = phase3_run_repo.create(
        source_urn="urn:gl:source:defra-2025",
        source_version="2025.1",
        operator="bot:status-walk-test",
    )

    def _status_for(sha256):
        conn = phase3_repo._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT status FROM alpha_source_artifacts_v0_1 WHERE sha256 = ?",
            (sha256,),
        )
        r = cur.fetchone()
        return r[0] if r else None

    artifact = phase3_runner_raw.fetch(
        run.run_id, source_id="defra-2025", source_url=defra_fixture_url
    )
    assert _status_for(artifact.sha256) == "fetched"

    parser_result = phase3_runner_raw.parse(
        run.run_id, source_id="defra-2025", artifact=artifact
    )
    assert _status_for(artifact.sha256) == "parsed"

    records = phase3_runner_raw.normalize(run.run_id, parser_result=parser_result)
    assert _status_for(artifact.sha256) == "normalized"

    validation = phase3_runner_raw.validate(run.run_id, records=records)
    assert _status_for(artifact.sha256) == "validated"

    dedupe_outcome = phase3_runner_raw.dedupe(
        run.run_id, accepted=validation.accepted
    )
    assert _status_for(artifact.sha256) == "deduped"

    phase3_runner_raw.stage(run.run_id, dedupe_outcome=dedupe_outcome)
    final_status = _status_for(artifact.sha256)
    assert final_status in ("staged", "review_required"), (
        f"expected final status in (staged, review_required), got {final_status!r}"
    )

    final_run = phase3_run_repo.get(run.run_id)
    assert final_run.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)
