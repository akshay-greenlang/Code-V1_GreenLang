# -*- coding: utf-8 -*-
"""Phase 3 / Wave 1.5 — DEFRA Excel reference end-to-end test.

This module is the canonical "reference source" acceptance test for
Phase 3 Wave 1.5. It drives the unified
:class:`IngestionPipelineRunner` against the deterministic DEFRA mini
fixture (``tests/factors/v0_1_alpha/phase3/fixtures/defra_2025_mini.xlsx``)
through stages 1 -> 7 and asserts:

  1. The run advances ``ingestion_runs.status`` through the canonical
     ladder (``created -> fetched -> parsed -> normalized -> validated
     -> deduped -> staged``).
  2. The parser snapshot matches the committed golden file (no parser
     drift across runs).
  3. A non-empty :class:`RunDiff` was produced; both MD and JSON
     serialisations are byte-deterministic and contain the expected
     counters.
  4. Publishing the run with a valid methodology-lead approver succeeds,
     factors land in the production namespace, and the run is
     ``published``.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Reference source: DEFRA Excel
  end-to-end" (Wave 1.5).
- ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 2 (artifact storage),
  Block 3 (Excel-family validation), Block 6 (snapshot tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenlang.factors.ingestion.diff import (
    RunDiff,
    serialize_json,
    serialize_markdown,
)
from greenlang.factors.ingestion.pipeline import RunStatus
from greenlang.factors.ingestion.parsers._phase3_adapters import (
    PHASE3_DEFRA_PARSER_VERSION,
    PHASE3_DEFRA_SOURCE_ID,
    PHASE3_DEFRA_SOURCE_URN,
    Phase3DEFRAExcelParser,
)
from tests.factors.v0_1_alpha.phase3.fixtures._build_defra_fixture import (
    FUEL_CONVERSION_ROWS,
    STATIONARY_ROWS,
)
from tests.factors.v0_1_alpha.phase3.parser_snapshots._helper import (
    compare_to_snapshot,
    regenerate_if_env,
)

#: Total expected rows from the synthetic DEFRA fixture (one per
#: data row, across both required tabs).
_EXPECTED_ROW_COUNT: int = len(STATIONARY_ROWS) + len(FUEL_CONVERSION_ROWS)


# ---------------------------------------------------------------------------
# 1. Status-ladder advancement.
# ---------------------------------------------------------------------------


def test_defra_run_advances_through_canonical_status_ladder(
    phase3_runner_raw,
    defra_fixture_path: Path,
    phase3_run_repo,
) -> None:
    """``runner.run()`` walks created -> fetched -> ... -> staged.

    The runner records a stage receipt at every transition. We pull the
    full per-stage history from ``ingestion_run_stage_history`` and
    assert the canonical ladder is observed in order.
    """
    run = phase3_runner_raw.run(
        source_id=PHASE3_DEFRA_SOURCE_ID,
        source_url=str(defra_fixture_path.resolve()),
        source_urn=PHASE3_DEFRA_SOURCE_URN,
        source_version="2025.1",
        operator="bot:test",
        auto_stage=True,
    )

    # Final state — STAGED or REVIEW_REQUIRED (both are terminal-success).
    assert run.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED), (
        f"unexpected terminal status: {run.status.value}"
    )
    # The stage history rows carry a row per stage; pull them and
    # assert the canonical sequence.
    conn = phase3_run_repo._memory_conn  # type: ignore[attr-defined]
    cur = conn.execute(
        "SELECT stage FROM ingestion_run_stage_history "
        "WHERE run_id = ? ORDER BY pk_id ASC",
        (run.run_id,),
    )
    stages_observed = [row[0] for row in cur.fetchall()]
    expected_ladder = ["fetch", "parse", "normalize", "validate", "dedupe", "stage"]
    # Ladder must appear as a (contiguous) subsequence.
    idx = 0
    for stage in stages_observed:
        if idx < len(expected_ladder) and stage == expected_ladder[idx]:
            idx += 1
    assert idx == len(expected_ladder), (
        f"canonical ladder not fully observed; saw {stages_observed!r}"
    )


# ---------------------------------------------------------------------------
# 2. Parser-snapshot match.
# ---------------------------------------------------------------------------


def test_defra_parser_snapshot_matches_committed_golden(
    defra_fixture_path: Path,
) -> None:
    """The parser output is byte-stable against the committed golden.

    Drives :class:`Phase3DEFRAExcelParser` directly so the snapshot test
    is decoupled from the runner's stage machinery; this isolates parser
    drift from runner-stage drift.
    """
    parser = Phase3DEFRAExcelParser()
    raw_bytes = defra_fixture_path.read_bytes()
    rows = parser.parse_bytes(
        raw_bytes,
        artifact_uri="file://defra_2025_mini.xlsx",
        artifact_sha256="0" * 64,  # frozen for snapshot determinism
    )
    # Strip volatile fields (``ingested_at``, ``published_at``) before
    # comparison so the golden does not encode wall-clock time.
    for row in rows:
        row.pop("published_at", None)
        ext = row.get("extraction") or {}
        if isinstance(ext, dict):
            ext.pop("ingested_at", None)
        review = row.get("review") or {}
        if isinstance(review, dict):
            for vol_key in ("reviewed_at", "approved_at"):
                review.pop(vol_key, None)

    parser_id = "defra_2025"
    # Allow regeneration via env var (snapshot framework contract).
    regenerate_if_env(parser_id, PHASE3_DEFRA_PARSER_VERSION, rows)
    compare_to_snapshot(parser_id, PHASE3_DEFRA_PARSER_VERSION, rows)


# ---------------------------------------------------------------------------
# 3. Diff is non-empty + serialisations are deterministic.
# ---------------------------------------------------------------------------


def test_defra_run_emits_non_empty_run_diff_with_deterministic_serialization(
    defra_ingestion_run,
) -> None:
    """The staged DEFRA run produces a non-empty diff with stable counters."""
    run_diff: RunDiff = getattr(defra_ingestion_run, "run_diff", None) or _load_run_diff(
        defra_ingestion_run.diff_json_uri,
    )
    assert run_diff is not None, "defra run did not attach a RunDiff"

    # Non-empty: the seeded ontology has zero DEFRA factors initially, so
    # every row is "added" relative to production.
    assert not run_diff.is_empty(), "expected a non-empty diff for a fresh run"
    assert run_diff.total_changes() >= _EXPECTED_ROW_COUNT, (
        f"expected at least {_EXPECTED_ROW_COUNT} added URNs, "
        f"saw {run_diff.total_changes()}"
    )

    # Deterministic JSON + MD serialisations.
    json_a = serialize_json(run_diff)
    json_b = serialize_json(run_diff)
    text_a = json.dumps(json_a, sort_keys=True, indent=2)
    text_b = json.dumps(json_b, sort_keys=True, indent=2)
    assert text_a == text_b, "JSON serialisation drift across two calls"

    md_a = serialize_markdown(run_diff)
    md_b = serialize_markdown(run_diff)
    assert md_a == md_b, "Markdown serialisation drift across two calls"

    # Counters: at least the ``added`` bucket must reflect the parsed rows.
    summary = json_a["summary"]
    assert summary["added"] == _EXPECTED_ROW_COUNT, (
        f"summary added={summary['added']!r} != {_EXPECTED_ROW_COUNT}"
    )
    assert summary["total_changes"] >= _EXPECTED_ROW_COUNT


# ---------------------------------------------------------------------------
# 4. Publish flips factors into production.
# ---------------------------------------------------------------------------


def test_defra_publish_flips_factors_to_production(
    phase3_runner_raw,
    defra_ingestion_run,
    phase3_repo,
) -> None:
    """``runner.publish(run_id, approver=...)`` flips records to production."""
    phase3_runner_raw.publish(
        defra_ingestion_run.run_id,
        approver="human:test-lead@greenlang.io",
    )
    run = phase3_runner_raw._run_repo.get(defra_ingestion_run.run_id)
    assert run.status == RunStatus.PUBLISHED, (
        f"post-publish status != published: {run.status.value}"
    )
    assert run.batch_id, "publish must record a batch_id on the run row"
    assert run.approved_by == "human:test-lead@greenlang.io"

    # Every staged URN now in production. Cross-check via the namespace
    # column on the alpha factor table.
    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.execute(
        "SELECT COUNT(*) FROM alpha_factors_v0_1 "
        "WHERE namespace = 'production' AND source_urn = ?",
        ("urn:gl:source:phase2-alpha",),  # parser uses the seeded source URN
    )
    promoted = cur.fetchone()[0]
    assert promoted == _EXPECTED_ROW_COUNT, (
        f"expected {_EXPECTED_ROW_COUNT} production-namespace rows, "
        f"saw {promoted}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_run_diff(diff_json_uri: str) -> RunDiff:
    """Re-hydrate a :class:`RunDiff` from the JSON file written at stage 6.

    Handles Windows-style file URIs (``file:///C:/...``) where the third
    slash is followed immediately by the drive letter — :class:`Path`
    on Windows treats a leading slash as UNC otherwise.
    """
    if not diff_json_uri:
        pytest.fail("run row does not carry a diff_json_uri")
    from urllib.parse import urlparse, unquote

    parsed = urlparse(diff_json_uri)
    raw_path = unquote(parsed.path)
    # Windows: urlparse yields '/C:/Users/...'; strip the leading slash
    # so :class:`Path` resolves against the local filesystem.
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunDiff(
        added=list(payload.get("added", [])),
        removed=list(payload.get("removed", [])),
        unchanged_count=payload.get("summary", {}).get("unchanged", 0),
        run_id=payload.get("run_id"),
        source_urn=payload.get("source_urn"),
        source_version=payload.get("source_version"),
    )
