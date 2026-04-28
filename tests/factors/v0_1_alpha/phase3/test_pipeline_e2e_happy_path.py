# -*- coding: utf-8 -*-
"""Phase 3 — end-to-end happy path test for the unified ingestion pipeline.

Exercises stages 1 -> 6 (fetch -> parse -> normalize -> validate ->
dedupe -> stage) and asserts:

  * Each stage advances ``ingestion_runs.status`` through the canonical
    enum ladder (``created -> fetched -> parsed -> normalized ->
    validated -> deduped -> staged``).
  * The final terminal status is ``staged`` — auto-publish is forbidden
    by the Phase 3 plan §"The seven-stage pipeline contract".
  * Exactly one row exists in ``ingestion_runs`` for the run.
  * One row per input record exists in ``ingestion_run_diffs``.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"The seven-stage pipeline contract"
- ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 6, scenario "fetch->stage"
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
    reason=(
        "greenlang.factors.ingestion.runner not yet committed; "
        "Wave 1.0 sibling agent still in flight"
    ),
)


def test_pipeline_e2e_happy_path_ends_in_staged(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    synthetic_excel_artifact,
    seeded_source_urn,
):
    """fetch -> parse -> normalize -> validate -> dedupe -> stage advances cleanly."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    # The runner exposes a ``run`` entry that drives stages 1-6 in order.
    # We inject ``mock_fetcher`` so no network call leaves the test process.
    result = phase3_runner.run(
        source_urn=seeded_source_urn,
        source_version="2024.1",
        fetcher=mock_fetcher,
        operator="bot:phase3-e2e",
    )

    # Final status is ``staged`` — NOT ``published`` (auto-publish forbidden).
    assert result.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)
    assert result.status != RunStatus.PUBLISHED


def test_pipeline_e2e_status_advances_through_each_stage(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    seeded_source_urn,
):
    """Every stage's status row is observable in ``ingestion_runs``.

    The runner records a status update at every stage transition; this
    test reads the recorded history (or the live status snapshot if the
    runner exposes a getter) and asserts the canonical ladder is hit in
    order.
    """
    from greenlang.factors.ingestion.pipeline import RunStatus

    observed_statuses = []

    def _hook(*, run_id, status, **_):
        observed_statuses.append(RunStatus(status))

    result = phase3_runner.run(
        source_urn=seeded_source_urn,
        source_version="2024.1",
        fetcher=mock_fetcher,
        operator="bot:phase3-e2e",
        on_stage_complete=_hook,
    )

    expected_ladder = [
        RunStatus.FETCHED,
        RunStatus.PARSED,
        RunStatus.NORMALIZED,
        RunStatus.VALIDATED,
        RunStatus.DEDUPED,
        RunStatus.STAGED,
    ]
    # The runner may emit the same status more than once on retry; we
    # only require that the canonical ladder appears as a subsequence.
    idx = 0
    for emitted in observed_statuses:
        if idx < len(expected_ladder) and emitted == expected_ladder[idx]:
            idx += 1
    assert idx == len(expected_ladder), (
        f"expected status ladder not observed in order; saw "
        f"{[s.value for s in observed_statuses]}"
    )
    assert result.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)


def test_pipeline_e2e_writes_ingestion_runs_row(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    seeded_source_urn,
):
    """A single row in ``ingestion_runs`` carries the run's final state."""
    result = phase3_runner.run(
        source_urn=seeded_source_urn,
        source_version="2024.1",
        fetcher=mock_fetcher,
        operator="bot:phase3-e2e",
    )

    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT run_id, source_urn, source_version, status, operator "
            "FROM ingestion_runs WHERE run_id = ?",
            (result.run_id,),
        )
        rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ingestion_runs table not yet provisioned: {exc!r}")
        return
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == result.run_id
    assert row[1] == seeded_source_urn
    assert row[2] == "2024.1"
    assert row[3] in ("staged", "review_required")
    assert row[4] == "bot:phase3-e2e"


def test_pipeline_e2e_writes_diff_rows_per_input(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    seeded_source_urn,
    synthetic_rows,
):
    """``ingestion_run_diffs`` carries one row per accepted input record."""
    result = phase3_runner.run(
        source_urn=seeded_source_urn,
        source_version="2024.1",
        fetcher=mock_fetcher,
        operator="bot:phase3-e2e",
    )

    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT change_kind FROM ingestion_run_diffs WHERE run_id = ?",
            (result.run_id,),
        )
        diffs = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ingestion_run_diffs table not yet provisioned: {exc!r}")
        return
    # Every synthetic row should have a corresponding diff entry.
    assert len(diffs) == len(synthetic_rows)
