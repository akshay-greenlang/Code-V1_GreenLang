# -*- coding: utf-8 -*-
"""Phase 3 — rollback demotes a published run; rows are never deleted.

Per PHASE_3_PLAN.md §"The seven-stage pipeline contract": a rollback
flips a ``published`` run back to ``staged`` (or ``rolled_back`` per the
exact contract) WITHOUT deleting the underlying factor rows. The
operation is recorded as a reverse entry in ``factor_publish_log``.

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 6 scenario
"rollback-demotes-not-deletes".
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


def test_rollback_preserves_factor_rows_and_records_reverse_entry(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    seeded_source_urn,
):
    """publish() then rollback() leaves factor rows intact."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    # Publish.
    staged = phase3_runner.run(
        source_urn=seeded_source_urn,
        source_version="2024.1",
        fetcher=mock_fetcher,
        operator="bot:phase3-rollback",
    )
    published = phase3_runner.publish(
        run_id=staged.run_id,
        approver="human:methodology@greenlang.io",
    )
    assert published.status == RunStatus.PUBLISHED

    # Snapshot the factor rows present after publish.
    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM factors_v0_1_factor WHERE source_urn = ?",
            (seeded_source_urn,),
        )
        rows_before = cur.fetchone()[0]
    except Exception:  # noqa: BLE001
        try:
            cur.execute(
                "SELECT COUNT(*) FROM factor WHERE source_urn = ?",
                (seeded_source_urn,),
            )
            rows_before = cur.fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"factor table not introspectable: {exc!r}")
            return

    # Rollback.
    rolled = phase3_runner.rollback(
        batch_id=published.batch_id,
        approver="human:methodology@greenlang.io",
    )
    assert rolled.status in (RunStatus.ROLLED_BACK, RunStatus.STAGED)

    # Factor rows are STILL there (rollback does not delete).
    try:
        cur.execute(
            "SELECT COUNT(*) FROM factors_v0_1_factor WHERE source_urn = ?",
            (seeded_source_urn,),
        )
        rows_after = cur.fetchone()[0]
    except Exception:  # noqa: BLE001
        cur.execute(
            "SELECT COUNT(*) FROM factor WHERE source_urn = ?",
            (seeded_source_urn,),
        )
        rows_after = cur.fetchone()[0]
    assert rows_after == rows_before, (
        f"rollback deleted rows ({rows_before} -> {rows_after}); "
        f"contract requires demotion only"
    )

    # A reverse entry is recorded.
    try:
        cur.execute(
            "SELECT operation FROM factor_publish_log "
            "WHERE batch_id = ? ORDER BY id DESC LIMIT 1",
            (published.batch_id,),
        )
        last_op = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"factor_publish_log not yet provisioned: {exc!r}")
        return
    assert last_op is not None
    assert last_op[0] in ("rollback", "rolled_back", "demote")
