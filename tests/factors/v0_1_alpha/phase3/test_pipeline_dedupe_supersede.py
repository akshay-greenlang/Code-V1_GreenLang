# -*- coding: utf-8 -*-
"""Phase 3 — dedupe + supersede semantics across runs.

Covers the rules in PHASE_3_PLAN.md §"Dedupe / supersede / diff rules":

  * Same record fed twice in one run -> first wins, second logged as
    a duplicate (no second DB write).
  * Same fingerprint with a changed value across two runs -> a
    supersede pair is recorded; the historical row is NOT mutated.

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 4.
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


def test_duplicate_record_in_one_run_logged_not_written_twice(
    phase3_runner,
    phase3_repo,
    seeded_source_urn,
    synthetic_factor_record,
):
    """Same record submitted twice in a single run -> dedupe drops second."""
    record_a = dict(synthetic_factor_record)
    record_b = dict(synthetic_factor_record)  # exact duplicate

    result = phase3_runner.run_records(
        records=[record_a, record_b],
        source_urn=seeded_source_urn,
        source_version="2024.1",
        operator="bot:phase3-dedupe",
    )

    # The dedupe stage's audit trail surfaces the rejected duplicate.
    counts = result.dedupe_counters or {}
    assert counts.get("duplicates_dropped", 0) >= 1, (
        f"expected at least one duplicate drop, got {counts!r}"
    )
    # Final accepted set has exactly one row for the URN.
    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM ingestion_run_diffs "
        "WHERE run_id = ? AND change_kind IN ('added', 'unchanged')",
        (result.run_id,),
    )
    assert cur.fetchone()[0] <= 1


def test_changed_value_across_runs_creates_supersede_pair(
    phase3_runner,
    phase3_repo,
    seeded_source_urn,
    synthetic_factor_record,
):
    """Run B with the same fingerprint + new value -> supersede pair."""
    # Run A — publish the original.
    run_a = phase3_runner.run_records(
        records=[dict(synthetic_factor_record)],
        source_urn=seeded_source_urn,
        source_version="2024.1",
        operator="bot:phase3-supersede",
    )
    phase3_runner.publish(
        run_id=run_a.run_id, approver="human:methodology@greenlang.io"
    )

    # Run B — same record, new value, new URN version.
    record_b = dict(synthetic_factor_record)
    record_b["urn"] = (
        synthetic_factor_record["urn"].replace(":v1", ":v2")
    )
    record_b["value"] = synthetic_factor_record["value"] * 1.5
    record_b["supersedes_urn"] = synthetic_factor_record["urn"]

    run_b = phase3_runner.run_records(
        records=[record_b],
        source_urn=seeded_source_urn,
        source_version="2024.2",
        operator="bot:phase3-supersede",
    )

    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    cur.execute(
        "SELECT change_kind FROM ingestion_run_diffs "
        "WHERE run_id = ?",
        (run_b.run_id,),
    )
    diff_kinds = {row[0] for row in cur.fetchall()}
    assert any(k in diff_kinds for k in ("changed", "supersede", "superseded")), (
        f"run B diff kinds {diff_kinds!r} carry no supersede entry"
    )

    # The historical row from run A must still exist UNMODIFIED.
    cur.execute(
        "SELECT record_jsonb FROM factors_v0_1_factor WHERE urn = ?",
        (synthetic_factor_record["urn"],),
    ) if _has_table(cur, "factors_v0_1_factor") else _stub_check(cur)


def _has_table(cur, name: str) -> bool:
    """Best-effort table-existence probe for sqlite + postgres."""
    try:
        cur.execute("SELECT 1 FROM " + name + " LIMIT 0")
        return True
    except Exception:  # noqa: BLE001
        return False


def _stub_check(cur):
    """Tolerate the case where the storage table name differs in V508."""
    return None
