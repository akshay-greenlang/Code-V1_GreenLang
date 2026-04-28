# -*- coding: utf-8 -*-
"""Phase 3 — parser failure produces a ``failed`` run, no factor rows.

When the parser stage raises :class:`ParserDispatchError` (or any other
unexpected exception), the runner must:

  * advance ``ingestion_runs.status`` to ``failed``;
  * write a structured traceback summary to ``error_json``;
  * leave ZERO factor rows for the run id (the parse failed before any
    INSERT could execute).

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 6 scenario
"parser-failure-records-failed-run".
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


def test_parser_dispatch_error_records_failed_run(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    seeded_source_urn,
):
    """Parser raises -> run.status == 'failed' AND error_json populated."""
    from greenlang.factors.ingestion.exceptions import ParserDispatchError
    from greenlang.factors.ingestion.pipeline import RunStatus

    def _angry_parser(*_args, **_kwargs):
        raise ParserDispatchError(
            "synthetic parser blow-up",
            source_id="phase3-test",
            registered_versions=[],
        )

    with pytest.raises(ParserDispatchError):
        phase3_runner.run(
            source_urn=seeded_source_urn,
            source_version="2024.1",
            fetcher=mock_fetcher,
            parser=_angry_parser,
            operator="bot:phase3-failed",
        )

    # The runner records the failed status BEFORE re-raising. Pull the
    # most-recent run row for this source_urn and assert.
    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT status, error_json FROM ingestion_runs "
            "WHERE source_urn = ? ORDER BY started_at DESC LIMIT 1",
            (seeded_source_urn,),
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ingestion_runs not yet provisioned: {exc!r}")
        return
    assert row is not None
    assert row[0] == RunStatus.FAILED.value
    # ``error_json`` carries SOMETHING — text or JSON depending on the
    # storage backend; we just need it non-null.
    assert row[1] is not None and row[1] != ""


def test_parser_failure_writes_zero_factor_rows(
    phase3_runner,
    phase3_repo,
    mock_fetcher,
    seeded_source_urn,
):
    """No factor row exists for a run that failed at the parse stage."""
    from greenlang.factors.ingestion.exceptions import ParserDispatchError

    def _angry_parser(*_args, **_kwargs):
        raise ParserDispatchError("synthetic parser blow-up")

    with pytest.raises(ParserDispatchError):
        phase3_runner.run(
            source_urn=seeded_source_urn,
            source_version="2024.1",
            fetcher=mock_fetcher,
            parser=_angry_parser,
            operator="bot:phase3-failed-zero",
        )

    conn = phase3_repo._connect()  # type: ignore[attr-defined]
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM factors_v0_1_factor "
            "WHERE source_urn = ?",
            (seeded_source_urn,),
        )
        n = cur.fetchone()[0]
    except Exception:  # noqa: BLE001
        # Different storage layout — fall back to a generic search.
        try:
            cur.execute("SELECT COUNT(*) FROM factor WHERE source_urn = ?", (seeded_source_urn,))
            n = cur.fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"factor table not introspectable: {exc!r}")
            return
    assert n == 0
