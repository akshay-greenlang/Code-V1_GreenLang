# -*- coding: utf-8 -*-
"""Phase 3 audit gap A — SQLite stage transitions are one atomic txn.

The Phase 3 plan §"Run-status enum (formal)" requires every multi-statement
stage advance in :class:`IngestionRunRepository` to commit atomically. The
SQLite mode (alpha + dev) historically opened the connection in
``isolation_level=None`` (autocommit), which committed each ``execute()``
immediately — so a crash between statements could leave the run row
half-applied (e.g. ``ingestion_runs.diff_json_uri`` set without a matching
``ingestion_run_diffs`` row, or vice versa).

This test forces a multi-statement stage advance (``set_diff``) to fail
mid-write by wrapping the underlying sqlite3 connection with a proxy
whose ``execute`` raises on the second statement, then asserts the first
``UPDATE ingestion_runs SET diff_json_uri`` was ROLLED BACK.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Callable

import pytest

from greenlang.factors.ingestion.pipeline import RunStatus
from greenlang.factors.ingestion.run_repository import IngestionRunRepository


class _FlakyConnProxy:
    """Wraps a sqlite3.Connection and raises on a chosen SQL prefix."""

    def __init__(self, real: sqlite3.Connection, predicate: Callable[[str], bool]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_pred", predicate)
        object.__setattr__(self, "_hit_count", 0)

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        pred = object.__getattribute__(self, "_pred")
        if isinstance(sql, str) and pred(sql):
            object.__setattr__(self, "_hit_count", object.__getattribute__(self, "_hit_count") + 1)
            raise sqlite3.OperationalError(
                "synthetic failure: simulated mid-stage crash"
            )
        return object.__getattribute__(self, "_real").execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


def _create_seed_run(repo: IngestionRunRepository) -> Any:
    """Create + advance a run to ``DEDUPED`` so ``set_diff`` is legal."""
    run = repo.create(
        source_urn="urn:gl:source:atomic-test",
        source_version="2024.1",
        operator="bot:atomic-test",
    )
    for st in (
        RunStatus.FETCHED,
        RunStatus.PARSED,
        RunStatus.NORMALIZED,
        RunStatus.VALIDATED,
        RunStatus.DEDUPED,
    ):
        repo.update_status(run.run_id, st)
    return run


def test_set_diff_rolls_back_on_second_statement_failure(tmp_path):
    """Mid-txn failure on the diffs INSERT must roll back the runs UPDATE.

    Pre-fix (autocommit) behaviour: the first UPDATE would have already
    committed when the second INSERT raised — leaving the run row's
    ``diff_json_uri`` set to a value with no matching diff summary row.

    Post-fix (BEGIN; ... COMMIT;) behaviour: the explicit transaction
    rolls back, both writes are reverted, and the run row's
    ``diff_json_uri`` stays NULL.
    """
    db_path = tmp_path / "atomic_txn.db"
    dsn = f"sqlite:///{db_path.as_posix()}"
    repo = IngestionRunRepository(dsn)
    run = _create_seed_run(repo)

    real_connect = repo._connect

    def _flaky_connect():
        real = real_connect()
        return _FlakyConnProxy(
            real,
            lambda s: s.strip().upper().startswith("INSERT INTO INGESTION_RUN_DIFFS"),
        )

    repo._connect = _flaky_connect  # type: ignore[assignment]

    try:
        with pytest.raises(sqlite3.OperationalError, match="synthetic failure"):
            repo.set_diff(
                run.run_id,
                diff_json_uri="file:///tmp/should-be-rolled-back.json",
                diff_md_uri="file:///tmp/should-be-rolled-back.md",
                summary_json={"changed": 1},
            )
    finally:
        repo._connect = real_connect  # type: ignore[assignment]

    # The first UPDATE inside set_diff() set ``diff_json_uri`` BEFORE the
    # second INSERT raised. With proper BEGIN/COMMIT semantics, the
    # rollback reverts that UPDATE so the run row reads NULL.
    fresh = repo.get(run.run_id)
    assert fresh.diff_json_uri is None, (
        f"expected diff_json_uri NULL after rollback, got {fresh.diff_json_uri!r}; "
        f"the first UPDATE leaked past the failed INSERT — atomic txn broken"
    )
    assert fresh.diff_md_uri is None, (
        "diff_md_uri leaked past rollback — atomic txn broken"
    )

    # Sanity: a clean retry (without the proxy) succeeds and the row
    # carries the diff URIs.
    repo.set_diff(
        run.run_id,
        diff_json_uri="file:///tmp/clean.json",
        diff_md_uri="file:///tmp/clean.md",
        summary_json={"changed": 0},
    )
    after = repo.get(run.run_id)
    assert after.diff_json_uri == "file:///tmp/clean.json"
    assert after.diff_md_uri == "file:///tmp/clean.md"


def test_update_status_rolls_back_on_failure(tmp_path):
    """A status-update failure must NOT leave a partial ``status`` write.

    Verifies the BEGIN/COMMIT wrapper holds for single-statement writes
    too — the UPDATE happens inside the explicit transaction so the
    rollback semantics are uniform across every stage write.
    """
    db_path = tmp_path / "atomic_status.db"
    dsn = f"sqlite:///{db_path.as_posix()}"
    repo = IngestionRunRepository(dsn)
    run = repo.create(
        source_urn="urn:gl:source:atomic-status",
        source_version="2024.1",
        operator="bot:atomic-status",
    )
    real_connect = repo._connect

    def _flaky_connect():
        real = real_connect()
        return _FlakyConnProxy(
            real,
            lambda s: s.strip().upper().startswith("UPDATE INGESTION_RUNS SET STATUS"),
        )

    repo._connect = _flaky_connect  # type: ignore[assignment]
    try:
        with pytest.raises(sqlite3.OperationalError, match="synthetic"):
            repo.update_status(run.run_id, RunStatus.FETCHED)
    finally:
        repo._connect = real_connect  # type: ignore[assignment]

    fresh = repo.get(run.run_id)
    assert fresh.status == RunStatus.CREATED, (
        f"expected status to remain CREATED after rollback, got {fresh.status!r}"
    )
