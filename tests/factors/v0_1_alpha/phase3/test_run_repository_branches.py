# -*- coding: utf-8 -*-
"""Phase 3 Block 6 — branch-coverage tests for ``ingestion/run_repository.py``.

Targets the residual ~27pp gap. The Postgres branches are gated on
``GL_TEST_POSTGRES_DSN``; this module installs a fake ``psycopg`` module
into ``sys.modules`` so the ``_*_pg`` callables execute under doubles.

Covers:

* ``IngestionRunNotFoundError`` raised by ``get`` / ``update_status``.
* ``_resolve_sqlite_path`` corner cases (memory, percent-encoded, raw
  path, sqlite-prefixed Windows path).
* ``_is_postgres_dsn`` for postgres + postgresql schemes.
* SQLite ``_set_diff_sqlite`` ON CONFLICT branch (re-set diff for same
  run twice).
* SQLite ``_append_stage_sqlite`` records stage rows.
* SQLite idempotent re-update of same status (no-op via
  ``assert_can_transition``).
* SQLite partial-row updates (``set_artifact`` with no parser fields,
  ``set_publish`` only stamps batch_id + approver).
* SQLite ``close()`` releases the in-memory connection.
* SQLite ``list_by_status`` / ``list_by_source`` returning multiple rows.
* ``_row_to_run`` / ``_parse_dt`` corner cases (None, empty string, ISO
  with Z suffix, naive datetime, garbage value).
* Postgres path: every method goes through the fake psycopg.
"""
from __future__ import annotations

import json
import sys
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Module-level helpers — run BEFORE any per-test fixture.
# ---------------------------------------------------------------------------


def test_resolve_sqlite_path_memory_default():
    from greenlang.factors.ingestion.run_repository import _resolve_sqlite_path

    assert _resolve_sqlite_path("") == ":memory:"
    assert _resolve_sqlite_path("   ") == ":memory:"
    assert _resolve_sqlite_path("sqlite:///") == ":memory:"
    assert _resolve_sqlite_path("sqlite:///:memory:") == ":memory:"


def test_resolve_sqlite_path_percent_encoded():
    from greenlang.factors.ingestion.run_repository import _resolve_sqlite_path

    out = _resolve_sqlite_path("sqlite:///tmp/some%20dir/db.sqlite")
    assert "some dir" in out


def test_resolve_sqlite_path_windows_drive():
    from greenlang.factors.ingestion.run_repository import _resolve_sqlite_path

    # Windows-style absolute path with drive letter.
    out = _resolve_sqlite_path("sqlite:///C:/temp/db.sqlite")
    assert out.startswith("C:/") or out.startswith("/C:")


def test_resolve_sqlite_path_raw_pass_through():
    from greenlang.factors.ingestion.run_repository import _resolve_sqlite_path

    # A bare path with no sqlite: prefix is returned as-is.
    out = _resolve_sqlite_path("/already/a/path.db")
    assert out == "/already/a/path.db"


def test_is_postgres_dsn_recognises_both_schemes():
    from greenlang.factors.ingestion.run_repository import _is_postgres_dsn

    assert _is_postgres_dsn("postgresql://x@y/db") is True
    assert _is_postgres_dsn("postgres://x@y/db") is True
    assert _is_postgres_dsn("sqlite:///:memory:") is False
    assert _is_postgres_dsn("") is False


# ---------------------------------------------------------------------------
# SQLite repository — happy + edge paths.
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo():
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository

    r = IngestionRunRepository("sqlite:///:memory:")
    yield r
    r.close()


def test_create_then_get_round_trip(repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    fetched = repo.get(run.run_id)
    assert fetched.run_id == run.run_id
    assert fetched.source_urn == "urn:gl:source:test"
    assert fetched.status == RunStatus.CREATED


def test_get_unknown_id_raises_not_found(repo):
    from greenlang.factors.ingestion.run_repository import IngestionRunNotFoundError

    with pytest.raises(IngestionRunNotFoundError) as exc_info:
        repo.get("does-not-exist")
    assert "does-not-exist" in str(exc_info.value)


def test_update_status_unknown_run_raises(repo):
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.run_repository import IngestionRunNotFoundError

    with pytest.raises(IngestionRunNotFoundError):
        repo.update_status("missing", RunStatus.FETCHED)


def test_update_status_idempotent_same_status(repo):
    """Updating to the same status MUST be a no-op (assert_can_transition)."""
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    repo.update_status(run.run_id, RunStatus.CREATED)
    repo.update_status(run.run_id, RunStatus.CREATED, current_stage=Stage.FETCH)
    fetched = repo.get(run.run_id)
    assert fetched.status == RunStatus.CREATED


def test_update_status_terminal_failure_stamps_finished_at(repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    repo.update_status(
        run.run_id, RunStatus.FAILED, error_json={"reason": "boom"}
    )
    fetched = repo.get(run.run_id)
    assert fetched.status == RunStatus.FAILED
    assert fetched.finished_at is not None
    assert fetched.error_json == {"reason": "boom"}


def test_update_status_published_stamps_finished_at(repo):
    """``PUBLISHED`` is a success state but finishes the run."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    # Walk the state machine to STAGED then PUBLISHED.
    for s in (
        RunStatus.FETCHED,
        RunStatus.PARSED,
        RunStatus.NORMALIZED,
        RunStatus.VALIDATED,
        RunStatus.DEDUPED,
        RunStatus.STAGED,
        RunStatus.PUBLISHED,
    ):
        repo.update_status(run.run_id, s)
    fetched = repo.get(run.run_id)
    assert fetched.status == RunStatus.PUBLISHED
    assert fetched.finished_at is not None


def test_set_artifact_with_no_parser_fields_preserves_existing(repo):
    """COALESCE means parser_module/version/commit stay if not passed."""
    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    repo.set_artifact(
        run.run_id,
        artifact_id="art-1",
        sha256="0" * 64,
        parser_module="my.parser",
        parser_version="1.0",
        parser_commit="abc",
    )
    # Re-set artifact without parser_* — the old values must remain.
    repo.set_artifact(run.run_id, artifact_id="art-2", sha256="1" * 64)
    fetched = repo.get(run.run_id)
    assert fetched.artifact_id == "art-2"
    assert fetched.artifact_sha256 == "1" * 64
    assert fetched.parser_module == "my.parser"
    assert fetched.parser_version == "1.0"
    assert fetched.parser_commit == "abc"


def test_set_diff_upserts_on_repeat_call(repo):
    """The ON CONFLICT branch in ``_set_diff_sqlite`` is exercised on re-set."""
    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    repo.set_diff(
        run.run_id,
        diff_json_uri="file:///j1.json",
        diff_md_uri="file:///m1.md",
        summary_json={"added": 1},
    )
    repo.set_diff(
        run.run_id,
        diff_json_uri="file:///j2.json",
        diff_md_uri="file:///m2.md",
        summary_json={"added": 2},
    )
    fetched = repo.get(run.run_id)
    assert fetched.diff_json_uri == "file:///j2.json"
    assert fetched.diff_md_uri == "file:///m2.md"


def test_set_publish_stamps_batch_and_approver(repo):
    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    repo.set_publish(
        run.run_id, batch_id="batch-1", approved_by="human:lead@x.com"
    )
    fetched = repo.get(run.run_id)
    assert fetched.batch_id == "batch-1"
    assert fetched.approved_by == "human:lead@x.com"


def test_append_stage_history_persists_row(repo):
    from greenlang.factors.ingestion.pipeline import Stage, StageResult

    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    sr = StageResult(
        stage=Stage.FETCH, ok=True, duration_s=0.5, details={"bytes": 100}
    )
    repo.append_stage_history(run.run_id, sr)
    # No public read API; assert the SELECT directly.
    conn = repo._connect()
    rows = list(conn.execute(
        "SELECT stage, ok, duration_s FROM ingestion_run_stage_history WHERE run_id = ?",
        (run.run_id,),
    ))
    assert len(rows) == 1
    assert rows[0]["stage"] == "fetch"
    assert rows[0]["ok"] == 1


def test_list_by_status_and_source_return_multiple_rows(repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    r1 = repo.create(source_urn="urn:gl:source:a", source_version="1", operator="o")
    r2 = repo.create(source_urn="urn:gl:source:a", source_version="2", operator="o")
    r3 = repo.create(source_urn="urn:gl:source:b", source_version="1", operator="o")

    same_source = repo.list_by_source("urn:gl:source:a")
    assert {r.run_id for r in same_source} == {r1.run_id, r2.run_id}

    same_status = repo.list_by_status(RunStatus.CREATED)
    assert {r.run_id for r in same_status} == {r1.run_id, r2.run_id, r3.run_id}


def test_list_by_status_returns_empty_when_no_match(repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    assert repo.list_by_status(RunStatus.PUBLISHED) == []


def test_close_releases_in_memory_connection(repo):
    """``close()`` is idempotent when called twice."""
    repo.close()
    repo.close()
    assert repo._memory_conn is None


def test_repository_with_filesystem_dsn(tmp_path):
    """Filesystem-backed SQLite path opens fresh connections per call."""
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository

    db = tmp_path / "subdir" / "runs.db"
    repo = IngestionRunRepository(f"sqlite:///{db}")
    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    fetched = repo.get(run.run_id)
    assert fetched.run_id == run.run_id
    # Parent directory was auto-created.
    assert db.parent.exists()


def test_repository_with_naked_path_dsn(tmp_path):
    """A bare path (no ``sqlite:`` prefix) is treated as SQLite."""
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository

    db = tmp_path / "runs.db"
    repo = IngestionRunRepository(str(db))
    run = repo.create(
        source_urn="urn:gl:source:test",
        source_version="1",
        operator="o",
    )
    assert repo.get(run.run_id).run_id == run.run_id


def test_set_artifact_unknown_run_is_silent_noop(repo):
    """SQLite UPDATE matches zero rows; no exception is raised."""
    repo.set_artifact("missing", artifact_id="a", sha256="b")
    # Still raises NotFound on get — UPDATE silently no-op'd.
    from greenlang.factors.ingestion.run_repository import IngestionRunNotFoundError
    with pytest.raises(IngestionRunNotFoundError):
        repo.get("missing")


# ---------------------------------------------------------------------------
# _row_to_run + _parse_dt corner cases
# ---------------------------------------------------------------------------


def test_row_to_run_handles_missing_optional_fields():
    from greenlang.factors.ingestion.run_repository import _row_to_run
    from greenlang.factors.ingestion.pipeline import RunStatus

    row = {
        "run_id": "r1",
        "source_urn": None,
        "source_version": None,
        "started_at": None,
        "finished_at": None,
        "status": None,
        "current_stage": None,
        "operator": None,
        "error_json": None,
    }
    run = _row_to_run(row)
    assert run.run_id == "r1"
    assert run.source_urn == ""
    assert run.source_version == ""
    assert run.status == RunStatus.CREATED
    assert run.current_stage is None
    assert run.error_json is None


def test_row_to_run_parses_error_json_string():
    from greenlang.factors.ingestion.run_repository import _row_to_run

    row = {
        "run_id": "r1",
        "started_at": "2026-04-28T12:00:00Z",
        "status": "fetched",
        "error_json": json.dumps({"reason": "x"}),
    }
    run = _row_to_run(row)
    assert run.error_json == {"reason": "x"}
    assert run.started_at is not None


def test_row_to_run_handles_garbage_error_json():
    from greenlang.factors.ingestion.run_repository import _row_to_run

    row = {
        "run_id": "r1",
        "started_at": "2026-04-28T12:00:00Z",
        "status": "fetched",
        "error_json": "<<not-json>>",
    }
    run = _row_to_run(row)
    assert run.error_json == {"raw": "<<not-json>>"}


def test_row_to_run_passes_through_dict_error_json():
    from greenlang.factors.ingestion.run_repository import _row_to_run

    row = {
        "run_id": "r1",
        "started_at": "2026-04-28T12:00:00Z",
        "status": "fetched",
        "error_json": {"already": "dict"},
    }
    run = _row_to_run(row)
    assert run.error_json == {"already": "dict"}


def test_row_to_run_resolves_current_stage_enum():
    from greenlang.factors.ingestion.run_repository import _row_to_run
    from greenlang.factors.ingestion.pipeline import Stage

    row = {
        "run_id": "r1",
        "started_at": "2026-04-28T12:00:00Z",
        "status": "fetched",
        "current_stage": "fetch",
    }
    run = _row_to_run(row)
    assert run.current_stage == Stage.FETCH


def test_parse_dt_handles_none_and_empty_and_garbage():
    from greenlang.factors.ingestion.run_repository import _parse_dt

    assert _parse_dt(None) is None
    assert _parse_dt("") is None
    assert _parse_dt("not-a-date") is None


def test_parse_dt_handles_naive_datetime_object():
    from greenlang.factors.ingestion.run_repository import _parse_dt

    naive = datetime(2026, 4, 28, 12, 0, 0)
    out = _parse_dt(naive)
    assert out is not None
    assert out.tzinfo is not None


def test_parse_dt_passes_through_aware_datetime():
    from greenlang.factors.ingestion.run_repository import _parse_dt

    aware = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert _parse_dt(aware) is aware


def test_parse_dt_handles_iso_with_z_suffix():
    from greenlang.factors.ingestion.run_repository import _parse_dt

    out = _parse_dt("2026-04-28T12:00:00Z")
    assert out is not None
    assert out.tzinfo is not None


def test_parse_dt_handles_naive_iso_string():
    from greenlang.factors.ingestion.run_repository import _parse_dt

    out = _parse_dt("2026-04-28T12:00:00")
    assert out is not None
    assert out.tzinfo is not None


# ---------------------------------------------------------------------------
# Postgres branches — drive every _*_pg method through a fake psycopg.
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn
        self._fetched: Optional[List[Tuple]] = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def execute(self, sql: str, params: Tuple = ()) -> None:
        self._conn.executed.append((sql, tuple(params or ())))
        # Stash a canned result for the next fetchone / fetchall call.
        sql_upper = sql.upper()
        if "SELECT" in sql_upper:
            # ``ORDER BY`` only appears in the list_* helpers; the
            # ``get_pg`` helper terminates with ``WHERE run_id = %s``.
            if "ORDER BY" in sql_upper:
                self._fetched = self._conn.canned_list_rows[:]
            else:
                self._fetched = self._conn.canned_get_rows[:]
        else:
            self._fetched = None

    def fetchone(self) -> Optional[Tuple]:
        if not self._fetched:
            return None
        return self._fetched.pop(0)

    def fetchall(self) -> List[Tuple]:
        rows = self._fetched or []
        self._fetched = []
        return rows


class _FakeConn:
    def __init__(self) -> None:
        self.executed: List[Tuple[str, Tuple]] = []
        self.committed = 0
        self.canned_get_rows: List[Tuple] = []
        self.canned_list_rows: List[Tuple] = []

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        return None


class _FakePsycopg:
    """Minimal stand-in for the ``psycopg`` module."""

    def __init__(self) -> None:
        self.conns: List[_FakeConn] = []

    def connect(self, dsn: str) -> _FakeConn:
        c = _FakeConn()
        self.conns.append(c)
        return c


@pytest.fixture()
def fake_psycopg(monkeypatch):
    fake = _FakePsycopg()
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    return fake


@pytest.fixture()
def pg_repo(fake_psycopg):
    """A repository wired to the fake psycopg module."""
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository

    return IngestionRunRepository("postgresql://test/db")


def _pg_canned_run_row() -> Tuple:
    return (
        "run-pg-1",                # run_id
        "urn:gl:source:test",     # source_urn
        "2024.1",                 # source_version
        "2026-04-28T12:00:00Z",   # started_at
        None,                      # finished_at
        "created",                # status
        None,                      # current_stage
        None,                      # artifact_id
        None,                      # artifact_sha256
        None, None, None,          # parser_module/version/commit
        "bot:test",               # operator
        None, None, None,          # batch_id/approved_by/error_json
        None, None,                # diff_*_uri
    )


def test_pg_create_inserts_and_commits(pg_repo, fake_psycopg):
    pg_repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    # One INSERT executed + one commit.
    sqls = [s for s, _ in fake_psycopg.conns[-1].executed]
    assert any("INSERT INTO ingestion_runs" in s for s in sqls)
    assert fake_psycopg.conns[-1].committed == 1


def test_pg_get_unknown_returns_not_found(pg_repo, fake_psycopg):
    """When fetchone() returns None, ``get`` raises ``IngestionRunNotFoundError``."""
    from greenlang.factors.ingestion.run_repository import IngestionRunNotFoundError

    # No canned row → fetchone() returns None.
    with pytest.raises(IngestionRunNotFoundError):
        pg_repo.get("missing-pg")


def test_pg_get_returns_run_when_row_found(pg_repo, fake_psycopg):
    # Pre-create a fake conn with canned data so the next get() returns it.
    fake = fake_psycopg
    canned = _pg_canned_run_row()

    # Patch ``connect`` so each call returns a fresh fake-conn loaded
    # with our canned row.
    def _connect_with_data(dsn: str) -> _FakeConn:
        c = _FakeConn()
        c.canned_get_rows = [canned]
        fake.conns.append(c)
        return c

    fake.connect = _connect_with_data  # type: ignore[assignment]

    run = pg_repo.get("run-pg-1")
    assert run.run_id == "run-pg-1"
    assert run.source_urn == "urn:gl:source:test"


def test_pg_update_status_executes_update(pg_repo, fake_psycopg):
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    canned = _pg_canned_run_row()

    def _connect_with_data(dsn: str) -> _FakeConn:
        c = _FakeConn()
        c.canned_get_rows = [canned]
        fake_psycopg.conns.append(c)
        return c

    fake_psycopg.connect = _connect_with_data  # type: ignore[assignment]

    pg_repo.update_status(
        "run-pg-1",
        RunStatus.FETCHED,
        current_stage=Stage.FETCH,
        error_json=None,
    )
    sqls = [s for s, _ in fake_psycopg.conns[-1].executed]
    assert any("UPDATE ingestion_runs SET status" in s for s in sqls)
    assert fake_psycopg.conns[-1].committed == 1


def test_pg_set_artifact_runs(pg_repo, fake_psycopg):
    pg_repo.set_artifact(
        "run-pg-1",
        artifact_id="a",
        sha256="b" * 64,
        parser_module="m",
        parser_version="1.0",
        parser_commit="c",
    )
    sqls = [s for s, _ in fake_psycopg.conns[-1].executed]
    assert any("UPDATE ingestion_runs" in s and "artifact_id" in s for s in sqls)
    assert fake_psycopg.conns[-1].committed == 1


def test_pg_set_diff_runs_update_and_upsert(pg_repo, fake_psycopg):
    pg_repo.set_diff(
        "run-pg-1",
        diff_json_uri="s3://j",
        diff_md_uri="s3://m",
        summary_json={"added": 1},
    )
    sqls = [s for s, _ in fake_psycopg.conns[-1].executed]
    assert any("UPDATE ingestion_runs SET diff_json_uri" in s for s in sqls)
    assert any("INSERT INTO ingestion_run_diffs" in s for s in sqls)
    assert fake_psycopg.conns[-1].committed == 1


def test_pg_set_publish_runs(pg_repo, fake_psycopg):
    pg_repo.set_publish(
        "run-pg-1", batch_id="bx", approved_by="human:x@y.z"
    )
    sqls = [s for s, _ in fake_psycopg.conns[-1].executed]
    assert any("UPDATE ingestion_runs SET batch_id" in s for s in sqls)
    assert fake_psycopg.conns[-1].committed == 1


def test_pg_append_stage_runs(pg_repo, fake_psycopg):
    from greenlang.factors.ingestion.pipeline import Stage, StageResult

    sr = StageResult(stage=Stage.FETCH, ok=True, duration_s=0.5)
    pg_repo.append_stage_history("run-pg-1", sr)
    sqls = [s for s, _ in fake_psycopg.conns[-1].executed]
    assert any("INSERT INTO ingestion_run_stage_history" in s for s in sqls)
    assert fake_psycopg.conns[-1].committed == 1


def test_pg_list_by_status_executes_select(pg_repo, fake_psycopg):
    from greenlang.factors.ingestion.pipeline import RunStatus

    canned = _pg_canned_run_row()

    def _connect_with_data(dsn: str) -> _FakeConn:
        c = _FakeConn()
        c.canned_list_rows = [canned]
        fake_psycopg.conns.append(c)
        return c

    fake_psycopg.connect = _connect_with_data  # type: ignore[assignment]

    rows = pg_repo.list_by_status(RunStatus.CREATED)
    assert len(rows) == 1
    assert rows[0].run_id == "run-pg-1"


def test_pg_list_by_source_executes_select(pg_repo, fake_psycopg):
    canned = _pg_canned_run_row()

    def _connect_with_data(dsn: str) -> _FakeConn:
        c = _FakeConn()
        c.canned_list_rows = [canned]
        fake_psycopg.conns.append(c)
        return c

    fake_psycopg.connect = _connect_with_data  # type: ignore[assignment]

    rows = pg_repo.list_by_source("urn:gl:source:test")
    assert len(rows) == 1


def test_pg_connect_path_calls_real_module(monkeypatch, fake_psycopg):
    """``_pg_connect`` lazy-imports psycopg via ``import psycopg`` line."""
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository

    repo = IngestionRunRepository("postgresql://test/db")
    conn = repo._pg_connect()
    assert isinstance(conn, _FakeConn)
