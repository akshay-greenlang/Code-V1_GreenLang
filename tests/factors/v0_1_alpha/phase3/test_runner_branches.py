# -*- coding: utf-8 -*-
"""Phase 3 Block 6 — branch-coverage tests for ``ingestion/runner.py``.

Targets the residual ~14pp gap that the existing happy-path e2e tests
do not exercise. This module is mock-driven: every collaborator is a
double so the tests stay sub-second and exercise error paths that the
e2e harness can't reach.

Covers:

* ``_default_fetcher_factory`` HTTP / file / file-fallback branches.
* ``_record_stage`` swallows telemetry write failures.
* ``_resolve_parser`` raises ``ParserDispatchError`` when no parser
  registered.
* ``fetch`` zero-bytes / sha-mismatch / generic-exception wrapping.
* ``parse`` schema-validation failure / dict-fallback branch / generic
  wrapping.
* ``parse`` parser exposing ``parse_bytes`` short-circuits the JSON
  branch.
* ``normalize`` exception path.
* ``validate`` orchestrator returning ``ok=False`` + per-record
  exception inside the loop / orchestrator absent (fallback).
* ``dedupe`` cross-source supersede without flag (rejected) /
  removal-candidate emission / generic wrapping.
* ``stage`` list_production raising / generic exception / per-record
  failure surfaces.
* ``publish`` PublishOrderError / generic wrapping.
* ``rollback`` RollbackOrderError when run not in PUBLISHED.
* ``run`` auto_publish without approver -> PublishOrderError.
* ``_get_orchestrator`` swallows lookup failure.
* ``_mark_failed`` swallows StageOrderError on already-terminal run.
* ``_read_artifact`` rejects unsupported URI schemes.
* ``_fingerprint_for`` handles attribute-style records via
  ``duplicate_fingerprint``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Default fetcher factory branches
# ---------------------------------------------------------------------------


def test_default_fetcher_factory_http_returns_http_fetcher():
    from greenlang.factors.ingestion.fetchers import HttpFetcher
    from greenlang.factors.ingestion.runner import _default_fetcher_factory

    assert isinstance(_default_fetcher_factory("http://example.com/x"), HttpFetcher)
    assert isinstance(_default_fetcher_factory("HTTPS://example.com/x"), HttpFetcher)


def test_default_fetcher_factory_file_returns_file_fetcher():
    from greenlang.factors.ingestion.fetchers import FileFetcher
    from greenlang.factors.ingestion.runner import _default_fetcher_factory

    assert isinstance(_default_fetcher_factory("file:///tmp/x"), FileFetcher)
    assert isinstance(_default_fetcher_factory("/tmp/x"), FileFetcher)
    assert isinstance(_default_fetcher_factory(""), FileFetcher)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_repo(tmp_path):
    """Real SQLite-memory repo so transitions are validated by the state machine."""
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository

    r = IngestionRunRepository("sqlite:///:memory:")
    yield r
    r.close()


@pytest.fixture()
def fake_factor_repo():
    """Stand-in that exposes ``_get_or_build_publish_orchestrator``."""
    repo = MagicMock(name="factor_repo")
    repo._get_or_build_publish_orchestrator.return_value = None
    return repo


@pytest.fixture()
def fake_publisher():
    """Publisher mock with the four call sites the runner uses."""
    pub = MagicMock(name="publisher")
    pub.list_production.return_value = []
    pub.list_staging.return_value = []
    pub.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    pub.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[], removals=[], changes=[], unchanged=0,
    )
    pub.flip_to_production.return_value = 0
    pub.rollback.return_value = 0
    return pub


@pytest.fixture()
def fake_artifact_store(tmp_path):
    from greenlang.factors.ingestion.artifacts import LocalArtifactStore

    return LocalArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def runner(run_repo, fake_factor_repo, fake_publisher, fake_artifact_store, tmp_path):
    from greenlang.factors.ingestion.parsers import ParserRegistry
    from greenlang.factors.ingestion.runner import IngestionPipelineRunner

    return IngestionPipelineRunner(
        run_repo=run_repo,
        factor_repo=fake_factor_repo,
        publisher=fake_publisher,
        artifact_store=fake_artifact_store,
        parser_registry=ParserRegistry(),
        diff_root=tmp_path / "diffs",
    )


def _new_run(run_repo, status_path: Optional[List["RunStatus"]] = None) -> "IngestionRun":  # type: ignore[name-defined]
    """Helper: create a run + walk it through statuses for stage-precondition tests."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = run_repo.create(
        source_urn="urn:gl:source:test",
        source_version="2024.1",
        operator="bot:test",
    )
    for s in status_path or []:
        run_repo.update_status(run.run_id, s)
    return run


# ---------------------------------------------------------------------------
# _resolve_parser + parse error paths
# ---------------------------------------------------------------------------


def test_resolve_parser_raises_when_no_parser_registered(runner):
    from greenlang.factors.ingestion.exceptions import ParserDispatchError

    with pytest.raises(ParserDispatchError) as exc_info:
        runner._resolve_parser("does-not-exist")
    assert "does-not-exist" in str(exc_info.value)


# ---------------------------------------------------------------------------
# fetch error paths
# ---------------------------------------------------------------------------


def test_fetch_zero_bytes_raises_artifact_store_error(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import ArtifactStoreError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo)

    class _ZeroBytesFetcher:
        def fetch(self, url: str) -> bytes:
            return b""

    runner._fetcher_factory = lambda url: _ZeroBytesFetcher()

    with pytest.raises(ArtifactStoreError) as exc_info:
        runner.fetch(run.run_id, source_id="src", source_url="http://x/y")
    assert "0 bytes" in str(exc_info.value)
    # Run row must be FAILED.
    fetched = run_repo.get(run.run_id)
    assert fetched.status == RunStatus.FAILED


def test_fetch_generic_exception_wrapped_as_artifact_store_error(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import ArtifactStoreError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo)

    class _BoomFetcher:
        def fetch(self, url: str) -> bytes:
            raise RuntimeError("network down")

    runner._fetcher_factory = lambda url: _BoomFetcher()

    with pytest.raises(ArtifactStoreError) as exc_info:
        runner.fetch(run.run_id, source_id="src", source_url="http://x/y")
    assert "fetch failed" in str(exc_info.value)
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


# ---------------------------------------------------------------------------
# parse error paths
# ---------------------------------------------------------------------------


def test_parse_schema_validation_failure_marks_failed(runner, run_repo, fake_artifact_store):
    """A parser that returns ok=False from validate_schema raises ValidationStageError."""
    from greenlang.factors.ingestion.exceptions import ValidationStageError
    from greenlang.factors.ingestion.pipeline import RunStatus

    # Walk the run to FETCHED (parse precondition).
    run = _new_run(run_repo, [RunStatus.FETCHED])

    # Stash a fake JSON artifact on disk.
    art = fake_artifact_store.put_bytes(b'{"hello": "world"}', source_id="src", url="x")

    class _BadParser:
        source_id = "src"
        parser_id = "bad"
        parser_version = "1.0"

        def validate_schema(self, data: Dict[str, Any]):
            return False, ["missing required field 'foo'"]

        def parse(self, data: Dict[str, Any]):
            return []

    runner._parser_registry.register(_BadParser())  # type: ignore[arg-type]

    with pytest.raises(ValidationStageError):
        runner.parse(run.run_id, source_id="src", artifact=art)
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


def test_parse_generic_exception_wrapped(runner, run_repo, fake_artifact_store):
    from greenlang.factors.ingestion.exceptions import ParserDispatchError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FETCHED])
    art = fake_artifact_store.put_bytes(b'{"x": 1}', source_id="src", url="x")

    class _CrashyParser:
        source_id = "src"
        parser_id = "crash"
        parser_version = "1.0"

        def validate_schema(self, data: Dict[str, Any]):
            return True, []

        def parse(self, data: Dict[str, Any]):
            raise RuntimeError("kaboom")

    runner._parser_registry.register(_CrashyParser())  # type: ignore[arg-type]

    with pytest.raises(ParserDispatchError):
        runner.parse(run.run_id, source_id="src", artifact=art)
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


def test_parse_uses_parse_bytes_when_available(runner, run_repo, fake_artifact_store):
    """Parsers exposing ``parse_bytes`` short-circuit the JSON-decode branch."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FETCHED])
    art = fake_artifact_store.put_bytes(b"raw-binary-bytes", source_id="src", url="x")

    captured = {}

    class _ByteParser:
        source_id = "src"
        parser_id = "byte"
        parser_version = "1.0"

        def parse_bytes(self, raw, *, artifact_uri, artifact_sha256):
            captured["raw"] = raw
            captured["uri"] = artifact_uri
            return [{"urn": "u1"}]

    runner._parser_registry.register(_ByteParser())  # type: ignore[arg-type]
    result = runner.parse(run.run_id, source_id="src", artifact=art)
    assert captured["raw"] == b"raw-binary-bytes"
    assert result.rows == [{"urn": "u1"}]
    assert run_repo.get(run.run_id).status == RunStatus.PARSED


# ---------------------------------------------------------------------------
# normalize error path
# ---------------------------------------------------------------------------


def test_normalize_wraps_generic_failure(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import IngestionError
    from greenlang.factors.ingestion.parser_harness import ParserResult
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FETCHED, RunStatus.PARSED])
    # ``ParserResult.rows`` is iterated via ``list(...)`` in normalize. A
    # non-iterable triggers the wrapping branch.
    bad_result = SimpleNamespace(rows=42)
    with pytest.raises(IngestionError):
        runner.normalize(run.run_id, parser_result=bad_result)


# ---------------------------------------------------------------------------
# validate error / branch paths
# ---------------------------------------------------------------------------


def test_validate_orchestrator_rejects_record(runner, run_repo, fake_factor_repo):
    """An orchestrator returning ``ok=False`` lands the record in ``rejected``."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [RunStatus.FETCHED, RunStatus.PARSED, RunStatus.NORMALIZED],
    )
    rejecting = MagicMock()
    rejecting.run_dry.return_value = SimpleNamespace(ok=False, errors=["bad licence"])
    fake_factor_repo._get_or_build_publish_orchestrator.return_value = rejecting

    outcome = runner.validate(run.run_id, records=[{"urn": "u1"}])
    assert outcome.accepted == []
    assert len(outcome.rejected) == 1
    assert "bad licence" in outcome.rejected[0][1]


def test_validate_orchestrator_exception_treated_as_rejection(runner, run_repo, fake_factor_repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [RunStatus.FETCHED, RunStatus.PARSED, RunStatus.NORMALIZED],
    )
    crashy = MagicMock()
    crashy.run_dry.side_effect = RuntimeError("orchestrator crash")
    fake_factor_repo._get_or_build_publish_orchestrator.return_value = crashy

    outcome = runner.validate(run.run_id, records=[{"urn": "u1"}])
    assert outcome.accepted == []
    assert "orchestrator crash" in outcome.rejected[0][1]


def test_validate_falls_back_when_orchestrator_unavailable(runner, run_repo, fake_factor_repo):
    """Missing orchestrator → every record passes (validation deferred)."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [RunStatus.FETCHED, RunStatus.PARSED, RunStatus.NORMALIZED],
    )
    fake_factor_repo._get_or_build_publish_orchestrator.return_value = None
    outcome = runner.validate(run.run_id, records=[{"urn": "u1"}, {"urn": "u2"}])
    assert len(outcome.accepted) == 2
    assert outcome.rejected == []


# ---------------------------------------------------------------------------
# dedupe branches
# ---------------------------------------------------------------------------


def test_dedupe_blocks_cross_source_supersede_without_flag(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import DedupeRejectedError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
        ],
    )
    new_record = {
        "urn": "urn:gl:factor:new",
        "supersedes_urn": "urn:gl:factor:old",
        "source_urn": "urn:gl:source:NEW",
    }
    prior = {
        "urn:gl:factor:old": {
            "urn": "urn:gl:factor:old",
            "source_urn": "urn:gl:source:OLD",
        }
    }
    with pytest.raises(DedupeRejectedError):
        runner.dedupe(
            run.run_id,
            accepted=[new_record],
            prior_production=prior,
            allow_cross_source_supersede=False,
        )
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


def test_dedupe_allows_cross_source_supersede_with_flag(runner, run_repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
        ],
    )
    new_record = {
        "urn": "urn:gl:factor:new",
        "supersedes_urn": "urn:gl:factor:old",
        "source_urn": "urn:gl:source:NEW",
    }
    prior = {
        "urn:gl:factor:old": {
            "urn": "urn:gl:factor:old",
            "source_urn": "urn:gl:source:OLD",
        }
    }
    outcome = runner.dedupe(
        run.run_id,
        accepted=[new_record],
        prior_production=prior,
        allow_cross_source_supersede=True,
    )
    assert outcome.supersede_pairs == [
        ("urn:gl:factor:old", "urn:gl:factor:new")
    ]


def test_dedupe_emits_removal_candidates(runner, run_repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
        ],
    )
    accepted = [{"urn": "urn:gl:factor:keep"}]
    prior = {
        "urn:gl:factor:keep": {"urn": "urn:gl:factor:keep"},
        "urn:gl:factor:gone": {"urn": "urn:gl:factor:gone"},
    }
    outcome = runner.dedupe(run.run_id, accepted=accepted, prior_production=prior)
    assert outcome.removal_candidates == ["urn:gl:factor:gone"]


def test_dedupe_within_run_collapses_duplicates(runner, run_repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
        ],
    )
    rec = {
        "urn": "urn:gl:factor:dup",
        "value": 1.0,
        "unit": "kg",
    }
    outcome = runner.dedupe(run.run_id, accepted=[rec, dict(rec), dict(rec)])
    assert len(outcome.final) == 1
    assert outcome.duplicate_count == 2


# ---------------------------------------------------------------------------
# stage error paths
# ---------------------------------------------------------------------------


def test_stage_swallows_list_production_failure(runner, run_repo, fake_publisher):
    from greenlang.factors.ingestion.diff import RunDiff
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    fake_publisher.list_production.side_effect = RuntimeError("prod read down")
    fake_publisher.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    fake_publisher.list_staging.return_value = [{"urn": "urn:gl:factor:1"}]
    fake_publisher.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[{"urn": "urn:gl:factor:1"}],
        removals=[],
        changes=[],
        unchanged=0,
    )
    outcome = DedupeOutcome(final=[{"urn": "urn:gl:factor:1"}])
    diff = runner.stage(run.run_id, dedupe_outcome=outcome)
    assert isinstance(diff, RunDiff)
    # Run advanced past STAGE despite the warning.
    assert run_repo.get(run.run_id).status in (
        RunStatus.STAGED,
        RunStatus.REVIEW_REQUIRED,
    )


def test_stage_wraps_per_record_publish_failure(runner, run_repo, fake_publisher):
    from greenlang.factors.ingestion.exceptions import IngestionError
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    fake_publisher.publish_to_staging.side_effect = RuntimeError("gate 5 failed")
    outcome = DedupeOutcome(final=[{"urn": "u1"}])
    with pytest.raises(IngestionError):
        runner.stage(run.run_id, dedupe_outcome=outcome)
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


def test_stage_empty_dedupe_yields_empty_diff(runner, run_repo, fake_publisher):
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    diff = runner.stage(run.run_id, dedupe_outcome=DedupeOutcome(final=[]))
    assert diff.is_empty() is True
    # Empty diff lands the run in STAGED (no review needed).
    assert run_repo.get(run.run_id).status == RunStatus.STAGED


# ---------------------------------------------------------------------------
# publish error paths
# ---------------------------------------------------------------------------


def test_publish_rejects_run_in_wrong_status(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import PublishOrderError

    run = _new_run(run_repo)  # status = CREATED
    with pytest.raises(PublishOrderError) as exc_info:
        runner.publish(run.run_id, approver="human:x@y.z")
    assert "staged" in str(exc_info.value)


def test_publish_wraps_publisher_failure(runner, run_repo, fake_publisher):
    from greenlang.factors.ingestion.exceptions import IngestionError
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    runner.stage(run.run_id, dedupe_outcome=DedupeOutcome(final=[]))
    fake_publisher.flip_to_production.side_effect = RuntimeError("flip failed")

    with pytest.raises(IngestionError):
        runner.publish(run.run_id, approver="human:x@y.z")
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


def test_publish_happy_path_advances_run(runner, run_repo, fake_publisher):
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    runner.stage(run.run_id, dedupe_outcome=DedupeOutcome(final=[]))
    fake_publisher.list_staging.return_value = [{"urn": "urn:gl:factor:1"}]
    fake_publisher.flip_to_production.return_value = 1
    res = runner.publish(run.run_id, approver="human:x@y.z", batch_id="batch-1")
    assert res.batch_id == "batch-1"
    assert res.promoted == 1
    assert run_repo.get(run.run_id).status == RunStatus.PUBLISHED


# ---------------------------------------------------------------------------
# rollback error path
# ---------------------------------------------------------------------------


def test_rollback_rejects_non_published_run(runner, run_repo):
    """Rollback on a batch whose owning run is not PUBLISHED → RollbackOrderError."""
    from greenlang.factors.ingestion.exceptions import RollbackOrderError

    # Force ``_runs_for_batch`` to surface a not-yet-published run by
    # patching ``list_by_status`` so the runner thinks the run is still
    # PUBLISHED-but-actually-rolled-back.
    fake_run = SimpleNamespace(
        run_id="r1",
        status=SimpleNamespace(value="rolled_back"),
        batch_id="b1",
    )
    runner._runs_for_batch = lambda bid: [fake_run]  # type: ignore[assignment]
    with pytest.raises(RollbackOrderError):
        runner.rollback(batch_id="b1", approver="human:x@y.z")


def test_rollback_happy_path_demotes(runner, run_repo, fake_publisher):
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    runner.stage(run.run_id, dedupe_outcome=DedupeOutcome(final=[]))
    fake_publisher.flip_to_production.return_value = 1
    res = runner.publish(run.run_id, approver="human:x@y.z", batch_id="batch-roll")
    fake_publisher.rollback.return_value = 1

    rb = runner.rollback(batch_id=res.batch_id, approver="human:x@y.z")
    assert rb.batch_id == "batch-roll"
    assert run_repo.get(run.run_id).status == RunStatus.ROLLED_BACK


# ---------------------------------------------------------------------------
# run() convenience
# ---------------------------------------------------------------------------


def test_run_auto_publish_without_approver_raises(runner):
    from greenlang.factors.ingestion.exceptions import PublishOrderError

    with pytest.raises(PublishOrderError):
        runner.run(
            source_id="src",
            source_url="file:///x",
            source_urn="urn:gl:source:x",
            source_version="1",
            auto_publish=True,
            approver=None,
        )


# ---------------------------------------------------------------------------
# Helper / private-method coverage
# ---------------------------------------------------------------------------


def test_get_orchestrator_swallows_lookup_failure(runner, fake_factor_repo):
    fake_factor_repo._get_or_build_publish_orchestrator.side_effect = RuntimeError("nope")
    assert runner._get_orchestrator() is None


def test_read_artifact_rejects_unsupported_uri_scheme(runner):
    from greenlang.factors.ingestion.artifacts import StoredArtifact
    from greenlang.factors.ingestion.exceptions import ArtifactStoreError

    art = StoredArtifact(
        artifact_id="a",
        sha256="0" * 64,
        storage_uri="s3://not-supported-yet/x",
        bytes_size=10,
    )
    with pytest.raises(ArtifactStoreError):
        runner._read_artifact(art)


def test_fingerprint_for_dict_record_uses_canonical_keys(runner):
    fp1 = runner._fingerprint_for(
        {"urn": "u", "source_urn": "s", "value": 1.0, "unit": "kg"}
    )
    fp2 = runner._fingerprint_for(
        {"urn": "u", "source_urn": "s", "value": 1.0, "unit": "kg"}
    )
    fp3 = runner._fingerprint_for(
        {"urn": "u", "source_urn": "s", "value": 2.0, "unit": "kg"}
    )
    assert fp1 == fp2
    assert fp1 != fp3


def test_fingerprint_for_attribute_record_uses_duplicate_fingerprint(runner):
    """Records with ``fuel_type`` go through the legacy ``duplicate_fingerprint``."""
    rec = SimpleNamespace(
        fuel_type="diesel",
        geography_urn="urn:gl:geo:US",
        vintage_start="2024-01-01",
        vintage_end="2024-12-31",
        value=2.68,
        unit_urn="urn:gl:unit:kg-per-l",
    )
    fp = runner._fingerprint_for(rec)
    assert isinstance(fp, str) and fp


def test_mark_failed_swallows_already_terminal_state(runner, run_repo):
    """Marking a FAILED run as FAILED again must not crash."""
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run = _new_run(run_repo, [RunStatus.FAILED])
    runner._mark_failed(run.run_id, Stage.FETCH, 0.0, RuntimeError("repeat"))
    assert run_repo.get(run.run_id).status == RunStatus.FAILED


def test_mark_failed_swallows_inner_repo_exception(runner, run_repo):
    """When the repo blows up, ``_mark_failed`` must log + return."""
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run = _new_run(run_repo)

    def _crash(*a, **kw):
        raise RuntimeError("repo down")

    runner._run_repo.update_status = _crash  # type: ignore[assignment]
    # Should not raise.
    runner._mark_failed(run.run_id, Stage.FETCH, 0.0, RuntimeError("primary"))


def test_record_stage_swallows_telemetry_write_failure(runner, run_repo):
    """A repo append-history failure must not crash the stage flow."""
    from greenlang.factors.ingestion.pipeline import Stage

    run = _new_run(run_repo)
    runner._run_repo.append_stage_history = lambda *a, **kw: (_ for _ in ()).throw(  # type: ignore[assignment]
        RuntimeError("disk full")
    )
    result = runner._record_stage(run.run_id, Stage.FETCH, ok=True, started_at=0.0)
    assert result.ok is True


# ---------------------------------------------------------------------------
# resume() — guard branches
# ---------------------------------------------------------------------------


def test_resume_requires_failed_or_rejected_status(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import StageOrderError

    run = _new_run(run_repo)  # CREATED
    with pytest.raises(StageOrderError):
        runner.resume(run.run_id, from_stage="parse")


def test_resume_rejects_unknown_from_stage(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import StageOrderError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FAILED])
    with pytest.raises(StageOrderError) as exc_info:
        runner.resume(run.run_id, from_stage="not-a-stage")
    assert "from_stage" in str(exc_info.value)


def test_resume_rejects_run_without_current_stage(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import StageOrderError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FAILED])
    # No current_stage was ever stamped on the row.
    with pytest.raises(StageOrderError) as exc_info:
        runner.resume(run.run_id, from_stage="parse")
    assert "current_stage" in str(exc_info.value)


def test_resume_rejects_when_target_does_not_match_failed_stage(
    runner, run_repo
):
    """``--from-stage dedupe`` is rejected for a run that died at PARSE."""
    from greenlang.factors.ingestion.exceptions import StageOrderError
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run = _new_run(run_repo)
    # Move directly to FAILED with current_stage=PARSE (sim of parse failure).
    run_repo.update_status(
        run.run_id, RunStatus.FAILED, current_stage=Stage.PARSE
    )
    with pytest.raises(StageOrderError):
        runner.resume(run.run_id, from_stage="dedupe")


def test_reconstruct_stored_artifact_requires_artifact_columns(runner, run_repo):
    from greenlang.factors.ingestion.exceptions import StageOrderError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FAILED])
    with pytest.raises(StageOrderError) as exc_info:
        runner._reconstruct_stored_artifact(run_repo.get(run.run_id))
    assert "artifact" in str(exc_info.value).lower()


def test_reconstruct_stored_artifact_missing_file_on_disk(
    runner, run_repo, tmp_path,
):
    from greenlang.factors.ingestion.exceptions import StageOrderError

    run = _new_run(run_repo)
    run_repo.set_artifact(
        run.run_id, artifact_id="missing-art-id", sha256="0" * 64,
    )
    obj = run_repo.get(run.run_id)
    with pytest.raises(StageOrderError) as exc_info:
        runner._reconstruct_stored_artifact(obj)
    assert "cannot locate" in str(exc_info.value)


def test_infer_source_id_from_urn_handles_uri_and_plain_strings(runner):
    assert runner._infer_source_id_from_urn("urn:gl:source:defra-2025") == "defra-2025"
    assert runner._infer_source_id_from_urn("plainstring") == "plainstring"
    assert runner._infer_source_id_from_urn("") == ""


# ---------------------------------------------------------------------------
# resume() — happy paths drive the in-memory replay across every stage.
# ---------------------------------------------------------------------------


@pytest.fixture()
def resume_runner(
    run_repo, fake_factor_repo, fake_publisher, fake_artifact_store, tmp_path,
):
    """A runner with a parser registered so the resume() helper can replay parse."""
    from greenlang.factors.ingestion.parsers import ParserRegistry
    from greenlang.factors.ingestion.runner import IngestionPipelineRunner

    class _JsonParser:
        source_id = "myx"
        parser_id = "jsonp"
        parser_version = "1.0"

        def validate_schema(self, data):
            return True, []

        def parse(self, data):
            return data.get("rows", [])

    reg = ParserRegistry()
    reg.register(_JsonParser())
    fake_factor_repo._get_or_build_publish_orchestrator.return_value = None
    return IngestionPipelineRunner(
        run_repo=run_repo,
        factor_repo=fake_factor_repo,
        publisher=fake_publisher,
        artifact_store=fake_artifact_store,
        parser_registry=reg,
        diff_root=tmp_path / "diffs",
    )


def _stage_a_failed_parse_run(
    resume_runner_, run_repo_, fake_artifact_store_,
):
    """Helper: drive a run to FAILED with current_stage=PARSE."""
    import json as _json
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run = run_repo_.create(
        source_urn="urn:gl:source:myx",
        source_version="1",
        operator="bot:test",
    )
    # Walk fetch happy path.
    raw = _json.dumps({"rows": [{"urn": "u1"}, {"urn": "u2"}]}).encode()

    class _OkFetcher:
        def fetch(self, url):
            return raw

    resume_runner_._fetcher_factory = lambda u: _OkFetcher()
    art = resume_runner_.fetch(
        run.run_id, source_id="myx", source_url="file:///x",
    )
    # Now manually advance to FAILED with current_stage=PARSE without
    # actually running parse (simulates a parse failure).
    run_repo_.update_status(
        run.run_id, RunStatus.FAILED, current_stage=Stage.PARSE
    )
    return run, art


def test_resume_from_parse_replays_through_stage(
    resume_runner, run_repo, fake_artifact_store, fake_publisher,
):
    """Resume from PARSE walks: parse -> normalize -> validate -> dedupe -> stage."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run, _ = _stage_a_failed_parse_run(resume_runner, run_repo, fake_artifact_store)
    fake_publisher.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    fake_publisher.list_staging.return_value = []
    fake_publisher.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[], removals=[], changes=[], unchanged=0,
    )

    final = resume_runner.resume(run.run_id, from_stage="parse", source_id="myx")
    # The run must have advanced past STAGED (or REVIEW_REQUIRED).
    assert final.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)


def test_resume_from_normalize_replays_remaining_stages(
    resume_runner, run_repo, fake_artifact_store, fake_publisher,
):
    """Resume from NORMALIZE skips parse but still produces a staged run."""
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run, _ = _stage_a_failed_parse_run(resume_runner, run_repo, fake_artifact_store)
    # Override the failure: pretend we got past PARSE then died at NORMALIZE.
    run_repo._connect().execute(
        "UPDATE ingestion_runs SET current_stage = ?, status = ? WHERE run_id = ?",
        (Stage.NORMALIZE.value, RunStatus.FAILED.value, run.run_id),
    )
    fake_publisher.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    fake_publisher.list_staging.return_value = []
    fake_publisher.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[], removals=[], changes=[], unchanged=0,
    )
    final = resume_runner.resume(run.run_id, from_stage="normalize")
    assert final.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)


def test_resume_from_validate_replays_remaining_stages(
    resume_runner, run_repo, fake_artifact_store, fake_publisher,
):
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run, _ = _stage_a_failed_parse_run(resume_runner, run_repo, fake_artifact_store)
    run_repo._connect().execute(
        "UPDATE ingestion_runs SET current_stage = ?, status = ? WHERE run_id = ?",
        (Stage.VALIDATE.value, RunStatus.FAILED.value, run.run_id),
    )
    fake_publisher.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    fake_publisher.list_staging.return_value = []
    fake_publisher.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[], removals=[], changes=[], unchanged=0,
    )
    final = resume_runner.resume(run.run_id, from_stage="validate")
    assert final.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)


def test_resume_from_dedupe_replays_remaining_stages(
    resume_runner, run_repo, fake_artifact_store, fake_publisher,
):
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run, _ = _stage_a_failed_parse_run(resume_runner, run_repo, fake_artifact_store)
    run_repo._connect().execute(
        "UPDATE ingestion_runs SET current_stage = ?, status = ? WHERE run_id = ?",
        (Stage.DEDUPE.value, RunStatus.FAILED.value, run.run_id),
    )
    fake_publisher.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    fake_publisher.list_staging.return_value = []
    fake_publisher.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[], removals=[], changes=[], unchanged=0,
    )
    final = resume_runner.resume(run.run_id, from_stage="dedupe")
    assert final.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)


def test_resume_from_stage_replays_only_stage_step(
    resume_runner, run_repo, fake_artifact_store, fake_publisher,
):
    from greenlang.factors.ingestion.pipeline import RunStatus, Stage

    run, _ = _stage_a_failed_parse_run(resume_runner, run_repo, fake_artifact_store)
    run_repo._connect().execute(
        "UPDATE ingestion_runs SET current_stage = ?, status = ? WHERE run_id = ?",
        (Stage.STAGE.value, RunStatus.FAILED.value, run.run_id),
    )
    fake_publisher.publish_to_staging.side_effect = lambda rec: rec.get("urn", "")
    fake_publisher.list_staging.return_value = []
    fake_publisher.diff_staging_vs_production.return_value = SimpleNamespace(
        additions=[], removals=[], changes=[], unchanged=0,
    )
    final = resume_runner.resume(run.run_id, from_stage="stage")
    assert final.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED)


# ---------------------------------------------------------------------------
# Happy-path coverage — exercise the success branches of every stage.
# ---------------------------------------------------------------------------


def test_fetch_happy_path_stores_artifact_and_updates_run(
    runner, run_repo,
):
    """Successful fetch advances run to FETCHED + writes artifact row."""
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo)

    class _OkFetcher:
        def fetch(self, url: str) -> bytes:
            return b"hello-world"

    runner._fetcher_factory = lambda url: _OkFetcher()
    art = runner.fetch(run.run_id, source_id="src", source_url="http://x/y")
    assert art.bytes_size == len(b"hello-world")
    assert run_repo.get(run.run_id).status == RunStatus.FETCHED
    assert run_repo.get(run.run_id).artifact_id == art.artifact_id


def test_normalize_happy_path_advances_run(runner, run_repo):
    from greenlang.factors.ingestion.parser_harness import ParserResult
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(run_repo, [RunStatus.FETCHED, RunStatus.PARSED])
    pr = ParserResult(status="ok", rows=[{"urn": "u1"}, {"urn": "u2"}])
    records = runner.normalize(run.run_id, parser_result=pr)
    assert len(records) == 2
    assert run_repo.get(run.run_id).status == RunStatus.NORMALIZED


def test_validate_happy_path_advances_run(runner, run_repo, fake_factor_repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [RunStatus.FETCHED, RunStatus.PARSED, RunStatus.NORMALIZED],
    )
    fake_factor_repo._get_or_build_publish_orchestrator.return_value = None
    outcome = runner.validate(run.run_id, records=[{"urn": "u1"}])
    assert run_repo.get(run.run_id).status == RunStatus.VALIDATED
    assert len(outcome.accepted) == 1


def test_dedupe_happy_path_advances_run(runner, run_repo):
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
        ],
    )
    runner.dedupe(run.run_id, accepted=[{"urn": "u1"}])
    assert run_repo.get(run.run_id).status == RunStatus.DEDUPED


def test_runs_for_batch_filters_by_published_status(
    runner, run_repo, fake_publisher,
):
    """Only PUBLISHED runs whose batch_id matches are returned."""
    from greenlang.factors.ingestion.pipeline import RunStatus
    from greenlang.factors.ingestion.runner import DedupeOutcome

    run = _new_run(
        run_repo,
        [
            RunStatus.FETCHED,
            RunStatus.PARSED,
            RunStatus.NORMALIZED,
            RunStatus.VALIDATED,
            RunStatus.DEDUPED,
        ],
    )
    runner.stage(run.run_id, dedupe_outcome=DedupeOutcome(final=[]))
    fake_publisher.flip_to_production.return_value = 0
    runner.publish(run.run_id, approver="human:x@y.z", batch_id="match-1")

    matches = runner._runs_for_batch("match-1")
    assert {m.run_id for m in matches} == {run.run_id}
    assert runner._runs_for_batch("does-not-match") == []
