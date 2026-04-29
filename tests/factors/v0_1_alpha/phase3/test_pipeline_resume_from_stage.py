# -*- coding: utf-8 -*-
"""Phase 3 audit gap B — `IngestionPipelineRunner.resume(...)` contract.

The Phase 3 plan §"CLI surface (Click group)" specifies:

  Resume mode requires explicit ``--from-stage <name>`` after a failed run.
  Allowed stages: ``parse|normalize|validate|dedupe|stage``.

This test exercises the runtime half of that contract:

* a failed run that died during PARSE can be resumed from ``parse`` and
  drive the pipeline through stage 6;
* a fresh ``CREATED`` run rejects ``--from-stage dedupe`` with
  :class:`StageOrderError` because the run is not in failed/rejected.

The CLI surface (``--from-stage`` flag wired in
:mod:`greenlang.factors.cli_ingest`) is exercised separately by the
``run_cmd`` regression suite.
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


def test_resume_rejects_fresh_created_run_with_stage_order_error(
    phase3_runner_raw,
    phase3_run_repo,
):
    """Resume on a fresh CREATED run must raise StageOrderError.

    A run that has not yet failed / been rejected has nothing to recover
    from. The runner refuses to bypass the canonical seven-stage ladder
    by jumping mid-pipeline on a clean run.
    """
    from greenlang.factors.ingestion.exceptions import StageOrderError

    fresh = phase3_run_repo.create(
        source_urn="urn:gl:source:resume-test-fresh",
        source_version="2024.1",
        operator="bot:resume-test",
    )
    # The run is CREATED. Resume from DEDUPE is illegal: (a) status is
    # not failed/rejected; (b) current_stage is None — the run never
    # ran, so the predecessor check never has anything to compare.
    with pytest.raises(StageOrderError) as exc_info:
        phase3_runner_raw.resume(fresh.run_id, from_stage="dedupe")
    msg = str(exc_info.value).lower()
    assert "failed" in msg or "rejected" in msg or "current_stage" in msg


def test_resume_picks_up_failed_run_from_parse(
    phase3_runner_raw,
    phase3_run_repo,
    defra_fixture_url,
    monkeypatch,
):
    """A failed run that died during PARSE can be resumed from ``parse``.

    Sequence:

    1. Drive a fresh DEFRA run; force ``parse`` to throw on the first
       attempt so the run lands in FAILED with current_stage=PARSE.
    2. Remove the patch.
    3. Call ``runner.resume(run_id, from_stage='parse')`` — the runner
       resets the row to FETCHED, reruns parse onward, and drives the
       run through to STAGED / REVIEW_REQUIRED.
    """
    from greenlang.factors.ingestion.exceptions import IngestionError
    from greenlang.factors.ingestion.pipeline import RunStatus

    # ---- Phase 1: drive a run that fails on parse ------------------------
    # Capture every created run_id so we can find the failed one even if
    # list_by_source's ordering doesn't surface it first.
    real_create = phase3_run_repo.create
    seen_runs = []

    def _capture_create(*args, **kwargs):
        run = real_create(*args, **kwargs)
        seen_runs.append(run.run_id)
        return run

    monkeypatch.setattr(phase3_run_repo, "create", _capture_create)

    # Patch the underlying parser in the registry — the runner's
    # parse() method's try/except block is what we want to exercise so
    # _mark_failed() actually runs and sets status=FAILED with
    # current_stage=PARSE.
    parser = phase3_runner_raw._parser_registry.get("defra-2025")
    assert parser is not None, "DEFRA parser missing from registry"
    fail_count = {"n": 0}
    real_parse_bytes = getattr(parser, "parse_bytes", None)

    def _flaky_parse_bytes(*args, **kwargs):
        fail_count["n"] += 1
        if fail_count["n"] == 1:
            raise RuntimeError("synthetic parse failure for resume test")
        return real_parse_bytes(*args, **kwargs)

    if real_parse_bytes is None:
        pytest.skip("DEFRA parser does not expose parse_bytes; cannot inject failure")
    monkeypatch.setattr(parser, "parse_bytes", _flaky_parse_bytes)

    # The runner wraps the parse() exception in a ParserDispatchError;
    # match on the inner message so our IngestionError rises cleanly.
    with pytest.raises(IngestionError):
        phase3_runner_raw.run(
            source_id="defra-2025",
            source_url=defra_fixture_url,
            source_urn="urn:gl:source:defra-2025",
            source_version="2025.1",
            operator="bot:resume-test",
        )
    assert seen_runs, "create() was never called"
    failed_run_id = seen_runs[-1]
    failed = phase3_run_repo.get(failed_run_id)
    assert failed.status == RunStatus.FAILED, (
        f"expected FAILED, got {failed.status!r}"
    )
    from greenlang.factors.ingestion.pipeline import Stage
    assert failed.current_stage == Stage.PARSE, (
        f"expected current_stage=PARSE on the failed run, got {failed.current_stage!r}"
    )
    assert failed.artifact_id is not None, (
        "fetch should have completed before parse failed"
    )

    # ---- Phase 2: clear the patch + resume ------------------------------
    # The fail_count is now > 0 so subsequent parse_bytes calls go
    # through to the real parser via the closure above.
    resumed = phase3_runner_raw.resume(
        failed.run_id, from_stage="parse", source_id="defra-2025"
    )
    assert resumed.status in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED), (
        f"resume did not drive run to staging; final status={resumed.status!r}"
    )
    # The artifact pinned on the row before the failure should still be
    # there — resume reuses the previously-fetched artifact.
    assert resumed.artifact_id == failed.artifact_id


def test_resume_rejects_unknown_from_stage(
    phase3_runner_raw,
    phase3_run_repo,
):
    """An unrecognised ``from_stage`` token raises StageOrderError."""
    from greenlang.factors.ingestion.exceptions import StageOrderError
    from greenlang.factors.ingestion.pipeline import RunStatus

    run = phase3_run_repo.create(
        source_urn="urn:gl:source:resume-test-bad",
        source_version="2024.1",
        operator="bot:resume-test",
    )
    # Coerce the run to FAILED via the backend writer so the resumable-
    # status guard passes; the next guard (whitelist) is what we're
    # testing.
    phase3_run_repo._update_status_sqlite(  # type: ignore[attr-defined]
        run.run_id, RunStatus.FAILED, None, None, None
    )
    with pytest.raises(StageOrderError, match="from_stage"):
        phase3_runner_raw.resume(run.run_id, from_stage="publish")
