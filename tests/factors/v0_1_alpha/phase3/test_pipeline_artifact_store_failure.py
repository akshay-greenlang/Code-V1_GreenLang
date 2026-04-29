# -*- coding: utf-8 -*-
"""Phase 3 audit gap D — artifact-store-failure produces no partial state.

The Phase 3 plan §"Artifact storage contract" requires that a fetch-time
storage failure (disk full, permission denied, S3 5xx) leaves the run in
a clean ``failed`` state — NEVER a half-fetched run, NEVER partial
factor rows, NEVER an orphaned source_artifacts row.

This test monkey-patches :meth:`LocalArtifactStore.put_bytes` to raise
``OSError("disk full")`` and asserts:

  (a) the run row's status is ``failed``;
  (b) ``error_json`` carries ``stage='fetch'`` and the OSError message;
  (c) zero factor rows landed in ``alpha_factors_v0_1``;
  (d) zero ``alpha_source_artifacts_v0_1`` rows committed for the run.
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


def test_artifact_store_failure_produces_clean_failed_run(
    phase3_runner_raw,
    phase3_run_repo,
    phase3_repo,
    defra_fixture_url,
    monkeypatch,
):
    """A LocalArtifactStore.put failure leaves the run in a clean failed state.

    Asserts the four invariants the Phase 3 plan requires for a stage-1
    storage failure:

      (a) run.status == 'failed';
      (b) run.error_json carries stage='fetch' + the OSError message;
      (c) zero factor rows in alpha_factors_v0_1;
      (d) zero source_artifacts rows for this run.
    """
    from greenlang.factors.ingestion.artifacts import LocalArtifactStore

    seen_run_ids = []
    real_create = phase3_run_repo.create

    def _capture(*args, **kwargs):
        run = real_create(*args, **kwargs)
        seen_run_ids.append(run.run_id)
        return run

    monkeypatch.setattr(phase3_run_repo, "create", _capture)

    def _explode_put(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(LocalArtifactStore, "put_bytes", _explode_put)

    # The runner wraps the OSError in an ArtifactStoreError.
    from greenlang.factors.ingestion.exceptions import ArtifactStoreError

    with pytest.raises(ArtifactStoreError):
        phase3_runner_raw.run(
            source_id="defra-2025",
            source_url=defra_fixture_url,
            source_urn="urn:gl:source:defra-2025",
            source_version="2025.1",
            operator="bot:gap-d-test",
            auto_stage=True,
        )

    assert seen_run_ids, "create() was never called"
    failed_run_id = seen_run_ids[-1]
    failed = phase3_run_repo.get(failed_run_id)

    # (a) Run status is FAILED.
    assert failed.status.value == "failed", (
        f"expected status=failed, got {failed.status.value!r}"
    )

    # (b) error_json carries stage='fetch' + the OSError message.
    assert failed.error_json is not None, "expected error_json on failed run"
    assert failed.error_json.get("stage") == "fetch", (
        f"expected error_json.stage='fetch', got {failed.error_json!r}"
    )
    error_msg = failed.error_json.get("message", "")
    assert "disk full" in error_msg, (
        f"expected error_json.message to carry the OSError text, "
        f"got {error_msg!r}"
    )

    # (c) Zero factor rows landed in alpha_factors_v0_1 for this run /
    # source_urn / version pair. We don't have a per-run factor index,
    # so we check the table is empty for the test source URN — the
    # fixture's seeded factors live under a different URN.
    conn = phase3_repo._connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM alpha_factors_v0_1 WHERE source_urn = ?",
        ("urn:gl:source:defra-2025",),
    )
    factor_count = cur.fetchone()[0]
    assert factor_count == 0, (
        f"expected 0 factor rows for failed DEFRA run, got {factor_count}"
    )

    # (d) Zero source_artifacts rows for this run.
    cur.execute(
        "SELECT COUNT(*) FROM alpha_source_artifacts_v0_1 "
        "WHERE ingestion_run_id = ?",
        (failed_run_id,),
    )
    artifact_count = cur.fetchone()[0]
    assert artifact_count == 0, (
        f"expected 0 source_artifacts rows for failed run, got {artifact_count}"
    )
