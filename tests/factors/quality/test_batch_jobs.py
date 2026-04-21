# -*- coding: utf-8 -*-
"""GAP-11 — Unit tests for the Factors Batch Job queue."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from greenlang.factors.batch_jobs import (
    MAX_REQUESTS_PER_TIER,
    BatchJob,
    BatchJobError,
    BatchJobHandle,
    BatchJobLimitError,
    BatchJobNotFoundError,
    BatchJobStateError,
    BatchJobStatus,
    BatchJobType,
    SQLiteBatchJobQueue,
    build_webhook_payload,
    cancel_batch_job,
    delete_batch_job,
    get_batch_job_results,
    get_batch_job_status,
    max_batch_size_for_tier,
    process_next_job,
    submit_batch_resolution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def queue(tmp_path: Path) -> SQLiteBatchJobQueue:
    return SQLiteBatchJobQueue(tmp_path / "batch.sqlite")


@pytest.fixture()
def fake_resolver():
    """A deterministic resolver that echoes back a result dict."""

    def _resolver(req: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "factor_id": req.get("activity", "unknown") + "-factor",
            "co2e_total": 1.234,
            "gas_breakdown": {
                "co2_kg": 1.0,
                "ch4_kg": 0.1,
                "n2o_kg": 0.0,
                "hfcs_kg": 0.0,
                "pfcs_kg": 0.0,
                "sf6_kg": 0.0,
                "nf3_kg": 0.0,
                "biogenic_co2_kg": 0.0,
                "co2e_total_kg": 1.234,
                "gwp_basis": "AR6-100yr",
            },
        }

    return _resolver


def _sample_requests(n: int) -> List[Dict[str, Any]]:
    return [{"activity": f"activity-{i}", "jurisdiction": "US"} for i in range(n)]


# ---------------------------------------------------------------------------
# Tier caps
# ---------------------------------------------------------------------------


class TestTierLimits:
    def test_default_limits(self):
        assert MAX_REQUESTS_PER_TIER["pro"] == 100
        assert MAX_REQUESTS_PER_TIER["consulting"] == 1000
        assert MAX_REQUESTS_PER_TIER["enterprise"] == 10_000

    def test_unknown_tier_falls_back_to_community(self):
        assert max_batch_size_for_tier("nonexistent") == MAX_REQUESTS_PER_TIER[
            "community"
        ]

    def test_none_tier_falls_back_to_community(self):
        assert max_batch_size_for_tier(None) == MAX_REQUESTS_PER_TIER[
            "community"
        ]


# ---------------------------------------------------------------------------
# submit_batch_resolution
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    def test_submit_returns_handle(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(10),
            tenant_id="tenant-A",
            tier="pro",
            created_by="alice",
        )
        assert isinstance(handle, BatchJobHandle)
        assert handle.request_count == 10
        assert handle.status == BatchJobStatus.QUEUED

    def test_submit_rejects_empty(self, queue):
        with pytest.raises(BatchJobError):
            submit_batch_resolution(
                queue,
                requests=[],
                tenant_id="tenant-A",
                tier="pro",
                created_by="alice",
            )

    def test_submit_enforces_tier_cap(self, queue):
        with pytest.raises(BatchJobLimitError):
            submit_batch_resolution(
                queue,
                requests=_sample_requests(101),
                tenant_id="tenant-A",
                tier="pro",
                created_by="alice",
            )

    def test_enterprise_allows_larger_batches(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(500),
            tenant_id="tenant-E",
            tier="enterprise",
            created_by="eve",
        )
        assert handle.request_count == 500

    def test_unknown_job_type_rejected(self, queue):
        with pytest.raises(ValueError):
            submit_batch_resolution(
                queue,
                requests=_sample_requests(5),
                tenant_id="tenant-A",
                tier="pro",
                created_by="alice",
                job_type="mystery",
            )


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_status_queued_then_completed(self, queue, fake_resolver):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(3),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        queued = get_batch_job_status(queue, handle.job_id)
        assert queued.status == BatchJobStatus.QUEUED

        done = process_next_job(queue, resolver=fake_resolver)
        assert done is not None
        assert done.status == BatchJobStatus.COMPLETED
        assert done.completed_count == 3
        assert done.failed_count == 0

    def test_process_next_job_empty_queue_returns_none(self, queue, fake_resolver):
        assert process_next_job(queue, resolver=fake_resolver) is None

    def test_failed_resolver_records_errors(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(2),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )

        def _bad_resolver(_: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError("boom")

        done = process_next_job(queue, resolver=_bad_resolver)
        assert done.status == BatchJobStatus.FAILED
        assert done.failed_count == 2
        assert done.error_log
        assert done.error_log[0]["error_type"] == "RuntimeError"

    def test_mixed_success_still_completes(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(3),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )

        def _flaky(req: Dict[str, Any]) -> Dict[str, Any]:
            if req.get("activity") == "activity-1":
                raise ValueError("skip middle")
            return {"ok": True}

        done = process_next_job(queue, resolver=_flaky)
        assert done.status == BatchJobStatus.COMPLETED  # at least one ok
        assert done.completed_count == 2
        assert done.failed_count == 1


# ---------------------------------------------------------------------------
# Results pagination
# ---------------------------------------------------------------------------


class TestResultsPagination:
    def test_pagination_returns_cursor(self, queue, fake_resolver):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(5),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        process_next_job(queue, resolver=fake_resolver)

        page1 = get_batch_job_results(queue, handle.job_id, cursor=0, limit=2)
        assert len(page1["results"]) == 2
        assert page1["has_more"] is True

        page2 = get_batch_job_results(
            queue, handle.job_id, cursor=page1["cursor"], limit=2
        )
        assert len(page2["results"]) == 2
        assert page2["has_more"] is True

        page3 = get_batch_job_results(
            queue, handle.job_id, cursor=page2["cursor"], limit=2
        )
        assert len(page3["results"]) == 1
        assert page3["has_more"] is False
        assert page3["total"] == 5

    def test_results_preserve_gas_breakdown(self, queue, fake_resolver):
        """CTO non-negotiable #1 — gas-level data must survive the queue."""
        submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        process_next_job(queue, resolver=fake_resolver)
        jobs, _ = queue.list_for_tenant("t")
        page = get_batch_job_results(queue, jobs[0].job_id, cursor=0, limit=10)
        gb = page["results"][0]["result"]["gas_breakdown"]
        # Every individual gas component must be present.
        for key in (
            "co2_kg",
            "ch4_kg",
            "n2o_kg",
            "hfcs_kg",
            "pfcs_kg",
            "sf6_kg",
            "nf3_kg",
            "biogenic_co2_kg",
            "co2e_total_kg",
            "gwp_basis",
        ):
            assert key in gb


# ---------------------------------------------------------------------------
# Cancel / delete
# ---------------------------------------------------------------------------


class TestCancelDelete:
    def test_cancel_queued_job(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(3),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        cancelled = cancel_batch_job(queue, handle.job_id)
        assert cancelled.status == BatchJobStatus.CANCELLED

    def test_cancel_running_then_delete(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(2),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        # Simulate a running job via next_queued.
        queue.next_queued()
        cancelled = cancel_batch_job(queue, handle.job_id)
        assert cancelled.status == BatchJobStatus.CANCELLED
        assert delete_batch_job(queue, handle.job_id) is True

    def test_delete_running_rejected(self, queue):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(2),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        queue.next_queued()
        with pytest.raises(BatchJobStateError):
            delete_batch_job(queue, handle.job_id)

    def test_cancel_completed_rejected(self, queue, fake_resolver):
        handle = submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        process_next_job(queue, resolver=fake_resolver)
        with pytest.raises(BatchJobStateError):
            cancel_batch_job(queue, handle.job_id)

    def test_missing_job_raises(self, queue):
        with pytest.raises(BatchJobNotFoundError):
            get_batch_job_status(queue, "does-not-exist")


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_list_filters_by_tenant(self, queue):
        submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="tenant-A",
            tier="pro",
            created_by="alice",
        )
        submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="tenant-B",
            tier="pro",
            created_by="bob",
        )
        a_jobs, a_total = queue.list_for_tenant("tenant-A")
        b_jobs, b_total = queue.list_for_tenant("tenant-B")
        assert a_total == 1 and len(a_jobs) == 1 and a_jobs[0].tenant_id == "tenant-A"
        assert b_total == 1 and len(b_jobs) == 1 and b_jobs[0].tenant_id == "tenant-B"

    def test_status_filter(self, queue, fake_resolver):
        for _ in range(3):
            submit_batch_resolution(
                queue,
                requests=_sample_requests(1),
                tenant_id="tenant-A",
                tier="pro",
                created_by="alice",
            )
        process_next_job(queue, resolver=fake_resolver)
        completed, total_c = queue.list_for_tenant(
            "tenant-A", status=BatchJobStatus.COMPLETED
        )
        queued, total_q = queue.list_for_tenant(
            "tenant-A", status=BatchJobStatus.QUEUED
        )
        assert total_c == 1 and total_q == 2


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


class TestWebhook:
    def test_webhook_fires_on_completion(self, queue, fake_resolver):
        captured: List[BatchJob] = []

        def _emit(job: BatchJob) -> None:
            captured.append(job)

        submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        done = process_next_job(queue, resolver=fake_resolver, webhook_emit=_emit)
        assert captured and captured[0].job_id == done.job_id

    def test_webhook_failure_does_not_break_completion(self, queue, fake_resolver):
        def _bad_emit(_: BatchJob) -> None:
            raise RuntimeError("webhook dead")

        submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        done = process_next_job(
            queue, resolver=fake_resolver, webhook_emit=_bad_emit
        )
        assert done.status == BatchJobStatus.COMPLETED

    def test_build_webhook_payload_shape(self, queue, fake_resolver):
        submit_batch_resolution(
            queue,
            requests=_sample_requests(1),
            tenant_id="t",
            tier="pro",
            created_by="alice",
        )
        done = process_next_job(queue, resolver=fake_resolver)
        payload = build_webhook_payload(done)
        assert payload["event_type"] == "batch_job.completed"
        for key in (
            "job_id",
            "tenant_id",
            "status",
            "completed_count",
            "failed_count",
            "request_count",
        ):
            assert key in payload


# ---------------------------------------------------------------------------
# Postgres queue + migration shape
# ---------------------------------------------------------------------------


class TestPostgresQueueWiring:
    def test_missing_hooks_raise(self):
        from greenlang.factors.batch_jobs import PostgresBatchJobQueue

        q = PostgresBatchJobQueue(
            connection_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("unused")
            )
        )
        with pytest.raises(BatchJobError):
            # results_writer hook is unwired by default -> raises.
            q.put_results("x", [{"ok": True}])


class TestV444MigrationShape:
    MIGRATION_PATH = (
        Path(__file__).resolve().parents[3]
        / "deployment"
        / "database"
        / "migrations"
        / "sql"
        / "V444__factors_batch_jobs.sql"
    )

    def test_migration_exists(self):
        assert self.MIGRATION_PATH.exists()

    def test_columns_present(self):
        sql = self.MIGRATION_PATH.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS factors_batch_jobs" in sql
        for col in (
            "job_id",
            "tenant_id",
            "job_type",
            "status",
            "submitted_at",
            "started_at",
            "completed_at",
            "request_count",
            "completed_count",
            "failed_count",
            "results_uri",
            "request_payload_uri",
            "error_log",
            "webhook_url",
            "webhook_secret_ref",
            "created_by",
        ):
            assert col in sql

    def test_status_and_job_type_constraints(self):
        sql = self.MIGRATION_PATH.read_text(encoding="utf-8")
        for v in ("queued", "running", "completed", "failed", "cancelled"):
            assert f"'{v}'" in sql
        for t in ("resolve", "search", "match", "diff"):
            assert f"'{t}'" in sql

    def test_indexes_present(self):
        sql = self.MIGRATION_PATH.read_text(encoding="utf-8")
        assert "idx_batch_tenant_status" in sql
        assert "idx_batch_status_submitted" in sql


# ---------------------------------------------------------------------------
# Error log + counts coherence
# ---------------------------------------------------------------------------


class TestEnumsAndDataclasses:
    def test_job_types_enumerated(self):
        assert {t.value for t in BatchJobType} == {
            "resolve",
            "search",
            "match",
            "diff",
        }

    def test_statuses_enumerated(self):
        assert {s.value for s in BatchJobStatus} == {
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
        }
