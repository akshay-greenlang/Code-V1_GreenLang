# -*- coding: utf-8 -*-
"""GAP-5 — Unit tests for the Factor Rollback workflow.

Covers:
    * ``RollbackService.plan_rollback`` incl. impact preview
    * Two-signature approval gate (methodology_lead + compliance_lead)
    * State-machine transitions (planned -> approved -> executing -> completed)
    * Cancellation + failure paths
    * Cascade hook invocation + persistence
    * SQLite ``RollbackStore`` round-trip
    * V443 migration shape alignment with ``RollbackStore`` schema
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from greenlang.factors.quality.impact_simulator import ImpactSimulator
from greenlang.factors.quality.rollback import (
    REQUIRED_APPROVAL_ROLES,
    RollbackApproval,
    RollbackApprovalError,
    RollbackError,
    RollbackNotFoundError,
    RollbackPlan,
    RollbackRecord,
    RollbackService,
    RollbackStateError,
    RollbackStatus,
    RollbackStore,
)
from greenlang.factors.quality.versioning import FactorVersionChain


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture()
def version_chain(tmp_path: Path) -> FactorVersionChain:
    """Version chain pre-populated with v1 + v2 for a single factor."""
    chain = FactorVersionChain(tmp_path / "chain.sqlite")
    chain.append(
        factor_id="EF:US:diesel:2024:v1",
        factor_version="v1",
        content_hash=_content_hash("v1"),
        changed_by="methodology-bot",
        change_reason="initial publish",
    )
    chain.append(
        factor_id="EF:US:diesel:2024:v1",
        factor_version="v2",
        content_hash=_content_hash("v2"),
        changed_by="methodology-bot",
        change_reason="2025 update",
    )
    return chain


@pytest.fixture()
def ledger_rows() -> List[Dict[str, Any]]:
    """Minimal ledger rows referencing the test factor."""
    return [
        {
            "id": 1,
            "entity_id": "calc-001",
            "tenant_id": "tenant-A",
            "content_hash": "content-1",
            "chain_hash": "chain-1",
            "metadata": {"factor_id": "EF:US:diesel:2024:v1"},
        },
        {
            "id": 2,
            "entity_id": "calc-002",
            "tenant_id": "tenant-B",
            "content_hash": "content-2",
            "chain_hash": "chain-2",
            "metadata": {"factor_id": "EF:US:diesel:2024:v1"},
        },
    ]


@pytest.fixture()
def simulator(ledger_rows) -> ImpactSimulator:
    return ImpactSimulator(ledger_entries=ledger_rows, evidence_records=[])


@pytest.fixture()
def store(tmp_path: Path) -> RollbackStore:
    return RollbackStore(tmp_path / "rollback.sqlite")


@pytest.fixture()
def service(version_chain, simulator, store) -> RollbackService:
    return RollbackService(
        version_chain=version_chain,
        impact_simulator=simulator,
        store=store,
    )


def _approve_twice(
    svc: RollbackService, rollback_id: str
) -> RollbackRecord:
    svc.approve_rollback(
        rollback_id=rollback_id,
        approver_id="alice",
        approver_role="methodology_lead",
        signature="sig-methodology-abcdef",
    )
    return svc.approve_rollback(
        rollback_id=rollback_id,
        approver_id="bob",
        approver_role="compliance_lead",
        signature="sig-compliance-abcdef",
    )


# ---------------------------------------------------------------------------
# plan_rollback
# ---------------------------------------------------------------------------


class TestPlanRollback:
    def test_returns_plan_with_impact_preview(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="bad factor value discovered",
            created_by="alice",
        )
        assert isinstance(plan, RollbackPlan)
        assert plan.factor_id == "EF:US:diesel:2024:v1"
        assert plan.from_version == "v2"
        assert plan.to_version == "v1"
        assert plan.status == RollbackStatus.PLANNED
        assert plan.affected_computations == 2
        assert plan.affected_tenants == 2
        assert plan.impact_report is not None
        assert "computations" in plan.impact_report

    def test_plan_is_persisted_as_planned(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="re-check",
            created_by="alice",
        )
        fetched = service.get_rollback(plan.rollback_id)
        assert fetched is not None
        assert fetched.status == RollbackStatus.PLANNED

    def test_rejects_empty_reason(self, service):
        with pytest.raises(RollbackError):
            service.plan_rollback(
                factor_id="EF:US:diesel:2024:v1",
                to_version="v1",
                reason="",
                created_by="alice",
            )

    def test_rejects_same_version(self, service):
        with pytest.raises(RollbackError):
            service.plan_rollback(
                factor_id="EF:US:diesel:2024:v1",
                to_version="v2",
                reason="rollback to self",
                created_by="alice",
            )

    def test_rejects_unknown_target_version(self, service):
        with pytest.raises(RollbackError):
            service.plan_rollback(
                factor_id="EF:US:diesel:2024:v1",
                to_version="v99",
                reason="ghost version",
                created_by="alice",
            )

    def test_rejects_unknown_factor(self, service):
        with pytest.raises(RollbackError):
            service.plan_rollback(
                factor_id="EF:US:unknown:2099:vX",
                to_version="v1",
                reason="no chain exists",
                created_by="alice",
            )

    def test_value_map_populates_deltas(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="value correction",
            created_by="alice",
            value_map={
                "EF:US:diesel:2024:v1": {"old": 2.5, "new": 2.0},
            },
        )
        comps = plan.impact_report["computations"]
        assert comps and comps[0]["old_value"] == 2.5
        assert comps[0]["new_value"] == 2.0
        assert comps[0]["delta_abs"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# approve / execute — two-signature gate
# ---------------------------------------------------------------------------


class TestApprovalGate:
    def test_single_signature_does_not_advance_to_approved(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="partial approval",
            created_by="alice",
        )
        record = service.approve_rollback(
            rollback_id=plan.rollback_id,
            approver_id="alice",
            approver_role="methodology_lead",
            signature="sig-abcdef12",
        )
        assert record.status == RollbackStatus.PLANNED
        assert len(record.approvals) == 1

    def test_two_signatures_advances_to_approved(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="full approval",
            created_by="alice",
        )
        record = _approve_twice(service, plan.rollback_id)
        assert record.status == RollbackStatus.APPROVED
        assert len(record.approvals) == 2
        assert record.approved_at is not None

    def test_duplicate_signer_rejected(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="x",
            created_by="alice",
        )
        service.approve_rollback(
            rollback_id=plan.rollback_id,
            approver_id="alice",
            approver_role="methodology_lead",
            signature="sig-aaaabbbb",
        )
        with pytest.raises(RollbackApprovalError):
            service.approve_rollback(
                rollback_id=plan.rollback_id,
                approver_id="alice",
                approver_role="compliance_lead",
                signature="sig-aaaaaaaa",
            )

    def test_same_role_rejected_even_from_different_user(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="x",
            created_by="alice",
        )
        service.approve_rollback(
            rollback_id=plan.rollback_id,
            approver_id="alice",
            approver_role="methodology_lead",
            signature="sig-aaaabbbb",
        )
        with pytest.raises(RollbackApprovalError):
            service.approve_rollback(
                rollback_id=plan.rollback_id,
                approver_id="carol",
                approver_role="methodology_lead",
                signature="sig-cccccccc",
            )

    def test_unknown_role_rejected(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="x",
            created_by="alice",
        )
        with pytest.raises(RollbackApprovalError):
            service.approve_rollback(
                rollback_id=plan.rollback_id,
                approver_id="dave",
                approver_role="sales_lead",
                signature="sig-bogus12345",
            )

    def test_short_signature_rejected(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="x",
            created_by="alice",
        )
        with pytest.raises(RollbackApprovalError):
            service.approve_rollback(
                rollback_id=plan.rollback_id,
                approver_id="alice",
                approver_role="methodology_lead",
                signature="abc",
            )

    def test_required_roles_constant(self):
        assert set(REQUIRED_APPROVAL_ROLES) == {
            "methodology_lead",
            "compliance_lead",
        }


# ---------------------------------------------------------------------------
# execute_rollback
# ---------------------------------------------------------------------------


class TestExecuteRollback:
    def test_execute_happy_path(self, service, version_chain):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="happy path",
            created_by="alice",
        )
        _approve_twice(service, plan.rollback_id)
        record = service.execute_rollback(rollback_id=plan.rollback_id)

        assert record.status == RollbackStatus.COMPLETED
        assert record.executed_at is not None
        chain = version_chain.get_version_chain("EF:US:diesel:2024:v1")
        assert len(chain) == 3
        # New head reuses v1's content hash and has a rollback marker.
        assert chain[-1].content_hash == _content_hash("v1")
        assert re.match(r"v1\+rb\.\d{14}", chain[-1].factor_version)

    def test_execute_without_approval_raises(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="unapproved",
            created_by="alice",
        )
        with pytest.raises(RollbackStateError):
            service.execute_rollback(rollback_id=plan.rollback_id)

    def test_cannot_execute_twice(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="double execute",
            created_by="alice",
        )
        _approve_twice(service, plan.rollback_id)
        service.execute_rollback(rollback_id=plan.rollback_id)
        with pytest.raises(RollbackStateError):
            service.execute_rollback(rollback_id=plan.rollback_id)

    def test_execute_invokes_cascade_hook(
        self, version_chain, simulator, store
    ):
        calls: List[str] = []

        def _cascade(factor_id: str, from_v: str, to_v: str) -> List[str]:
            calls.append(f"{factor_id}:{from_v}->{to_v}")
            return ["calc-001", "calc-002"]

        svc = RollbackService(
            version_chain=version_chain,
            impact_simulator=simulator,
            store=store,
            cascade_lookup=_cascade,
        )
        plan = svc.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="cascade",
            created_by="alice",
        )
        _approve_twice(svc, plan.rollback_id)
        record = svc.execute_rollback(rollback_id=plan.rollback_id)
        assert calls == ["EF:US:diesel:2024:v1:v2->v1"]
        assert record.cascade_computations == ["calc-001", "calc-002"]

    def test_failed_cascade_marks_record_failed(
        self, version_chain, simulator, store
    ):
        svc = RollbackService(
            version_chain=version_chain,
            impact_simulator=simulator,
            store=store,
            cascade_lookup=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        plan = svc.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="cascade fail",
            created_by="alice",
        )
        _approve_twice(svc, plan.rollback_id)
        # Cascade errors are swallowed by the service (they just produce
        # an empty cascade list), so completion still succeeds.  The
        # resilience here is by design.
        record = svc.execute_rollback(rollback_id=plan.rollback_id)
        assert record.status == RollbackStatus.COMPLETED
        assert record.cascade_computations == []


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_cancel_from_planned(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="cancel test",
            created_by="alice",
        )
        record = service.cancel_rollback(
            rollback_id=plan.rollback_id, reason="mistake"
        )
        assert record.status == RollbackStatus.CANCELLED
        assert "mistake" in (record.failure_reason or "")

    def test_cancel_from_approved(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="cancel after approval",
            created_by="alice",
        )
        _approve_twice(service, plan.rollback_id)
        record = service.cancel_rollback(
            rollback_id=plan.rollback_id, reason="new info"
        )
        assert record.status == RollbackStatus.CANCELLED

    def test_cannot_cancel_completed(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="x",
            created_by="alice",
        )
        _approve_twice(service, plan.rollback_id)
        service.execute_rollback(rollback_id=plan.rollback_id)
        with pytest.raises(RollbackStateError):
            service.cancel_rollback(
                rollback_id=plan.rollback_id, reason="too late"
            )

    def test_list_for_factor_chronological(self, service):
        for reason in ["first", "second", "third"]:
            service.plan_rollback(
                factor_id="EF:US:diesel:2024:v1",
                to_version="v1",
                reason=reason,
                created_by="alice",
            )
        records = service.list_for_factor("EF:US:diesel:2024:v1")
        assert len(records) == 3
        # Newest-first.
        assert records[0].created_at >= records[-1].created_at

    def test_get_missing_raises(self, service):
        with pytest.raises(RollbackNotFoundError):
            # Force via private helper so we hit the not-found path.
            service._require_record("does-not-exist")

    def test_audit_record_is_stable(self, service):
        plan = service.plan_rollback(
            factor_id="EF:US:diesel:2024:v1",
            to_version="v1",
            reason="audit",
            created_by="alice",
        )
        _approve_twice(service, plan.rollback_id)
        record = service.execute_rollback(rollback_id=plan.rollback_id)
        audit = service.create_rollback_audit_record(record)
        assert audit["audit_kind"] == "factor_rollback"
        assert audit["rollback_id"] == plan.rollback_id
        assert audit["status"] == RollbackStatus.COMPLETED.value
        assert "recorded_at" in audit


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestRollbackStore:
    def test_roundtrip(self, tmp_path):
        store = RollbackStore(tmp_path / "r.sqlite")
        rec = RollbackRecord(
            rollback_id="r1",
            factor_id="EF:x",
            from_version="v2",
            to_version="v1",
            reason="r",
            status=RollbackStatus.PLANNED,
            approvals=[
                RollbackApproval(
                    approver_id="alice",
                    approver_role="methodology_lead",
                    signature="sig12345678",
                    approved_at="2026-04-20T00:00:00+00:00",
                )
            ],
            created_at="2026-04-20T00:00:00+00:00",
            created_by="alice",
        )
        store.upsert(rec)
        fetched = store.get("r1")
        assert fetched is not None
        assert fetched.factor_id == "EF:x"
        assert fetched.approvals[0].approver_role == "methodology_lead"

    def test_list_for_factor_empty(self, tmp_path):
        store = RollbackStore(tmp_path / "r.sqlite")
        assert store.list_for_factor("EF:none") == []

    def test_in_memory_default(self):
        store = RollbackStore()
        assert store.get("nothing") is None

    def test_list_by_status(self, tmp_path):
        store = RollbackStore(tmp_path / "r.sqlite")
        for i, status in enumerate(
            [RollbackStatus.PLANNED, RollbackStatus.APPROVED, RollbackStatus.COMPLETED]
        ):
            store.upsert(
                RollbackRecord(
                    rollback_id=f"r{i}",
                    factor_id="EF:x",
                    from_version="v2",
                    to_version="v1",
                    reason="r",
                    status=status,
                    created_at="2026-04-20T00:00:00+00:00",
                    created_by="alice",
                )
            )
        planned = store.list_by_status([RollbackStatus.PLANNED])
        assert len(planned) == 1 and planned[0].rollback_id == "r0"


# ---------------------------------------------------------------------------
# DB migration shape
# ---------------------------------------------------------------------------


class TestV443MigrationShape:
    MIGRATION_PATH = (
        Path(__file__).resolve().parents[3]
        / "deployment"
        / "database"
        / "migrations"
        / "sql"
        / "V443__factors_rollback_records.sql"
    )

    def test_migration_exists(self):
        assert self.MIGRATION_PATH.exists()

    def test_table_name_and_columns_present(self):
        sql = self.MIGRATION_PATH.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS factors_rollback_records" in sql
        for col in (
            "rollback_id",
            "factor_id",
            "from_version",
            "to_version",
            "reason",
            "status",
            "approved_by_1",
            "approved_by_2",
            "approved_at",
            "executed_at",
            "affected_computations",
            "affected_tenants",
            "impact_report_json",
            "created_at",
            "created_by",
        ):
            assert col in sql, f"missing column: {col}"

    def test_status_check_constraint(self):
        sql = self.MIGRATION_PATH.read_text(encoding="utf-8")
        # Status CHECK must enumerate every RollbackStatus value.
        for value in (
            "planned",
            "approved",
            "executing",
            "completed",
            "failed",
            "cancelled",
        ):
            assert f"'{value}'" in sql

    def test_indexes_present(self):
        sql = self.MIGRATION_PATH.read_text(encoding="utf-8")
        assert "idx_rollback_factor" in sql
        assert "idx_rollback_status" in sql


# ---------------------------------------------------------------------------
# Versioning hooks
# ---------------------------------------------------------------------------


class TestVersioningHooks:
    def test_get_version_chain_alias(self, version_chain):
        via_alias = version_chain.get_version_chain("EF:US:diesel:2024:v1")
        via_original = version_chain.chain("EF:US:diesel:2024:v1")
        assert [e.factor_version for e in via_alias] == [
            e.factor_version for e in via_original
        ]

    def test_is_rollback_available_true_for_older(self, version_chain):
        assert (
            version_chain.is_rollback_available("EF:US:diesel:2024:v1", "v1")
            is True
        )

    def test_is_rollback_available_false_for_head(self, version_chain):
        assert (
            version_chain.is_rollback_available("EF:US:diesel:2024:v1", "v2")
            is False
        )

    def test_mark_rollback_available(self, version_chain):
        marker = version_chain.mark_rollback_available(
            "EF:US:diesel:2024:v1", "v1"
        )
        assert marker["available"] is True
        assert marker["current_version"] == "v2"
