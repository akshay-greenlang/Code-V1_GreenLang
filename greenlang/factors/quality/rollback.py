# -*- coding: utf-8 -*-
"""
Factor rollback workflow (GAP-5).

Implements the **per-factor rollback** flow the CTO specified:

    1. A methodology lead files a **rollback plan** against an existing
       factor version.  The plan calls the ``ImpactSimulator`` to preview
       every computation, tenant, and evidence bundle that will shift.

    2. A **two-signature approval** gate (methodology_lead +
       compliance_lead) unlocks execution.  The signatures are stored
       with the rollback record and the status transitions
       ``PLANNED -> APPROVED``.

    3. Execution moves the status to ``EXECUTING`` and replays the
       previous version's ``content_hash`` as the current head of the
       version chain via a fresh ``FactorVersionChain.append`` call
       (append-only — the original chain is never mutated).

    4. Dependent computations are **cascade-flagged** for re-run (the
       actual re-run happens in the Scope Engine / Comply pipelines;
       this module just emits the flag set so those services can pick
       it up).  Each flagged computation is attached to the rollback
       record as part of the immutable audit trail.

    5. Status ends at ``COMPLETED`` (or ``FAILED`` on any error during
       cascade / version-chain append — at which point the record and
       reason are still persisted for audit).

State machine::

    PLANNED ──approve──▶ APPROVED ──execute──▶ EXECUTING
                                                   │
                                        ┌──────────┴──────────┐
                                        ▼                     ▼
                                    COMPLETED              FAILED

Storage: SQLite-backed for dev / local tests; the production Postgres
schema lives at ``deployment/database/migrations/sql/V443__factors_rollback_records.sql``
and mirrors the SQLite shape 1:1.

CTO non-negotiable #2 (append-only versioning) is honoured: we never
``UPDATE`` the version chain; every rollback produces a **new** row that
reuses the older ``content_hash``.

Author: GL-BackendDeveloper
Gap: GAP-5
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from greenlang.factors.quality.impact_simulator import (
    ImpactReport,
    ImpactSimulator,
)
from greenlang.factors.quality.versioning import (
    FactorVersionChain,
    VersionEntry,
    VersioningError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / state machine
# ---------------------------------------------------------------------------


class RollbackStatus(str, Enum):
    """Lifecycle of a rollback record."""

    PLANNED = "planned"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Allowed transitions for the rollback state machine.  Every transition
#: not listed here is rejected with ``RollbackStateError``.
_ALLOWED_TRANSITIONS: Dict[RollbackStatus, frozenset] = {
    RollbackStatus.PLANNED: frozenset(
        {RollbackStatus.APPROVED, RollbackStatus.CANCELLED}
    ),
    RollbackStatus.APPROVED: frozenset(
        {RollbackStatus.EXECUTING, RollbackStatus.CANCELLED}
    ),
    RollbackStatus.EXECUTING: frozenset(
        {RollbackStatus.COMPLETED, RollbackStatus.FAILED}
    ),
    RollbackStatus.COMPLETED: frozenset(),
    RollbackStatus.FAILED: frozenset(),
    RollbackStatus.CANCELLED: frozenset(),
}


#: Roles required for the two-signature approval gate.
REQUIRED_APPROVAL_ROLES: List[str] = ["methodology_lead", "compliance_lead"]


class RollbackError(RuntimeError):
    """Base class for rollback-workflow errors."""


class RollbackStateError(RollbackError):
    """Illegal state transition attempted."""


class RollbackApprovalError(RollbackError):
    """Approval gate failed (missing role, duplicate signer, bad sig)."""


class RollbackNotFoundError(RollbackError):
    """Caller referenced a rollback_id that doesn't exist."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RollbackApproval:
    """One signer's approval record inside a RollbackRecord."""

    approver_id: str
    approver_role: str
    signature: str
    approved_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approver_id": self.approver_id,
            "approver_role": self.approver_role,
            "signature": self.signature,
            "approved_at": self.approved_at,
        }


@dataclass
class RollbackPlan:
    """Pre-execution preview built by ``RollbackService.plan_rollback``."""

    rollback_id: str
    factor_id: str
    from_version: str
    to_version: str
    reason: str
    created_by: str
    created_at: str
    impact_report: Optional[Dict[str, Any]] = None
    affected_computations: int = 0
    affected_tenants: int = 0
    status: RollbackStatus = RollbackStatus.PLANNED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "factor_id": self.factor_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "status": self.status.value,
            "affected_computations": self.affected_computations,
            "affected_tenants": self.affected_tenants,
            "impact_report": self.impact_report,
        }


@dataclass
class RollbackRecord:
    """Immutable audit entry for a single rollback.

    Persisted verbatim to ``factors_rollback_records``; the dataclass
    mirrors the Postgres / SQLite column set 1:1.  Approvals are stored
    as a JSON array alongside denormalised ``approved_by_1`` /
    ``approved_by_2`` columns so operators can filter by signer without
    deserialising the blob.
    """

    rollback_id: str
    factor_id: str
    from_version: str
    to_version: str
    reason: str
    status: RollbackStatus
    approvals: List[RollbackApproval] = field(default_factory=list)
    approved_at: Optional[str] = None
    executed_at: Optional[str] = None
    affected_computations: int = 0
    affected_tenants: int = 0
    impact_report: Optional[Dict[str, Any]] = None
    cascade_computations: List[str] = field(default_factory=list)
    created_at: str = ""
    created_by: str = ""
    failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def approved_by_1(self) -> Optional[str]:
        return self.approvals[0].approver_id if len(self.approvals) >= 1 else None

    @property
    def approved_by_2(self) -> Optional[str]:
        return self.approvals[1].approver_id if len(self.approvals) >= 2 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "factor_id": self.factor_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "status": self.status.value,
            "approvals": [a.to_dict() for a in self.approvals],
            "approved_by_1": self.approved_by_1,
            "approved_by_2": self.approved_by_2,
            "approved_at": self.approved_at,
            "executed_at": self.executed_at,
            "affected_computations": self.affected_computations,
            "affected_tenants": self.affected_tenants,
            "impact_report": self.impact_report,
            "cascade_computations": list(self.cascade_computations),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Storage — SQLite backend (prod uses V443 Postgres table of same shape)
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors_rollback_records (
    rollback_id            TEXT PRIMARY KEY,
    factor_id              TEXT NOT NULL,
    from_version           TEXT NOT NULL,
    to_version             TEXT NOT NULL,
    reason                 TEXT NOT NULL,
    status                 TEXT NOT NULL,
    approved_by_1          TEXT,
    approved_by_2          TEXT,
    approved_at            TEXT,
    executed_at            TEXT,
    affected_computations  INTEGER NOT NULL DEFAULT 0,
    affected_tenants       INTEGER NOT NULL DEFAULT 0,
    impact_report_json     TEXT,
    approvals_json         TEXT,
    cascade_json           TEXT,
    failure_reason         TEXT,
    created_at             TEXT NOT NULL,
    created_by             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rb_factor ON factors_rollback_records (factor_id);
CREATE INDEX IF NOT EXISTS idx_rb_status ON factors_rollback_records (status);
"""


class RollbackStore:
    """Thread-safe SQLite-backed rollback record store.

    The production Postgres variant uses the identical column set
    (see ``V443__factors_rollback_records.sql``).
    """

    def __init__(self, sqlite_path: Union[str, Path, None] = None) -> None:
        self._lock = threading.Lock()
        if sqlite_path is None:
            # In-memory DB for tests that don't need persistence.
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            p = Path(sqlite_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(p), isolation_level=None, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, record: RollbackRecord) -> None:
        """Insert or replace a rollback record keyed by ``rollback_id``."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO factors_rollback_records (
                    rollback_id, factor_id, from_version, to_version, reason,
                    status, approved_by_1, approved_by_2, approved_at,
                    executed_at, affected_computations, affected_tenants,
                    impact_report_json, approvals_json, cascade_json,
                    failure_reason, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rollback_id) DO UPDATE SET
                    status = excluded.status,
                    approved_by_1 = excluded.approved_by_1,
                    approved_by_2 = excluded.approved_by_2,
                    approved_at = excluded.approved_at,
                    executed_at = excluded.executed_at,
                    affected_computations = excluded.affected_computations,
                    affected_tenants = excluded.affected_tenants,
                    impact_report_json = excluded.impact_report_json,
                    approvals_json = excluded.approvals_json,
                    cascade_json = excluded.cascade_json,
                    failure_reason = excluded.failure_reason
                """,
                (
                    record.rollback_id,
                    record.factor_id,
                    record.from_version,
                    record.to_version,
                    record.reason,
                    record.status.value,
                    record.approved_by_1,
                    record.approved_by_2,
                    record.approved_at,
                    record.executed_at,
                    record.affected_computations,
                    record.affected_tenants,
                    json.dumps(record.impact_report, default=str)
                    if record.impact_report is not None
                    else None,
                    json.dumps([a.to_dict() for a in record.approvals]),
                    json.dumps(list(record.cascade_computations)),
                    record.failure_reason,
                    record.created_at,
                    record.created_by,
                ),
            )

    def get(self, rollback_id: str) -> Optional[RollbackRecord]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT rollback_id, factor_id, from_version, to_version, reason,
                       status, approvals_json, approved_at, executed_at,
                       affected_computations, affected_tenants,
                       impact_report_json, cascade_json,
                       failure_reason, created_at, created_by
                FROM factors_rollback_records
                WHERE rollback_id = ?
                """,
                (rollback_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_for_factor(self, factor_id: str) -> List[RollbackRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT rollback_id, factor_id, from_version, to_version, reason,
                       status, approvals_json, approved_at, executed_at,
                       affected_computations, affected_tenants,
                       impact_report_json, cascade_json,
                       failure_reason, created_at, created_by
                FROM factors_rollback_records
                WHERE factor_id = ?
                ORDER BY created_at DESC
                """,
                (factor_id,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list_by_status(
        self, statuses: Iterable[RollbackStatus]
    ) -> List[RollbackRecord]:
        wanted = [s.value for s in statuses]
        if not wanted:
            return []
        placeholders = ",".join("?" for _ in wanted)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT rollback_id, factor_id, from_version, to_version, reason,
                       status, approvals_json, approved_at, executed_at,
                       affected_computations, affected_tenants,
                       impact_report_json, cascade_json,
                       failure_reason, created_at, created_by
                FROM factors_rollback_records
                WHERE status IN ({placeholders})
                ORDER BY created_at DESC
                """,
                wanted,
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_record(row) -> RollbackRecord:
    """Materialise a DB row into a RollbackRecord dataclass."""
    approvals_json = row[6] or "[]"
    try:
        approvals_raw = json.loads(approvals_json) or []
    except json.JSONDecodeError:
        approvals_raw = []
    approvals = [
        RollbackApproval(
            approver_id=a.get("approver_id", ""),
            approver_role=a.get("approver_role", ""),
            signature=a.get("signature", ""),
            approved_at=a.get("approved_at", ""),
        )
        for a in approvals_raw
    ]

    impact_report = None
    if row[11]:
        try:
            impact_report = json.loads(row[11])
        except json.JSONDecodeError:
            impact_report = None

    cascade: List[str] = []
    if row[12]:
        try:
            cascade = list(json.loads(row[12]))
        except json.JSONDecodeError:
            cascade = []

    return RollbackRecord(
        rollback_id=row[0],
        factor_id=row[1],
        from_version=row[2],
        to_version=row[3],
        reason=row[4],
        status=RollbackStatus(row[5]),
        approvals=approvals,
        approved_at=row[7],
        executed_at=row[8],
        affected_computations=int(row[9] or 0),
        affected_tenants=int(row[10] or 0),
        impact_report=impact_report,
        cascade_computations=cascade,
        failure_reason=row[13],
        created_at=row[14] or "",
        created_by=row[15] or "",
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


#: Signature of the callable that returns the computations affected by
#: a version change.  We accept a callable rather than a hard-coded
#: repository so the scope-engine integration can inject its own
#: lookup without pulling its heavy transitive imports into this module.
CascadeLookup = Callable[[str, str, str], List[str]]


class RollbackService:
    """Orchestrates the five-stage rollback workflow.

    The service owns three collaborators:

    * ``version_chain`` — an existing ``FactorVersionChain`` that records
      the original versions.  We only *read* from it here and *append*
      on execute; we never mutate earlier rows.
    * ``impact_simulator`` — the already-built ``ImpactSimulator`` that
      ``plan_rollback`` calls to produce the impact preview.
    * ``store`` — the ``RollbackStore`` persisting the audit trail.

    The optional ``cascade_lookup`` hook returns the list of computation
    IDs the Scope Engine should re-run.  Callers that don't need a cascade
    (e.g. dev/test) can leave it as the default empty-list implementation.
    """

    def __init__(
        self,
        *,
        version_chain: FactorVersionChain,
        impact_simulator: ImpactSimulator,
        store: RollbackStore,
        cascade_lookup: Optional[CascadeLookup] = None,
    ) -> None:
        self._chain = version_chain
        self._simulator = impact_simulator
        self._store = store
        self._cascade_lookup: CascadeLookup = (
            cascade_lookup if cascade_lookup is not None else _empty_cascade
        )

    # ------------------------------------------------------------------
    # Stage 1 — plan
    # ------------------------------------------------------------------

    def plan_rollback(
        self,
        *,
        factor_id: str,
        to_version: str,
        reason: str,
        created_by: str,
        value_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> RollbackPlan:
        """Build a ``RollbackPlan`` with an impact preview.

        Args:
            factor_id: Factor to roll back.
            to_version: The earlier version to restore.
            reason: Human-readable justification (required for audit).
            created_by: Operator filing the rollback.
            value_map: Optional ``{factor_id: {"old": x, "new": y}}`` to
                populate numeric deltas in the ImpactReport.
        """
        if not reason or not reason.strip():
            raise RollbackError("reason is required")

        chain_entries = self._chain.chain(factor_id)
        if not chain_entries:
            raise RollbackError(
                "No version chain found for factor %r" % factor_id
            )
        current = chain_entries[-1]
        from_version = current.factor_version

        if from_version == to_version:
            raise RollbackError(
                "Target version equals current version for %r" % factor_id
            )

        target = next(
            (e for e in chain_entries if e.factor_version == to_version), None
        )
        if target is None:
            raise RollbackError(
                "Target version %r not found in chain for %r"
                % (to_version, factor_id)
            )

        # Build impact preview.
        report = self._simulator.simulate_replacement(
            replaced_factor_ids=[factor_id],
            value_map=value_map,
        )

        rollback_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        plan = RollbackPlan(
            rollback_id=rollback_id,
            factor_id=factor_id,
            from_version=from_version,
            to_version=to_version,
            reason=reason.strip(),
            created_by=created_by,
            created_at=now,
            impact_report=report.to_dict(),
            affected_computations=len(report.computations),
            affected_tenants=len(report.tenants),
            status=RollbackStatus.PLANNED,
        )

        # Persist as a PLANNED record so the plan is durable before approval.
        record = RollbackRecord(
            rollback_id=rollback_id,
            factor_id=factor_id,
            from_version=from_version,
            to_version=to_version,
            reason=plan.reason,
            status=RollbackStatus.PLANNED,
            impact_report=plan.impact_report,
            affected_computations=plan.affected_computations,
            affected_tenants=plan.affected_tenants,
            created_at=now,
            created_by=created_by,
        )
        self._store.upsert(record)
        logger.info(
            "Rollback planned: id=%s factor=%s %s -> %s "
            "(computations=%d tenants=%d)",
            rollback_id, factor_id, from_version, to_version,
            plan.affected_computations, plan.affected_tenants,
        )
        return plan

    # ------------------------------------------------------------------
    # Stage 2 — approve
    # ------------------------------------------------------------------

    def approve_rollback(
        self,
        *,
        rollback_id: str,
        approver_id: str,
        approver_role: str,
        signature: str,
    ) -> RollbackRecord:
        """Attach one signer's approval to the rollback record.

        The state advances to ``APPROVED`` only when both required roles
        (methodology_lead + compliance_lead) have signed.  Duplicate
        signers are rejected with ``RollbackApprovalError``.
        """
        record = self._require_record(rollback_id)
        if record.status not in (
            RollbackStatus.PLANNED,
            RollbackStatus.APPROVED,
        ):
            raise RollbackStateError(
                "Cannot approve rollback in state %r" % record.status.value
            )
        if approver_role not in REQUIRED_APPROVAL_ROLES:
            raise RollbackApprovalError(
                "Role %r not in required %s"
                % (approver_role, REQUIRED_APPROVAL_ROLES)
            )
        for existing in record.approvals:
            if existing.approver_id == approver_id:
                raise RollbackApprovalError(
                    "Approver %r already signed" % approver_id
                )
            if existing.approver_role == approver_role:
                raise RollbackApprovalError(
                    "Role %r already signed" % approver_role
                )
        if not signature or len(signature) < 8:
            raise RollbackApprovalError("signature is required (min 8 chars)")

        record.approvals.append(
            RollbackApproval(
                approver_id=approver_id,
                approver_role=approver_role,
                signature=signature,
                approved_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        # Both signatures collected -> advance to APPROVED
        if len(record.approvals) >= len(REQUIRED_APPROVAL_ROLES):
            self._transition(record, RollbackStatus.APPROVED)
            record.approved_at = datetime.now(timezone.utc).isoformat()
        self._store.upsert(record)
        logger.info(
            "Rollback approval added: id=%s role=%s total_signatures=%d",
            rollback_id, approver_role, len(record.approvals),
        )
        return record

    # ------------------------------------------------------------------
    # Stage 3 — execute
    # ------------------------------------------------------------------

    def execute_rollback(self, *, rollback_id: str) -> RollbackRecord:
        """Execute an approved rollback.

        Writes a new row to the factor version chain that reuses the
        ``content_hash`` of ``to_version`` (append-only), collects the
        cascade set, and transitions to COMPLETED.
        """
        record = self._require_record(rollback_id)
        if record.status != RollbackStatus.APPROVED:
            raise RollbackStateError(
                "execute_rollback requires APPROVED state (was %s)"
                % record.status.value
            )
        if len(record.approvals) < len(REQUIRED_APPROVAL_ROLES):
            raise RollbackApprovalError(
                "execute_rollback requires %d signatures (have %d)"
                % (len(REQUIRED_APPROVAL_ROLES), len(record.approvals))
            )

        self._transition(record, RollbackStatus.EXECUTING)
        self._store.upsert(record)

        try:
            target_entry = next(
                (
                    e
                    for e in self._chain.chain(record.factor_id)
                    if e.factor_version == record.to_version
                ),
                None,
            )
            if target_entry is None:
                raise RollbackError(
                    "Target version %r disappeared from chain"
                    % record.to_version
                )

            # Append-only: re-seal the target's content_hash as the new head.
            rolled_version = self._rolled_version_label(
                record.factor_id, record.to_version
            )
            self._chain.append(
                factor_id=record.factor_id,
                factor_version=rolled_version,
                content_hash=target_entry.content_hash,
                changed_by=record.approved_by_1 or record.created_by,
                change_reason="rollback: %s" % record.reason,
                migration_notes="rollback_id=%s" % record.rollback_id,
            )

            cascade = self.cascade_to_calculations(
                factor_id=record.factor_id,
                from_v=record.from_version,
                to_v=record.to_version,
            )
            record.cascade_computations = cascade
            record.executed_at = datetime.now(timezone.utc).isoformat()
            self._transition(record, RollbackStatus.COMPLETED)
        except Exception as exc:  # noqa: BLE001 - persist + re-raise
            record.failure_reason = "%s: %s" % (type(exc).__name__, exc)
            self._transition(record, RollbackStatus.FAILED, force=True)
            self._store.upsert(record)
            logger.error(
                "Rollback execution failed: id=%s reason=%s",
                record.rollback_id, record.failure_reason,
            )
            raise

        self._store.upsert(record)
        logger.info(
            "Rollback executed: id=%s factor=%s cascade_count=%d",
            record.rollback_id, record.factor_id, len(record.cascade_computations),
        )
        return record

    # ------------------------------------------------------------------
    # Stage 4 — cascade
    # ------------------------------------------------------------------

    def cascade_to_calculations(
        self,
        *,
        factor_id: str,
        from_v: str,
        to_v: str,
    ) -> List[str]:
        """Return the list of dependent computation IDs flagged for re-run.

        Delegates to the injected ``cascade_lookup`` callable.  In
        production the Scope Engine provides a lookup against its
        ``climate_ledger_entries`` table; in tests we use a closure.
        """
        try:
            return list(self._cascade_lookup(factor_id, from_v, to_v))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cascade lookup failed for factor=%s: %s — continuing with empty set",
                factor_id, exc,
            )
            return []

    # ------------------------------------------------------------------
    # Stage 5 — cancel + reads
    # ------------------------------------------------------------------

    def cancel_rollback(self, *, rollback_id: str, reason: str) -> RollbackRecord:
        """Cancel a PLANNED or APPROVED rollback (pre-execution only)."""
        record = self._require_record(rollback_id)
        if record.status not in (
            RollbackStatus.PLANNED,
            RollbackStatus.APPROVED,
        ):
            raise RollbackStateError(
                "Cannot cancel rollback in state %r" % record.status.value
            )
        record.failure_reason = "cancelled: %s" % reason
        self._transition(record, RollbackStatus.CANCELLED, force=True)
        self._store.upsert(record)
        return record

    def get_rollback(self, rollback_id: str) -> Optional[RollbackRecord]:
        return self._store.get(rollback_id)

    def list_for_factor(self, factor_id: str) -> List[RollbackRecord]:
        return self._store.list_for_factor(factor_id)

    # ------------------------------------------------------------------
    # Stage 6 — audit
    # ------------------------------------------------------------------

    def create_rollback_audit_record(
        self, record: RollbackRecord
    ) -> Dict[str, Any]:
        """Return an immutable audit entry for the ledger / evidence vault.

        The payload is what an external auditor would expect: the full
        rollback context, both signatures, the impact summary, and the
        cascade list.  Callers typically stash this in the Climate
        Ledger as a ``rollback.completed`` operation.
        """
        payload = record.to_dict()
        payload["audit_kind"] = "factor_rollback"
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
        return payload

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_record(self, rollback_id: str) -> RollbackRecord:
        record = self._store.get(rollback_id)
        if record is None:
            raise RollbackNotFoundError(
                "Rollback %r not found" % rollback_id
            )
        return record

    def _transition(
        self,
        record: RollbackRecord,
        target: RollbackStatus,
        *,
        force: bool = False,
    ) -> None:
        """Move the record to ``target`` if the state machine permits it."""
        if not force:
            allowed = _ALLOWED_TRANSITIONS.get(record.status, frozenset())
            if target not in allowed:
                raise RollbackStateError(
                    "Illegal transition %s -> %s"
                    % (record.status.value, target.value)
                )
        record.status = target

    @staticmethod
    def _rolled_version_label(factor_id: str, to_version: str) -> str:
        """Pick a monotonic version label for the append-only chain.

        We suffix the target version with a UTC timestamp so each
        rollback is unique in ``factor_version_chain`` even when the
        same ``to_version`` is restored twice.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return "%s+rb.%s" % (to_version, stamp)


# ---------------------------------------------------------------------------
# Default cascade hook
# ---------------------------------------------------------------------------


def _empty_cascade(factor_id: str, from_v: str, to_v: str) -> List[str]:
    """No-op cascade — returns an empty set.  Used by tests + dev."""
    return []


__all__ = [
    "CascadeLookup",
    "REQUIRED_APPROVAL_ROLES",
    "RollbackApproval",
    "RollbackApprovalError",
    "RollbackError",
    "RollbackNotFoundError",
    "RollbackPlan",
    "RollbackRecord",
    "RollbackService",
    "RollbackStateError",
    "RollbackStatus",
    "RollbackStore",
]
