# -*- coding: utf-8 -*-
"""Run-status enum, stage enum, and the stage transition matrix.

This module is the **contract** half of the unified Phase 3 ingestion
pipeline (the runtime composition lives in :mod:`runner`). It encodes:

1. :class:`RunStatus` — the canonical run lifecycle states. Mirrors the
   Postgres ENUM that V507 will introduce so the Python and SQL surfaces
   stay 1:1.
2. :class:`Stage` — the seven ordered stages an ingestion run executes,
   plus the implicit ``ROLLBACK`` operation handled out-of-band.
3. :class:`IngestionRun` — the canonical in-memory snapshot of a row in
   the ``ingestion_runs`` table.
4. :class:`StageResult` — the structured per-stage execution receipt
   that the runner persists to ``ingestion_runs.stage_history`` (V508).
5. The stage transition matrix that enforces the lifecycle ordering
   contract — ``StageOrderError`` is raised on every illegal transition.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"The seven-stage pipeline contract"
- ``docs/factors/PHASE_3_PLAN.md`` §"Run-status enum (formal)"
- ``docs/factors/PHASE_3_PLAN.md`` §"Wave 1.0 Framework"
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional

from greenlang.factors.ingestion.exceptions import StageOrderError

logger = logging.getLogger(__name__)


__all__ = [
    "RunStatus",
    "Stage",
    "IngestionRun",
    "StageResult",
    "TERMINAL_FAILURE_STATUSES",
    "TERMINAL_SUCCESS_STATUSES",
    "ALLOWED_TRANSITIONS",
    "assert_can_transition",
    "stage_for_target_status",
    "now_utc",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):
    """Canonical lifecycle states for an ``ingestion_runs`` row.

    Mirrors the Postgres ENUM that Alembic V507 introduces. Using ``str``
    inheritance keeps the JSON / row serialization byte-identical to the
    Postgres representation (``'fetched'`` not ``'RunStatus.FETCHED'``).

    See ``docs/factors/PHASE_3_PLAN.md`` §"Run-status enum (formal)".
    """

    CREATED = "created"
    FETCHED = "fetched"
    PARSED = "parsed"
    NORMALIZED = "normalized"
    VALIDATED = "validated"
    DEDUPED = "deduped"
    STAGED = "staged"
    REVIEW_REQUIRED = "review_required"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Stage(str, Enum):
    """Pipeline stages — the seven ordered units of work in a run.

    Each entry corresponds to one row in the table at
    ``docs/factors/PHASE_3_PLAN.md`` §"The seven-stage pipeline contract".
    The ``ROLLBACK`` member is provided so :class:`StageResult` can record
    the rollback operation without needing a parallel enum.
    """

    FETCH = "fetch"
    PARSE = "parse"
    NORMALIZE = "normalize"
    VALIDATE = "validate"
    DEDUPE = "dedupe"
    STAGE = "stage"
    PUBLISH = "publish"


# ---------------------------------------------------------------------------
# Terminal status sets
# ---------------------------------------------------------------------------


#: Statuses that represent a fully successful run. Exit code 0 if the CLI
#: returns one of these.
TERMINAL_SUCCESS_STATUSES: FrozenSet[RunStatus] = frozenset(
    {RunStatus.PUBLISHED, RunStatus.STAGED, RunStatus.REVIEW_REQUIRED}
)

#: Statuses that represent a terminal failure. Exit code >= 1.
TERMINAL_FAILURE_STATUSES: FrozenSet[RunStatus] = frozenset(
    {RunStatus.REJECTED, RunStatus.FAILED, RunStatus.ROLLED_BACK}
)


# ---------------------------------------------------------------------------
# Stage transition matrix
# ---------------------------------------------------------------------------


#: Adjacency list: each ``RunStatus`` maps to the set of statuses that may
#: legally follow it. The seven happy-path edges plus the failure edges
#: (every non-terminal status may transition to ``failed`` or ``rejected``,
#: every ``review_required`` transition must come from ``staged``, and only
#: ``published`` may transition to ``rolled_back``).
ALLOWED_TRANSITIONS: Dict[RunStatus, FrozenSet[RunStatus]] = {
    RunStatus.CREATED: frozenset(
        {RunStatus.FETCHED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.FETCHED: frozenset(
        {RunStatus.PARSED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.PARSED: frozenset(
        {RunStatus.NORMALIZED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.NORMALIZED: frozenset(
        {RunStatus.VALIDATED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.VALIDATED: frozenset(
        {RunStatus.DEDUPED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.DEDUPED: frozenset(
        {RunStatus.STAGED, RunStatus.REVIEW_REQUIRED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.STAGED: frozenset(
        {RunStatus.REVIEW_REQUIRED, RunStatus.PUBLISHED, RunStatus.FAILED, RunStatus.REJECTED}
    ),
    RunStatus.REVIEW_REQUIRED: frozenset(
        {RunStatus.PUBLISHED, RunStatus.REJECTED, RunStatus.FAILED}
    ),
    RunStatus.PUBLISHED: frozenset({RunStatus.ROLLED_BACK}),
    # Terminal states — no further transitions.
    RunStatus.REJECTED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.ROLLED_BACK: frozenset({RunStatus.STAGED}),  # re-stage after rollback
}


#: Stage -> the status it produces on success. Used by the runner to map
#: a target stage to the ``status`` value it must write on completion.
_STAGE_SUCCESS_STATUS: Dict[Stage, RunStatus] = {
    Stage.FETCH: RunStatus.FETCHED,
    Stage.PARSE: RunStatus.PARSED,
    Stage.NORMALIZE: RunStatus.NORMALIZED,
    Stage.VALIDATE: RunStatus.VALIDATED,
    Stage.DEDUPE: RunStatus.DEDUPED,
    Stage.STAGE: RunStatus.STAGED,
    Stage.PUBLISH: RunStatus.PUBLISHED,
}


#: Stage -> the status the run MUST already be in before that stage may run.
#: Mirrors the seven-stage contract: stage N's precondition is the success
#: status of stage N-1 (FETCH being the special case that requires CREATED).
_STAGE_REQUIRED_PREDECESSOR: Dict[Stage, RunStatus] = {
    Stage.FETCH: RunStatus.CREATED,
    Stage.PARSE: RunStatus.FETCHED,
    Stage.NORMALIZE: RunStatus.PARSED,
    Stage.VALIDATE: RunStatus.NORMALIZED,
    Stage.DEDUPE: RunStatus.VALIDATED,
    Stage.STAGE: RunStatus.DEDUPED,
    Stage.PUBLISH: RunStatus.STAGED,
}


def stage_for_target_status(stage: Stage) -> RunStatus:
    """Return the ``RunStatus`` value that ``stage`` produces on success.

    Helper exposed so the runner does not need to hold the matrix copy
    in two places.
    """
    return _STAGE_SUCCESS_STATUS[stage]


def assert_can_transition(
    current: RunStatus,
    desired: RunStatus,
    *,
    run_id: Optional[str] = None,
) -> None:
    """Validate that ``current -> desired`` is a legal transition.

    Raises :class:`StageOrderError` on a violation. ``run_id`` is included
    in the error payload so the operator can locate the offending row in
    ``ingestion_runs`` without a follow-up query.

    Per Phase 3 plan §"The seven-stage pipeline contract": no stage may
    be skipped except via explicit ``--from-stage`` resume mode after a
    failed run. Even that resume path replays from the failed stage onward;
    it never jumps over an intermediate state.
    """
    if current == desired:
        # Idempotent re-write of the same status (e.g. logging a retry).
        return
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if desired in allowed:
        return
    raise StageOrderError(
        "illegal status transition: %s -> %s" % (current.value, desired.value),
        from_status=current.value,
        to_status=desired.value,
        run_id=run_id,
    )


def assert_stage_precondition(
    stage: Stage,
    current_status: RunStatus,
    *,
    run_id: Optional[str] = None,
) -> None:
    """Validate that ``current_status`` permits running ``stage`` next.

    Raises :class:`StageOrderError` if ``current_status`` is not the stage's
    required predecessor. The runner calls this BEFORE executing any stage
    work so a malformed sequence aborts before side-effects begin.
    """
    required = _STAGE_REQUIRED_PREDECESSOR[stage]
    if current_status == required:
        return
    raise StageOrderError(
        "stage %s requires predecessor status %s; run currently at %s"
        % (stage.value, required.value, current_status.value),
        from_status=current_status.value,
        to_status=_STAGE_SUCCESS_STATUS[stage].value,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Dataclasses — IngestionRun, StageResult
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    """Return a timezone-aware UTC ``datetime``.

    Centralised so every dataclass + repository write uses the same clock
    helper (no naive ``datetime.utcnow()`` calls in Phase 3 code).
    """
    return datetime.now(timezone.utc)


@dataclass
class IngestionRun:
    """In-memory snapshot of an ``ingestion_runs`` row.

    Field set is the authoritative Phase 3 contract; the V507 migration
    introduces these columns 1:1 (with ``status`` backed by a Postgres
    ENUM matching :class:`RunStatus`). Mirrors the Phase 3 plan's
    §"Artifact storage contract" requirement that every certified factor
    traces back to a ``raw_artifact_uri`` + ``raw_artifact_sha256`` pair
    plus the parser metadata that produced it.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_urn: str = ""
    source_version: str = ""
    started_at: datetime = field(default_factory=now_utc)
    status: RunStatus = RunStatus.CREATED
    current_stage: Optional[Stage] = None
    artifact_id: Optional[str] = None
    artifact_sha256: Optional[str] = None
    parser_module: Optional[str] = None
    parser_version: Optional[str] = None
    parser_commit: Optional[str] = None
    operator: str = ""
    batch_id: Optional[str] = None
    approved_by: Optional[str] = None
    error_json: Optional[Dict[str, Any]] = None
    diff_json_uri: Optional[str] = None
    diff_md_uri: Optional[str] = None
    finished_at: Optional[datetime] = None

    def to_row(self) -> Dict[str, Any]:
        """Serialise to a column-aligned dict the run repository persists.

        Datetimes use ISO-8601 with timezone; ``status`` and ``current_stage``
        are emitted as enum values (the ``str`` mixin handles the cast).
        """
        return {
            "run_id": self.run_id,
            "source_urn": self.source_urn,
            "source_version": self.source_version,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "status": self.status.value,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "parser_module": self.parser_module,
            "parser_version": self.parser_version,
            "parser_commit": self.parser_commit,
            "operator": self.operator,
            "batch_id": self.batch_id,
            "approved_by": self.approved_by,
            "error_json": self.error_json,
            "diff_json_uri": self.diff_json_uri,
            "diff_md_uri": self.diff_md_uri,
        }


@dataclass
class StageResult:
    """Structured receipt for a single stage execution.

    Persisted as one row in ``ingestion_run_stage_history`` (V508) so a
    CLI ``status`` command can render the full per-stage timeline. The
    ``details`` blob carries stage-specific counters: row counts for
    parse/normalize, accept/reject counts for validate, supersede pairs
    for dedupe, etc.
    """

    stage: Stage
    ok: bool
    duration_s: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=now_utc)
    finished_at: Optional[datetime] = None

    def to_row(self) -> Dict[str, Any]:
        """Serialise to a column-aligned dict for run repository persistence."""
        return {
            "stage": self.stage.value,
            "ok": bool(self.ok),
            "duration_s": float(self.duration_s),
            "error": self.error,
            "details": dict(self.details),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
