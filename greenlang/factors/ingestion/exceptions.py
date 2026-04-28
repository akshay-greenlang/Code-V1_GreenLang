# -*- coding: utf-8 -*-
"""Typed exceptions for the unified Phase 3 ingestion pipeline.

Every exception in this module derives from :class:`IngestionError` so a
caller can catch the entire pipeline failure surface with a single
``except IngestionError`` clause. Sub-classes pinpoint *which* of the seven
stages (or the run-status order check) rejected the call so the CLI agent
can map exit codes 1:1 to status enums.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"The seven-stage pipeline contract":
  every stage commits a row to ``ingestion_runs`` advancing the status
  enum; a failure short-circuits with the stage name + structured error.
- ``docs/factors/PHASE_3_PLAN.md`` §"Run-status enum (formal)":
  the ``rejected`` / ``failed`` / ``rolled_back`` terminal states are
  mirrored by the exception types raised here.
"""

from typing import Any, Dict, List, Optional


class IngestionError(Exception):
    """Base class for every error raised by the Phase 3 ingestion pipeline.

    Carries an optional ``details`` dict that the runner attaches to
    ``ingestion_runs.error_json`` so a CLI ``status`` command can render
    the structured failure verbatim.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stage = stage
        self.details: Dict[str, Any] = dict(details or {})
        super().__init__(message)


class StageOrderError(IngestionError):
    """Caller invoked a stage out of order.

    Phase 3 enforces a strict ``created -> fetched -> parsed -> normalized
    -> validated -> deduped -> staged -> published`` progression. Any call
    that would advance the status from a non-adjacent (or earlier) state
    raises this exception. The runner logs the offending transition before
    re-raising so the operator can replay from the last good stage via
    ``--from-stage``.
    """

    def __init__(
        self,
        message: str,
        *,
        from_status: Optional[str] = None,
        to_status: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            stage="order",
            details={
                "from_status": from_status,
                "to_status": to_status,
                "run_id": run_id,
            },
        )


class ArtifactStoreError(IngestionError):
    """Fetch wrote zero bytes, or the post-store re-hash disagrees with the
    pre-store sha256.

    Per the Phase 3 plan §"Artifact storage contract", checksum verification
    happens at every stage transition; a mismatch is fatal because every
    certified factor MUST resolve to a stable ``raw_artifact_uri`` +
    ``raw_artifact_sha256`` pair.
    """

    def __init__(
        self,
        message: str,
        *,
        sha256_expected: Optional[str] = None,
        sha256_actual: Optional[str] = None,
        bytes_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            message,
            stage="fetch",
            details={
                "sha256_expected": sha256_expected,
                "sha256_actual": sha256_actual,
                "bytes_size": bytes_size,
            },
        )


class ParserDispatchError(IngestionError):
    """The :class:`ParserRegistry` has no entry for the requested source_id,
    or the registered parser's ``parser_version`` disagrees with the
    ``source_registry.yaml`` pin.

    Phase 3 plan §"Fetcher / parser families" makes the registry the single
    dispatch authority; an unregistered source MUST be rejected at parse
    time, not silently skipped (the legacy ``bulk_ingest`` fallback).
    """

    def __init__(
        self,
        message: str,
        *,
        source_id: Optional[str] = None,
        registered_versions: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            stage="parse",
            details={
                "source_id": source_id,
                "registered_versions": list(registered_versions or []),
            },
        )


class ValidationStageError(IngestionError):
    """One or more accepted records failed schema, licence, or ontology checks.

    Distinguishes from per-record validation rejection (which the runner
    captures in the ``rejected`` list returned from ``validate``); this
    exception fires only when the validation layer itself is mis-wired
    (e.g. orchestrator missing, schema unparseable, ontology table absent).

    Phase 3 plan §"The seven-stage pipeline contract" stage 4: the gate
    must run cleanly before the run can transition to ``validated``.
    """

    def __init__(
        self,
        message: str,
        *,
        rejected_count: int = 0,
        first_reasons: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            stage="validate",
            details={
                "rejected_count": rejected_count,
                "first_reasons": list(first_reasons or [])[:10],
            },
        )


class DedupeRejectedError(IngestionError):
    """A cross-source supersede was attempted without a methodology-lead
    approval flag.

    Per Phase 3 plan §"Dedupe / supersede / diff rules": cross-source
    supersedes are blocked by default and require explicit approval. The
    runner refuses to advance the run to ``staged`` until the operator
    re-runs with ``--allow-cross-source-supersede`` and an approver email.
    """

    def __init__(
        self,
        message: str,
        *,
        offending_pairs: Optional[List[tuple]] = None,
    ) -> None:
        super().__init__(
            message,
            stage="dedupe",
            details={
                "offending_pairs": list(offending_pairs or [])[:20],
            },
        )


class PublishOrderError(IngestionError):
    """``publish()`` was called on a run not in ``staged`` or
    ``review_required``.

    Phase 3 plan §"The seven-stage pipeline contract" stage 7 requires a
    fully-staged run + an explicit human approver before the publish-batch
    flip can run. This exception is raised before the
    :class:`AlphaPublisher.flip_to_production` call, so no DB writes
    occur on a malformed publish request.
    """

    def __init__(
        self,
        message: str,
        *,
        current_status: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            stage="publish",
            details={
                "current_status": current_status,
                "run_id": run_id,
            },
        )


class RollbackOrderError(IngestionError):
    """``rollback()`` was called on a run not in ``published``.

    Per Phase 3 plan §"The seven-stage pipeline contract": a rollback
    demotes a *published* run back to ``staged`` (the underlying records
    are never deleted; the rollback is recorded in
    ``factor_publish_log``). Calling rollback on a run that never reached
    ``published`` is a programming error.
    """

    def __init__(
        self,
        message: str,
        *,
        current_status: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            stage="rollback",
            details={
                "current_status": current_status,
                "batch_id": batch_id,
            },
        )


__all__ = [
    "IngestionError",
    "StageOrderError",
    "ArtifactStoreError",
    "ParserDispatchError",
    "ValidationStageError",
    "DedupeRejectedError",
    "PublishOrderError",
    "RollbackOrderError",
]
