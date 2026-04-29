# -*- coding: utf-8 -*-
"""Unified Phase 3 ingestion-pipeline runner — composes existing components.

This module is the **runtime** half of Phase 3. It wires together five
pre-existing primitives — :class:`HttpFetcher` / :class:`FileFetcher`,
:class:`LocalArtifactStore`, :class:`ParserRegistry`, :class:`CanonicalNormalizer`,
:class:`AlphaPublisher` + :class:`AlphaFactorRepository` (with the Phase 2
:class:`PublishGateOrchestrator` running inside the repo's ``publish()``) —
into one strict 7-stage runner that enforces the run-status order matrix
defined in :mod:`pipeline`.

Per the Phase 3 plan §"Reality check": ~60% of the primitives already
exist and are NOT rewritten here. This runner only orchestrates; every
stage delegates to the pre-existing implementation.

Stage map (mirrors ``docs/factors/PHASE_3_PLAN.md`` §"The seven-stage
pipeline contract"):

1. ``fetch``     → :class:`BaseFetcher` + :class:`LocalArtifactStore`
2. ``parse``     → :class:`ParserRegistry` + :class:`run_parser`
3. ``normalize`` → :class:`CanonicalNormalizer`
4. ``validate``  → :class:`PublishGateOrchestrator` (per-record dry-run)
5. ``dedupe``    → :func:`duplicate_fingerprint` + cross-source rules
6. ``stage``     → :class:`AlphaPublisher.publish_to_staging` + diff export
7. ``publish``   → :class:`AlphaPublisher.flip_to_production`
``rollback``   → :class:`AlphaPublisher.rollback`

Constructor injection is enforced for the heavyweight collaborators
(repository, publisher, gate orchestrator). The runner deliberately does
NOT instantiate :class:`PublishGateOrchestrator` itself — Phase 2 already
wires it inside :meth:`AlphaFactorRepository.publish`, and re-instantiating
it here would duplicate the gate. Tests pass an in-memory repository and a
fake publisher, so DI is mandatory for testability anyway.
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from greenlang.factors.dedupe_rules import duplicate_fingerprint
from greenlang.factors.ingestion.artifacts import LocalArtifactStore, StoredArtifact
from greenlang.factors.ingestion.diff import (
    RunDiff,
    from_staging_diff,
    serialize_json,
    serialize_markdown,
)
from greenlang.factors.ingestion.exceptions import (
    ArtifactStoreError,
    DedupeRejectedError,
    IngestionError,
    ParserDispatchError,
    PublishOrderError,
    RollbackOrderError,
    StageOrderError,
    ValidationStageError,
)
from greenlang.factors.ingestion.fetchers import BaseFetcher, FileFetcher, HttpFetcher
from greenlang.factors.ingestion.parser_harness import ParserContext, ParserResult
from greenlang.factors.ingestion.parsers import (
    BaseSourceParser,
    ParserRegistry,
    build_default_registry,
)
from greenlang.factors.ingestion.pipeline import (
    IngestionRun,
    RunStatus,
    Stage,
    StageResult,
    assert_stage_precondition,
    now_utc,
    stage_for_target_status,
)
from greenlang.factors.ingestion.run_repository import IngestionRunRepository
from greenlang.factors.ingestion.source_safety import (
    assert_source_safe_for_env,
)

logger = logging.getLogger(__name__)


__all__ = [
    "DedupeOutcome",
    "PublishResult",
    "RollbackResult",
    "ValidationOutcome",
    "IngestionPipelineRunner",
]


# ---------------------------------------------------------------------------
# Stage output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    """Stage-4 result: per-record accept / reject lists.

    Phase 3 plan stage 4 row table calls for "reject list + accept list +
    per-record gate result". The reject list carries the record dict
    alongside the failure reason so the operator can re-run with patched
    inputs without re-fetching.
    """

    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Tuple[Dict[str, Any], str]] = field(default_factory=list)


@dataclass
class DedupeOutcome:
    """Stage-5 result: post-dedupe record set + supersede + removal hints.

    Per the Phase 3 plan §"Dedupe / supersede / diff rules", the stage
    must emit a **final** list (records that survive into staging),
    **supersede pairs** (old_urn -> new_urn within the same source),
    and **removal candidates** (production URNs absent from this run that
    a methodology lead must explicitly approve before they disappear from
    the catalogue — they are NEVER auto-deleted).
    """

    final: List[Dict[str, Any]] = field(default_factory=list)
    supersede_pairs: List[Tuple[str, str]] = field(default_factory=list)
    removal_candidates: List[str] = field(default_factory=list)
    duplicate_count: int = 0


@dataclass
class PublishResult:
    """Stage-7 result: batch identifier + count of URNs flipped to prod."""

    batch_id: str
    promoted: int
    approved_by: str


@dataclass
class RollbackResult:
    """Rollback receipt: the batch demoted + count of URNs returned to staging."""

    batch_id: str
    demoted: int
    approved_by: str


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class IngestionPipelineRunner:
    """Single-class orchestrator for the seven-stage Phase 3 ingestion contract.

    The runner is intentionally thin: every stage method does the
    minimum amount of glue work needed to (a) validate the
    run-status precondition, (b) delegate to the pre-existing primitive,
    (c) update the run repository with the new status + stage history
    receipt, and (d) return a structured result.

    Constructor parameters
    ----------------------
    run_repo
        The :class:`IngestionRunRepository` instance backing
        ``ingestion_runs`` / ``ingestion_run_stage_history`` /
        ``ingestion_run_diffs`` writes.
    factor_repo
        The :class:`AlphaFactorRepository` whose ``publish()`` runs the
        Phase 2 :class:`PublishGateOrchestrator` and writes records to
        the ``alpha_factors_v0_1`` table.
    publisher
        The :class:`AlphaPublisher` instance attached to ``factor_repo``.
        Owns the staging vs production namespace flip + the
        :class:`StagingDiff` snapshot.
    artifact_store
        The :class:`LocalArtifactStore` (or any :class:`ArtifactStore`)
        that persists the raw fetched bytes. The runner SHA-256s the
        bytes BEFORE handing them to the store and re-checks the result
        for tamper detection.
    parser_registry
        Optional :class:`ParserRegistry`. Defaults to the built-in registry
        from :func:`build_default_registry`.
    fetcher_factory
        Optional callable returning a :class:`BaseFetcher` for a given URL.
        Defaults to :func:`_default_fetcher_factory` which inspects the
        URL scheme.
    diff_root
        Filesystem directory where stage-6 diff JSON + MD artefacts are
        written. Created on first use; defaults to
        ``{cwd}/.greenlang/factors/diffs``.

    Phase 3 / WS1 references
    ------------------------
    - §"The seven-stage pipeline contract"
    - §"Run-status enum (formal)"
    - §"Artifact storage contract"
    - §"CLI surface (Click group)"
    """

    def __init__(
        self,
        *,
        run_repo: IngestionRunRepository,
        factor_repo: Any,  # AlphaFactorRepository (typed as Any to avoid heavy import)
        publisher: Any,  # AlphaPublisher
        artifact_store: LocalArtifactStore,
        parser_registry: Optional[ParserRegistry] = None,
        fetcher_factory: Optional[Callable[[str], BaseFetcher]] = None,
        diff_root: Optional[Path] = None,
        operator: str = "bot:ingestion-runner",
        env: str = "dev",
    ) -> None:
        self._run_repo = run_repo
        self._factor_repo = factor_repo
        self._publisher = publisher
        self._artifact_store = artifact_store
        self._parser_registry = parser_registry or build_default_registry()
        self._fetcher_factory = fetcher_factory or _default_fetcher_factory
        self._diff_root = Path(
            diff_root or Path.cwd() / ".greenlang" / "factors" / "diffs"
        )
        self._operator = operator
        # Phase 3 / Wave 3.0 (Block 7 / Gate 4) — environment label used by
        # ``run()`` and ``publish()`` to call
        # :func:`assert_source_safe_for_env`. Tests + dev workstations leave
        # the default ``"dev"``, which is a no-op.
        self._env = (env or "dev").strip().lower()

    # -- helpers ----------------------------------------------------------

    def _record_stage(
        self,
        run_id: str,
        stage: Stage,
        ok: bool,
        started_at: float,
        *,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        """Persist a :class:`StageResult` row and return it for the caller."""
        duration = time.monotonic() - started_at
        result = StageResult(
            stage=stage,
            ok=ok,
            duration_s=duration,
            error=error,
            details=dict(details or {}),
            finished_at=now_utc(),
        )
        try:
            self._run_repo.append_stage_history(run_id, result)
        except Exception as exc:  # noqa: BLE001 — telemetry must not crash run
            logger.warning(
                "ingestion_runner: stage-history write failed run=%s stage=%s err=%s",
                run_id, stage.value, exc,
            )
        return result

    def _resolve_parser(self, source_id: str) -> BaseSourceParser:
        parser = self._parser_registry.get(source_id)
        if parser is None:
            raise ParserDispatchError(
                "no parser registered for source_id=%s" % source_id,
                source_id=source_id,
                registered_versions=self._parser_registry.list_source_ids(),
            )
        return parser

    # -- stage 1: fetch ---------------------------------------------------

    def fetch(
        self,
        run_id: str,
        *,
        source_id: str,
        source_url: str,
    ) -> StoredArtifact:
        """Stage 1 — fetch raw bytes + persist to the artifact store.

        Validates the run is currently in :attr:`RunStatus.CREATED`,
        downloads via the appropriate :class:`BaseFetcher`, hashes the
        bytes, persists, then re-hashes the stored copy. A length-zero
        download or a sha256 mismatch raises :class:`ArtifactStoreError`
        and transitions the run to :attr:`RunStatus.FAILED`.

        Phase 3 audit gap C — also upserts the full 16+ field row into
        ``source_artifacts`` so the lineage table is the canonical
        per-run record (parser metadata, licence, operator, status).
        """
        run = self._run_repo.get(run_id)
        assert_stage_precondition(Stage.FETCH, run.status, run_id=run_id)
        started = time.monotonic()
        try:
            fetcher = self._fetcher_factory(source_url)
            data = fetcher.fetch(source_url)
            if not data:
                raise ArtifactStoreError(
                    "fetch returned 0 bytes",
                    bytes_size=0,
                )
            sha = hashlib.sha256(data).hexdigest()
            artifact = self._artifact_store.put_bytes(
                data, source_id=source_id, url=source_url
            )
            if artifact.sha256 != sha:
                raise ArtifactStoreError(
                    "sha256 mismatch after artifact store write",
                    sha256_expected=sha,
                    sha256_actual=artifact.sha256,
                    bytes_size=len(data),
                )
            self._run_repo.set_artifact(
                run_id,
                artifact_id=artifact.artifact_id,
                sha256=artifact.sha256,
            )
            # Phase 3 audit gap C — land the full source_artifacts row.
            # The runner pulls the parser_function / licence_class /
            # redistribution_class / source_publication_date from the
            # source registry. Failures here MUST NOT fail the fetch
            # stage (the source_artifacts table is an audit aid, not a
            # blocker — the runner's run row remains the canonical
            # forensic record). We log + continue.
            try:
                self._upsert_source_artifact_row(
                    run=run,
                    source_id=source_id,
                    source_url=source_url,
                    artifact=artifact,
                    operator=run.operator,
                    status="fetched",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fetch: source_artifacts upsert failed run=%s err=%s",
                    run_id, exc,
                )
            self._run_repo.update_status(
                run_id, RunStatus.FETCHED, current_stage=Stage.FETCH
            )
            self._record_stage(
                run_id, Stage.FETCH, ok=True, started_at=started,
                details={
                    "artifact_id": artifact.artifact_id,
                    "sha256": artifact.sha256,
                    "bytes_size": artifact.bytes_size,
                    "storage_uri": artifact.storage_uri,
                },
            )
            logger.info(
                "fetch ok run=%s artifact=%s sha=%s bytes=%d",
                run_id, artifact.artifact_id, artifact.sha256, artifact.bytes_size,
            )
            return artifact
        except IngestionError as exc:
            self._mark_failed(run_id, Stage.FETCH, started, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = ArtifactStoreError("fetch failed: %s" % exc)
            self._mark_failed(run_id, Stage.FETCH, started, wrapped)
            raise wrapped from exc

    # -- stage 2: parse ---------------------------------------------------

    def parse(
        self,
        run_id: str,
        *,
        source_id: str,
        artifact: StoredArtifact,
    ) -> ParserResult:
        """Stage 2 — dispatch the artifact to its registered parser.

        Reads the raw bytes back from the artifact store URI (file://
        only at v0.1 — Wave 4 will add S3), runs the parser, and records
        row counts on the stage receipt.
        """
        run = self._run_repo.get(run_id)
        assert_stage_precondition(Stage.PARSE, run.status, run_id=run_id)
        started = time.monotonic()
        try:
            parser = self._resolve_parser(source_id)
            raw_bytes = self._read_artifact(artifact)
            # Family-aware parse dispatch (Phase 3 / Wave 1.5):
            # - Excel-family parsers expose ``parse_bytes(raw, *,
            #   artifact_uri, artifact_sha256)`` because their inputs are
            #   binary workbooks, not JSON text.
            # - JSON-family parsers continue to expose the legacy
            #   ``parse(data: dict)`` ABC method; the runner JSON-loads
            #   the bytes before calling them.
            if hasattr(parser, "parse_bytes"):
                rows = parser.parse_bytes(  # type: ignore[attr-defined]
                    raw_bytes,
                    artifact_uri=artifact.storage_uri,
                    artifact_sha256=artifact.sha256,
                )
            else:
                data = json.loads(raw_bytes.decode("utf-8"))
                ok, schema_issues = parser.validate_schema(data)
                if not ok:
                    raise ValidationStageError(
                        "parser schema validation failed",
                        rejected_count=len(schema_issues),
                        first_reasons=schema_issues,
                    )
                rows = parser.parse(data)
            result = ParserResult(status="ok", rows=rows)
            self._run_repo.set_artifact(
                run_id,
                artifact_id=artifact.artifact_id,
                sha256=artifact.sha256,
                parser_module=parser.__class__.__module__,
                parser_version=getattr(parser, "parser_version", "0"),
                parser_commit=None,
            )
            self._run_repo.update_status(
                run_id, RunStatus.PARSED, current_stage=Stage.PARSE
            )
            self._advance_source_artifact_status(run_id, "parsed")
            self._record_stage(
                run_id, Stage.PARSE, ok=True, started_at=started,
                details={
                    "row_count": len(rows),
                    "parser_id": parser.parser_id,
                    "parser_version": getattr(parser, "parser_version", "0"),
                },
            )
            logger.info(
                "parse ok run=%s parser=%s rows=%d",
                run_id, parser.parser_id, len(rows),
            )
            return result
        except IngestionError as exc:
            self._mark_failed(run_id, Stage.PARSE, started, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = ParserDispatchError(
                "parse failed: %s" % exc, source_id=source_id
            )
            self._mark_failed(run_id, Stage.PARSE, started, wrapped)
            raise wrapped from exc

    # -- stage 3: normalize -----------------------------------------------

    def normalize(
        self,
        run_id: str,
        *,
        parser_result: ParserResult,
    ) -> List[Dict[str, Any]]:
        """Stage 3 — pass-through normalization for v0.1 alpha records.

        The 30 in-tree parsers already return dicts shaped to the
        ``factor_record_v0_1`` schema (Phase 1 contract), so the v0.1
        normalizer is a structural pass-through. Wave 2 will swap in
        :class:`CanonicalNormalizer` once the EmissionFactorRecord ↔
        v0.1 mapping is finalised.
        """
        run = self._run_repo.get(run_id)
        assert_stage_precondition(Stage.NORMALIZE, run.status, run_id=run_id)
        started = time.monotonic()
        try:
            records = list(parser_result.rows)
            self._run_repo.update_status(
                run_id, RunStatus.NORMALIZED, current_stage=Stage.NORMALIZE
            )
            self._advance_source_artifact_status(run_id, "normalized")
            self._record_stage(
                run_id, Stage.NORMALIZE, ok=True, started_at=started,
                details={"record_count": len(records)},
            )
            return records
        except Exception as exc:  # noqa: BLE001
            wrapped = IngestionError("normalize failed: %s" % exc, stage="normalize")
            self._mark_failed(run_id, Stage.NORMALIZE, started, wrapped)
            raise wrapped from exc

    # -- stage 4: validate ------------------------------------------------

    def validate(
        self,
        run_id: str,
        *,
        records: Sequence[Dict[str, Any]],
    ) -> ValidationOutcome:
        """Stage 4 — per-record dry-run against the Phase 2 publish gates.

        Calls :meth:`PublishGateOrchestrator.run_dry` (or its equivalent)
        per record so a single bad row does not poison the whole batch.
        Records that pass land in ``accepted``; records that fail land
        in ``rejected`` with a reason string. The stage transitions the
        run to :attr:`RunStatus.VALIDATED` even when ``rejected`` is
        non-empty — partial rejection is normal; the runner records the
        counts on the stage receipt.
        """
        run = self._run_repo.get(run_id)
        assert_stage_precondition(Stage.VALIDATE, run.status, run_id=run_id)
        started = time.monotonic()
        try:
            accepted: List[Dict[str, Any]] = []
            rejected: List[Tuple[Dict[str, Any], str]] = []
            orchestrator = self._get_orchestrator()
            for rec in records:
                try:
                    if orchestrator is not None and hasattr(orchestrator, "run_dry"):
                        result = orchestrator.run_dry(rec)
                        if getattr(result, "ok", True):
                            accepted.append(rec)
                        else:
                            reasons = getattr(result, "errors", []) or [str(result)]
                            rejected.append((rec, "; ".join(map(str, reasons))[:500]))
                    else:
                        # Fallback: schema-only check happens at publish
                        # time via the repository. Accept everything here
                        # so the dedupe stage can still run.
                        accepted.append(rec)
                except Exception as exc:  # noqa: BLE001
                    rejected.append((rec, "validate exception: %s" % exc))

            outcome = ValidationOutcome(accepted=accepted, rejected=rejected)
            self._run_repo.update_status(
                run_id, RunStatus.VALIDATED, current_stage=Stage.VALIDATE
            )
            self._advance_source_artifact_status(run_id, "validated")
            self._record_stage(
                run_id, Stage.VALIDATE, ok=True, started_at=started,
                details={
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "first_reject_reasons": [r for _, r in rejected[:10]],
                },
            )
            logger.info(
                "validate run=%s accepted=%d rejected=%d",
                run_id, len(accepted), len(rejected),
            )
            return outcome
        except IngestionError as exc:
            self._mark_failed(run_id, Stage.VALIDATE, started, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = ValidationStageError("validate failed: %s" % exc)
            self._mark_failed(run_id, Stage.VALIDATE, started, wrapped)
            raise wrapped from exc

    # -- stage 5: dedupe --------------------------------------------------

    def dedupe(
        self,
        run_id: str,
        *,
        accepted: Sequence[Dict[str, Any]],
        prior_production: Optional[Dict[str, Dict[str, Any]]] = None,
        allow_cross_source_supersede: bool = False,
    ) -> DedupeOutcome:
        """Stage 5 — dedupe within run + compute supersede + removal hints.

        Within-run duplicates use :func:`duplicate_fingerprint` to fold
        rows that describe the same activity / geography / vintage tuple.
        Cross-source supersedes (a record in this run with
        ``supersedes_urn`` pointing at a production URN from a different
        source) are blocked unless ``allow_cross_source_supersede=True``.
        Removal candidates are production URNs absent from this run; they
        are flagged but never deleted (Phase 3 plan: "never auto-deleted").
        """
        run = self._run_repo.get(run_id)
        assert_stage_precondition(Stage.DEDUPE, run.status, run_id=run_id)
        started = time.monotonic()
        try:
            seen_fingerprints: Dict[str, Dict[str, Any]] = {}
            duplicate_count = 0
            for rec in accepted:
                fp = self._fingerprint_for(rec)
                if fp in seen_fingerprints:
                    duplicate_count += 1
                    continue
                seen_fingerprints[fp] = rec
            final = list(seen_fingerprints.values())

            # Supersede pairs.
            supersede_pairs: List[Tuple[str, str]] = []
            cross_source_violations: List[Tuple[str, str]] = []
            prior = prior_production or {}
            for rec in final:
                supersedes_urn = rec.get("supersedes_urn")
                new_urn = rec.get("urn")
                if not (
                    isinstance(supersedes_urn, str)
                    and supersedes_urn
                    and isinstance(new_urn, str)
                ):
                    continue
                old_rec = prior.get(supersedes_urn)
                if old_rec is None:
                    continue
                supersede_pairs.append((supersedes_urn, new_urn))
                old_source = (old_rec.get("source_urn") or "")
                new_source = (rec.get("source_urn") or "")
                if old_source and new_source and old_source != new_source:
                    cross_source_violations.append((supersedes_urn, new_urn))

            if cross_source_violations and not allow_cross_source_supersede:
                raise DedupeRejectedError(
                    "cross-source supersede attempted without explicit approval",
                    offending_pairs=cross_source_violations,
                )

            # Removal candidates: in production but not in this run's URN set.
            current_urns = {
                rec.get("urn") for rec in final if isinstance(rec.get("urn"), str)
            }
            removal_candidates = sorted(
                u for u in prior if u not in current_urns and u
            )

            outcome = DedupeOutcome(
                final=final,
                supersede_pairs=supersede_pairs,
                removal_candidates=removal_candidates,
                duplicate_count=duplicate_count,
            )
            self._run_repo.update_status(
                run_id, RunStatus.DEDUPED, current_stage=Stage.DEDUPE
            )
            self._advance_source_artifact_status(run_id, "deduped")
            self._record_stage(
                run_id, Stage.DEDUPE, ok=True, started_at=started,
                details={
                    "final_count": len(final),
                    "duplicate_count": duplicate_count,
                    "supersede_count": len(supersede_pairs),
                    "removal_candidate_count": len(removal_candidates),
                },
            )
            return outcome
        except IngestionError as exc:
            self._mark_failed(run_id, Stage.DEDUPE, started, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = IngestionError("dedupe failed: %s" % exc, stage="dedupe")
            self._mark_failed(run_id, Stage.DEDUPE, started, wrapped)
            raise wrapped from exc

    # -- stage 6: stage + diff -------------------------------------------

    def stage(
        self,
        run_id: str,
        *,
        dedupe_outcome: DedupeOutcome,
    ) -> RunDiff:
        """Stage 6 — write records to staging namespace + emit JSON+MD diff.

        The Phase 3 plan §"Dedupe / supersede / diff rules" makes the
        Markdown diff the SOLE artefact a methodology lead reads. The
        runner persists both JSON and MD to disk, stores the URIs on
        the run row, and returns the in-memory diff.

        Records that the publisher's per-record gate rejects bubble up
        as exceptions; one bad record fails the whole stage so the
        operator gets a clear "fix this row" signal.
        """
        run = self._run_repo.get(run_id)
        assert_stage_precondition(Stage.STAGE, run.status, run_id=run_id)
        started = time.monotonic()
        try:
            # Capture the current production-side records BEFORE staging
            # so the diff can show per-attribute deltas where applicable.
            production_records: Dict[str, Dict[str, Any]] = {}
            try:
                production_records = {
                    rec.get("urn", ""): rec
                    for rec in self._publisher.list_production()
                    if isinstance(rec, dict) and rec.get("urn")
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "stage: list_production failed (%s); diff will be URN-only",
                    exc,
                )

            # Push every record to staging via the publisher (this runs the
            # Phase 2 gates inside AlphaFactorRepository.publish).
            staged_urns: List[str] = []
            for rec in dedupe_outcome.final:
                urn = self._publisher.publish_to_staging(rec)
                staged_urns.append(urn)

            # Compute the staging-vs-production diff.
            staging_diff = self._publisher.diff_staging_vs_production()
            staging_records = {
                rec.get("urn", ""): rec
                for rec in self._publisher.list_staging()
                if isinstance(rec, dict) and rec.get("urn")
            }
            run_diff = from_staging_diff(
                staging_diff,
                run_id=run_id,
                source_urn=run.source_urn,
                source_version=run.source_version,
                production_records=production_records,
                staging_records=staging_records,
            )

            # Persist the diff to disk + run row.
            json_uri, md_uri = self._write_diff_artefacts(run_id, run_diff)
            self._run_repo.set_diff(
                run_id,
                diff_json_uri=json_uri,
                diff_md_uri=md_uri,
                summary_json=serialize_json(run_diff)["summary"],
            )

            # Choose the next status: REVIEW_REQUIRED if there is anything
            # to review, STAGED if the run is a no-op (parser-version-only
            # change with no factor deltas).
            next_status = (
                RunStatus.REVIEW_REQUIRED
                if not run_diff.is_empty()
                else RunStatus.STAGED
            )
            self._run_repo.update_status(
                run_id, next_status, current_stage=Stage.STAGE
            )
            self._advance_source_artifact_status(run_id, next_status.value)
            self._record_stage(
                run_id, Stage.STAGE, ok=True, started_at=started,
                details={
                    "staged_count": len(staged_urns),
                    "diff_summary": serialize_json(run_diff)["summary"],
                    "diff_json_uri": json_uri,
                    "diff_md_uri": md_uri,
                },
            )
            logger.info(
                "stage ok run=%s staged=%d diff=%s",
                run_id, len(staged_urns),
                "non-empty" if not run_diff.is_empty() else "empty",
            )
            return run_diff
        except IngestionError as exc:
            self._mark_failed(run_id, Stage.STAGE, started, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = IngestionError("stage failed: %s" % exc, stage="stage")
            self._mark_failed(run_id, Stage.STAGE, started, wrapped)
            raise wrapped from exc

    # -- stage 7: publish -------------------------------------------------

    def publish(
        self,
        run_id: str,
        *,
        approver: str,
        batch_id: Optional[str] = None,
        source_entry: Optional[Dict[str, Any]] = None,
    ) -> PublishResult:
        """Stage 7 — flip every staged URN for this run into production.

        Refuses to run unless the run is in :attr:`RunStatus.STAGED` or
        :attr:`RunStatus.REVIEW_REQUIRED`. Calls
        :meth:`AlphaPublisher.flip_to_production` with every URN currently
        in staging that belongs to this run; on success records
        ``batch_id`` + ``approved_by`` on the run row and transitions to
        :attr:`RunStatus.PUBLISHED`.
        """
        run = self._run_repo.get(run_id)
        if run.status not in (RunStatus.STAGED, RunStatus.REVIEW_REQUIRED):
            raise PublishOrderError(
                "publish requires status in {staged, review_required}; got %s"
                % run.status.value,
                current_status=run.status.value,
                run_id=run_id,
            )
        # Phase 3 / Wave 3.0 (Block 7 / Gate 4) — last-ditch source-safety check
        # at production publish time. The ``run()`` convenience already enforces
        # this at the start of the run, but operators that drive the runner
        # stage-by-stage land here directly.
        if self._env == "production" and source_entry is not None:
            assert_source_safe_for_env(source_entry, self._env)
        started = time.monotonic()
        try:
            staging_urns = [
                rec.get("urn")
                for rec in self._publisher.list_staging()
                if isinstance(rec, dict) and isinstance(rec.get("urn"), str)
            ]
            bid = batch_id or "ingrun-%s" % uuid.uuid4().hex
            promoted = self._publisher.flip_to_production(
                urns=staging_urns,
                approved_by=approver,
                batch_id=bid,
            )
            self._run_repo.set_publish(run_id, batch_id=bid, approved_by=approver)
            self._run_repo.update_status(
                run_id, RunStatus.PUBLISHED, current_stage=Stage.PUBLISH
            )
            self._record_stage(
                run_id, Stage.PUBLISH, ok=True, started_at=started,
                details={
                    "batch_id": bid,
                    "promoted": promoted,
                    "approver": approver,
                },
            )
            logger.info(
                "publish ok run=%s batch=%s promoted=%d approver=%s",
                run_id, bid, promoted, approver,
            )
            return PublishResult(batch_id=bid, promoted=promoted, approved_by=approver)
        except IngestionError as exc:
            self._mark_failed(run_id, Stage.PUBLISH, started, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = IngestionError("publish failed: %s" % exc, stage="publish")
            self._mark_failed(run_id, Stage.PUBLISH, started, wrapped)
            raise wrapped from exc

    # -- rollback ---------------------------------------------------------

    def rollback(
        self,
        *,
        batch_id: str,
        approver: str,
    ) -> RollbackResult:
        """Demote every URN in ``batch_id`` back to staging.

        Rollback is run-scoped at the API surface but batch-scoped at the
        publisher (the underlying :meth:`AlphaPublisher.rollback` operates
        on batches). The runner looks up the run associated with the
        batch and refuses if it is not currently in
        :attr:`RunStatus.PUBLISHED`.
        """
        runs = self._runs_for_batch(batch_id)
        for run in runs:
            if run.status != RunStatus.PUBLISHED:
                raise RollbackOrderError(
                    "rollback requires status=published; run %s is %s"
                    % (run.run_id, run.status.value),
                    current_status=run.status.value,
                    batch_id=batch_id,
                )
        demoted = self._publisher.rollback(
            batch_id=batch_id, approved_by=approver
        )
        for run in runs:
            self._run_repo.update_status(
                run.run_id, RunStatus.ROLLED_BACK, current_stage=Stage.PUBLISH
            )
        logger.info(
            "rollback ok batch=%s demoted=%d approver=%s",
            batch_id, demoted, approver,
        )
        return RollbackResult(
            batch_id=batch_id, demoted=demoted, approved_by=approver
        )

    # -- whole-run convenience -------------------------------------------

    def run(
        self,
        *,
        source_id: str,
        source_url: str,
        source_urn: str,
        source_version: str,
        operator: Optional[str] = None,
        auto_stage: bool = True,
        auto_publish: bool = False,
        approver: Optional[str] = None,
        allow_cross_source_supersede: bool = False,
        source_entry: Optional[Dict[str, Any]] = None,
    ) -> IngestionRun:
        """Drive stages 1-6 (default) or 1-7 (with ``approver``) end-to-end.

        Per the Phase 3 plan §"CLI surface (Click group)" — the
        ``gl factors ingest run`` command runs stages 1-6 by default;
        stage 7 only fires when the operator passes ``--auto-publish``
        and an explicit ``--approved-by``. This method enforces both
        gates: ``auto_publish=True`` without ``approver`` raises
        :class:`PublishOrderError`.
        """
        if auto_publish and not approver:
            raise PublishOrderError(
                "auto_publish=True requires an explicit approver",
                current_status=None,
                run_id=None,
            )

        # Phase 3 / Wave 3.0 (Block 7 / Gate 4) — when this runner targets
        # ``production``, the source entry MUST clear status + release_milestone
        # gates. dev/test/staging is a no-op.
        if self._env == "production" and source_entry is not None:
            assert_source_safe_for_env(source_entry, self._env)

        run = self._run_repo.create(
            source_urn=source_urn,
            source_version=source_version,
            operator=operator or self._operator,
        )
        artifact = self.fetch(run.run_id, source_id=source_id, source_url=source_url)
        parser_result = self.parse(
            run.run_id, source_id=source_id, artifact=artifact
        )
        records = self.normalize(run.run_id, parser_result=parser_result)
        validation = self.validate(run.run_id, records=records)
        if auto_stage:
            dedupe_outcome = self.dedupe(
                run.run_id,
                accepted=validation.accepted,
                allow_cross_source_supersede=allow_cross_source_supersede,
            )
            self.stage(run.run_id, dedupe_outcome=dedupe_outcome)
            if auto_publish and approver:
                self.publish(run.run_id, approver=approver)
        return self._run_repo.get(run.run_id)

    # -- resume mode (Phase 3 audit gap B) --------------------------------

    #: Whitelist of stages the operator can resume from. Per the Phase 3
    #: plan §"The seven-stage pipeline contract", FETCH is not resumable
    #: (a fresh fetch creates a new artifact and a new run); PUBLISH is
    #: never auto-resumed (always behind explicit human approval).
    _RESUMABLE_STAGES: Dict[str, Stage] = {
        "parse": Stage.PARSE,
        "normalize": Stage.NORMALIZE,
        "validate": Stage.VALIDATE,
        "dedupe": Stage.DEDUPE,
        "stage": Stage.STAGE,
    }

    def resume(
        self,
        run_id: str,
        *,
        from_stage: str,
        source_id: Optional[str] = None,
        allow_cross_source_supersede: bool = False,
    ) -> IngestionRun:
        """Re-execute the pipeline from ``from_stage`` onward on a failed run.

        Phase 3 audit gap B — the plan says "resume mode requires explicit
        ``--from-stage <name>`` after a failed run". This method is the
        runtime half of that contract.

        Preconditions
        -------------
        * ``run_id`` must resolve to an existing run row.
        * The run's status MUST be in :attr:`RunStatus.FAILED` or
          :attr:`RunStatus.REJECTED` (resume is a recovery operation; a
          fresh ``CREATED`` run that hasn't run yet is rejected).
        * ``from_stage`` MUST be one of {``parse``, ``normalize``,
          ``validate``, ``dedupe``, ``stage``}. ``fetch`` is not
          resumable (a re-fetch creates a new run + new artifact);
          ``publish`` is always behind explicit human approval.
        * The stage's predecessor must match the run's last known
          ``current_stage`` (the stage that failed). For example, a run
          that failed during PARSE may be resumed from ``parse`` — but
          NOT from ``dedupe``, because dedupe's predecessor (VALIDATED)
          was never reached.

        Behaviour
        ---------
        On a clean precondition check the method:

        1. Resets the run's status to the predecessor of ``from_stage``
           (so :func:`assert_stage_precondition` accepts the next call).
        2. Re-fetches the artifact metadata from the run row + storage.
        3. Drives every stage from ``from_stage`` through ``stage`` (it
           never auto-publishes — publish always needs explicit approval).
        4. Returns the freshly-loaded :class:`IngestionRun`.

        Raises
        ------
        StageOrderError
            * Run is not in failed/rejected.
            * ``from_stage`` is not in the resumable whitelist.
            * The requested stage's predecessor does not match the
              run's current_stage (i.e. the run never reached the prior
              stage on its first attempt).
        """
        run = self._run_repo.get(run_id)

        # Guard 1: only failed/rejected runs are resumable.
        if run.status not in (RunStatus.FAILED, RunStatus.REJECTED):
            raise StageOrderError(
                "resume requires status in {failed, rejected}; got %s"
                % run.status.value,
                from_status=run.status.value,
                to_status=None,
                run_id=run_id,
            )

        # Guard 2: from_stage must be in the resumable whitelist.
        key = (from_stage or "").strip().lower()
        if key not in self._RESUMABLE_STAGES:
            raise StageOrderError(
                "resume from_stage must be one of %s; got %r"
                % (sorted(self._RESUMABLE_STAGES), from_stage),
                from_status=run.status.value,
                to_status=None,
                run_id=run_id,
            )
        target_stage = self._RESUMABLE_STAGES[key]

        # Guard 3: the requested stage's predecessor must match the run's
        # current_stage (i.e. the run actually reached the prior stage on
        # its first attempt). A run that died during PARSE has
        # current_stage=PARSE; it can be resumed from ``parse`` (rerun
        # the same stage) but NOT from ``dedupe``.
        from greenlang.factors.ingestion.pipeline import (  # noqa: PLC0415
            _STAGE_REQUIRED_PREDECESSOR,
            _STAGE_SUCCESS_STATUS,
        )
        if run.current_stage is None:
            raise StageOrderError(
                "resume requires a run that recorded current_stage; "
                "the run row carries no current_stage marker",
                from_status=run.status.value,
                to_status=None,
                run_id=run_id,
            )
        # The "predecessor" check: the requested stage must be either
        # the failed stage itself OR a stage downstream of it. Resuming
        # UPSTREAM of the failed stage is not allowed (would re-run
        # already-successful work and lose receipts).
        stage_order = list(self._RESUMABLE_STAGES.values())
        # Allow target_stage to equal current_stage OR the runner needs
        # the stage that failed to be the same one we resume.
        if run.current_stage != target_stage:
            raise StageOrderError(
                "resume from_stage=%s does not match the failed stage %s; "
                "the run never reached %s on its first attempt"
                % (target_stage.value, run.current_stage.value, target_stage.value),
                from_status=run.current_stage.value,
                to_status=target_stage.value,
                run_id=run_id,
            )

        # Reset the run row's status to the predecessor of the target
        # stage so :func:`assert_stage_precondition` will accept the
        # rerun. We bypass :meth:`update_status`'s transition matrix
        # (which forbids FAILED -> FETCHED) by calling the backend
        # writer directly — resume is the documented exception.
        predecessor_status = _STAGE_REQUIRED_PREDECESSOR[target_stage]
        if self._run_repo._is_postgres:
            self._run_repo._update_status_pg(  # type: ignore[attr-defined]
                run_id,
                predecessor_status,
                target_stage,
                None,
                None,
            )
        else:
            self._run_repo._update_status_sqlite(  # type: ignore[attr-defined]
                run_id,
                predecessor_status,
                target_stage,
                None,
                None,
            )
        logger.info(
            "resume run=%s reset to %s for from_stage=%s",
            run_id, predecessor_status.value, target_stage.value,
        )

        # Re-execute every stage from ``target_stage`` through the
        # default end-of-pipeline (stage 6 = STAGE). Re-load any state
        # that the resumed stages need — most stages take their input
        # from the prior stage's return value, so we mirror the run()
        # entry point but skip stages we don't need.
        artifact = self._reconstruct_stored_artifact(run)
        parser_result: Optional[ParserResult] = None
        records: List[Dict[str, Any]] = []
        validation: Optional[ValidationOutcome] = None
        dedupe_outcome: Optional[DedupeOutcome] = None
        # Stages that come BEFORE the resume target need to be replayed
        # in-memory only (no re-write to the DB) so the next stage can
        # consume their outputs.
        if target_stage == Stage.PARSE:
            sid = source_id or self._infer_source_id_from_urn(run.source_urn)
            parser_result = self.parse(run_id, source_id=sid, artifact=artifact)
            records = self.normalize(run_id, parser_result=parser_result)
            validation = self.validate(run_id, records=records)
            dedupe_outcome = self.dedupe(
                run_id, accepted=validation.accepted,
                allow_cross_source_supersede=allow_cross_source_supersede,
            )
            self.stage(run_id, dedupe_outcome=dedupe_outcome)
        elif target_stage == Stage.NORMALIZE:
            sid = source_id or self._infer_source_id_from_urn(run.source_urn)
            parser_result = self._reparse_for_resume(run, sid, artifact)
            records = self.normalize(run_id, parser_result=parser_result)
            validation = self.validate(run_id, records=records)
            dedupe_outcome = self.dedupe(
                run_id, accepted=validation.accepted,
                allow_cross_source_supersede=allow_cross_source_supersede,
            )
            self.stage(run_id, dedupe_outcome=dedupe_outcome)
        elif target_stage == Stage.VALIDATE:
            sid = source_id or self._infer_source_id_from_urn(run.source_urn)
            parser_result = self._reparse_for_resume(run, sid, artifact)
            records = list(parser_result.rows)
            validation = self.validate(run_id, records=records)
            dedupe_outcome = self.dedupe(
                run_id, accepted=validation.accepted,
                allow_cross_source_supersede=allow_cross_source_supersede,
            )
            self.stage(run_id, dedupe_outcome=dedupe_outcome)
        elif target_stage == Stage.DEDUPE:
            sid = source_id or self._infer_source_id_from_urn(run.source_urn)
            parser_result = self._reparse_for_resume(run, sid, artifact)
            records = list(parser_result.rows)
            # In resume-from-DEDUPE we trust the validate stage's prior
            # acceptance: every parsed row is treated as accepted (the
            # validation receipt sits in stage_history for audit).
            dedupe_outcome = self.dedupe(
                run_id, accepted=records,
                allow_cross_source_supersede=allow_cross_source_supersede,
            )
            self.stage(run_id, dedupe_outcome=dedupe_outcome)
        elif target_stage == Stage.STAGE:
            sid = source_id or self._infer_source_id_from_urn(run.source_urn)
            parser_result = self._reparse_for_resume(run, sid, artifact)
            records = list(parser_result.rows)
            dedupe_outcome = DedupeOutcome(
                final=records,
                supersede_pairs=[],
                removal_candidates=[],
                duplicate_count=0,
            )
            self.stage(run_id, dedupe_outcome=dedupe_outcome)
        return self._run_repo.get(run_id)

    def _reconstruct_stored_artifact(self, run: IngestionRun) -> StoredArtifact:
        """Build a :class:`StoredArtifact` from the run row's stored fields."""
        if not (run.artifact_id and run.artifact_sha256):
            raise StageOrderError(
                "resume requires a previously-fetched artifact; run %s "
                "has no artifact_id/sha256 on its row (fetch never completed)"
                % run.run_id,
                from_status=run.status.value,
                to_status=None,
                run_id=run.run_id,
            )
        # The artifact store path is reconstructed from the artifact_id
        # via the same scheme :class:`LocalArtifactStore.put_bytes` uses
        # (source_id/<sha-prefix>/<artifact_id>.bin). We don't have the
        # source_id on the row, so we glob for the artifact_id under the
        # store root.
        from urllib.parse import urlparse  # noqa: PLC0415
        store_root = self._artifact_store.root
        candidates = list(store_root.rglob(f"{run.artifact_id}.bin"))
        if not candidates:
            raise StageOrderError(
                "resume cannot locate stored artifact %s under %s"
                % (run.artifact_id, store_root),
                from_status=run.status.value,
                to_status=None,
                run_id=run.run_id,
            )
        path = candidates[0]
        bytes_size = path.stat().st_size
        uri = path.resolve().as_uri()
        _ = urlparse  # quiet linter
        return StoredArtifact(
            artifact_id=run.artifact_id,
            sha256=run.artifact_sha256,
            storage_uri=uri,
            bytes_size=bytes_size,
        )

    def _reparse_for_resume(
        self,
        run: IngestionRun,
        source_id: str,
        artifact: StoredArtifact,
    ) -> ParserResult:
        """Re-run parse() in-memory without advancing the run row.

        Resume-from-stage-N (N > parse) needs the parsed rows so the
        next stage can run, but we must NOT touch the run row's status
        (which is already past PARSED). This is a pure re-parse: the
        bytes are read, the parser runs, the rows come back. No DB write.
        """
        parser = self._resolve_parser(source_id)
        raw_bytes = self._read_artifact(artifact)
        if hasattr(parser, "parse_bytes"):
            rows = parser.parse_bytes(  # type: ignore[attr-defined]
                raw_bytes,
                artifact_uri=artifact.storage_uri,
                artifact_sha256=artifact.sha256,
            )
        else:
            data = json.loads(raw_bytes.decode("utf-8"))
            rows = parser.parse(data)
        return ParserResult(status="ok", rows=rows)

    def _infer_source_id_from_urn(self, source_urn: str) -> str:
        """Parse the ``source_id`` (registry key) out of a source URN.

        DEFRA's URN ``urn:gl:source:defra-2025`` -> ``defra-2025``. Used
        so the resume() path doesn't need to hold the source_id from the
        original run; the run row stores the URN.
        """
        if not source_urn or ":" not in source_urn:
            return source_urn
        return source_urn.rsplit(":", 1)[-1]

    # -- internal helpers -------------------------------------------------

    # -- Phase 3 audit gap C — source_artifacts row writer + status walk ---

    @staticmethod
    def _load_source_registry_entry(source_id: str) -> Dict[str, Any]:
        """Look up one entry from ``greenlang/factors/data/source_registry.yaml``.

        Returns an empty dict when the registry file is missing, the
        source_id has no entry, or YAML parsing fails — callers must
        treat the result as best-effort context, never as a gate.
        """
        from pathlib import Path  # noqa: PLC0415
        try:
            import yaml  # type: ignore  # noqa: PLC0415
        except ImportError:
            return {}
        registry_path = (
            Path(__file__).resolve().parents[1] / "data" / "source_registry.yaml"
        )
        if not registry_path.exists():
            return {}
        try:
            doc = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(doc, dict):
            return {}
        sources = doc.get("sources") or []
        if not isinstance(sources, list):
            return {}
        for entry in sources:
            if isinstance(entry, dict) and entry.get("source_id") == source_id:
                return dict(entry)
        return {}

    def _upsert_source_artifact_row(
        self,
        *,
        run: IngestionRun,
        source_id: str,
        source_url: str,
        artifact: StoredArtifact,
        operator: Optional[str],
        status: str,
    ) -> None:
        """Land the full Phase 3 contract row in source_artifacts.

        Pulls registry-resident metadata (parser_function /
        licence_class / redistribution_class / source_publication_date)
        from ``source_registry.yaml`` and combines it with the in-flight
        artifact + run state. Idempotent on sha256 — re-fetches refresh
        the row's run-scoped fields.
        """
        # Skip when the run repo doesn't expose the upsert API (e.g.
        # legacy fixtures with patched ``set_diff`` shims that don't
        # carry the new method).
        if not hasattr(self._run_repo, "upsert_source_artifact"):
            return
        registry = self._load_source_registry_entry(source_id)
        parser = self._parser_registry.get(source_id)
        parser_module = (
            (parser.__class__.__module__ if parser else None)
            or registry.get("parser_module")
        )
        parser_function = (
            registry.get("parser_function")
            or (parser.__class__.__name__ if parser else None)
        )
        parser_version = (
            getattr(parser, "parser_version", None)
            or registry.get("parser_version")
            or "0"
        )
        # Some sources expose a ``publication_date``, others a ``source_publication_date``;
        # we accept either.
        pub_date = (
            registry.get("source_publication_date")
            or registry.get("publication_date")
        )
        # Content type — best effort from URL suffix; the artifact store
        # does not retain it but we can guess from extension.
        content_type = self._guess_content_type(source_url)
        self._run_repo.upsert_source_artifact(
            sha256=artifact.sha256,
            source_urn=run.source_urn,
            source_version=run.source_version,
            source_url=source_url,
            bytes_size=artifact.bytes_size,
            content_type=content_type,
            parser_module=parser_module,
            parser_function=parser_function,
            parser_version=parser_version,
            parser_commit=registry.get("parser_commit"),
            operator=operator,
            licence_class=registry.get("licence_class") or registry.get("license_class"),
            redistribution_class=registry.get("redistribution_class"),
            source_publication_date=str(pub_date) if pub_date else None,
            ingestion_run_id=run.run_id,
            status=status,
            uri=artifact.storage_uri,
        )

    @staticmethod
    def _guess_content_type(url: str) -> Optional[str]:
        """Best-effort MIME-type guess for the fetched URL."""
        lowered = (url or "").lower()
        for suffix, ct in (
            (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            (".xls", "application/vnd.ms-excel"),
            (".csv", "text/csv"),
            (".json", "application/json"),
            (".xml", "application/xml"),
            (".pdf", "application/pdf"),
            (".zip", "application/zip"),
        ):
            if lowered.endswith(suffix):
                return ct
        return None

    def _advance_source_artifact_status(
        self, run_id: str, status: str
    ) -> None:
        """Walk the source_artifacts row's status forward by one stage.

        Best-effort — a missing repo method or a Postgres connection
        failure does NOT fail the calling stage.
        """
        if not hasattr(self._run_repo, "set_source_artifact_status"):
            return
        try:
            self._run_repo.set_source_artifact_status(
                ingestion_run_id=run_id, status=status
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "source_artifacts status update failed run=%s status=%s err=%s",
                run_id, status, exc,
            )

    def _read_artifact(self, artifact: StoredArtifact) -> bytes:
        """Read the raw bytes of a stored artifact via its file:// URI."""
        uri = artifact.storage_uri
        if uri.startswith("file://"):
            from urllib.parse import urlparse  # noqa: PLC0415
            parsed = urlparse(uri)
            path = parsed.path
            # Windows: urlparse yields '/C:/...' — strip the leading slash.
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return Path(path).read_bytes()
        raise ArtifactStoreError(
            "unsupported artifact URI scheme: %s" % uri,
        )

    def _fingerprint_for(self, rec: Dict[str, Any]) -> str:
        """Compute a dedupe fingerprint for a v0.1 record dict."""
        # Reuse :func:`duplicate_fingerprint` by attribute lookup; the
        # function already supports both records-with-attrs and dicts via
        # getattr (which falls through to __getattribute__ on dicts in
        # this repo's record types). For plain dicts we call a simpler
        # fingerprint that hashes the canonical key set.
        if hasattr(rec, "fuel_type"):
            return duplicate_fingerprint(rec)
        canonical = {
            "urn": rec.get("urn"),
            "source_urn": rec.get("source_urn"),
            "geography_urn": rec.get("geography_urn"),
            "vintage_start": rec.get("vintage_start"),
            "vintage_end": rec.get("vintage_end"),
            "value": rec.get("value"),
            "unit": rec.get("unit"),
        }
        text = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]

    def _get_orchestrator(self) -> Any:
        """Return the Phase 2 publish-gate orchestrator from the repo."""
        try:
            return self._factor_repo._get_or_build_publish_orchestrator()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingestion_runner: orchestrator lookup failed (%s); "
                "validate stage will accept all records",
                exc,
            )
            return None

    def _runs_for_batch(self, batch_id: str) -> List[IngestionRun]:
        """Return every run currently in PUBLISHED that owns ``batch_id``."""
        runs = self._run_repo.list_by_status(RunStatus.PUBLISHED)
        return [r for r in runs if r.batch_id == batch_id]

    def _write_diff_artefacts(
        self, run_id: str, diff: RunDiff
    ) -> Tuple[str, str]:
        """Persist diff JSON + Markdown to ``self._diff_root`` and return URIs."""
        self._diff_root.mkdir(parents=True, exist_ok=True)
        json_path = self._diff_root / ("%s.diff.json" % run_id)
        md_path = self._diff_root / ("%s.diff.md" % run_id)
        json_path.write_text(
            json.dumps(serialize_json(diff), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(serialize_markdown(diff), encoding="utf-8")
        return json_path.resolve().as_uri(), md_path.resolve().as_uri()

    def _mark_failed(
        self,
        run_id: str,
        stage: Stage,
        started_at: float,
        exc: Exception,
    ) -> None:
        """Record a stage failure receipt + transition the run to FAILED."""
        try:
            self._record_stage(
                run_id, stage, ok=False, started_at=started_at,
                error=str(exc),
                details=getattr(exc, "details", {}),
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            self._run_repo.update_status(
                run_id,
                RunStatus.FAILED,
                current_stage=stage,
                error_json={
                    "stage": stage.value,
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "details": getattr(exc, "details", {}),
                },
            )
        except StageOrderError:
            # The run may already be in a terminal state; ignore.
            pass
        except Exception as inner:  # noqa: BLE001
            logger.warning(
                "ingestion_runner: failed to mark run %s as FAILED (%s)",
                run_id, inner,
            )


# ---------------------------------------------------------------------------
# Default fetcher dispatch
# ---------------------------------------------------------------------------


def _default_fetcher_factory(url: str) -> BaseFetcher:
    """Pick the right fetcher implementation for a URL.

    Phase 3 v0.1 ships HTTP + file fetchers; the webhook + S3 fetchers
    arrive in Wave 3. The factory degrades to file:// for any non-http
    URL.
    """
    lowered = (url or "").lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return HttpFetcher()
    return FileFetcher()
