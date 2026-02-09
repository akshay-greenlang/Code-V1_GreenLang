# -*- coding: utf-8 -*-
"""
Deduplication Pipeline - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

End-to-end orchestration of the 6 core deduplication engines:
RecordFingerprinter, BlockingEngine, SimilarityScorer, MatchClassifier,
ClusterResolver, and MergeEngine. Executes the 7-stage pipeline
(fingerprint, block, compare, classify, cluster, merge, complete)
with checkpointing, resumability, and comprehensive reporting.

Zero-Hallucination Guarantees:
    - Pipeline orchestration is deterministic stage-by-stage execution
    - No ML/LLM calls in the orchestration path
    - All sub-engines individually enforce zero-hallucination
    - Provenance chain maintained across all pipeline stages
    - Checkpoints use serializable state only

Pipeline Stages (7):
    FINGERPRINT: Generate record fingerprints (SHA-256/SimHash/MinHash)
    BLOCK:       Partition records into candidate blocks
    COMPARE:     Pairwise similarity scoring within blocks
    CLASSIFY:    Classify comparisons into MATCH/POSSIBLE/NON_MATCH
    CLUSTER:     Group matched pairs into duplicate clusters
    MERGE:       Merge duplicate clusters into canonical records
    COMPLETE:    Pipeline finished, generate report

Example:
    >>> from greenlang.duplicate_detector.deduplication_pipeline import DeduplicationPipeline
    >>> pipeline = DeduplicationPipeline()
    >>> report = pipeline.run_pipeline(
    ...     records=[{"id": "1", "name": "Alice"}, {"id": "2", "name": "Alice"}],
    ...     rule=dedup_rule,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from greenlang.duplicate_detector.blocking_engine import BlockingEngine
from greenlang.duplicate_detector.cluster_resolver import ClusterResolver
from greenlang.duplicate_detector.match_classifier import MatchClassifier
from greenlang.duplicate_detector.merge_engine import MergeEngine
from greenlang.duplicate_detector.models import (
    BlockResult,
    BlockingStrategy,
    ClusterAlgorithm,
    ConflictResolution,
    DedupJob,
    DedupReport,
    DedupRule,
    DedupStatistics,
    DuplicateCluster,
    FingerprintAlgorithm,
    IssueSeverity,
    DedupIssueSummary,
    JobStatus,
    MatchClassification,
    MatchResult,
    MergeDecision,
    MergeStrategy,
    PipelineCheckpoint,
    PipelineStage,
    RecordFingerprint,
    RecordSource,
    ReportFormat,
    SimilarityResult,
)
from greenlang.duplicate_detector.record_fingerprinter import RecordFingerprinter
from greenlang.duplicate_detector.similarity_scorer import SimilarityScorer

logger = logging.getLogger(__name__)

__all__ = [
    "DeduplicationPipeline",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a pipeline operation."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE: int = 10_000
_MAX_COMPARISONS_PER_BLOCK: int = 50_000
_CHECKPOINT_INTERVAL: int = 1000


# =============================================================================
# DeduplicationPipeline
# =============================================================================


class DeduplicationPipeline:
    """End-to-end deduplication pipeline orchestrating 6 core engines.

    Executes a 7-stage pipeline: fingerprint -> block -> compare ->
    classify -> cluster -> merge -> complete. Supports checkpointing
    for resumability, batch processing for memory efficiency, and
    comprehensive reporting for audit compliance.

    This pipeline follows GreenLang's zero-hallucination principle
    by delegating all calculations to deterministic sub-engines.

    Attributes:
        fingerprinter: RecordFingerprinter engine instance.
        blocker: BlockingEngine instance.
        scorer: SimilarityScorer engine instance.
        classifier: MatchClassifier engine instance.
        cluster_resolver: ClusterResolver engine instance.
        merge_engine: MergeEngine engine instance.

    Example:
        >>> pipeline = DeduplicationPipeline()
        >>> report = pipeline.run_pipeline(records, rule)
        >>> print(report.summary)
    """

    def __init__(self) -> None:
        """Initialize DeduplicationPipeline with all 6 sub-engines."""
        self.fingerprinter = RecordFingerprinter()
        self.blocker = BlockingEngine()
        self.scorer = SimilarityScorer()
        self.classifier = MatchClassifier()
        self.cluster_resolver = ClusterResolver()
        self.merge_engine = MergeEngine()

        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None

        self._checkpoints: Dict[str, PipelineCheckpoint] = {}

        logger.info("DeduplicationPipeline initialized with all 6 engines")

    # ------------------------------------------------------------------
    # Public API - Full pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        records: List[Dict[str, Any]],
        rule: DedupRule,
        dataset_ids: Optional[List[str]] = None,
        record_source: RecordSource = RecordSource.SINGLE_DATASET,
        checkpoint_interval: int = _CHECKPOINT_INTERVAL,
        max_comparisons_per_block: int = _MAX_COMPARISONS_PER_BLOCK,
    ) -> DedupReport:
        """Run the full deduplication pipeline end-to-end.

        Executes all 7 stages sequentially: fingerprint, block,
        compare, classify, cluster, merge, complete.

        Args:
            records: List of record dictionaries to deduplicate.
            rule: Dedup rule defining comparison and merge behavior.
            dataset_ids: Optional dataset identifiers.
            record_source: Single-dataset or cross-dataset mode.
            checkpoint_interval: Records between checkpoint saves.
            max_comparisons_per_block: Max comparisons per block.

        Returns:
            DedupReport with full results.

        Raises:
            ValueError: If records or rule are invalid.
        """
        start_time = time.monotonic()
        job = DedupJob(
            dataset_ids=dataset_ids or [],
            rule_id=rule.rule_id,
            status=JobStatus.RUNNING,
            stage=PipelineStage.FINGERPRINT,
            total_records=len(records),
            started_at=_utcnow(),
        )

        issues: List[DedupIssueSummary] = []

        try:
            # Validate input
            self.validate_input(records, rule)

            # Build record lookup
            record_map: Dict[str, Dict[str, Any]] = {}
            for idx, rec in enumerate(records):
                rid = str(rec.get("id", f"rec-{idx}"))
                if "id" not in rec:
                    rec["id"] = rid
                record_map[rid] = rec

            # Stage 1: Fingerprint
            job.stage = PipelineStage.FINGERPRINT
            fingerprints = self._run_fingerprint_stage(
                records, rule, job, issues,
            )

            # Stage 2: Block
            job.stage = PipelineStage.BLOCK
            blocks = self._run_block_stage(
                records, rule, job, issues,
            )

            # Stage 3: Compare
            job.stage = PipelineStage.COMPARE
            comparisons = self._run_compare_stage(
                blocks, record_map, rule, job, issues,
                max_comparisons_per_block,
            )

            # Stage 4: Classify
            job.stage = PipelineStage.CLASSIFY
            match_results = self._run_classify_stage(
                comparisons, rule, job, issues,
            )

            # Stage 5: Cluster
            job.stage = PipelineStage.CLUSTER
            clusters = self._run_cluster_stage(
                match_results, record_map, rule, job, issues,
            )

            # Stage 6: Merge
            job.stage = PipelineStage.MERGE
            merge_decisions = self._run_merge_stage(
                clusters, record_map, rule, job, issues,
            )

            # Stage 7: Complete
            job.stage = PipelineStage.COMPLETE
            job.status = JobStatus.COMPLETED
            job.completed_at = _utcnow()

            # Generate report
            report = self.generate_report(
                job, clusters, merge_decisions, issues,
            )

            self._record_success(time.monotonic() - start_time)
            logger.info(
                "Pipeline completed: %d records -> %d clusters, "
                "%d merges, %d issues",
                len(records), len(clusters),
                len(merge_decisions), len(issues),
            )
            return report

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = _utcnow()
            self._record_failure(time.monotonic() - start_time)
            logger.error("Pipeline failed: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Public API - Individual stage execution
    # ------------------------------------------------------------------

    def run_stage(
        self,
        stage: PipelineStage,
        records: List[Dict[str, Any]],
        rule: DedupRule,
        **kwargs: Any,
    ) -> Any:
        """Run a single pipeline stage.

        Args:
            stage: Pipeline stage to execute.
            records: Input records.
            rule: Dedup rule.
            **kwargs: Additional stage-specific parameters.

        Returns:
            Stage-specific output.

        Raises:
            ValueError: If stage is unknown.
        """
        job = DedupJob(
            status=JobStatus.RUNNING,
            stage=stage,
            total_records=len(records),
        )
        issues: List[DedupIssueSummary] = []

        record_map = {
            str(r.get("id", f"rec-{i}")): r
            for i, r in enumerate(records)
        }

        if stage == PipelineStage.FINGERPRINT:
            return self._run_fingerprint_stage(records, rule, job, issues)
        elif stage == PipelineStage.BLOCK:
            return self._run_block_stage(records, rule, job, issues)
        elif stage == PipelineStage.COMPARE:
            blocks = kwargs.get("blocks", [])
            return self._run_compare_stage(
                blocks, record_map, rule, job, issues,
            )
        elif stage == PipelineStage.CLASSIFY:
            comparisons = kwargs.get("comparisons", [])
            return self._run_classify_stage(comparisons, rule, job, issues)
        elif stage == PipelineStage.CLUSTER:
            match_results = kwargs.get("match_results", [])
            return self._run_cluster_stage(
                match_results, record_map, rule, job, issues,
            )
        elif stage == PipelineStage.MERGE:
            clusters = kwargs.get("clusters", [])
            return self._run_merge_stage(
                clusters, record_map, rule, job, issues,
            )
        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")

    # ------------------------------------------------------------------
    # Public API - Checkpointing
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        job_id: str,
        stage: PipelineStage,
        records_processed: int,
        state_data: Dict[str, Any],
    ) -> PipelineCheckpoint:
        """Save a pipeline checkpoint for resumability.

        Args:
            job_id: Identifier of the running job.
            stage: Current pipeline stage.
            records_processed: Number of records processed so far.
            state_data: Serializable pipeline state.

        Returns:
            PipelineCheckpoint instance.
        """
        provenance = _compute_provenance(
            "checkpoint", f"{job_id}:{stage.value}:{records_processed}",
        )

        cp = PipelineCheckpoint(
            job_id=job_id,
            stage=stage,
            records_processed=records_processed,
            state_data=state_data,
            provenance_hash=provenance,
        )

        self._checkpoints[job_id] = cp
        logger.info(
            "Checkpoint saved for job %s at stage %s (%d records)",
            job_id, stage.value, records_processed,
        )
        return cp

    def resume_from_checkpoint(
        self,
        job_id: str,
    ) -> Optional[PipelineCheckpoint]:
        """Retrieve the latest checkpoint for a job.

        Args:
            job_id: Identifier of the job to resume.

        Returns:
            PipelineCheckpoint if found, None otherwise.
        """
        cp = self._checkpoints.get(job_id)
        if cp:
            logger.info(
                "Resuming job %s from stage %s (%d records)",
                job_id, cp.stage.value, cp.records_processed,
            )
        else:
            logger.warning("No checkpoint found for job %s", job_id)
        return cp

    # ------------------------------------------------------------------
    # Public API - Statistics and reporting
    # ------------------------------------------------------------------

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Return operational statistics for the pipeline and all sub-engines.

        Returns:
            Dictionary with pipeline and per-engine statistics.
        """
        with self._stats_lock:
            avg_ms = 0.0
            if self._invocations > 0:
                avg_ms = self._total_duration_ms / self._invocations

            return {
                "pipeline": {
                    "invocations": self._invocations,
                    "successes": self._successes,
                    "failures": self._failures,
                    "total_duration_ms": round(self._total_duration_ms, 3),
                    "avg_duration_ms": round(avg_ms, 3),
                    "last_invoked_at": (
                        self._last_invoked_at.isoformat()
                        if self._last_invoked_at else None
                    ),
                    "active_checkpoints": len(self._checkpoints),
                },
                "engines": {
                    "fingerprinter": self.fingerprinter.get_statistics(),
                    "blocker": self.blocker.get_statistics(),
                    "scorer": self.scorer.get_statistics(),
                    "classifier": self.classifier.get_statistics(),
                    "cluster_resolver": self.cluster_resolver.get_statistics(),
                    "merge_engine": self.merge_engine.get_statistics(),
                },
            }

    def generate_report(
        self,
        job: DedupJob,
        clusters: List[DuplicateCluster],
        merge_decisions: List[MergeDecision],
        issues: List[DedupIssueSummary],
    ) -> DedupReport:
        """Generate a comprehensive deduplication report.

        Args:
            job: The dedup job metadata.
            clusters: List of duplicate clusters found.
            merge_decisions: List of merge decisions made.
            issues: List of issues encountered.

        Returns:
            DedupReport instance.
        """
        # Compute statistics
        total_duplicates = sum(c.member_count for c in clusters)
        avg_cluster_size = (
            total_duplicates / len(clusters) if clusters else 0.0
        )
        avg_similarity = 0.0
        if clusters:
            qualities = [c.cluster_quality for c in clusters]
            avg_similarity = sum(qualities) / len(qualities)

        dup_rate = 0.0
        if job.total_records > 0:
            dup_rate = min(1.0, job.matched / job.total_records)

        statistics = DedupStatistics(
            total_jobs=1,
            total_records=job.total_records,
            total_duplicates=total_duplicates,
            total_merges=len(merge_decisions),
            avg_similarity=round(avg_similarity, 4),
            duplicate_rate=round(dup_rate, 4),
            avg_cluster_size=round(avg_cluster_size, 2),
            by_status={job.status.value: 1},
        )

        # Build summary text
        summary = (
            f"Deduplication completed: {job.total_records} records processed, "
            f"{len(clusters)} duplicate clusters found, "
            f"{len(merge_decisions)} merges performed. "
            f"Duplicate rate: {dup_rate:.1%}. "
            f"Average cluster size: {avg_cluster_size:.1f}. "
            f"{len(issues)} issues detected."
        )

        provenance = _compute_provenance(
            "generate_report", f"{job.job_id}:{len(clusters)}:{len(merge_decisions)}",
        )

        return DedupReport(
            job_id=job.job_id,
            format=ReportFormat.JSON,
            summary=summary,
            clusters=clusters,
            merge_decisions=merge_decisions,
            statistics=statistics,
            issues=issues,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Public API - Input validation
    # ------------------------------------------------------------------

    def validate_input(
        self,
        records: List[Dict[str, Any]],
        rule: DedupRule,
    ) -> None:
        """Validate pipeline input records and rule.

        Args:
            records: Input records to validate.
            rule: Dedup rule to validate.

        Raises:
            ValueError: If input is invalid.
        """
        if not records:
            raise ValueError("records list must not be empty")
        if len(records) < 2:
            raise ValueError("At least 2 records are required for deduplication")

        if not rule.field_configs:
            raise ValueError("rule must have at least one field_config")

        # Check that field configs reference existing fields
        available_fields: set = set()
        for rec in records:
            available_fields.update(rec.keys())

        for fc in rule.field_configs:
            if fc.field_name not in available_fields:
                logger.warning(
                    "Field '%s' in rule not found in any record",
                    fc.field_name,
                )

    # ------------------------------------------------------------------
    # Public API - Comparison estimation
    # ------------------------------------------------------------------

    def estimate_comparisons(
        self,
        num_records: int,
        blocking_strategy: BlockingStrategy = BlockingStrategy.SORTED_NEIGHBORHOOD,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """Estimate the number of pairwise comparisons.

        Args:
            num_records: Total number of records.
            blocking_strategy: Blocking strategy to estimate for.
            window_size: Window size for sorted neighborhood.

        Returns:
            Dictionary with estimation results.
        """
        all_pairs = num_records * (num_records - 1) // 2

        if blocking_strategy == BlockingStrategy.NONE:
            estimated = all_pairs
            reduction = 0.0
        elif blocking_strategy == BlockingStrategy.SORTED_NEIGHBORHOOD:
            w = min(window_size, num_records)
            estimated = num_records * w * (w - 1) // 2
            estimated = min(estimated, all_pairs)
            reduction = 1.0 - (estimated / all_pairs) if all_pairs > 0 else 0.0
        else:
            # Conservative estimate: assume ~10% of all pairs
            estimated = max(1, all_pairs // 10)
            reduction = 0.9

        return {
            "total_records": num_records,
            "all_pairs": all_pairs,
            "estimated_comparisons": estimated,
            "reduction_ratio": round(reduction, 4),
            "blocking_strategy": blocking_strategy.value,
        }

    # ------------------------------------------------------------------
    # Public API - Cross-dataset dedup
    # ------------------------------------------------------------------

    def cross_dataset_dedup(
        self,
        dataset_a: List[Dict[str, Any]],
        dataset_b: List[Dict[str, Any]],
        rule: DedupRule,
    ) -> DedupReport:
        """Deduplicate records across two datasets.

        Tags each record with a dataset source marker, then runs
        the standard pipeline on the combined set.

        Args:
            dataset_a: First dataset records.
            dataset_b: Second dataset records.
            rule: Dedup rule to apply.

        Returns:
            DedupReport with cross-dataset results.
        """
        # Tag records with dataset source
        for idx, rec in enumerate(dataset_a):
            rec.setdefault("id", f"a-{idx}")
            rec["_dataset"] = "dataset_a"

        for idx, rec in enumerate(dataset_b):
            rec.setdefault("id", f"b-{idx}")
            rec["_dataset"] = "dataset_b"

        combined = dataset_a + dataset_b

        return self.run_pipeline(
            records=combined,
            rule=rule,
            dataset_ids=["dataset_a", "dataset_b"],
            record_source=RecordSource.CROSS_DATASET,
        )

    # ------------------------------------------------------------------
    # Private - Stage implementations
    # ------------------------------------------------------------------

    def _run_fingerprint_stage(
        self,
        records: List[Dict[str, Any]],
        rule: DedupRule,
        job: DedupJob,
        issues: List[DedupIssueSummary],
    ) -> List[RecordFingerprint]:
        """Execute the fingerprinting stage.

        Args:
            records: Input records.
            rule: Dedup rule.
            job: Job tracker.
            issues: Issues list to append to.

        Returns:
            List of RecordFingerprint instances.
        """
        start = time.monotonic()
        logger.info("Stage FINGERPRINT: processing %d records", len(records))

        field_set = [fc.field_name for fc in rule.field_configs]

        fingerprints = self.fingerprinter.fingerprint_batch(
            records=records,
            field_set=field_set,
            algorithm=FingerprintAlgorithm.SHA256,
            id_field="id",
        )

        job.fingerprinted = len(fingerprints)
        elapsed = (time.monotonic() - start) * 1000.0

        # Check for duplicate fingerprints (exact duplicates)
        fp_groups: Dict[str, List[str]] = {}
        for fp in fingerprints:
            fp_groups.setdefault(fp.fingerprint_hash, []).append(fp.record_id)

        exact_dup_count = sum(
            len(ids) - 1 for ids in fp_groups.values() if len(ids) > 1
        )

        if exact_dup_count > 0:
            issues.append(DedupIssueSummary(
                severity=IssueSeverity.INFO,
                stage=PipelineStage.FINGERPRINT,
                description=(
                    f"Found {exact_dup_count} exact duplicate fingerprints "
                    f"across {len(records)} records"
                ),
                affected_records=exact_dup_count,
            ))

        logger.info(
            "Stage FINGERPRINT complete: %d fingerprints in %.1fms "
            "(%d exact duplicates)",
            len(fingerprints), elapsed, exact_dup_count,
        )
        return fingerprints

    def _run_block_stage(
        self,
        records: List[Dict[str, Any]],
        rule: DedupRule,
        job: DedupJob,
        issues: List[DedupIssueSummary],
    ) -> List[BlockResult]:
        """Execute the blocking stage.

        Args:
            records: Input records.
            rule: Dedup rule.
            job: Job tracker.
            issues: Issues list to append to.

        Returns:
            List of BlockResult instances.
        """
        start = time.monotonic()
        logger.info(
            "Stage BLOCK: partitioning %d records with %s",
            len(records), rule.blocking_strategy.value,
        )

        key_fields = rule.blocking_fields if rule.blocking_fields else [
            fc.field_name for fc in rule.field_configs
        ]

        blocks = self.blocker.create_blocks(
            records=records,
            strategy=rule.blocking_strategy,
            key_fields=key_fields,
            id_field="id",
        )

        elapsed = (time.monotonic() - start) * 1000.0

        # Check for oversized blocks
        oversized = [
            b for b in blocks if b.record_count > _MAX_COMPARISONS_PER_BLOCK
        ]
        if oversized:
            issues.append(DedupIssueSummary(
                severity=IssueSeverity.HIGH,
                stage=PipelineStage.BLOCK,
                description=(
                    f"{len(oversized)} blocks exceed max comparisons limit "
                    f"({_MAX_COMPARISONS_PER_BLOCK})"
                ),
                affected_records=sum(b.record_count for b in oversized),
                suggested_fix="Consider using a more selective blocking strategy",
            ))

        logger.info(
            "Stage BLOCK complete: %d blocks in %.1fms", len(blocks), elapsed,
        )
        return blocks

    def _run_compare_stage(
        self,
        blocks: List[BlockResult],
        record_map: Dict[str, Dict[str, Any]],
        rule: DedupRule,
        job: DedupJob,
        issues: List[DedupIssueSummary],
        max_comparisons_per_block: int = _MAX_COMPARISONS_PER_BLOCK,
    ) -> List[SimilarityResult]:
        """Execute the comparison stage.

        Args:
            blocks: Block results from blocking stage.
            record_map: Records keyed by ID.
            rule: Dedup rule with field configs.
            job: Job tracker.
            issues: Issues list to append to.
            max_comparisons_per_block: Max comparisons per block.

        Returns:
            List of SimilarityResult instances.
        """
        start = time.monotonic()
        total_pairs = sum(
            b.record_count * (b.record_count - 1) // 2 for b in blocks
        )
        logger.info(
            "Stage COMPARE: scoring %d candidate pairs across %d blocks",
            total_pairs, len(blocks),
        )

        all_results: List[SimilarityResult] = []
        compared = 0

        for block in blocks:
            if len(block.record_ids) < 2:
                continue

            # Limit comparisons per block
            block_pairs = len(block.record_ids) * (len(block.record_ids) - 1) // 2
            if block_pairs > max_comparisons_per_block:
                issues.append(DedupIssueSummary(
                    severity=IssueSeverity.MEDIUM,
                    stage=PipelineStage.COMPARE,
                    description=(
                        f"Block '{block.block_key}' has {block_pairs} pairs, "
                        f"truncating to {max_comparisons_per_block}"
                    ),
                    affected_records=block.record_count,
                ))

            pair_count = 0
            for id_a, id_b in combinations(block.record_ids, 2):
                if pair_count >= max_comparisons_per_block:
                    break

                rec_a = record_map.get(id_a)
                rec_b = record_map.get(id_b)
                if rec_a is None or rec_b is None:
                    continue

                result = self.scorer.score_pair(
                    record_a=rec_a,
                    record_b=rec_b,
                    field_configs=rule.field_configs,
                    record_a_id=id_a,
                    record_b_id=id_b,
                )
                all_results.append(result)
                pair_count += 1
                compared += 1

        job.compared = compared
        elapsed = (time.monotonic() - start) * 1000.0

        logger.info(
            "Stage COMPARE complete: %d comparisons in %.1fms",
            compared, elapsed,
        )
        return all_results

    def _run_classify_stage(
        self,
        comparisons: List[SimilarityResult],
        rule: DedupRule,
        job: DedupJob,
        issues: List[DedupIssueSummary],
    ) -> List[MatchResult]:
        """Execute the classification stage.

        Args:
            comparisons: Similarity results from comparison stage.
            rule: Dedup rule with thresholds.
            job: Job tracker.
            issues: Issues list to append to.

        Returns:
            List of MatchResult instances.
        """
        start = time.monotonic()
        logger.info(
            "Stage CLASSIFY: classifying %d comparisons (thresholds: "
            "match=%.2f, possible=%.2f)",
            len(comparisons), rule.match_threshold, rule.possible_threshold,
        )

        if not comparisons:
            return []

        match_results = self.classifier.classify_batch(
            similarity_results=comparisons,
            match_threshold=rule.match_threshold,
            possible_threshold=rule.possible_threshold,
        )

        match_count = sum(
            1 for r in match_results
            if r.classification == MatchClassification.MATCH
        )
        possible_count = sum(
            1 for r in match_results
            if r.classification == MatchClassification.POSSIBLE
        )

        job.matched = match_count
        elapsed = (time.monotonic() - start) * 1000.0

        if possible_count > match_count * 2:
            issues.append(DedupIssueSummary(
                severity=IssueSeverity.MEDIUM,
                stage=PipelineStage.CLASSIFY,
                description=(
                    f"High number of POSSIBLE matches ({possible_count}) "
                    f"relative to MATCH ({match_count}). Consider adjusting "
                    f"thresholds."
                ),
                affected_records=possible_count,
                suggested_fix="Lower match_threshold or raise possible_threshold",
            ))

        logger.info(
            "Stage CLASSIFY complete: %d matches, %d possible, "
            "%d non-matches in %.1fms",
            match_count, possible_count,
            len(match_results) - match_count - possible_count, elapsed,
        )
        return match_results

    def _run_cluster_stage(
        self,
        match_results: List[MatchResult],
        record_map: Dict[str, Dict[str, Any]],
        rule: DedupRule,
        job: DedupJob,
        issues: List[DedupIssueSummary],
    ) -> List[DuplicateCluster]:
        """Execute the clustering stage.

        Args:
            match_results: Match results from classification stage.
            record_map: Records keyed by ID.
            rule: Dedup rule.
            job: Job tracker.
            issues: Issues list to append to.

        Returns:
            List of DuplicateCluster instances.
        """
        start = time.monotonic()

        # Filter to MATCH only
        matches_only = [
            r for r in match_results
            if r.classification == MatchClassification.MATCH
        ]

        if not matches_only:
            logger.info("Stage CLUSTER: no matches to cluster")
            return []

        logger.info(
            "Stage CLUSTER: clustering %d matches", len(matches_only),
        )

        clusters = self.cluster_resolver.form_clusters(
            match_results=matches_only,
            algorithm=ClusterAlgorithm.UNION_FIND,
            records=record_map,
        )

        job.clustered = len(clusters)
        elapsed = (time.monotonic() - start) * 1000.0

        # Check for large clusters
        large_clusters = [c for c in clusters if c.member_count > 10]
        if large_clusters:
            issues.append(DedupIssueSummary(
                severity=IssueSeverity.LOW,
                stage=PipelineStage.CLUSTER,
                description=(
                    f"{len(large_clusters)} clusters have more than 10 members. "
                    f"Largest cluster has {max(c.member_count for c in large_clusters)} "
                    f"members."
                ),
                affected_records=sum(c.member_count for c in large_clusters),
                suggested_fix="Review large clusters for false positives",
            ))

        logger.info(
            "Stage CLUSTER complete: %d clusters in %.1fms",
            len(clusters), elapsed,
        )
        return clusters

    def _run_merge_stage(
        self,
        clusters: List[DuplicateCluster],
        record_map: Dict[str, Dict[str, Any]],
        rule: DedupRule,
        job: DedupJob,
        issues: List[DedupIssueSummary],
    ) -> List[MergeDecision]:
        """Execute the merge stage.

        Args:
            clusters: Duplicate clusters from clustering stage.
            record_map: Records keyed by ID.
            rule: Dedup rule.
            job: Job tracker.
            issues: Issues list to append to.

        Returns:
            List of MergeDecision instances.
        """
        start = time.monotonic()

        if not clusters:
            logger.info("Stage MERGE: no clusters to merge")
            return []

        logger.info(
            "Stage MERGE: merging %d clusters using %s",
            len(clusters), rule.merge_strategy.value,
        )

        decisions = self.merge_engine.merge_batch(
            clusters=clusters,
            records=record_map,
            strategy=rule.merge_strategy,
        )

        job.merged = len(decisions)
        elapsed = (time.monotonic() - start) * 1000.0

        # Check for high conflict rates
        total_conflicts = sum(len(d.conflicts) for d in decisions)
        if total_conflicts > len(decisions) * 3:
            issues.append(DedupIssueSummary(
                severity=IssueSeverity.MEDIUM,
                stage=PipelineStage.MERGE,
                description=(
                    f"High conflict rate during merge: {total_conflicts} "
                    f"conflicts across {len(decisions)} merges "
                    f"(avg {total_conflicts / len(decisions):.1f} per merge)"
                ),
                affected_records=total_conflicts,
                suggested_fix=(
                    "Review data quality or adjust merge strategy"
                ),
            ))

        logger.info(
            "Stage MERGE complete: %d merges (%d conflicts) in %.1fms",
            len(decisions), total_conflicts, elapsed,
        )
        return decisions

    # ------------------------------------------------------------------
    # Public API - Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return current pipeline operational statistics."""
        return self.get_pipeline_stats()

    def reset_statistics(self) -> None:
        """Reset all pipeline and engine statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

        self.fingerprinter.reset_statistics()
        self.blocker.reset_statistics()
        self.scorer.reset_statistics()
        self.classifier.reset_statistics()
        self.cluster_resolver.reset_statistics()
        self.merge_engine.reset_statistics()

    # ------------------------------------------------------------------
    # Private methods - Stats tracking
    # ------------------------------------------------------------------

    def _record_success(self, elapsed_seconds: float) -> None:
        """Record a successful invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._successes += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()

    def _record_failure(self, elapsed_seconds: float) -> None:
        """Record a failed invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._failures += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()
