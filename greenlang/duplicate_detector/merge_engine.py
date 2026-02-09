# -*- coding: utf-8 -*-
"""
Merge Engine - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

Merges duplicate clusters into single canonical records using 6 merge
strategies: keep_first, keep_latest, keep_most_complete, merge_fields,
golden_record, and custom. Handles field-level conflict resolution
with full provenance and undo capability.

Zero-Hallucination Guarantees:
    - All merge strategies use deterministic selection logic
    - Conflict resolution uses rule-based field picking
    - Golden record uses field-level completeness/length scoring
    - No ML/LLM calls in merge path
    - Provenance recorded for every merge operation

Supported Merge Strategies:
    KEEP_FIRST:         Keep the first record (alphabetical ID order)
    KEEP_LATEST:        Keep the most recently updated record
    KEEP_MOST_COMPLETE: Keep the record with fewest null/empty fields
    MERGE_FIELDS:       Combine non-null fields from all records
    GOLDEN_RECORD:      Best-of-breed field selection with priority
    CUSTOM:             User-defined merge function (callback)

Conflict Resolution Methods:
    FIRST:          Use value from the first record (alphabetical)
    LATEST:         Use value from the most recently updated record
    MOST_COMPLETE:  Use value from the record with fewest nulls
    LONGEST:        Use the longest non-null string value
    SHORTEST:       Use the shortest non-null string value

Example:
    >>> from greenlang.duplicate_detector.merge_engine import MergeEngine
    >>> engine = MergeEngine()
    >>> decision = engine.merge_cluster(
    ...     cluster=cluster,
    ...     records={"r1": {...}, "r2": {...}},
    ...     strategy=MergeStrategy.KEEP_MOST_COMPLETE,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from greenlang.duplicate_detector.models import (
    ConflictResolution,
    DuplicateCluster,
    MergeConflict,
    MergeDecision,
    MergeStrategy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MergeEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a merge operation."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_empty(value: Any) -> bool:
    """Check if a value is empty (None or empty string after strip)."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _record_completeness(record: Dict[str, Any]) -> int:
    """Count non-empty fields in a record."""
    return sum(1 for v in record.values() if not _is_empty(v))


# =============================================================================
# MergeEngine
# =============================================================================


class MergeEngine:
    """Merge engine for combining duplicate records.

    Merges duplicate clusters into single canonical records using
    configurable strategies and conflict resolution. Maintains full
    provenance and supports undo via merge decision history.

    This engine follows GreenLang's zero-hallucination principle:
    all merge decisions use deterministic rule-based logic with no ML.

    Attributes:
        _stats_lock: Threading lock for stats updates.
        _invocations: Total invocation count.
        _successes: Total successful invocations.
        _failures: Total failed invocations.
        _total_duration_ms: Cumulative processing time.
        _merge_history: List of past merge decisions for undo.

    Example:
        >>> engine = MergeEngine()
        >>> decision = engine.merge_cluster(cluster, records)
        >>> print(decision.merged_record)
    """

    def __init__(self) -> None:
        """Initialize MergeEngine with empty statistics and history."""
        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None
        self._merge_history: List[MergeDecision] = []
        self._custom_merge_fn: Optional[
            Callable[[List[Dict[str, Any]]], Dict[str, Any]]
        ] = None
        logger.info("MergeEngine initialized")

    # ------------------------------------------------------------------
    # Public API - Custom merge registration
    # ------------------------------------------------------------------

    def register_custom_merge(
        self,
        merge_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    ) -> None:
        """Register a custom merge function for CUSTOM strategy.

        Args:
            merge_fn: Function taking list of records and returning merged record.
        """
        self._custom_merge_fn = merge_fn
        logger.info("Custom merge function registered")

    # ------------------------------------------------------------------
    # Public API - Cluster merge
    # ------------------------------------------------------------------

    def merge_cluster(
        self,
        cluster: DuplicateCluster,
        records: Dict[str, Dict[str, Any]],
        strategy: MergeStrategy = MergeStrategy.KEEP_MOST_COMPLETE,
        conflict_resolution: ConflictResolution = ConflictResolution.MOST_COMPLETE,
        timestamp_field: Optional[str] = None,
    ) -> MergeDecision:
        """Merge a duplicate cluster into a single canonical record.

        Args:
            cluster: The duplicate cluster to merge.
            records: Full records keyed by record_id.
            strategy: Merge strategy to apply.
            conflict_resolution: Field-level conflict resolution method.
            timestamp_field: Field name for record timestamps (for LATEST).

        Returns:
            MergeDecision with merged record and conflict details.

        Raises:
            ValueError: If cluster has fewer than 2 members or records missing.
        """
        start_time = time.monotonic()
        try:
            if len(cluster.member_record_ids) < 2:
                raise ValueError("Cluster must have at least 2 members to merge")

            # Gather source records in order
            source_records = self._gather_source_records(
                cluster.member_record_ids, records,
            )

            # Execute merge strategy
            if strategy == MergeStrategy.KEEP_FIRST:
                merged, conflicts = self.keep_first(source_records)
            elif strategy == MergeStrategy.KEEP_LATEST:
                merged, conflicts = self.keep_latest(
                    source_records, timestamp_field,
                )
            elif strategy == MergeStrategy.KEEP_MOST_COMPLETE:
                merged, conflicts = self.keep_most_complete(source_records)
            elif strategy == MergeStrategy.MERGE_FIELDS:
                merged, conflicts = self.merge_fields(
                    source_records, conflict_resolution,
                )
            elif strategy == MergeStrategy.GOLDEN_RECORD:
                merged, conflicts = self.golden_record(
                    source_records, conflict_resolution,
                )
            elif strategy == MergeStrategy.CUSTOM:
                merged, conflicts = self._custom_merge(source_records)
            else:
                raise ValueError(f"Unknown merge strategy: {strategy}")

            provenance = _compute_provenance(
                "merge_cluster",
                f"{cluster.cluster_id}:{strategy.value}:{len(source_records)}",
            )

            decision = MergeDecision(
                cluster_id=cluster.cluster_id,
                strategy=strategy,
                merged_record=merged,
                conflicts=conflicts,
                source_records=source_records,
                provenance_hash=provenance,
            )

            self._merge_history.append(decision)
            self._record_success(time.monotonic() - start_time)

            logger.info(
                "Merged cluster %s using %s: %d records -> 1, %d conflicts",
                cluster.cluster_id, strategy.value,
                len(source_records), len(conflicts),
            )
            return decision

        except Exception as e:
            self._record_failure(time.monotonic() - start_time)
            logger.error(
                "Merge failed for cluster %s: %s", cluster.cluster_id, e,
            )
            raise

    def merge_batch(
        self,
        clusters: List[DuplicateCluster],
        records: Dict[str, Dict[str, Any]],
        strategy: MergeStrategy = MergeStrategy.KEEP_MOST_COMPLETE,
        conflict_resolution: ConflictResolution = ConflictResolution.MOST_COMPLETE,
        timestamp_field: Optional[str] = None,
    ) -> List[MergeDecision]:
        """Merge a batch of clusters.

        Args:
            clusters: List of duplicate clusters to merge.
            records: Full records keyed by record_id.
            strategy: Merge strategy to apply.
            conflict_resolution: Field-level conflict resolution method.
            timestamp_field: Field name for record timestamps.

        Returns:
            List of MergeDecision instances.
        """
        if not clusters:
            return []

        logger.info("Merging batch of %d clusters using %s", len(clusters), strategy.value)

        decisions: List[MergeDecision] = []
        for cluster in clusters:
            decision = self.merge_cluster(
                cluster, records, strategy,
                conflict_resolution, timestamp_field,
            )
            decisions.append(decision)

        return decisions

    # ------------------------------------------------------------------
    # Public API - Merge strategies
    # ------------------------------------------------------------------

    def keep_first(
        self,
        source_records: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Keep the first record (by alphabetical ID order).

        Args:
            source_records: List of source records (already sorted by ID).

        Returns:
            Tuple of (merged_record, conflicts).
        """
        if not source_records:
            return ({}, [])

        merged = copy.deepcopy(source_records[0])
        conflicts = self._detect_conflicts(source_records, ConflictResolution.FIRST)
        return (merged, conflicts)

    def keep_latest(
        self,
        source_records: List[Dict[str, Any]],
        timestamp_field: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Keep the most recently updated record.

        Uses timestamp_field to determine recency, or falls back to
        the last record in the list.

        Args:
            source_records: List of source records.
            timestamp_field: Field name for record timestamps.

        Returns:
            Tuple of (merged_record, conflicts).
        """
        if not source_records:
            return ({}, [])

        if timestamp_field:
            latest = self._find_latest_record(source_records, timestamp_field)
        else:
            latest = source_records[-1]

        merged = copy.deepcopy(latest)
        conflicts = self._detect_conflicts(
            source_records, ConflictResolution.LATEST,
        )
        return (merged, conflicts)

    def keep_most_complete(
        self,
        source_records: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Keep the record with the fewest null/empty fields.

        Args:
            source_records: List of source records.

        Returns:
            Tuple of (merged_record, conflicts).
        """
        if not source_records:
            return ({}, [])

        best_record = source_records[0]
        best_completeness = _record_completeness(best_record)

        for rec in source_records[1:]:
            completeness = _record_completeness(rec)
            if completeness > best_completeness:
                best_completeness = completeness
                best_record = rec

        merged = copy.deepcopy(best_record)
        conflicts = self._detect_conflicts(
            source_records, ConflictResolution.MOST_COMPLETE,
        )
        return (merged, conflicts)

    def merge_fields(
        self,
        source_records: List[Dict[str, Any]],
        conflict_resolution: ConflictResolution = ConflictResolution.MOST_COMPLETE,
    ) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Combine non-null fields from all records.

        For each field, uses the first non-null value found across
        all records. When multiple records have non-null values,
        applies the specified conflict resolution.

        Args:
            source_records: List of source records.
            conflict_resolution: Method for resolving field conflicts.

        Returns:
            Tuple of (merged_record, conflicts).
        """
        if not source_records:
            return ({}, [])

        # Collect all field names
        all_fields: Set[str] = set()
        for rec in source_records:
            all_fields.update(rec.keys())

        merged: Dict[str, Any] = {}
        conflicts: List[MergeConflict] = []

        for field in sorted(all_fields):
            non_empty_values: Dict[str, Any] = {}
            for rec in source_records:
                rid = str(rec.get("id", ""))
                val = rec.get(field)
                if not _is_empty(val):
                    non_empty_values[rid] = val

            if not non_empty_values:
                merged[field] = None
                continue

            if len(non_empty_values) == 1:
                # No conflict: use the single non-null value
                merged[field] = next(iter(non_empty_values.values()))
                continue

            # Multiple non-null values: resolve conflict
            chosen_value, chosen_rid = self.resolve_conflict(
                field, non_empty_values, source_records, conflict_resolution,
            )
            merged[field] = chosen_value
            conflicts.append(MergeConflict(
                field_name=field,
                values=non_empty_values,
                chosen_value=chosen_value,
                resolution_method=conflict_resolution,
                source_record_id=chosen_rid,
            ))

        return (merged, conflicts)

    def golden_record(
        self,
        source_records: List[Dict[str, Any]],
        conflict_resolution: ConflictResolution = ConflictResolution.MOST_COMPLETE,
    ) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Create a best-of-breed golden record from all sources.

        For each field, selects the best value using field-level
        quality scoring: non-null > null, longer strings > shorter,
        more complete records get priority.

        Args:
            source_records: List of source records.
            conflict_resolution: Fallback conflict resolution method.

        Returns:
            Tuple of (golden_record, conflicts).
        """
        if not source_records:
            return ({}, [])

        # Rank records by completeness
        ranked_records = sorted(
            source_records,
            key=lambda r: _record_completeness(r),
            reverse=True,
        )

        # Collect all field names
        all_fields: Set[str] = set()
        for rec in source_records:
            all_fields.update(rec.keys())

        golden: Dict[str, Any] = {}
        conflicts: List[MergeConflict] = []

        for field in sorted(all_fields):
            candidates: Dict[str, Any] = {}
            for rec in source_records:
                rid = str(rec.get("id", ""))
                val = rec.get(field)
                if not _is_empty(val):
                    candidates[rid] = val

            if not candidates:
                golden[field] = None
                continue

            if len(candidates) == 1:
                golden[field] = next(iter(candidates.values()))
                continue

            # Score each candidate value for quality
            best_value = None
            best_rid = ""
            best_score = -1.0

            for rid, val in candidates.items():
                score = self._score_field_value(val, field, rid, ranked_records)
                if score > best_score:
                    best_score = score
                    best_value = val
                    best_rid = rid

            golden[field] = best_value
            conflicts.append(MergeConflict(
                field_name=field,
                values=candidates,
                chosen_value=best_value,
                resolution_method=conflict_resolution,
                source_record_id=best_rid,
            ))

        return (golden, conflicts)

    # ------------------------------------------------------------------
    # Public API - Conflict resolution
    # ------------------------------------------------------------------

    def resolve_conflict(
        self,
        field_name: str,
        values: Dict[str, Any],
        source_records: List[Dict[str, Any]],
        method: ConflictResolution = ConflictResolution.MOST_COMPLETE,
        timestamp_field: Optional[str] = None,
    ) -> Tuple[Any, Optional[str]]:
        """Resolve a field-level conflict among multiple values.

        Args:
            field_name: Name of the conflicting field.
            values: Mapping of record_id to field value.
            source_records: Full source records for context.
            method: Conflict resolution method.
            timestamp_field: Field for timestamp-based resolution.

        Returns:
            Tuple of (chosen_value, source_record_id).
        """
        if not values:
            return (None, None)

        if len(values) == 1:
            rid = next(iter(values))
            return (values[rid], rid)

        if method == ConflictResolution.FIRST:
            rid = sorted(values.keys())[0]
            return (values[rid], rid)

        elif method == ConflictResolution.LATEST:
            if timestamp_field:
                latest_rec = self._find_latest_record(
                    source_records, timestamp_field,
                )
                rid = str(latest_rec.get("id", ""))
                if rid in values:
                    return (values[rid], rid)
            # Fallback: last key
            rid = list(values.keys())[-1]
            return (values[rid], rid)

        elif method == ConflictResolution.MOST_COMPLETE:
            best_rid = ""
            best_completeness = -1
            for rid in values:
                for rec in source_records:
                    if str(rec.get("id", "")) == rid:
                        c = _record_completeness(rec)
                        if c > best_completeness:
                            best_completeness = c
                            best_rid = rid
                        break
            if best_rid and best_rid in values:
                return (values[best_rid], best_rid)
            rid = sorted(values.keys())[0]
            return (values[rid], rid)

        elif method == ConflictResolution.LONGEST:
            best_rid = ""
            best_length = -1
            for rid, val in values.items():
                length = len(str(val)) if val is not None else 0
                if length > best_length:
                    best_length = length
                    best_rid = rid
            return (values[best_rid], best_rid)

        elif method == ConflictResolution.SHORTEST:
            best_rid = ""
            best_length = float("inf")
            for rid, val in values.items():
                length = len(str(val)) if val is not None else 0
                if length < best_length:
                    best_length = length
                    best_rid = rid
            return (values[best_rid], best_rid)

        # Fallback
        rid = sorted(values.keys())[0]
        return (values[rid], rid)

    # ------------------------------------------------------------------
    # Public API - Validation
    # ------------------------------------------------------------------

    def validate_merge(
        self,
        decision: MergeDecision,
    ) -> Dict[str, Any]:
        """Validate a merge decision for completeness and correctness.

        Checks that:
        - Merged record is not empty
        - All source record fields are accounted for
        - Conflicts are properly resolved

        Args:
            decision: The merge decision to validate.

        Returns:
            Dictionary with validation results.
        """
        issues: List[str] = []

        if not decision.merged_record:
            issues.append("Merged record is empty")

        if not decision.source_records:
            issues.append("No source records")

        # Check all fields from source records appear in merged
        all_source_fields: Set[str] = set()
        for rec in decision.source_records:
            all_source_fields.update(rec.keys())

        merged_fields = set(decision.merged_record.keys())
        missing_fields = all_source_fields - merged_fields
        if missing_fields:
            issues.append(
                f"Fields missing from merged record: {sorted(missing_fields)}"
            )

        # Check all conflicts have chosen values
        unresolved = [
            c.field_name for c in decision.conflicts if c.chosen_value is None
        ]
        if unresolved:
            issues.append(
                f"Unresolved conflicts for fields: {unresolved}"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "source_record_count": len(decision.source_records),
            "merged_field_count": len(decision.merged_record),
            "conflict_count": len(decision.conflicts),
            "strategy": decision.strategy.value,
        }

    # ------------------------------------------------------------------
    # Public API - Undo
    # ------------------------------------------------------------------

    def undo_merge(
        self,
        cluster_id: str,
    ) -> Optional[MergeDecision]:
        """Undo the most recent merge for a given cluster.

        Removes the merge decision from history and returns it.

        Args:
            cluster_id: Identifier of the cluster to undo merge for.

        Returns:
            The removed MergeDecision, or None if not found.
        """
        for i in range(len(self._merge_history) - 1, -1, -1):
            if self._merge_history[i].cluster_id == cluster_id:
                decision = self._merge_history.pop(i)
                logger.info(
                    "Undid merge for cluster %s (strategy: %s)",
                    cluster_id, decision.strategy.value,
                )
                return decision

        logger.warning("No merge found for cluster %s to undo", cluster_id)
        return None

    def get_merge_history(self) -> List[MergeDecision]:
        """Return the full merge decision history.

        Returns:
            List of MergeDecision instances.
        """
        return list(self._merge_history)

    # ------------------------------------------------------------------
    # Public API - Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return current engine operational statistics."""
        with self._stats_lock:
            avg_ms = 0.0
            if self._invocations > 0:
                avg_ms = self._total_duration_ms / self._invocations
            return {
                "engine_name": "MergeEngine",
                "invocations": self._invocations,
                "successes": self._successes,
                "failures": self._failures,
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_ms, 3),
                "last_invoked_at": (
                    self._last_invoked_at.isoformat()
                    if self._last_invoked_at else None
                ),
                "merge_history_size": len(self._merge_history),
            }

    def reset_statistics(self) -> None:
        """Reset all operational statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _gather_source_records(
        self,
        member_ids: List[str],
        records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Gather and validate source records for a cluster.

        Args:
            member_ids: List of record IDs in the cluster.
            records: Full records keyed by record_id.

        Returns:
            List of source record dicts sorted by ID.

        Raises:
            ValueError: If any member record is missing.
        """
        source: List[Dict[str, Any]] = []
        missing: List[str] = []

        for rid in sorted(member_ids):
            if rid in records:
                rec = copy.deepcopy(records[rid])
                # Ensure ID is set
                if "id" not in rec:
                    rec["id"] = rid
                source.append(rec)
            else:
                missing.append(rid)

        if missing:
            raise ValueError(
                f"Missing records for merge: {missing}"
            )

        return source

    def _find_latest_record(
        self,
        source_records: List[Dict[str, Any]],
        timestamp_field: str,
    ) -> Dict[str, Any]:
        """Find the most recently updated record by timestamp field.

        Args:
            source_records: List of source records.
            timestamp_field: Field name containing the timestamp.

        Returns:
            The most recent record.
        """
        latest = source_records[0]
        latest_ts = str(latest.get(timestamp_field, ""))

        for rec in source_records[1:]:
            ts = str(rec.get(timestamp_field, ""))
            if ts > latest_ts:
                latest_ts = ts
                latest = rec

        return latest

    def _detect_conflicts(
        self,
        source_records: List[Dict[str, Any]],
        resolution_method: ConflictResolution,
    ) -> List[MergeConflict]:
        """Detect field-level conflicts across source records.

        A conflict exists when two or more records have different
        non-empty values for the same field.

        Args:
            source_records: List of source records.
            resolution_method: Resolution method applied.

        Returns:
            List of MergeConflict instances.
        """
        all_fields: Set[str] = set()
        for rec in source_records:
            all_fields.update(rec.keys())

        conflicts: List[MergeConflict] = []

        for field in sorted(all_fields):
            non_empty: Dict[str, Any] = {}
            for rec in source_records:
                rid = str(rec.get("id", ""))
                val = rec.get(field)
                if not _is_empty(val):
                    non_empty[rid] = val

            if len(non_empty) <= 1:
                continue

            # Check if values actually differ
            unique_values = set(str(v) for v in non_empty.values())
            if len(unique_values) <= 1:
                continue

            # Resolve
            chosen_value, chosen_rid = self.resolve_conflict(
                field, non_empty, source_records, resolution_method,
            )
            conflicts.append(MergeConflict(
                field_name=field,
                values=non_empty,
                chosen_value=chosen_value,
                resolution_method=resolution_method,
                source_record_id=chosen_rid,
            ))

        return conflicts

    def _custom_merge(
        self,
        source_records: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Execute custom merge function.

        Args:
            source_records: List of source records.

        Returns:
            Tuple of (merged_record, empty_conflicts).

        Raises:
            ValueError: If no custom merge function is registered.
        """
        if self._custom_merge_fn is None:
            raise ValueError(
                "No custom merge function registered. "
                "Call register_custom_merge() first."
            )

        merged = self._custom_merge_fn(source_records)
        return (merged, [])

    def _score_field_value(
        self,
        value: Any,
        field_name: str,
        record_id: str,
        ranked_records: List[Dict[str, Any]],
    ) -> float:
        """Score a field value for golden record selection.

        Scoring criteria:
        - Non-null: +1.0
        - String length (normalized): +0.0 to +0.5
        - Record rank (by completeness): +0.0 to +0.5

        Args:
            value: The field value to score.
            field_name: Name of the field.
            record_id: ID of the record providing this value.
            ranked_records: Records sorted by completeness (descending).

        Returns:
            Quality score for this field value.
        """
        score = 0.0

        # Non-null bonus
        if not _is_empty(value):
            score += 1.0

        # String length bonus
        if isinstance(value, str):
            length = len(value.strip())
            score += min(0.5, length / 200.0)

        # Record rank bonus (more complete records score higher)
        for rank, rec in enumerate(ranked_records):
            if str(rec.get("id", "")) == record_id:
                rank_score = (len(ranked_records) - rank) / len(ranked_records)
                score += rank_score * 0.5
                break

        return score

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
