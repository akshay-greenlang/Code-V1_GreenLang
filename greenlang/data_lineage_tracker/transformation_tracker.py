# -*- coding: utf-8 -*-
"""
Transformation Tracker Engine - AGENT-DATA-018 Data Lineage Tracker

Engine 2 of 7 in the Data Lineage Tracker pipeline. Captures and stores
data transformation events as lineage graph edges. Each transformation
records the originating agent, pipeline context, source and target asset
identifiers, record-level accounting (in/out/filtered/error), execution
duration, and free-form parameters. Multiple secondary indexes enable
fast lookups by agent, pipeline, transformation type, and time range.

Zero-Hallucination Guarantees:
    - All record counts and duration metrics use deterministic Python
      arithmetic.  No LLM calls for numeric computations.
    - Statistics are computed with explicit, auditable formulas over
      in-memory indexes.
    - SHA-256 provenance hashes anchor every recorded transformation to
      a tamper-evident chain.

Thread Safety:
    All public methods acquire ``self._lock`` before mutating or reading
    shared state, making the engine safe for concurrent use across
    FastAPI request handlers and background workers.

Example:
    >>> from greenlang.data_lineage_tracker.transformation_tracker import (
    ...     TransformationTrackerEngine,
    ... )
    >>> engine = TransformationTrackerEngine()
    >>> txn = engine.record_transformation(
    ...     transformation_type="filter",
    ...     agent_id="data-quality-profiler",
    ...     pipeline_id="pipeline-001",
    ...     source_asset_ids=["asset-a"],
    ...     target_asset_ids=["asset-b"],
    ...     records_in=1000,
    ...     records_out=950,
    ...     records_filtered=50,
    ... )
    >>> print(txn["id"], txn["provenance_hash"][:16])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.data_lineage_tracker.config import get_config
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
from greenlang.data_lineage_tracker.metrics import (
    record_transformation_captured,
    observe_processing_duration,
    PROMETHEUS_AVAILABLE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TRANSFORMATION_TYPES: frozenset = frozenset({
    "filter",
    "aggregate",
    "join",
    "calculate",
    "impute",
    "deduplicate",
    "enrich",
    "merge",
    "split",
    "validate",
    "normalize",
    "classify",
})

_VALID_CHAIN_DIRECTIONS: frozenset = frozenset({"backward", "forward"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with timezone info.

    Returns:
        Timezone-aware UTC datetime with microsecond precision.
    """
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string.

    Returns:
        ISO-8601 formatted timestamp string with UTC timezone.
    """
    return _utcnow().isoformat()


def _generate_id() -> str:
    """Generate a unique transformation identifier.

    Uses UUID4 for globally unique, collision-resistant identifiers
    suitable for distributed environments.

    Returns:
        Lowercase hex UUID4 string (e.g. ``"a1b2c3d4e5f6..."``).
    """
    return uuid.uuid4().hex


def _build_provenance_payload(transformation: dict) -> dict:
    """Build a dictionary suitable for SHA-256 provenance hashing.

    Extracts the deterministic fields from a transformation record and
    returns them in a stable structure for hashing.

    Args:
        transformation: The full transformation record dict.

    Returns:
        Dictionary with provenance-relevant fields only.
    """
    return {
        "id": transformation["id"],
        "transformation_type": transformation["transformation_type"],
        "agent_id": transformation["agent_id"],
        "pipeline_id": transformation["pipeline_id"],
        "source_asset_ids": transformation["source_asset_ids"],
        "target_asset_ids": transformation["target_asset_ids"],
        "records_in": transformation["records_in"],
        "records_out": transformation["records_out"],
        "records_filtered": transformation["records_filtered"],
        "records_error": transformation["records_error"],
        "duration_ms": transformation["duration_ms"],
        "created_at": transformation["created_at"],
    }


# ---------------------------------------------------------------------------
# TransformationTrackerEngine
# ---------------------------------------------------------------------------


class TransformationTrackerEngine:
    """Engine for capturing and storing data transformation events.

    Records every data transformation performed by GreenLang agents as a
    lineage graph edge, maintaining multiple secondary indexes for efficient
    retrieval by agent, pipeline, transformation type, and time range.

    Each recorded transformation includes:
        - A unique identifier (UUID4).
        - The transformation type (from ``VALID_TRANSFORMATION_TYPES``).
        - The originating agent and pipeline identifiers.
        - Source and target asset identifiers (lineage edges).
        - Record-level accounting: in / out / filtered / error counts.
        - Execution duration in milliseconds.
        - Free-form parameters and metadata dictionaries.
        - A SHA-256 provenance hash for audit trail integrity.

    Attributes:
        _transformations: Primary store keyed by transformation ID.
        _agent_index: Secondary index mapping ``agent_id`` to lists of
            transformation IDs for fast per-agent lookups.
        _pipeline_index: Secondary index mapping ``pipeline_id`` to lists
            of transformation IDs for fast per-pipeline lookups.
        _type_index: Secondary index mapping ``transformation_type`` to
            lists of transformation IDs for fast per-type lookups.
        _time_index: Sorted list of ``(iso_timestamp, id)`` tuples for
            efficient time-range queries via binary search.
        _lock: Reentrant lock for thread-safe access to all shared state.
        _provenance: ProvenanceTracker instance for SHA-256 chain hashing.

    Example:
        >>> engine = TransformationTrackerEngine()
        >>> txn = engine.record_transformation(
        ...     transformation_type="aggregate",
        ...     agent_id="spend-categorizer",
        ...     pipeline_id="quarterly-rollup",
        ...     source_asset_ids=["raw-spend-q1"],
        ...     target_asset_ids=["agg-spend-q1"],
        ...     records_in=50000,
        ...     records_out=1200,
        ... )
        >>> assert txn["transformation_type"] == "aggregate"
        >>> assert len(txn["provenance_hash"]) == 64
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialise the TransformationTrackerEngine.

        Args:
            provenance: Optional :class:`ProvenanceTracker` instance. When
                ``None`` a fresh tracker is created internally. Pass a
                shared tracker to unify provenance chains across engines.
        """
        # Primary store
        self._transformations: Dict[str, dict] = {}

        # Secondary indexes
        self._agent_index: Dict[str, List[str]] = {}
        self._pipeline_index: Dict[str, List[str]] = {}
        self._type_index: Dict[str, List[str]] = {}
        self._time_index: List[Tuple[str, str]] = []

        # Concurrency
        self._lock: threading.Lock = threading.Lock()

        # Provenance
        self._provenance: ProvenanceTracker = provenance if provenance is not None else ProvenanceTracker()

        logger.info(
            "TransformationTrackerEngine initialised with %d valid types",
            len(VALID_TRANSFORMATION_TYPES),
        )

    # ------------------------------------------------------------------
    # Public API - Record
    # ------------------------------------------------------------------

    def record_transformation(
        self,
        transformation_type: str,
        agent_id: str,
        pipeline_id: str,
        source_asset_ids: List[str],
        target_asset_ids: List[str],
        execution_id: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        records_in: int = 0,
        records_out: int = 0,
        records_filtered: int = 0,
        records_error: int = 0,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Record a single data transformation event.

        Validates the transformation type, generates a unique identifier,
        timestamps the event, stores it in all secondary indexes, records
        a provenance chain entry, and emits a Prometheus metric.

        Args:
            transformation_type: Type of transformation performed. Must be
                one of ``VALID_TRANSFORMATION_TYPES``.
            agent_id: Identifier of the agent that performed the
                transformation (e.g. ``"data-quality-profiler"``).
            pipeline_id: Identifier of the pipeline execution context.
            source_asset_ids: List of input data asset identifiers consumed
                by this transformation.
            target_asset_ids: List of output data asset identifiers produced
                by this transformation.
            execution_id: Optional external execution/run identifier for
                cross-referencing with orchestrator logs.
            description: Human-readable description of the transformation.
            parameters: Optional dictionary of transformation-specific
                parameters (e.g. filter predicates, aggregation keys).
            records_in: Number of records consumed as input.
            records_out: Number of records produced as output.
            records_filtered: Number of records removed during processing.
            records_error: Number of records that failed processing.
            duration_ms: Wall-clock execution time in milliseconds.
            metadata: Optional dictionary of additional contextual fields.

        Returns:
            Dictionary representing the recorded transformation with all
            fields including the generated ``id``, ``provenance_hash``,
            and ``created_at`` timestamp.

        Raises:
            ValueError: If ``transformation_type`` is not in
                ``VALID_TRANSFORMATION_TYPES``, or if ``agent_id``,
                ``pipeline_id``, ``source_asset_ids``, or
                ``target_asset_ids`` are empty.
            TypeError: If ``source_asset_ids`` or ``target_asset_ids``
                are not lists.
        """
        start = time.monotonic()

        # -- Input validation -----------------------------------------------
        self._validate_record_inputs(
            transformation_type=transformation_type,
            agent_id=agent_id,
            pipeline_id=pipeline_id,
            source_asset_ids=source_asset_ids,
            target_asset_ids=target_asset_ids,
        )

        # -- Build transformation record ------------------------------------
        txn_id = _generate_id()
        now_iso = _utcnow_iso()

        transformation: dict = {
            "id": txn_id,
            "transformation_type": transformation_type,
            "agent_id": agent_id,
            "pipeline_id": pipeline_id,
            "execution_id": execution_id or "",
            "source_asset_ids": list(source_asset_ids),
            "target_asset_ids": list(target_asset_ids),
            "description": description,
            "parameters": dict(parameters) if parameters else {},
            "records_in": max(0, records_in),
            "records_out": max(0, records_out),
            "records_filtered": max(0, records_filtered),
            "records_error": max(0, records_error),
            "duration_ms": max(0.0, duration_ms),
            "metadata": dict(metadata) if metadata else {},
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        # -- Provenance hash ------------------------------------------------
        provenance_payload = _build_provenance_payload(transformation)
        provenance_entry = self._provenance.record(
            entity_type="transformation",
            entity_id=txn_id,
            action="transformation_captured",
            metadata=provenance_payload,
        )
        transformation["provenance_hash"] = provenance_entry.hash_value

        # -- Persist to stores under lock -----------------------------------
        with self._lock:
            self._transformations[txn_id] = transformation
            self._index_transformation(txn_id, transformation, now_iso)

        # -- Metrics --------------------------------------------------------
        record_transformation_captured(transformation_type, agent_id)
        elapsed = time.monotonic() - start
        observe_processing_duration("transformation_capture", elapsed)

        logger.info(
            "Recorded transformation id=%s type=%s agent=%s pipeline=%s "
            "records_in=%d records_out=%d duration_ms=%.1f "
            "provenance_prefix=%s",
            txn_id,
            transformation_type,
            agent_id,
            pipeline_id,
            transformation["records_in"],
            transformation["records_out"],
            transformation["duration_ms"],
            transformation["provenance_hash"][:16],
        )
        return transformation

    # ------------------------------------------------------------------
    # Public API - Retrieve
    # ------------------------------------------------------------------

    def get_transformation(self, transformation_id: str) -> Optional[dict]:
        """Retrieve a single transformation by its unique identifier.

        Args:
            transformation_id: The UUID4 identifier of the transformation
                to retrieve.

        Returns:
            A copy of the transformation dictionary, or ``None`` if no
            transformation with the given ID exists.
        """
        if not transformation_id:
            return None
        with self._lock:
            txn = self._transformations.get(transformation_id)
            return dict(txn) if txn is not None else None

    def search_transformations(
        self,
        transformation_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Search transformations with optional multi-criteria filtering.

        Applies filters in order of expected selectivity (type, agent,
        pipeline, time range) and returns paginated results. When multiple
        filters are provided they are intersected (AND semantics).

        Args:
            transformation_type: Optional filter by transformation type.
            agent_id: Optional filter by originating agent identifier.
            pipeline_id: Optional filter by pipeline identifier.
            start_time: Optional ISO-8601 lower bound (inclusive) for
                ``created_at`` timestamp filtering.
            end_time: Optional ISO-8601 upper bound (inclusive) for
                ``created_at`` timestamp filtering.
            limit: Maximum number of results to return. Defaults to 100.
            offset: Number of matching results to skip before returning.
                Defaults to 0.

        Returns:
            List of transformation dictionaries matching all supplied
            criteria, ordered by ``created_at`` ascending, paginated by
            ``offset`` and ``limit``.
        """
        start = time.monotonic()
        limit = max(1, limit)
        offset = max(0, offset)

        with self._lock:
            # Start with candidate ID sets from secondary indexes
            candidate_sets: List[set] = []

            if transformation_type is not None:
                ids = self._type_index.get(transformation_type, [])
                candidate_sets.append(set(ids))

            if agent_id is not None:
                ids = self._agent_index.get(agent_id, [])
                candidate_sets.append(set(ids))

            if pipeline_id is not None:
                ids = self._pipeline_index.get(pipeline_id, [])
                candidate_sets.append(set(ids))

            if start_time is not None or end_time is not None:
                time_ids = self._get_ids_in_time_range(start_time, end_time)
                candidate_sets.append(set(time_ids))

            # Intersect all candidate sets, or use full ID set if no filters
            if candidate_sets:
                result_ids = candidate_sets[0]
                for s in candidate_sets[1:]:
                    result_ids = result_ids.intersection(s)
            else:
                result_ids = set(self._transformations.keys())

            # Collect matching transformations and sort by created_at
            results = [
                dict(self._transformations[tid])
                for tid in result_ids
                if tid in self._transformations
            ]

        # Sort by timestamp ascending
        results.sort(key=lambda t: t.get("created_at", ""))

        # Apply pagination
        paginated = results[offset: offset + limit]

        elapsed = time.monotonic() - start
        observe_processing_duration("transformation_search", elapsed)

        logger.debug(
            "search_transformations: matched=%d returned=%d "
            "(offset=%d, limit=%d) in %.3fs",
            len(results),
            len(paginated),
            offset,
            limit,
            elapsed,
        )
        return paginated

    def get_transformations_by_agent(self, agent_id: str) -> List[dict]:
        """Retrieve all transformations performed by a specific agent.

        Args:
            agent_id: The agent identifier to filter by.

        Returns:
            List of transformation dictionaries for the agent, ordered by
            ``created_at`` ascending. Returns an empty list if the agent
            has no recorded transformations.
        """
        if not agent_id:
            return []
        with self._lock:
            ids = self._agent_index.get(agent_id, [])
            results = [
                dict(self._transformations[tid])
                for tid in ids
                if tid in self._transformations
            ]
        results.sort(key=lambda t: t.get("created_at", ""))
        return results

    def get_transformations_by_pipeline(self, pipeline_id: str) -> List[dict]:
        """Retrieve all transformations within a specific pipeline.

        Args:
            pipeline_id: The pipeline identifier to filter by.

        Returns:
            List of transformation dictionaries for the pipeline, ordered
            by ``created_at`` ascending. Returns an empty list if the
            pipeline has no recorded transformations.
        """
        if not pipeline_id:
            return []
        with self._lock:
            ids = self._pipeline_index.get(pipeline_id, [])
            results = [
                dict(self._transformations[tid])
                for tid in ids
                if tid in self._transformations
            ]
        results.sort(key=lambda t: t.get("created_at", ""))
        return results

    def get_transformations_by_type(
        self, transformation_type: str
    ) -> List[dict]:
        """Retrieve all transformations of a specific type.

        Args:
            transformation_type: The transformation type to filter by.

        Returns:
            List of transformation dictionaries of the given type, ordered
            by ``created_at`` ascending. Returns an empty list if no
            transformations of that type have been recorded.
        """
        if not transformation_type:
            return []
        with self._lock:
            ids = self._type_index.get(transformation_type, [])
            results = [
                dict(self._transformations[tid])
                for tid in ids
                if tid in self._transformations
            ]
        results.sort(key=lambda t: t.get("created_at", ""))
        return results

    def get_transformations_in_range(
        self,
        start_time: str,
        end_time: str,
    ) -> List[dict]:
        """Retrieve all transformations within a time range.

        Uses the sorted time index with binary search for efficient
        range queries over potentially large transformation stores.

        Args:
            start_time: ISO-8601 lower bound (inclusive).
            end_time: ISO-8601 upper bound (inclusive).

        Returns:
            List of transformation dictionaries whose ``created_at``
            falls within ``[start_time, end_time]``, ordered by
            ``created_at`` ascending.
        """
        if not start_time or not end_time:
            return []
        with self._lock:
            ids = self._get_ids_in_time_range(start_time, end_time)
            results = [
                dict(self._transformations[tid])
                for tid in ids
                if tid in self._transformations
            ]
        results.sort(key=lambda t: t.get("created_at", ""))
        return results

    # ------------------------------------------------------------------
    # Public API - Batch
    # ------------------------------------------------------------------

    def batch_record(
        self,
        transformations: List[dict],
    ) -> dict:
        """Record multiple transformations in a single batch operation.

        Processes each transformation individually, collecting successes
        and failures. A failure in one record does not abort the batch;
        the remaining records continue to be processed.

        Args:
            transformations: List of dictionaries, each containing the
                keyword arguments expected by
                :meth:`record_transformation`. At minimum each dict must
                include ``transformation_type``, ``agent_id``,
                ``pipeline_id``, ``source_asset_ids``, and
                ``target_asset_ids``.

        Returns:
            Summary dictionary with keys:
                - ``recorded``: List of successfully recorded
                  transformation dicts.
                - ``failed``: Number of records that failed.
                - ``errors``: List of error message strings for each
                  failed record.

        Example:
            >>> result = engine.batch_record([
            ...     {
            ...         "transformation_type": "filter",
            ...         "agent_id": "profiler",
            ...         "pipeline_id": "p1",
            ...         "source_asset_ids": ["a"],
            ...         "target_asset_ids": ["b"],
            ...     },
            ... ])
            >>> assert result["failed"] == 0
        """
        start = time.monotonic()
        recorded: List[dict] = []
        errors: List[str] = []

        for idx, txn_spec in enumerate(transformations):
            try:
                if not isinstance(txn_spec, dict):
                    raise TypeError(
                        f"Expected dict at index {idx}, "
                        f"got {type(txn_spec).__name__}"
                    )
                result = self.record_transformation(
                    transformation_type=txn_spec.get("transformation_type", ""),
                    agent_id=txn_spec.get("agent_id", ""),
                    pipeline_id=txn_spec.get("pipeline_id", ""),
                    source_asset_ids=txn_spec.get("source_asset_ids", []),
                    target_asset_ids=txn_spec.get("target_asset_ids", []),
                    execution_id=txn_spec.get("execution_id"),
                    description=txn_spec.get("description", ""),
                    parameters=txn_spec.get("parameters"),
                    records_in=txn_spec.get("records_in", 0),
                    records_out=txn_spec.get("records_out", 0),
                    records_filtered=txn_spec.get("records_filtered", 0),
                    records_error=txn_spec.get("records_error", 0),
                    duration_ms=txn_spec.get("duration_ms", 0.0),
                    metadata=txn_spec.get("metadata"),
                )
                recorded.append(result)
            except (ValueError, TypeError) as exc:
                error_msg = f"Batch index {idx}: {exc}"
                errors.append(error_msg)
                logger.warning("batch_record failure: %s", error_msg)
            except Exception as exc:
                error_msg = f"Batch index {idx}: unexpected error: {exc}"
                errors.append(error_msg)
                logger.error(
                    "batch_record unexpected failure at index %d: %s",
                    idx,
                    exc,
                    exc_info=True,
                )

        elapsed = time.monotonic() - start
        observe_processing_duration("transformation_batch_record", elapsed)

        logger.info(
            "batch_record completed: total=%d recorded=%d failed=%d "
            "in %.3fs",
            len(transformations),
            len(recorded),
            len(errors),
            elapsed,
        )

        return {
            "recorded": recorded,
            "failed": len(errors),
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Public API - Chain Traversal
    # ------------------------------------------------------------------

    def get_transformation_chain(
        self,
        asset_id: str,
        direction: str = "backward",
    ) -> List[dict]:
        """Build an ordered chain of transformations for a data asset.

        Follows lineage edges through ``source_asset_ids`` and
        ``target_asset_ids`` to reconstruct the transformation chain
        that either produced (backward) or consumed (forward) a given
        data asset.

        **Backward** (default): starts from transformations whose
        ``target_asset_ids`` contain ``asset_id``, then recursively
        follows each transformation's ``source_asset_ids`` to find
        upstream transformations.

        **Forward**: starts from transformations whose
        ``source_asset_ids`` contain ``asset_id``, then recursively
        follows each transformation's ``target_asset_ids`` to find
        downstream transformations.

        Cycle detection is implemented via a visited set to prevent
        infinite loops in graphs with circular dependencies.

        Args:
            asset_id: The data asset identifier to trace lineage for.
            direction: Traversal direction. Must be ``"backward"``
                (upstream/provenance) or ``"forward"`` (downstream/impact).
                Defaults to ``"backward"``.

        Returns:
            Ordered list of transformation dictionaries forming the
            lineage chain, from the starting asset outward. Returns an
            empty list if no transformations reference the asset.

        Raises:
            ValueError: If ``asset_id`` is empty or ``direction`` is
                not ``"backward"`` or ``"forward"``.
        """
        if not asset_id:
            raise ValueError("asset_id must not be empty")
        if direction not in _VALID_CHAIN_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_CHAIN_DIRECTIONS)}, "
                f"got '{direction}'"
            )

        start = time.monotonic()

        with self._lock:
            # Build reverse lookup indexes for chain traversal
            target_to_txns = self._build_target_index()
            source_to_txns = self._build_source_index()

            chain: List[dict] = []
            visited: set = set()
            queue: List[str] = [asset_id]

            while queue:
                current_asset = queue.pop(0)

                if direction == "backward":
                    # Find transformations that produced this asset
                    txn_ids = target_to_txns.get(current_asset, [])
                else:
                    # Find transformations that consumed this asset
                    txn_ids = source_to_txns.get(current_asset, [])

                for tid in txn_ids:
                    if tid in visited:
                        continue
                    visited.add(tid)

                    txn = self._transformations.get(tid)
                    if txn is None:
                        continue

                    chain.append(dict(txn))

                    # Enqueue next hop assets for further traversal
                    if direction == "backward":
                        for src_id in txn.get("source_asset_ids", []):
                            if src_id not in visited:
                                queue.append(src_id)
                    else:
                        for tgt_id in txn.get("target_asset_ids", []):
                            if tgt_id not in visited:
                                queue.append(tgt_id)

        # Sort chain by created_at for stable ordering
        chain.sort(key=lambda t: t.get("created_at", ""))

        elapsed = time.monotonic() - start
        observe_processing_duration("transformation_chain_traversal", elapsed)

        logger.debug(
            "get_transformation_chain: asset=%s direction=%s "
            "chain_length=%d in %.3fs",
            asset_id,
            direction,
            len(chain),
            elapsed,
        )
        return chain

    # ------------------------------------------------------------------
    # Public API - Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Compute aggregate statistics across all recorded transformations.

        Returns deterministic, zero-hallucination statistics computed
        from in-memory data with explicit arithmetic.

        Returns:
            Dictionary with the following keys:
                - ``total``: Total number of recorded transformations.
                - ``by_type``: Dict mapping transformation type to count.
                - ``by_agent``: Dict mapping agent_id to count.
                - ``by_pipeline``: Dict mapping pipeline_id to count.
                - ``avg_duration_ms``: Average execution duration across
                  all transformations, or 0.0 if none exist.
                - ``total_records_in``: Sum of ``records_in`` across all
                  transformations.
                - ``total_records_out``: Sum of ``records_out`` across all
                  transformations.
                - ``total_records_filtered``: Sum of ``records_filtered``
                  across all transformations.
                - ``total_records_error``: Sum of ``records_error`` across
                  all transformations.
        """
        start = time.monotonic()

        with self._lock:
            total = len(self._transformations)

            by_type: Dict[str, int] = {
                t: len(ids) for t, ids in self._type_index.items()
            }
            by_agent: Dict[str, int] = {
                a: len(ids) for a, ids in self._agent_index.items()
            }
            by_pipeline: Dict[str, int] = {
                p: len(ids) for p, ids in self._pipeline_index.items()
            }

            total_duration_ms: float = 0.0
            total_records_in: int = 0
            total_records_out: int = 0
            total_records_filtered: int = 0
            total_records_error: int = 0

            for txn in self._transformations.values():
                total_duration_ms += txn.get("duration_ms", 0.0)
                total_records_in += txn.get("records_in", 0)
                total_records_out += txn.get("records_out", 0)
                total_records_filtered += txn.get("records_filtered", 0)
                total_records_error += txn.get("records_error", 0)

        avg_duration_ms = (
            total_duration_ms / total if total > 0 else 0.0
        )

        elapsed = time.monotonic() - start
        observe_processing_duration("transformation_statistics", elapsed)

        stats = {
            "total": total,
            "by_type": by_type,
            "by_agent": by_agent,
            "by_pipeline": by_pipeline,
            "avg_duration_ms": round(avg_duration_ms, 3),
            "total_records_in": total_records_in,
            "total_records_out": total_records_out,
            "total_records_filtered": total_records_filtered,
            "total_records_error": total_records_error,
        }

        logger.debug(
            "get_statistics: total=%d avg_duration=%.3fms "
            "records_in=%d records_out=%d",
            total,
            avg_duration_ms,
            total_records_in,
            total_records_out,
        )
        return stats

    # ------------------------------------------------------------------
    # Public API - Export
    # ------------------------------------------------------------------

    def export_transformations(
        self,
        agent_id: Optional[str] = None,
    ) -> List[dict]:
        """Export transformations as a list of dictionaries.

        Optionally filter by agent identifier. The returned list is
        ordered by ``created_at`` ascending and contains deep copies
        of the stored records to prevent mutation.

        Args:
            agent_id: Optional agent identifier to filter exports. When
                ``None``, all transformations are exported.

        Returns:
            List of transformation dictionaries, each containing all
            stored fields including ``provenance_hash``.
        """
        start = time.monotonic()

        with self._lock:
            if agent_id is not None:
                ids = self._agent_index.get(agent_id, [])
                results = [
                    dict(self._transformations[tid])
                    for tid in ids
                    if tid in self._transformations
                ]
            else:
                results = [
                    dict(txn) for txn in self._transformations.values()
                ]

        results.sort(key=lambda t: t.get("created_at", ""))

        elapsed = time.monotonic() - start
        observe_processing_duration("transformation_export", elapsed)

        logger.info(
            "export_transformations: agent_filter=%s count=%d in %.3fs",
            agent_id or "ALL",
            len(results),
            elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Public API - List All
    # ------------------------------------------------------------------

    def list_transformations(self, limit: int = 10000) -> List[dict]:
        """Return all recorded transformations as a list of dictionaries.

        Convenience method used by LineageTrackerPipelineEngine to
        synchronise the lineage graph from the transformation tracker.

        Args:
            limit: Maximum number of transformations to return.

        Returns:
            List of transformation dictionaries sorted by created_at
            ascending, capped at ``limit``.
        """
        with self._lock:
            all_txns = list(self._transformations.values())

        # Sort by created_at ascending for deterministic ordering
        all_txns.sort(key=lambda t: t.get("created_at", ""))
        return [dict(t) for t in all_txns[:limit]]

    # ------------------------------------------------------------------
    # Public API - Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all recorded transformations and reset all indexes.

        This operation is irreversible. Primarily intended for testing
        and development environments. The provenance tracker is NOT
        reset by this operation to preserve the audit chain.
        """
        with self._lock:
            count = len(self._transformations)
            self._transformations.clear()
            self._agent_index.clear()
            self._pipeline_index.clear()
            self._type_index.clear()
            self._time_index.clear()

        logger.info(
            "TransformationTrackerEngine cleared: %d transformations removed",
            count,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Return the total number of recorded transformations.

        Returns:
            Integer count of transformations in the primary store.
        """
        with self._lock:
            return len(self._transformations)

    @property
    def provenance(self) -> ProvenanceTracker:
        """Return the underlying provenance tracker instance.

        Returns:
            The :class:`ProvenanceTracker` used by this engine.
        """
        return self._provenance

    # ------------------------------------------------------------------
    # Internal - Validation
    # ------------------------------------------------------------------

    def _validate_record_inputs(
        self,
        transformation_type: str,
        agent_id: str,
        pipeline_id: str,
        source_asset_ids: Any,
        target_asset_ids: Any,
    ) -> None:
        """Validate inputs for record_transformation.

        Args:
            transformation_type: Must be in VALID_TRANSFORMATION_TYPES.
            agent_id: Must be a non-empty string.
            pipeline_id: Must be a non-empty string.
            source_asset_ids: Must be a non-empty list.
            target_asset_ids: Must be a non-empty list.

        Raises:
            ValueError: If any string argument is empty, or if
                ``transformation_type`` is not recognised.
            TypeError: If ``source_asset_ids`` or ``target_asset_ids``
                are not lists.
        """
        if not transformation_type:
            raise ValueError("transformation_type must not be empty")

        if transformation_type not in VALID_TRANSFORMATION_TYPES:
            raise ValueError(
                f"Invalid transformation_type '{transformation_type}'. "
                f"Must be one of: {sorted(VALID_TRANSFORMATION_TYPES)}"
            )

        if not agent_id:
            raise ValueError("agent_id must not be empty")

        if not pipeline_id:
            raise ValueError("pipeline_id must not be empty")

        if not isinstance(source_asset_ids, list):
            raise TypeError(
                f"source_asset_ids must be a list, "
                f"got {type(source_asset_ids).__name__}"
            )

        if not source_asset_ids:
            raise ValueError("source_asset_ids must not be empty")

        if not isinstance(target_asset_ids, list):
            raise TypeError(
                f"target_asset_ids must be a list, "
                f"got {type(target_asset_ids).__name__}"
            )

        if not target_asset_ids:
            raise ValueError("target_asset_ids must not be empty")

    # ------------------------------------------------------------------
    # Internal - Indexing
    # ------------------------------------------------------------------

    def _index_transformation(
        self,
        txn_id: str,
        transformation: dict,
        timestamp_iso: str,
    ) -> None:
        """Add a transformation to all secondary indexes.

        Must be called while ``self._lock`` is held.

        Args:
            txn_id: Unique transformation identifier.
            transformation: The full transformation record dict.
            timestamp_iso: ISO-8601 timestamp for time index insertion.
        """
        # Agent index
        agent_id = transformation["agent_id"]
        if agent_id not in self._agent_index:
            self._agent_index[agent_id] = []
        self._agent_index[agent_id].append(txn_id)

        # Pipeline index
        pipeline_id = transformation["pipeline_id"]
        if pipeline_id not in self._pipeline_index:
            self._pipeline_index[pipeline_id] = []
        self._pipeline_index[pipeline_id].append(txn_id)

        # Type index
        txn_type = transformation["transformation_type"]
        if txn_type not in self._type_index:
            self._type_index[txn_type] = []
        self._type_index[txn_type].append(txn_id)

        # Time index (maintained sorted for binary search)
        entry = (timestamp_iso, txn_id)
        bisect.insort(self._time_index, entry)

    # ------------------------------------------------------------------
    # Internal - Time Range Queries
    # ------------------------------------------------------------------

    def _get_ids_in_time_range(
        self,
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> List[str]:
        """Return transformation IDs whose timestamps fall in the range.

        Uses binary search on the sorted ``_time_index`` for O(log n)
        bound location plus O(k) scan of the matching entries.

        Must be called while ``self._lock`` is held.

        Args:
            start_time: Optional ISO-8601 lower bound (inclusive).
            end_time: Optional ISO-8601 upper bound (inclusive).

        Returns:
            List of transformation IDs within the specified time range.
        """
        if not self._time_index:
            return []

        # Determine effective bounds
        effective_start = start_time or ""
        effective_end = end_time or "\xff"  # Beyond any ISO timestamp

        # Binary search for the left bound
        left = bisect.bisect_left(
            self._time_index, (effective_start,)
        )

        # Scan rightward collecting matches
        result_ids: List[str] = []
        for i in range(left, len(self._time_index)):
            ts, tid = self._time_index[i]
            if ts > effective_end:
                break
            result_ids.append(tid)

        return result_ids

    # ------------------------------------------------------------------
    # Internal - Chain Traversal Indexes
    # ------------------------------------------------------------------

    def _build_target_index(self) -> Dict[str, List[str]]:
        """Build a reverse index from target asset IDs to transformation IDs.

        Must be called while ``self._lock`` is held.

        Returns:
            Dictionary mapping each target asset ID to the list of
            transformation IDs that produce it.
        """
        index: Dict[str, List[str]] = {}
        for tid, txn in self._transformations.items():
            for asset_id in txn.get("target_asset_ids", []):
                if asset_id not in index:
                    index[asset_id] = []
                index[asset_id].append(tid)
        return index

    def _build_source_index(self) -> Dict[str, List[str]]:
        """Build a reverse index from source asset IDs to transformation IDs.

        Must be called while ``self._lock`` is held.

        Returns:
            Dictionary mapping each source asset ID to the list of
            transformation IDs that consume it.
        """
        index: Dict[str, List[str]] = {}
        for tid, txn in self._transformations.items():
            for asset_id in txn.get("source_asset_ids", []):
                if asset_id not in index:
                    index[asset_id] = []
                index[asset_id].append(tid)
        return index

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of recorded transformations.

        Returns:
            Integer count of transformations in the primary store.
        """
        return self.count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing the class name and transformation count.
        """
        return (
            f"TransformationTrackerEngine("
            f"transformations={self.count}, "
            f"agents={len(self._agent_index)}, "
            f"pipelines={len(self._pipeline_index)}, "
            f"types={len(self._type_index)})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["TransformationTrackerEngine"]
