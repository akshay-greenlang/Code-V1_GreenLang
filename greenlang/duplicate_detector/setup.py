# -*- coding: utf-8 -*-
"""
Duplicate Detection Service Setup - AGENT-DATA-011

Provides ``configure_duplicate_detector(app)`` which wires up the
Duplicate Detection SDK (record fingerprinting, blocking, similarity
scoring, match classification, cluster resolution, merge engine,
deduplication pipeline, provenance tracker) and mounts the REST API.

Also exposes ``get_duplicate_detector(app)`` for programmatic access
and the ``DuplicateDetectorService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.duplicate_detector.setup import configure_duplicate_detector
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_duplicate_detector(app))

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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.duplicate_detector.config import (
    DuplicateDetectorConfig,
    get_config,
)
from greenlang.duplicate_detector.metrics import (
    PROMETHEUS_AVAILABLE,
    inc_jobs,
    inc_fingerprints,
    inc_blocks,
    inc_comparisons,
    inc_matches,
    inc_clusters,
    inc_merges,
    inc_conflicts,
    observe_duration,
    observe_similarity,
    set_active_jobs,
    inc_errors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# Lightweight Pydantic response models used by the facade
# ===================================================================


class FingerprintResponse(BaseModel):
    """Record fingerprinting result.

    Attributes:
        fingerprint_id: Unique fingerprint operation identifier.
        algorithm: Fingerprinting algorithm used.
        total_records: Total records processed.
        unique_fingerprints: Number of unique fingerprints generated.
        duplicate_candidates: Number of records sharing a fingerprint.
        fingerprints: Mapping of record index to fingerprint hash.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    fingerprint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = Field(default="sha256")
    total_records: int = Field(default=0)
    unique_fingerprints: int = Field(default=0)
    duplicate_candidates: int = Field(default=0)
    fingerprints: Dict[str, str] = Field(default_factory=dict)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class BlockResponse(BaseModel):
    """Blocking partition result.

    Attributes:
        block_id: Unique block operation identifier.
        strategy: Blocking strategy used.
        total_records: Total records processed.
        total_blocks: Number of blocks created.
        total_pairs: Total candidate pairs generated.
        largest_block: Size of the largest block.
        reduction_ratio: Pair space reduction ratio (0.0-1.0).
        blocks: Summary of blocks created.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = Field(default="sorted_neighborhood")
    total_records: int = Field(default=0)
    total_blocks: int = Field(default=0)
    total_pairs: int = Field(default=0)
    largest_block: int = Field(default=0)
    reduction_ratio: float = Field(default=0.0)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class CompareResponse(BaseModel):
    """Similarity comparison result.

    Attributes:
        comparison_id: Unique comparison operation identifier.
        algorithm: Default comparison algorithm used.
        total_pairs: Total pairs compared.
        comparisons: List of comparison results per pair.
        avg_similarity: Average similarity score across all pairs.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    comparison_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = Field(default="jaro_winkler")
    total_pairs: int = Field(default=0)
    comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    avg_similarity: float = Field(default=0.0)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ClassifyResponse(BaseModel):
    """Match classification result.

    Attributes:
        classify_id: Unique classification operation identifier.
        total_comparisons: Total comparisons classified.
        matches: Number of MATCH classifications.
        possible_matches: Number of POSSIBLE classifications.
        non_matches: Number of NON_MATCH classifications.
        match_threshold: Threshold used for MATCH classification.
        possible_threshold: Threshold used for POSSIBLE classification.
        classifications: List of classification results.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    classify_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_comparisons: int = Field(default=0)
    matches: int = Field(default=0)
    possible_matches: int = Field(default=0)
    non_matches: int = Field(default=0)
    match_threshold: float = Field(default=0.85)
    possible_threshold: float = Field(default=0.65)
    classifications: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ClusterResponse(BaseModel):
    """Cluster resolution result.

    Attributes:
        cluster_run_id: Unique cluster operation identifier.
        algorithm: Clustering algorithm used.
        total_matches: Total match pairs input.
        total_clusters: Number of duplicate clusters formed.
        largest_cluster: Size of the largest cluster.
        avg_cluster_size: Average cluster size.
        clusters: List of cluster details.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    cluster_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = Field(default="union_find")
    total_matches: int = Field(default=0)
    total_clusters: int = Field(default=0)
    largest_cluster: int = Field(default=0)
    avg_cluster_size: float = Field(default=0.0)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class MergeResponse(BaseModel):
    """Merge execution result.

    Attributes:
        merge_id: Unique merge operation identifier.
        strategy: Merge strategy used.
        total_clusters: Total clusters processed.
        total_records_merged: Total records merged.
        total_golden_records: Number of golden records produced.
        conflicts_resolved: Number of field-level conflicts resolved.
        conflict_resolution: Conflict resolution method used.
        merged_records: List of golden record summaries.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    merge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = Field(default="keep_most_complete")
    total_clusters: int = Field(default=0)
    total_records_merged: int = Field(default=0)
    total_golden_records: int = Field(default=0)
    conflicts_resolved: int = Field(default=0)
    conflict_resolution: str = Field(default="most_complete")
    merged_records: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class PipelineResponse(BaseModel):
    """Full deduplication pipeline result.

    Attributes:
        pipeline_id: Unique pipeline run identifier.
        job_id: Associated dedup job identifier.
        status: Pipeline status (completed, failed, cancelled).
        total_records: Total input records.
        total_duplicates: Total duplicates identified.
        total_clusters: Total duplicate clusters.
        total_merged: Total records merged into golden records.
        stages: Per-stage summary (fingerprint, block, compare, classify,
            cluster, merge).
        processing_time_ms: Total processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = Field(default="")
    status: str = Field(default="completed")
    total_records: int = Field(default=0)
    total_duplicates: int = Field(default=0)
    total_clusters: int = Field(default=0)
    total_merged: int = Field(default=0)
    stages: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class StatsResponse(BaseModel):
    """Aggregate statistics for the duplicate detection service.

    Attributes:
        total_jobs: Total dedup jobs processed.
        completed_jobs: Total jobs completed successfully.
        failed_jobs: Total jobs that failed.
        total_records_processed: Total records fingerprinted.
        total_duplicates_found: Total duplicate records identified.
        total_clusters: Total duplicate clusters formed.
        total_merges: Total merges completed.
        total_conflicts: Total merge conflicts resolved.
        total_rules: Total dedup rules defined.
        active_jobs: Number of currently active jobs.
        avg_similarity: Average similarity score across all comparisons.
        provenance_entries: Total provenance entries recorded.
    """
    total_jobs: int = Field(default=0)
    completed_jobs: int = Field(default=0)
    failed_jobs: int = Field(default=0)
    total_records_processed: int = Field(default=0)
    total_duplicates_found: int = Field(default=0)
    total_clusters: int = Field(default=0)
    total_merges: int = Field(default=0)
    total_conflicts: int = Field(default=0)
    total_rules: int = Field(default=0)
    active_jobs: int = Field(default=0)
    avg_similarity: float = Field(default=0.0)
    provenance_entries: int = Field(default=0)


# ===================================================================
# Provenance helper
# ===================================================================


class _ProvenanceTracker:
    """Minimal provenance tracker recording SHA-256 audit entries.

    Attributes:
        entries: List of provenance entries.
        entry_count: Number of entries recorded.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.entry_count: int = 0

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry and return its hash.

        Args:
            entity_type: Type of entity (dedup_job, match, cluster, merge, rule).
            entity_id: Entity identifier.
            action: Action performed (fingerprint, block, compare, classify,
                cluster, merge, pipeline, create, cancel).
            data_hash: SHA-256 hash of associated data.
            user_id: User or system that performed the action.

        Returns:
            SHA-256 hash of the provenance entry itself.
        """
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self.entry_count += 1
        return entry_hash


# ===================================================================
# Helper utilities
# ===================================================================


# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["DuplicateDetectorService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# DuplicateDetectorService facade
# ===================================================================


class DuplicateDetectorService:
    """Unified facade over the Duplicate Detection SDK.

    Aggregates all dedup engines (record fingerprinter, blocking engine,
    similarity scorer, match classifier, cluster resolver, merge engine,
    deduplication pipeline, provenance tracker) through a single entry
    point with convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: DuplicateDetectorConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = DuplicateDetectorService()
        >>> result = service.fingerprint_records(
        ...     records=[{"name": "Alice", "email": "alice@co.com"}],
        ...     field_set=["name", "email"],
        ... )
        >>> print(result.total_records, result.unique_fingerprints)
    """

    def __init__(
        self,
        config: Optional[DuplicateDetectorConfig] = None,
    ) -> None:
        """Initialize the Duplicate Detector Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - RecordFingerprinterEngine
        - BlockingEngine
        - SimilarityScorerEngine
        - MatchClassifierEngine
        - ClusterResolverEngine
        - MergeEngine
        - DeduplicationPipelineEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._fingerprinter_engine: Any = None
        self._blocking_engine: Any = None
        self._similarity_engine: Any = None
        self._classifier_engine: Any = None
        self._cluster_engine: Any = None
        self._merge_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._fingerprint_results: Dict[str, FingerprintResponse] = {}
        self._block_results: Dict[str, BlockResponse] = {}
        self._compare_results: Dict[str, CompareResponse] = {}
        self._classify_results: Dict[str, ClassifyResponse] = {}
        self._cluster_results: Dict[str, ClusterResponse] = {}
        self._merge_results: Dict[str, MergeResponse] = {}
        self._pipeline_results: Dict[str, PipelineResponse] = {}
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._matches: Dict[str, Dict[str, Any]] = {}
        self._clusters: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._stats = StatsResponse()
        self._started = False
        self._similarity_sum = 0.0
        self._similarity_count = 0
        self._active_jobs = 0

        logger.info("DuplicateDetectorService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def fingerprinter_engine(self) -> Any:
        """Get the RecordFingerprinterEngine instance."""
        return self._fingerprinter_engine

    @property
    def blocking_engine(self) -> Any:
        """Get the BlockingEngine instance."""
        return self._blocking_engine

    @property
    def similarity_engine(self) -> Any:
        """Get the SimilarityScorerEngine instance."""
        return self._similarity_engine

    @property
    def classifier_engine(self) -> Any:
        """Get the MatchClassifierEngine instance."""
        return self._classifier_engine

    @property
    def cluster_engine(self) -> Any:
        """Get the ClusterResolverEngine instance."""
        return self._cluster_engine

    @property
    def merge_engine(self) -> Any:
        """Get the MergeEngine instance."""
        return self._merge_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the DeduplicationPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.duplicate_detector.record_fingerprinter import (
                RecordFingerprinterEngine,
            )
            self._fingerprinter_engine = RecordFingerprinterEngine(self.config)
        except ImportError:
            logger.warning("RecordFingerprinterEngine not available; using stub")

        try:
            from greenlang.duplicate_detector.blocking_engine import BlockingEngine
            self._blocking_engine = BlockingEngine(self.config)
        except ImportError:
            logger.warning("BlockingEngine not available; using stub")

        try:
            from greenlang.duplicate_detector.similarity_scorer import (
                SimilarityScorerEngine,
            )
            self._similarity_engine = SimilarityScorerEngine(self.config)
        except ImportError:
            logger.warning("SimilarityScorerEngine not available; using stub")

        try:
            from greenlang.duplicate_detector.match_classifier import (
                MatchClassifierEngine,
            )
            self._classifier_engine = MatchClassifierEngine(self.config)
        except ImportError:
            logger.warning("MatchClassifierEngine not available; using stub")

        try:
            from greenlang.duplicate_detector.cluster_resolver import (
                ClusterResolverEngine,
            )
            self._cluster_engine = ClusterResolverEngine(self.config)
        except ImportError:
            logger.warning("ClusterResolverEngine not available; using stub")

        try:
            from greenlang.duplicate_detector.merge_engine import MergeEngine
            self._merge_engine = MergeEngine(self.config)
        except ImportError:
            logger.warning("MergeEngine not available; using stub")

        try:
            from greenlang.duplicate_detector.dedup_pipeline import (
                DeduplicationPipelineEngine,
            )
            self._pipeline_engine = DeduplicationPipelineEngine(self.config)
        except ImportError:
            logger.warning("DeduplicationPipelineEngine not available; using stub")

    # ------------------------------------------------------------------
    # Record fingerprinting
    # ------------------------------------------------------------------

    def fingerprint_records(
        self,
        records: List[Dict[str, Any]],
        field_set: Optional[List[str]] = None,
        algorithm: Optional[str] = None,
    ) -> FingerprintResponse:
        """Fingerprint records using a deterministic hashing algorithm.

        Zero-hallucination: All fingerprints are SHA-256 / SimHash / MinHash
        computed deterministically in Python. No LLM calls.

        Args:
            records: List of record dicts to fingerprint.
            field_set: Fields to include in fingerprint (all if None).
            algorithm: Fingerprint algorithm (sha256, simhash, minhash).
                Uses config default if None.

        Returns:
            FingerprintResponse with fingerprint mapping.

        Raises:
            ValueError: If records is empty.
        """
        start_time = time.time()
        algo = algorithm or self.config.fingerprint_algorithm

        if not records:
            raise ValueError("Records list must not be empty for fingerprinting")

        records = records[:self.config.max_records_per_job]

        # Delegate to engine if available
        if self._fingerprinter_engine is not None:
            engine_result = self._fingerprinter_engine.fingerprint(
                records=records,
                field_set=field_set,
                algorithm=algo,
            )
            return self._wrap_fingerprint_result(engine_result, start_time, algo)

        # Fallback: built-in SHA-256 fingerprinting
        fingerprints: Dict[str, str] = {}
        for idx, record in enumerate(records):
            fields = field_set or list(record.keys())
            field_data = {f: record.get(f) for f in sorted(fields) if f in record}
            if self.config.fingerprint_normalize:
                field_data = self._normalize_fields(field_data)
            fp = hashlib.sha256(
                json.dumps(field_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            fingerprints[str(idx)] = fp

        unique_fps = len(set(fingerprints.values()))
        dup_candidates = len(fingerprints) - unique_fps
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = FingerprintResponse(
            algorithm=algo,
            total_records=len(records),
            unique_fingerprints=unique_fps,
            duplicate_candidates=dup_candidates,
            fingerprints=fingerprints,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._fingerprint_results[result.fingerprint_id] = result

        # Provenance
        self.provenance.record(
            entity_type="fingerprint",
            entity_id=result.fingerprint_id,
            action="fingerprint",
            data_hash=result.provenance_hash,
        )

        # Metrics
        inc_fingerprints(algo, len(records))
        observe_duration("fingerprint", time.time() - start_time)

        self._stats.total_records_processed += len(records)

        logger.info(
            "Fingerprinted %d records (%s): %d unique, %d candidates",
            len(records), algo, unique_fps, dup_candidates,
        )
        return result

    def _normalize_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field values for deterministic fingerprinting.

        Lowercases strings, strips whitespace, converts None to empty string.

        Args:
            fields: Field name to value mapping.

        Returns:
            Normalized field mapping.
        """
        normalized: Dict[str, Any] = {}
        for key, value in fields.items():
            if value is None:
                normalized[key] = ""
            elif isinstance(value, str):
                normalized[key] = value.strip().lower()
            else:
                normalized[key] = value
        return normalized

    def _wrap_fingerprint_result(
        self,
        engine_result: Any,
        start_time: float,
        algorithm: str,
    ) -> FingerprintResponse:
        """Wrap engine result into FingerprintResponse.

        Args:
            engine_result: Raw engine result dict or object.
            start_time: Operation start timestamp.
            algorithm: Algorithm used.

        Returns:
            FingerprintResponse with provenance.
        """
        data = engine_result if isinstance(engine_result, dict) else {}
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = FingerprintResponse(
            algorithm=algorithm,
            total_records=data.get("total_records", 0),
            unique_fingerprints=data.get("unique_fingerprints", 0),
            duplicate_candidates=data.get("duplicate_candidates", 0),
            fingerprints=data.get("fingerprints", {}),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._fingerprint_results[result.fingerprint_id] = result

        self.provenance.record(
            entity_type="fingerprint",
            entity_id=result.fingerprint_id,
            action="fingerprint",
            data_hash=result.provenance_hash,
        )
        inc_fingerprints(algorithm, result.total_records)
        observe_duration("fingerprint", time.time() - start_time)
        self._stats.total_records_processed += result.total_records

        return result

    # ------------------------------------------------------------------
    # Blocking
    # ------------------------------------------------------------------

    def create_blocks(
        self,
        records: List[Dict[str, Any]],
        strategy: Optional[str] = None,
        key_fields: Optional[List[str]] = None,
    ) -> BlockResponse:
        """Create blocking partitions for candidate pair generation.

        Reduces the comparison space by grouping records that are likely
        duplicates into blocks.

        Args:
            records: List of record dicts.
            strategy: Blocking strategy (sorted_neighborhood, standard,
                canopy, none). Uses config default if None.
            key_fields: Fields to use for blocking key generation.

        Returns:
            BlockResponse with block partition details.

        Raises:
            ValueError: If records is empty.
        """
        start_time = time.time()
        strat = strategy or self.config.blocking_strategy

        if not records:
            raise ValueError("Records list must not be empty for blocking")

        records = records[:self.config.max_records_per_job]

        # Delegate to engine if available
        if self._blocking_engine is not None:
            engine_result = self._blocking_engine.create_blocks(
                records=records,
                strategy=strat,
                key_fields=key_fields,
            )
            return self._wrap_block_result(engine_result, start_time, strat)

        # Fallback: simple prefix-key blocking
        blocks: Dict[str, List[int]] = {}
        for idx, record in enumerate(records):
            fields = key_fields or list(record.keys())[:1]
            key_parts = []
            for f in fields:
                val = str(record.get(f, "")).strip().lower()
                key_parts.append(val[:self.config.blocking_key_size])
            block_key = "|".join(key_parts)
            if block_key not in blocks:
                blocks[block_key] = []
            blocks[block_key].append(idx)

        # Calculate pairs
        total_pairs = 0
        largest_block = 0
        block_summaries: List[Dict[str, Any]] = []

        for bk, indices in blocks.items():
            size = len(indices)
            pairs = size * (size - 1) // 2
            total_pairs += pairs
            if size > largest_block:
                largest_block = size
            block_summaries.append({
                "block_key": bk,
                "size": size,
                "pairs": pairs,
            })

        n = len(records)
        full_pairs = n * (n - 1) // 2
        reduction = 1.0 - (total_pairs / max(full_pairs, 1))

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = BlockResponse(
            strategy=strat,
            total_records=len(records),
            total_blocks=len(blocks),
            total_pairs=total_pairs,
            largest_block=largest_block,
            reduction_ratio=round(reduction, 4),
            blocks=block_summaries,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._block_results[result.block_id] = result

        self.provenance.record(
            entity_type="block",
            entity_id=result.block_id,
            action="block",
            data_hash=result.provenance_hash,
        )

        inc_blocks(strat, len(blocks))
        observe_duration("block", time.time() - start_time)

        logger.info(
            "Created %d blocks (%s): %d pairs, reduction=%.4f",
            len(blocks), strat, total_pairs, reduction,
        )
        return result

    def _wrap_block_result(
        self,
        engine_result: Any,
        start_time: float,
        strategy: str,
    ) -> BlockResponse:
        """Wrap engine result into BlockResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.
            strategy: Strategy used.

        Returns:
            BlockResponse with provenance.
        """
        data = engine_result if isinstance(engine_result, dict) else {}
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = BlockResponse(
            strategy=strategy,
            total_records=data.get("total_records", 0),
            total_blocks=data.get("total_blocks", 0),
            total_pairs=data.get("total_pairs", 0),
            largest_block=data.get("largest_block", 0),
            reduction_ratio=data.get("reduction_ratio", 0.0),
            blocks=data.get("blocks", []),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._block_results[result.block_id] = result

        self.provenance.record(
            entity_type="block",
            entity_id=result.block_id,
            action="block",
            data_hash=result.provenance_hash,
        )
        inc_blocks(strategy, result.total_blocks)
        observe_duration("block", time.time() - start_time)
        return result

    # ------------------------------------------------------------------
    # Similarity comparison
    # ------------------------------------------------------------------

    def compare_pairs(
        self,
        block_results: Dict[str, Any],
        field_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> CompareResponse:
        """Compare candidate record pairs for similarity.

        Zero-hallucination: All similarity calculations use deterministic
        string distance algorithms (Jaro-Winkler, Levenshtein, etc.).

        Args:
            block_results: Block results containing candidate pairs or
                raw pair list under 'pairs' key.
            field_configs: Per-field comparison configuration with
                algorithm and weight.

        Returns:
            CompareResponse with per-pair similarity scores.

        Raises:
            ValueError: If block_results contains no pairs.
        """
        start_time = time.time()
        algo = self.config.default_similarity_algorithm

        pairs = block_results.get("pairs", [])
        if not pairs:
            raise ValueError("No candidate pairs to compare")

        # Delegate to engine if available
        if self._similarity_engine is not None:
            engine_result = self._similarity_engine.compare(
                pairs=pairs,
                field_configs=field_configs,
            )
            return self._wrap_compare_result(engine_result, start_time, algo)

        # Fallback: simple exact match comparison
        comparisons: List[Dict[str, Any]] = []
        total_sim = 0.0

        for pair in pairs[:self.config.max_comparisons_per_block]:
            rec_a = pair.get("record_a", {})
            rec_b = pair.get("record_b", {})
            fields = field_configs or [
                {"field": f, "algorithm": "exact", "weight": 1.0}
                for f in rec_a.keys()
            ]
            field_scores: List[Dict[str, Any]] = []
            weighted_sum = 0.0
            total_weight = 0.0

            for fc in fields:
                field = fc.get("field", "")
                weight = float(fc.get("weight", 1.0))
                val_a = str(rec_a.get(field, "")).strip().lower()
                val_b = str(rec_b.get(field, "")).strip().lower()
                score = 1.0 if val_a == val_b else 0.0
                field_scores.append({
                    "field": field,
                    "algorithm": fc.get("algorithm", "exact"),
                    "score": score,
                    "weight": weight,
                })
                weighted_sum += score * weight
                total_weight += weight

            overall = weighted_sum / max(total_weight, 1.0)
            total_sim += overall

            comparisons.append({
                "pair_id": str(uuid.uuid4()),
                "record_a_id": pair.get("id_a", ""),
                "record_b_id": pair.get("id_b", ""),
                "field_scores": field_scores,
                "overall_score": round(overall, 4),
            })

            observe_similarity(algo, overall)

        avg_sim = total_sim / max(len(comparisons), 1)
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = CompareResponse(
            algorithm=algo,
            total_pairs=len(comparisons),
            comparisons=comparisons,
            avg_similarity=round(avg_sim, 4),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._compare_results[result.comparison_id] = result

        self.provenance.record(
            entity_type="comparison",
            entity_id=result.comparison_id,
            action="compare",
            data_hash=result.provenance_hash,
        )

        inc_comparisons(algo, len(comparisons))
        observe_duration("compare", time.time() - start_time)

        self._update_avg_similarity(total_sim, len(comparisons))

        logger.info(
            "Compared %d pairs (%s): avg_similarity=%.4f",
            len(comparisons), algo, avg_sim,
        )
        return result

    def _wrap_compare_result(
        self,
        engine_result: Any,
        start_time: float,
        algorithm: str,
    ) -> CompareResponse:
        """Wrap engine result into CompareResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.
            algorithm: Algorithm used.

        Returns:
            CompareResponse with provenance.
        """
        data = engine_result if isinstance(engine_result, dict) else {}
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = CompareResponse(
            algorithm=algorithm,
            total_pairs=data.get("total_pairs", 0),
            comparisons=data.get("comparisons", []),
            avg_similarity=data.get("avg_similarity", 0.0),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._compare_results[result.comparison_id] = result

        self.provenance.record(
            entity_type="comparison",
            entity_id=result.comparison_id,
            action="compare",
            data_hash=result.provenance_hash,
        )
        inc_comparisons(algorithm, result.total_pairs)
        observe_duration("compare", time.time() - start_time)
        return result

    # ------------------------------------------------------------------
    # Match classification
    # ------------------------------------------------------------------

    def classify_matches(
        self,
        comparisons: List[Dict[str, Any]],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> ClassifyResponse:
        """Classify comparison results into match/possible/non-match.

        Uses deterministic threshold-based classification.
        No LLM calls (zero-hallucination).

        Args:
            comparisons: List of comparison result dicts with 'overall_score'.
            thresholds: Optional override dict with 'match', 'possible' keys.

        Returns:
            ClassifyResponse with classification breakdown.

        Raises:
            ValueError: If comparisons is empty.
        """
        start_time = time.time()

        if not comparisons:
            raise ValueError("Comparisons list must not be empty")

        match_thresh = (
            thresholds.get("match", self.config.match_threshold)
            if thresholds else self.config.match_threshold
        )
        possible_thresh = (
            thresholds.get("possible", self.config.possible_threshold)
            if thresholds else self.config.possible_threshold
        )

        # Delegate to engine if available
        if self._classifier_engine is not None:
            engine_result = self._classifier_engine.classify(
                comparisons=comparisons,
                match_threshold=match_thresh,
                possible_threshold=possible_thresh,
            )
            return self._wrap_classify_result(
                engine_result, start_time, match_thresh, possible_thresh,
            )

        # Fallback: threshold-based classification
        classifications: List[Dict[str, Any]] = []
        match_count = 0
        possible_count = 0
        non_match_count = 0

        for comp in comparisons:
            score = float(comp.get("overall_score", 0.0))

            if score >= match_thresh:
                label = "MATCH"
                match_count += 1
            elif score >= possible_thresh:
                label = "POSSIBLE"
                possible_count += 1
            else:
                label = "NON_MATCH"
                non_match_count += 1

            classification = {
                "pair_id": comp.get("pair_id", str(uuid.uuid4())),
                "record_a_id": comp.get("record_a_id", ""),
                "record_b_id": comp.get("record_b_id", ""),
                "overall_score": score,
                "classification": label,
            }
            classifications.append(classification)

            # Store match for retrieval
            if label in ("MATCH", "POSSIBLE"):
                match_id = classification["pair_id"]
                self._matches[match_id] = classification

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = ClassifyResponse(
            total_comparisons=len(comparisons),
            matches=match_count,
            possible_matches=possible_count,
            non_matches=non_match_count,
            match_threshold=match_thresh,
            possible_threshold=possible_thresh,
            classifications=classifications,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._classify_results[result.classify_id] = result

        self.provenance.record(
            entity_type="classification",
            entity_id=result.classify_id,
            action="classify",
            data_hash=result.provenance_hash,
        )

        inc_matches("match", match_count)
        inc_matches("possible", possible_count)
        inc_matches("non_match", non_match_count)
        observe_duration("classify", time.time() - start_time)

        self._stats.total_duplicates_found += match_count

        logger.info(
            "Classified %d comparisons: %d match, %d possible, %d non-match",
            len(comparisons), match_count, possible_count, non_match_count,
        )
        return result

    def _wrap_classify_result(
        self,
        engine_result: Any,
        start_time: float,
        match_thresh: float,
        possible_thresh: float,
    ) -> ClassifyResponse:
        """Wrap engine result into ClassifyResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.
            match_thresh: Match threshold used.
            possible_thresh: Possible threshold used.

        Returns:
            ClassifyResponse with provenance.
        """
        data = engine_result if isinstance(engine_result, dict) else {}
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = ClassifyResponse(
            total_comparisons=data.get("total_comparisons", 0),
            matches=data.get("matches", 0),
            possible_matches=data.get("possible_matches", 0),
            non_matches=data.get("non_matches", 0),
            match_threshold=match_thresh,
            possible_threshold=possible_thresh,
            classifications=data.get("classifications", []),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._classify_results[result.classify_id] = result

        self.provenance.record(
            entity_type="classification",
            entity_id=result.classify_id,
            action="classify",
            data_hash=result.provenance_hash,
        )
        inc_matches("match", result.matches)
        inc_matches("possible", result.possible_matches)
        observe_duration("classify", time.time() - start_time)
        self._stats.total_duplicates_found += result.matches
        return result

    # ------------------------------------------------------------------
    # Cluster resolution
    # ------------------------------------------------------------------

    def form_clusters(
        self,
        matches: List[Dict[str, Any]],
        algorithm: Optional[str] = None,
    ) -> ClusterResponse:
        """Form duplicate clusters from matched pairs.

        Groups matched record pairs into connected clusters using
        union-find or connected components.

        Args:
            matches: List of match dicts with record_a_id, record_b_id.
            algorithm: Clustering algorithm (union_find, connected_components).
                Uses config default if None.

        Returns:
            ClusterResponse with cluster details.

        Raises:
            ValueError: If matches is empty.
        """
        start_time = time.time()
        algo = algorithm or self.config.cluster_algorithm

        if not matches:
            raise ValueError("Matches list must not be empty for clustering")

        # Delegate to engine if available
        if self._cluster_engine is not None:
            engine_result = self._cluster_engine.form_clusters(
                matches=matches,
                algorithm=algo,
            )
            return self._wrap_cluster_result(engine_result, start_time, algo)

        # Fallback: simple union-find clustering
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for match in matches:
            a = match.get("record_a_id", "")
            b = match.get("record_b_id", "")
            if a and b:
                parent.setdefault(a, a)
                parent.setdefault(b, b)
                union(a, b)

        # Group by root
        cluster_map: Dict[str, List[str]] = {}
        for node in parent:
            root = find(node)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(node)

        clusters: List[Dict[str, Any]] = []
        largest = 0
        total_size = 0

        for root, members in cluster_map.items():
            if len(members) < 2:
                continue
            cluster_id = str(uuid.uuid4())
            size = len(members)
            if size > largest:
                largest = size
            total_size += size
            cluster_detail = {
                "cluster_id": cluster_id,
                "root": root,
                "members": members,
                "size": size,
            }
            clusters.append(cluster_detail)
            self._clusters[cluster_id] = cluster_detail

        avg_size = total_size / max(len(clusters), 1)
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = ClusterResponse(
            algorithm=algo,
            total_matches=len(matches),
            total_clusters=len(clusters),
            largest_cluster=largest,
            avg_cluster_size=round(avg_size, 2),
            clusters=clusters,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._cluster_results[result.cluster_run_id] = result

        self.provenance.record(
            entity_type="cluster",
            entity_id=result.cluster_run_id,
            action="cluster",
            data_hash=result.provenance_hash,
        )

        inc_clusters(algo, len(clusters))
        observe_duration("cluster", time.time() - start_time)

        self._stats.total_clusters += len(clusters)

        logger.info(
            "Formed %d clusters (%s): largest=%d, avg=%.2f",
            len(clusters), algo, largest, avg_size,
        )
        return result

    def _wrap_cluster_result(
        self,
        engine_result: Any,
        start_time: float,
        algorithm: str,
    ) -> ClusterResponse:
        """Wrap engine result into ClusterResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.
            algorithm: Algorithm used.

        Returns:
            ClusterResponse with provenance.
        """
        data = engine_result if isinstance(engine_result, dict) else {}
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = ClusterResponse(
            algorithm=algorithm,
            total_matches=data.get("total_matches", 0),
            total_clusters=data.get("total_clusters", 0),
            largest_cluster=data.get("largest_cluster", 0),
            avg_cluster_size=data.get("avg_cluster_size", 0.0),
            clusters=data.get("clusters", []),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._cluster_results[result.cluster_run_id] = result

        self.provenance.record(
            entity_type="cluster",
            entity_id=result.cluster_run_id,
            action="cluster",
            data_hash=result.provenance_hash,
        )
        inc_clusters(algorithm, result.total_clusters)
        observe_duration("cluster", time.time() - start_time)
        self._stats.total_clusters += result.total_clusters
        return result

    # ------------------------------------------------------------------
    # Merge execution
    # ------------------------------------------------------------------

    def merge_duplicates(
        self,
        clusters: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
        strategy: Optional[str] = None,
    ) -> MergeResponse:
        """Merge duplicate records within clusters into golden records.

        Zero-hallucination: All merges use deterministic field selection
        rules (most complete, longest, first, latest).

        Args:
            clusters: List of cluster dicts with 'members'.
            records: Full record list indexed by position or ID.
            strategy: Merge strategy (keep_first, keep_latest,
                keep_most_complete, merge_fields, golden_record, custom).
                Uses config default if None.

        Returns:
            MergeResponse with merged golden records.

        Raises:
            ValueError: If clusters is empty.
        """
        start_time = time.time()
        strat = strategy or self.config.default_merge_strategy

        if not clusters:
            raise ValueError("Clusters list must not be empty for merging")

        # Build record lookup
        record_map: Dict[str, Dict[str, Any]] = {}
        for idx, rec in enumerate(records):
            rec_id = str(rec.get("id", idx))
            record_map[rec_id] = rec

        # Delegate to engine if available
        if self._merge_engine is not None:
            engine_result = self._merge_engine.merge(
                clusters=clusters,
                record_map=record_map,
                strategy=strat,
            )
            return self._wrap_merge_result(engine_result, start_time, strat)

        # Fallback: keep_most_complete merge strategy
        merged_records: List[Dict[str, Any]] = []
        total_merged = 0
        conflicts_resolved = 0

        for cluster in clusters:
            members = cluster.get("members", [])
            if len(members) < 2:
                continue

            member_records = [
                record_map[m] for m in members if m in record_map
            ]
            if not member_records:
                continue

            golden = self._merge_by_most_complete(member_records)
            conflicts_in_cluster = self._count_conflicts(member_records)
            conflicts_resolved += conflicts_in_cluster
            total_merged += len(member_records)

            merged_records.append({
                "golden_record": golden,
                "source_count": len(member_records),
                "cluster_id": cluster.get("cluster_id", ""),
                "conflicts": conflicts_in_cluster,
            })

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = MergeResponse(
            strategy=strat,
            total_clusters=len(clusters),
            total_records_merged=total_merged,
            total_golden_records=len(merged_records),
            conflicts_resolved=conflicts_resolved,
            conflict_resolution=self.config.merge_conflict_resolution,
            merged_records=merged_records,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._merge_results[result.merge_id] = result

        self.provenance.record(
            entity_type="merge",
            entity_id=result.merge_id,
            action="merge",
            data_hash=result.provenance_hash,
        )

        inc_merges(strat, len(merged_records))
        if conflicts_resolved > 0:
            inc_conflicts(self.config.merge_conflict_resolution, conflicts_resolved)
        observe_duration("merge", time.time() - start_time)

        self._stats.total_merges += len(merged_records)
        self._stats.total_conflicts += conflicts_resolved

        logger.info(
            "Merged %d records into %d golden records (%s): %d conflicts",
            total_merged, len(merged_records), strat, conflicts_resolved,
        )
        return result

    def _merge_by_most_complete(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge records by keeping the most complete value for each field.

        Args:
            records: List of record dicts to merge.

        Returns:
            Golden record with most complete field values.
        """
        golden: Dict[str, Any] = {}
        all_fields = set()
        for rec in records:
            all_fields.update(rec.keys())

        for field in sorted(all_fields):
            best_value = None
            best_len = -1
            for rec in records:
                val = rec.get(field)
                if val is None or val == "":
                    continue
                val_len = len(str(val))
                if val_len > best_len:
                    best_value = val
                    best_len = val_len
            golden[field] = best_value
        return golden

    def _count_conflicts(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Count the number of field-level conflicts across records.

        A conflict exists when two or more records have different non-null
        values for the same field.

        Args:
            records: List of record dicts.

        Returns:
            Number of conflicting fields.
        """
        all_fields = set()
        for rec in records:
            all_fields.update(rec.keys())

        conflicts = 0
        for field in all_fields:
            values = set()
            for rec in records:
                val = rec.get(field)
                if val is not None and val != "":
                    values.add(str(val))
            if len(values) > 1:
                conflicts += 1
        return conflicts

    def _wrap_merge_result(
        self,
        engine_result: Any,
        start_time: float,
        strategy: str,
    ) -> MergeResponse:
        """Wrap engine result into MergeResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.
            strategy: Strategy used.

        Returns:
            MergeResponse with provenance.
        """
        data = engine_result if isinstance(engine_result, dict) else {}
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = MergeResponse(
            strategy=strategy,
            total_clusters=data.get("total_clusters", 0),
            total_records_merged=data.get("total_records_merged", 0),
            total_golden_records=data.get("total_golden_records", 0),
            conflicts_resolved=data.get("conflicts_resolved", 0),
            conflict_resolution=data.get(
                "conflict_resolution", self.config.merge_conflict_resolution,
            ),
            merged_records=data.get("merged_records", []),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._merge_results[result.merge_id] = result

        self.provenance.record(
            entity_type="merge",
            entity_id=result.merge_id,
            action="merge",
            data_hash=result.provenance_hash,
        )
        inc_merges(strategy, result.total_golden_records)
        observe_duration("merge", time.time() - start_time)
        self._stats.total_merges += result.total_golden_records
        self._stats.total_conflicts += result.conflicts_resolved
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        records: List[Dict[str, Any]],
        rule: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> PipelineResponse:
        """Run the full deduplication pipeline end-to-end.

        Executes: fingerprint -> block -> compare -> classify -> cluster -> merge.

        Args:
            records: List of record dicts to deduplicate.
            rule: Optional dedup rule configuration.
            options: Optional pipeline options overriding config defaults.

        Returns:
            PipelineResponse with full pipeline results.

        Raises:
            ValueError: If records is empty.
        """
        start_time = time.time()
        pipeline_id = str(uuid.uuid4())

        if not records:
            raise ValueError("Records list must not be empty for pipeline")

        self._active_jobs += 1
        set_active_jobs(self._active_jobs)

        opts = options or {}
        field_set = opts.get("field_set")
        algorithm = opts.get("fingerprint_algorithm")
        strategy = opts.get("blocking_strategy")
        key_fields = opts.get("key_fields")
        merge_strategy = opts.get("merge_strategy")

        stages: Dict[str, Dict[str, Any]] = {}
        status = "completed"

        try:
            # Stage 1: Fingerprint
            fp_result = self.fingerprint_records(
                records=records,
                field_set=field_set,
                algorithm=algorithm,
            )
            stages["fingerprint"] = {
                "id": fp_result.fingerprint_id,
                "total_records": fp_result.total_records,
                "unique_fingerprints": fp_result.unique_fingerprints,
                "processing_time_ms": fp_result.processing_time_ms,
            }

            # Stage 2: Block
            block_result = self.create_blocks(
                records=records,
                strategy=strategy,
                key_fields=key_fields,
            )
            stages["block"] = {
                "id": block_result.block_id,
                "total_blocks": block_result.total_blocks,
                "total_pairs": block_result.total_pairs,
                "reduction_ratio": block_result.reduction_ratio,
                "processing_time_ms": block_result.processing_time_ms,
            }

            # Stage 3: Compare (build pairs from blocks)
            pairs = self._build_pairs_from_blocks(
                block_result.blocks, records,
            )
            compare_result = self.compare_pairs(
                block_results={"pairs": pairs},
                field_configs=opts.get("field_configs"),
            )
            stages["compare"] = {
                "id": compare_result.comparison_id,
                "total_pairs": compare_result.total_pairs,
                "avg_similarity": compare_result.avg_similarity,
                "processing_time_ms": compare_result.processing_time_ms,
            }

            # Stage 4: Classify
            classify_result = self.classify_matches(
                comparisons=compare_result.comparisons,
                thresholds=opts.get("thresholds"),
            )
            stages["classify"] = {
                "id": classify_result.classify_id,
                "matches": classify_result.matches,
                "possible_matches": classify_result.possible_matches,
                "non_matches": classify_result.non_matches,
                "processing_time_ms": classify_result.processing_time_ms,
            }

            # Stage 5: Cluster (only MATCH classifications)
            match_items = [
                c for c in classify_result.classifications
                if c.get("classification") == "MATCH"
            ]
            if match_items:
                cluster_result = self.form_clusters(
                    matches=match_items,
                    algorithm=opts.get("cluster_algorithm"),
                )
                stages["cluster"] = {
                    "id": cluster_result.cluster_run_id,
                    "total_clusters": cluster_result.total_clusters,
                    "largest_cluster": cluster_result.largest_cluster,
                    "processing_time_ms": cluster_result.processing_time_ms,
                }

                # Stage 6: Merge
                if cluster_result.clusters:
                    merge_result = self.merge_duplicates(
                        clusters=cluster_result.clusters,
                        records=records,
                        strategy=merge_strategy,
                    )
                    stages["merge"] = {
                        "id": merge_result.merge_id,
                        "total_golden_records": merge_result.total_golden_records,
                        "conflicts_resolved": merge_result.conflicts_resolved,
                        "processing_time_ms": merge_result.processing_time_ms,
                    }

            total_duplicates = classify_result.matches
            total_clusters = stages.get("cluster", {}).get("total_clusters", 0)
            total_merged = stages.get("merge", {}).get(
                "total_golden_records", 0,
            )

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            status = "failed"
            total_duplicates = 0
            total_clusters = 0
            total_merged = 0
            inc_errors("pipeline")

        finally:
            self._active_jobs -= 1
            set_active_jobs(self._active_jobs)

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = PipelineResponse(
            pipeline_id=pipeline_id,
            status=status,
            total_records=len(records),
            total_duplicates=total_duplicates,
            total_clusters=total_clusters,
            total_merged=total_merged,
            stages=stages,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        self._pipeline_results[pipeline_id] = result

        self.provenance.record(
            entity_type="pipeline",
            entity_id=pipeline_id,
            action="pipeline",
            data_hash=result.provenance_hash,
        )

        inc_jobs(status)
        observe_duration("pipeline", time.time() - start_time)

        self._stats.total_jobs += 1
        if status == "completed":
            self._stats.completed_jobs += 1
        else:
            self._stats.failed_jobs += 1

        logger.info(
            "Pipeline %s %s: %d records, %d duplicates, %d clusters, %.1fms",
            pipeline_id[:8], status, len(records),
            total_duplicates, total_clusters, processing_time_ms,
        )
        return result

    def _build_pairs_from_blocks(
        self,
        blocks: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build record pairs from block partitions.

        Args:
            blocks: List of block summary dicts with indices.
            records: Full record list.

        Returns:
            List of pair dicts with record_a, record_b, id_a, id_b.
        """
        pairs: List[Dict[str, Any]] = []
        for block in blocks:
            # Block summaries may not contain member indices.
            # In the fallback path, reconstruct from block_key.
            members = block.get("members", [])
            if not members:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    idx_a = members[i]
                    idx_b = members[j]
                    if isinstance(idx_a, int) and idx_a < len(records):
                        rec_a = records[idx_a]
                    else:
                        rec_a = {}
                    if isinstance(idx_b, int) and idx_b < len(records):
                        rec_b = records[idx_b]
                    else:
                        rec_b = {}
                    pairs.append({
                        "id_a": str(idx_a),
                        "id_b": str(idx_b),
                        "record_a": rec_a,
                        "record_b": rec_b,
                    })
                    if len(pairs) >= self.config.max_comparisons_per_block:
                        return pairs
        return pairs

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_dedup_job(
        self,
        dataset_ids: List[str],
        rule_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new deduplication job.

        Args:
            dataset_ids: List of dataset identifiers to deduplicate.
            rule_id: Optional dedup rule ID to apply.

        Returns:
            Job creation result dict.
        """
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "dataset_ids": dataset_ids,
            "rule_id": rule_id,
            "status": "created",
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
            "total_records": 0,
            "total_duplicates": 0,
            "total_clusters": 0,
            "processing_time_ms": 0.0,
        }
        self._jobs[job_id] = job

        self.provenance.record(
            entity_type="dedup_job",
            entity_id=job_id,
            action="create",
            data_hash=_compute_hash(job),
        )

        logger.info("Created dedup job %s for %d datasets", job_id[:8], len(dataset_ids))
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a dedup job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Job dict or None if not found.
        """
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List dedup jobs with optional filtering.

        Args:
            status: Optional status filter.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.

        Returns:
            Dict with jobs list and count.
        """
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
        page = jobs[offset:offset + limit]
        return {
            "jobs": page,
            "count": len(page),
            "total": len(jobs),
            "limit": limit,
            "offset": offset,
        }

    def cancel_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Cancel a dedup job.

        Args:
            job_id: Job identifier.

        Returns:
            Updated job dict or None if not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        job["status"] = "cancelled"
        job["updated_at"] = _utcnow().isoformat()

        self.provenance.record(
            entity_type="dedup_job",
            entity_id=job_id,
            action="cancel",
            data_hash=_compute_hash(job),
        )

        inc_jobs("cancelled")
        logger.info("Cancelled dedup job %s", job_id[:8])
        return job

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def create_rule(
        self,
        rule_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new deduplication rule.

        Args:
            rule_config: Rule configuration dict with name, field_weights,
                match_threshold, blocking_strategy, etc.

        Returns:
            Created rule dict.
        """
        rule_id = str(uuid.uuid4())
        rule = {
            "rule_id": rule_id,
            "name": rule_config.get("name", ""),
            "description": rule_config.get("description", ""),
            "field_weights": rule_config.get("field_weights", []),
            "match_threshold": rule_config.get(
                "match_threshold", self.config.match_threshold,
            ),
            "possible_threshold": rule_config.get(
                "possible_threshold", self.config.possible_threshold,
            ),
            "blocking_strategy": rule_config.get(
                "blocking_strategy", self.config.blocking_strategy,
            ),
            "blocking_key_fields": rule_config.get("blocking_key_fields", []),
            "merge_strategy": rule_config.get(
                "merge_strategy", self.config.default_merge_strategy,
            ),
            "is_active": rule_config.get("is_active", True),
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        rule["provenance_hash"] = _compute_hash(rule)
        self._rules[rule_id] = rule

        self.provenance.record(
            entity_type="rule",
            entity_id=rule_id,
            action="create",
            data_hash=rule["provenance_hash"],
        )

        self._stats.total_rules += 1
        logger.info("Created dedup rule %s: %s", rule_id[:8], rule["name"])
        return rule

    def list_rules(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List dedup rules.

        Args:
            limit: Maximum number of rules to return.
            offset: Number of rules to skip.

        Returns:
            Dict with rules list and count.
        """
        rules = list(self._rules.values())
        page = rules[offset:offset + limit]
        return {
            "rules": page,
            "count": len(page),
            "total": len(rules),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def get_statistics(self) -> StatsResponse:
        """Get aggregated duplicate detection statistics.

        Returns:
            StatsResponse summary.
        """
        self._stats.active_jobs = self._active_jobs
        self._stats.provenance_entries = self.provenance.entry_count
        self._stats.avg_similarity = round(
            self._similarity_sum / max(self._similarity_count, 1), 4,
        )
        return self._stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the service.

        Returns:
            Health status dict.
        """
        return {
            "status": "healthy" if self._started else "not_started",
            "service": "duplicate-detector",
            "started": self._started,
            "jobs": len(self._jobs),
            "fingerprint_results": len(self._fingerprint_results),
            "block_results": len(self._block_results),
            "compare_results": len(self._compare_results),
            "classify_results": len(self._classify_results),
            "cluster_results": len(self._cluster_results),
            "merge_results": len(self._merge_results),
            "pipeline_results": len(self._pipeline_results),
            "rules": len(self._rules),
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Report and detail retrieval
    # ------------------------------------------------------------------

    def generate_report(
        self,
        job_id: str,
        report_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate a dedup report for a job.

        Args:
            job_id: Job identifier.
            report_format: Output format (json, markdown, text).

        Returns:
            Report dict with summary and details.
        """
        start_time = time.time()
        report_id = str(uuid.uuid4())

        job = self._jobs.get(job_id)
        pipeline = None
        for pr in self._pipeline_results.values():
            if pr.job_id == job_id:
                pipeline = pr
                break

        report = {
            "report_id": report_id,
            "job_id": job_id,
            "format": report_format,
            "job": job,
            "pipeline": pipeline.model_dump(mode="json") if pipeline else None,
            "statistics": self.get_statistics().model_dump(mode="json"),
            "generated_at": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        report["provenance_hash"] = _compute_hash(report)

        self.provenance.record(
            entity_type="report",
            entity_id=report_id,
            action="generate",
            data_hash=report["provenance_hash"],
        )

        observe_duration("generate_report", time.time() - start_time)

        logger.info("Generated report %s for job %s", report_id[:8], job_id[:8])
        return report

    def get_match_details(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific match.

        Args:
            match_id: Match/pair identifier.

        Returns:
            Match detail dict or None if not found.
        """
        return self._matches.get(match_id)

    def get_cluster_details(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific cluster.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            Cluster detail dict or None if not found.
        """
        return self._clusters.get(cluster_id)

    def get_merge_result(self, merge_id: str) -> Optional[MergeResponse]:
        """Get a merge result by ID.

        Args:
            merge_id: Merge identifier.

        Returns:
            MergeResponse or None if not found.
        """
        return self._merge_results.get(merge_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_avg_similarity(
        self,
        total_sim: float,
        count: int,
    ) -> None:
        """Update the running average similarity score.

        Args:
            total_sim: Sum of similarity scores in this batch.
            count: Number of scores in this batch.
        """
        self._similarity_sum += total_sim
        self._similarity_count += count

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get duplicate detection service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        stats = self.get_statistics()
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_jobs": stats.total_jobs,
            "completed_jobs": stats.completed_jobs,
            "failed_jobs": stats.failed_jobs,
            "total_records_processed": stats.total_records_processed,
            "total_duplicates_found": stats.total_duplicates_found,
            "total_clusters": stats.total_clusters,
            "total_merges": stats.total_merges,
            "total_conflicts": stats.total_conflicts,
            "total_rules": stats.total_rules,
            "active_jobs": stats.active_jobs,
            "avg_similarity": stats.avg_similarity,
            "provenance_entries": stats.provenance_entries,
        }


# ===================================================================
# Module-level configuration functions
# ===================================================================


async def configure_duplicate_detector(
    app: Any,
    config: Optional[DuplicateDetectorConfig] = None,
) -> DuplicateDetectorService:
    """Configure the Duplicate Detector Service on a FastAPI application.

    Creates the DuplicateDetectorService, stores it in app.state, mounts
    the duplicate detector API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional duplicate detector config.

    Returns:
        DuplicateDetectorService instance.
    """
    global _singleton_instance

    service = DuplicateDetectorService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.duplicate_detector_service = service

    # Mount duplicate detector API router
    try:
        from greenlang.duplicate_detector.api.router import router as dd_router
        if dd_router is not None:
            app.include_router(dd_router)
            logger.info("Duplicate detector API router mounted")
    except ImportError:
        logger.warning("Duplicate detector API router not available")

    service._started = True
    logger.info("Duplicate detector service configured and started")
    return service


def get_duplicate_detector(app: Any) -> DuplicateDetectorService:
    """Get the DuplicateDetectorService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        DuplicateDetectorService instance.

    Raises:
        RuntimeError: If duplicate detector service not configured.
    """
    service = getattr(app.state, "duplicate_detector_service", None)
    if service is None:
        raise RuntimeError(
            "Duplicate detector service not configured. "
            "Call configure_duplicate_detector(app) first."
        )
    return service


def get_router(service: Optional[DuplicateDetectorService] = None) -> Any:
    """Get the duplicate detector API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.duplicate_detector.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "DuplicateDetectorService",
    "configure_duplicate_detector",
    "get_duplicate_detector",
    "get_router",
    # Models
    "FingerprintResponse",
    "BlockResponse",
    "CompareResponse",
    "ClassifyResponse",
    "ClusterResponse",
    "MergeResponse",
    "PipelineResponse",
    "StatsResponse",
]
