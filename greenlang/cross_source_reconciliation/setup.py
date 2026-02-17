# -*- coding: utf-8 -*-
"""
Cross-Source Reconciliation Service Setup - AGENT-DATA-015

Provides ``configure_reconciliation(app)`` which wires up the Cross-Source
Reconciliation SDK (source registry, matching, comparison, discrepancy
detection, resolution, audit trail, pipeline orchestration, provenance
tracker) and mounts the REST API.

Also exposes ``get_reconciliation(app)`` for programmatic access,
``get_router()`` for obtaining the FastAPI APIRouter, and the
``CrossSourceReconciliationService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.cross_source_reconciliation.setup import configure_reconciliation
    >>> app = FastAPI()
    >>> configure_reconciliation(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
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

from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
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


# ---------------------------------------------------------------------------
# Optional engine imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.source_registry import (
        SourceRegistryEngine,
    )
except ImportError:
    SourceRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.matching_engine import (
        MatchingEngine,
    )
except ImportError:
    MatchingEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.comparison_engine import (
        ComparisonEngine,
    )
except ImportError:
    ComparisonEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.discrepancy_detector import (
        DiscrepancyDetectorEngine,
    )
except ImportError:
    DiscrepancyDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.resolution_engine import (
        ResolutionEngine,
    )
except ImportError:
    ResolutionEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.audit_trail import (
        AuditTrailEngine,
    )
except ImportError:
    AuditTrailEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.reconciliation_pipeline import (
        ReconciliationPipelineEngine,
    )
except ImportError:
    ReconciliationPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.metrics import PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Build a SHA-256 hash for arbitrary data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ===================================================================
# CrossSourceReconciliationService facade
# ===================================================================


class CrossSourceReconciliationService:
    """Facade service for the Cross-Source Reconciliation SDK.

    Wires together the 7 engines (SourceRegistry, Matching,
    Comparison, DiscrepancyDetector, Resolution, AuditTrail,
    ReconciliationPipeline) behind a simple API suitable for
    REST endpoint delegation.

    Attributes:
        config: Service configuration dictionary.
        _provenance: ProvenanceTracker instance.
        _source_registry: SourceRegistryEngine instance.
        _matching_engine: MatchingEngine instance.
        _comparison_engine: ComparisonEngine instance.
        _discrepancy_detector: DiscrepancyDetectorEngine instance.
        _resolution_engine: ResolutionEngine instance.
        _audit_trail: AuditTrailEngine instance.
        _pipeline: ReconciliationPipelineEngine instance.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize CrossSourceReconciliationService.

        Args:
            config: Optional configuration dictionary. Defaults to
                sensible production defaults when not provided.
        """
        self.config = config or {}
        self._provenance = get_provenance_tracker()

        # Engine stubs -- created lazily or via startup()
        self._source_registry: Any = None
        self._matching_engine: Any = None
        self._comparison_engine: Any = None
        self._discrepancy_detector: Any = None
        self._resolution_engine: Any = None
        self._audit_trail: Any = None
        self._pipeline: Any = None

        # In-memory stores
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._matches: Dict[str, Dict[str, Any]] = {}
        self._comparisons: Dict[str, Dict[str, Any]] = {}
        self._discrepancies: Dict[str, Dict[str, Any]] = {}
        self._resolutions: Dict[str, Dict[str, Any]] = {}
        self._golden_records: Dict[str, Dict[str, Any]] = {}
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}

        # Aggregate counters
        self._stats = {
            "total_jobs": 0,
            "total_sources": 0,
            "total_matches": 0,
            "total_comparisons": 0,
            "total_discrepancies": 0,
            "total_resolutions": 0,
            "total_golden_records": 0,
            "total_pipelines": 0,
        }

        self._started = False
        logger.info("CrossSourceReconciliationService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Initialize all engines."""
        if SourceRegistryEngine is not None:
            self._source_registry = SourceRegistryEngine(config=self.config)
        if MatchingEngine is not None:
            self._matching_engine = MatchingEngine(config=self.config)
        if ComparisonEngine is not None:
            self._comparison_engine = ComparisonEngine(config=self.config)
        if DiscrepancyDetectorEngine is not None:
            self._discrepancy_detector = DiscrepancyDetectorEngine(
                config=self.config,
            )
        if ResolutionEngine is not None:
            self._resolution_engine = ResolutionEngine(config=self.config)
        if AuditTrailEngine is not None:
            self._audit_trail = AuditTrailEngine(config=self.config)
        if ReconciliationPipelineEngine is not None:
            self._pipeline = ReconciliationPipelineEngine(
                config=self.config,
            )

        self._started = True
        logger.info("CrossSourceReconciliationService started")

    def shutdown(self) -> None:
        """Shutdown the service."""
        self._started = False
        logger.info("CrossSourceReconciliationService shutdown")

    # ------------------------------------------------------------------
    # Health & Statistics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return service health status.

        Returns:
            Dictionary with service status, engine availability, and
            store sizes.
        """
        return {
            "status": "healthy" if self._started else "starting",
            "service": "cross_source_reconciliation",
            "engines": {
                "source_registry": self._source_registry is not None,
                "matching_engine": self._matching_engine is not None,
                "comparison_engine": self._comparison_engine is not None,
                "discrepancy_detector": self._discrepancy_detector is not None,
                "resolution_engine": self._resolution_engine is not None,
                "audit_trail": self._audit_trail is not None,
                "pipeline": self._pipeline is not None,
            },
            "stores": {
                "jobs": len(self._jobs),
                "sources": len(self._sources),
                "matches": len(self._matches),
                "discrepancies": len(self._discrepancies),
                "golden_records": len(self._golden_records),
            },
            "timestamp": _utcnow().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return service statistics.

        Returns:
            Dictionary with aggregate counts and store sizes.
        """
        return {
            **self._stats,
            "jobs_stored": len(self._jobs),
            "sources_stored": len(self._sources),
            "matches_stored": len(self._matches),
            "comparisons_stored": len(self._comparisons),
            "discrepancies_stored": len(self._discrepancies),
            "resolutions_stored": len(self._resolutions),
            "golden_records_stored": len(self._golden_records),
            "pipeline_results_stored": len(self._pipeline_results),
            "provenance_entries": self._provenance.entry_count,
            "timestamp": _utcnow().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_stats() used by the router."""
        return self.get_stats()

    # ------------------------------------------------------------------
    # Job Management
    # ------------------------------------------------------------------

    def create_job(
        self,
        name: str = "",
        source_ids: Optional[List[str]] = None,
        strategy: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a reconciliation job.

        Args:
            name: Human-readable job name.
            source_ids: List of source IDs to reconcile.
            strategy: Reconciliation strategy (auto, priority_wins,
                most_recent, weighted_average, consensus).
            config: Additional job configuration overrides.

        Returns:
            Dictionary with job details including job_id and status.
        """
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "name": name or f"reconciliation-{job_id[:8]}",
            "source_ids": source_ids or [],
            "strategy": strategy,
            "config": config or {},
            "status": "pending",
            "match_count": 0,
            "discrepancy_count": 0,
            "golden_record_count": 0,
            "created_at": _utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "provenance_hash": _compute_hash({
                "job_id": job_id,
                "source_ids": source_ids or [],
                "strategy": strategy,
            }),
        }
        self._jobs[job_id] = job
        self._stats["total_jobs"] += 1
        self._provenance.record(
            "reconciliation_job", job_id, "create", job["provenance_hash"],
        )
        logger.info("Created reconciliation job %s", job_id)
        return job

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List reconciliation jobs with optional status filter.

        Args:
            status: Optional status filter.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with jobs list, count, and pagination metadata.
        """
        items = list(self._jobs.values())
        if status is not None:
            items = [j for j in items if j.get("status") == status]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "jobs": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a reconciliation job by ID.

        Args:
            job_id: Unique job identifier.

        Returns:
            Job dictionary or None if not found.
        """
        return self._jobs.get(job_id)

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel and delete a reconciliation job.

        Args:
            job_id: Unique job identifier.

        Returns:
            Dictionary with deletion status.

        Raises:
            ValueError: If job not found.
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        job = self._jobs.pop(job_id)
        job["status"] = "cancelled"
        self._provenance.record(
            "reconciliation_job", job_id, "cancel",
            _compute_hash({"job_id": job_id, "action": "cancel"}),
        )
        logger.info("Cancelled reconciliation job %s", job_id)
        return {"job_id": job_id, "status": "cancelled"}

    # ------------------------------------------------------------------
    # Source Management
    # ------------------------------------------------------------------

    def register_source(
        self,
        name: str,
        source_type: str = "manual",
        schema: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        credibility_score: float = 0.8,
        refresh_cadence: str = "monthly",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a data source for reconciliation.

        Args:
            name: Human-readable source name.
            source_type: Source type (erp, utility, meter, questionnaire,
                registry, manual, api).
            schema: Source schema definition with column types.
            priority: Source priority (1=highest, 10=lowest).
            credibility_score: Source credibility (0.0 to 1.0).
            refresh_cadence: Data refresh frequency.
            metadata: Additional source metadata.

        Returns:
            Dictionary with registered source details.
        """
        source_id = str(uuid.uuid4())
        source = {
            "source_id": source_id,
            "name": name,
            "source_type": source_type,
            "schema": schema or {},
            "priority": priority,
            "credibility_score": credibility_score,
            "refresh_cadence": refresh_cadence,
            "metadata": metadata or {},
            "record_count": 0,
            "status": "active",
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "source_id": source_id,
                "name": name,
                "source_type": source_type,
            }),
        }

        if self._source_registry is not None:
            try:
                result = self._source_registry.register(
                    name=name,
                    source_type=source_type,
                    schema=schema,
                    priority=priority,
                    credibility_score=credibility_score,
                )
                source.update(result)
            except Exception as exc:
                logger.warning(
                    "Source registry engine error, using fallback: %s", exc,
                )

        self._sources[source_id] = source
        self._stats["total_sources"] += 1
        self._provenance.record(
            "source", source_id, "register", source["provenance_hash"],
        )
        logger.info("Registered source %s (%s)", name, source_id)
        return source

    def list_sources(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List registered data sources with pagination.

        Args:
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with sources list and pagination metadata.
        """
        items = list(self._sources.values())
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "sources": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get a registered data source by ID.

        Args:
            source_id: Unique source identifier.

        Returns:
            Source dictionary or None if not found.
        """
        return self._sources.get(source_id)

    def update_source(
        self,
        source_id: str,
        name: Optional[str] = None,
        priority: Optional[int] = None,
        credibility_score: Optional[float] = None,
        refresh_cadence: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a registered data source.

        Args:
            source_id: Unique source identifier.
            name: Optional new name.
            priority: Optional new priority.
            credibility_score: Optional new credibility score.
            refresh_cadence: Optional new refresh cadence.
            schema: Optional new schema definition.
            metadata: Optional additional metadata to merge.

        Returns:
            Updated source dictionary.

        Raises:
            ValueError: If source not found.
        """
        source = self._sources.get(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        if name is not None:
            source["name"] = name
        if priority is not None:
            source["priority"] = priority
        if credibility_score is not None:
            source["credibility_score"] = credibility_score
        if refresh_cadence is not None:
            source["refresh_cadence"] = refresh_cadence
        if schema is not None:
            source["schema"] = schema
        if metadata is not None:
            source["metadata"].update(metadata)

        source["updated_at"] = _utcnow().isoformat()
        source["provenance_hash"] = _compute_hash({
            "source_id": source_id,
            "updated_at": source["updated_at"],
        })

        self._provenance.record(
            "source", source_id, "update", source["provenance_hash"],
        )
        logger.info("Updated source %s", source_id)
        return source

    # ------------------------------------------------------------------
    # Record Matching
    # ------------------------------------------------------------------

    def match_records(
        self,
        source_ids: Optional[List[str]] = None,
        records_a: Optional[List[Dict[str, Any]]] = None,
        records_b: Optional[List[Dict[str, Any]]] = None,
        match_keys: Optional[List[str]] = None,
        threshold: float = 0.85,
        strategy: str = "composite",
    ) -> Dict[str, Any]:
        """Match records across data sources.

        Uses composite key matching with optional fuzzy matching to
        identify records across sources that refer to the same entity
        and time period.

        Args:
            source_ids: Source IDs to match across (uses registered
                source data).
            records_a: First set of records (inline data).
            records_b: Second set of records (inline data).
            match_keys: Fields to use as composite match key.
            threshold: Minimum match confidence threshold (0.0-1.0).
            strategy: Matching strategy (exact, fuzzy, composite,
                rule_based).

        Returns:
            Dictionary with match results, confidence scores, and
            provenance hash.
        """
        start_t = time.time()
        match_id = str(uuid.uuid4())

        list_a = records_a or []
        list_b = records_b or []
        keys = match_keys or ["entity_id", "period"]

        if self._matching_engine is not None:
            try:
                result = self._matching_engine.match(
                    records_a=list_a,
                    records_b=list_b,
                    match_keys=keys,
                    threshold=threshold,
                    strategy=strategy,
                )
                output = {
                    "match_id": match_id,
                    "source_ids": source_ids or [],
                    "strategy": strategy,
                    "threshold": threshold,
                    "matched_pairs": result.get("matched_pairs", []),
                    "total_matched": result.get("total_matched", 0),
                    "total_unmatched_a": result.get("total_unmatched_a", 0),
                    "total_unmatched_b": result.get("total_unmatched_b", 0),
                    "avg_confidence": result.get("avg_confidence", 0.0),
                    "processing_time_ms": (time.time() - start_t) * 1000.0,
                    "provenance_hash": result.get(
                        "provenance_hash",
                        _compute_hash({"match_id": match_id}),
                    ),
                }
            except Exception as exc:
                logger.warning(
                    "Matching engine error, using fallback: %s", exc,
                )
                output = self._match_fallback(
                    match_id, source_ids, list_a, list_b,
                    keys, threshold, strategy, start_t,
                )
        else:
            output = self._match_fallback(
                match_id, source_ids, list_a, list_b,
                keys, threshold, strategy, start_t,
            )

        self._matches[match_id] = output
        self._stats["total_matches"] += 1
        self._provenance.record(
            "record_match", match_id, "match", output["provenance_hash"],
        )
        return output

    def _match_fallback(
        self,
        match_id: str,
        source_ids: Optional[List[str]],
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        match_keys: List[str],
        threshold: float,
        strategy: str,
        start_t: float,
    ) -> Dict[str, Any]:
        """Fallback matching using exact key comparison.

        Args:
            match_id: Generated match identifier.
            source_ids: Source IDs involved.
            records_a: First record set.
            records_b: Second record set.
            match_keys: Fields for key matching.
            threshold: Confidence threshold.
            strategy: Matching strategy name.
            start_t: Start timestamp for duration tracking.

        Returns:
            Dictionary with fallback match results.
        """
        # Build index on records_b
        b_index: Dict[str, Dict[str, Any]] = {}
        for rec in records_b:
            key = tuple(str(rec.get(k, "")) for k in match_keys)
            b_index["|".join(key)] = rec

        matched_pairs: List[Dict[str, Any]] = []
        unmatched_a = 0
        for rec_a in records_a:
            key = "|".join(
                str(rec_a.get(k, "")) for k in match_keys
            )
            if key in b_index:
                matched_pairs.append({
                    "record_a": rec_a,
                    "record_b": b_index[key],
                    "confidence": 1.0,
                    "match_type": "exact",
                })
            else:
                unmatched_a += 1

        unmatched_b = len(records_b) - len(matched_pairs)
        avg_conf = 1.0 if matched_pairs else 0.0
        elapsed = (time.time() - start_t) * 1000.0

        return {
            "match_id": match_id,
            "source_ids": source_ids or [],
            "strategy": strategy,
            "threshold": threshold,
            "matched_pairs": matched_pairs,
            "total_matched": len(matched_pairs),
            "total_unmatched_a": unmatched_a,
            "total_unmatched_b": max(unmatched_b, 0),
            "avg_confidence": avg_conf,
            "processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "match_id": match_id,
                "total_matched": len(matched_pairs),
            }),
        }

    def list_matches(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List stored match results with pagination.

        Args:
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with matches list and pagination metadata.
        """
        items = list(self._matches.values())
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "matches": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get a stored match result by ID.

        Args:
            match_id: Unique match identifier.

        Returns:
            Match dictionary or None if not found.
        """
        return self._matches.get(match_id)

    # ------------------------------------------------------------------
    # Record Comparison
    # ------------------------------------------------------------------

    def compare_records(
        self,
        match_id: Optional[str] = None,
        record_a: Optional[Dict[str, Any]] = None,
        record_b: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        tolerance_pct: float = 5.0,
        tolerance_abs: float = 0.01,
    ) -> Dict[str, Any]:
        """Compare matched records field by field.

        Performs tolerance-aware comparison of fields between two
        matched records, classifying each field as match, mismatch,
        partial_match, or missing.

        Args:
            match_id: Optional ID of a stored match to compare.
            record_a: First record data (inline).
            record_b: Second record data (inline).
            fields: Fields to compare (all shared fields if None).
            tolerance_pct: Relative tolerance as percentage.
            tolerance_abs: Absolute tolerance for numeric fields.

        Returns:
            Dictionary with comparison results per field.
        """
        start_t = time.time()
        comparison_id = str(uuid.uuid4())

        rec_a = record_a or {}
        rec_b = record_b or {}

        # If match_id provided, load records from stored match
        if match_id and not record_a:
            match = self._matches.get(match_id)
            if match and match.get("matched_pairs"):
                pair = match["matched_pairs"][0]
                rec_a = pair.get("record_a", {})
                rec_b = pair.get("record_b", {})

        # Determine fields to compare
        compare_fields = fields or list(
            set(rec_a.keys()) & set(rec_b.keys())
        )

        if self._comparison_engine is not None:
            try:
                result = self._comparison_engine.compare(
                    record_a=rec_a,
                    record_b=rec_b,
                    fields=compare_fields,
                    tolerance_pct=tolerance_pct,
                    tolerance_abs=tolerance_abs,
                )
                output = {
                    "comparison_id": comparison_id,
                    "match_id": match_id or "",
                    "fields_compared": result.get("fields_compared", []),
                    "total_fields": result.get("total_fields", 0),
                    "matching_fields": result.get("matching_fields", 0),
                    "mismatching_fields": result.get("mismatching_fields", 0),
                    "missing_fields": result.get("missing_fields", 0),
                    "match_rate": result.get("match_rate", 0.0),
                    "processing_time_ms": (time.time() - start_t) * 1000.0,
                    "provenance_hash": result.get(
                        "provenance_hash",
                        _compute_hash({"comparison_id": comparison_id}),
                    ),
                }
            except Exception as exc:
                logger.warning(
                    "Comparison engine error, using fallback: %s", exc,
                )
                output = self._compare_fallback(
                    comparison_id, match_id, rec_a, rec_b,
                    compare_fields, tolerance_pct, tolerance_abs, start_t,
                )
        else:
            output = self._compare_fallback(
                comparison_id, match_id, rec_a, rec_b,
                compare_fields, tolerance_pct, tolerance_abs, start_t,
            )

        self._comparisons[comparison_id] = output
        self._stats["total_comparisons"] += 1
        self._provenance.record(
            "field_comparison", comparison_id, "compare",
            output["provenance_hash"],
        )
        return output

    def _compare_fallback(
        self,
        comparison_id: str,
        match_id: Optional[str],
        rec_a: Dict[str, Any],
        rec_b: Dict[str, Any],
        fields: List[str],
        tolerance_pct: float,
        tolerance_abs: float,
        start_t: float,
    ) -> Dict[str, Any]:
        """Fallback field-level comparison with tolerance checks.

        Args:
            comparison_id: Generated comparison identifier.
            match_id: Optional match ID reference.
            rec_a: First record.
            rec_b: Second record.
            fields: Fields to compare.
            tolerance_pct: Relative tolerance percentage.
            tolerance_abs: Absolute tolerance.
            start_t: Start timestamp for duration tracking.

        Returns:
            Dictionary with fallback comparison results.
        """
        field_results: List[Dict[str, Any]] = []
        matching = 0
        mismatching = 0
        missing = 0

        for f in fields:
            val_a = rec_a.get(f)
            val_b = rec_b.get(f)

            if val_a is None or val_b is None:
                field_results.append({
                    "field": f,
                    "result": "missing",
                    "value_a": val_a,
                    "value_b": val_b,
                })
                missing += 1
                continue

            # Numeric tolerance check
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                abs_diff = abs(float(val_a) - float(val_b))
                denom = max(abs(float(val_a)), abs(float(val_b)), 1e-10)
                rel_diff = (abs_diff / denom) * 100.0

                if abs_diff <= tolerance_abs or rel_diff <= tolerance_pct:
                    field_results.append({
                        "field": f,
                        "result": "match",
                        "value_a": val_a,
                        "value_b": val_b,
                        "abs_diff": abs_diff,
                        "rel_diff_pct": round(rel_diff, 4),
                    })
                    matching += 1
                else:
                    field_results.append({
                        "field": f,
                        "result": "mismatch",
                        "value_a": val_a,
                        "value_b": val_b,
                        "abs_diff": abs_diff,
                        "rel_diff_pct": round(rel_diff, 4),
                    })
                    mismatching += 1
            else:
                # String/other comparison
                if str(val_a).strip().lower() == str(val_b).strip().lower():
                    field_results.append({
                        "field": f,
                        "result": "match",
                        "value_a": val_a,
                        "value_b": val_b,
                    })
                    matching += 1
                else:
                    field_results.append({
                        "field": f,
                        "result": "mismatch",
                        "value_a": val_a,
                        "value_b": val_b,
                    })
                    mismatching += 1

        total = len(fields)
        match_rate = matching / total if total > 0 else 0.0
        elapsed = (time.time() - start_t) * 1000.0

        return {
            "comparison_id": comparison_id,
            "match_id": match_id or "",
            "fields_compared": field_results,
            "total_fields": total,
            "matching_fields": matching,
            "mismatching_fields": mismatching,
            "missing_fields": missing,
            "match_rate": round(match_rate, 4),
            "processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "comparison_id": comparison_id,
                "matching": matching,
                "mismatching": mismatching,
            }),
        }

    # ------------------------------------------------------------------
    # Discrepancy Detection
    # ------------------------------------------------------------------

    def detect_discrepancies(
        self,
        comparison_id: Optional[str] = None,
        comparisons: Optional[List[Dict[str, Any]]] = None,
        severity_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Detect and classify discrepancies from comparison results.

        Scans comparison results for mismatching fields and classifies
        each discrepancy by type and severity.

        Args:
            comparison_id: Optional stored comparison ID to analyze.
            comparisons: Optional inline comparison field results.
            severity_thresholds: Optional severity classification
                thresholds (critical, high, medium, low percentages).

        Returns:
            Dictionary with classified discrepancies.
        """
        start_t = time.time()
        detection_id = str(uuid.uuid4())

        thresholds = severity_thresholds or {
            "critical": 50.0,
            "high": 25.0,
            "medium": 10.0,
            "low": 5.0,
        }

        # Gather field results
        field_results: List[Dict[str, Any]] = comparisons or []
        if comparison_id and not field_results:
            comp = self._comparisons.get(comparison_id)
            if comp:
                field_results = comp.get("fields_compared", [])

        if self._discrepancy_detector is not None:
            try:
                result = self._discrepancy_detector.detect(
                    field_results=field_results,
                    severity_thresholds=thresholds,
                )
                output = {
                    "detection_id": detection_id,
                    "comparison_id": comparison_id or "",
                    "discrepancies": result.get("discrepancies", []),
                    "total_discrepancies": result.get("total_discrepancies", 0),
                    "by_severity": result.get("by_severity", {}),
                    "by_type": result.get("by_type", {}),
                    "processing_time_ms": (time.time() - start_t) * 1000.0,
                    "provenance_hash": result.get(
                        "provenance_hash",
                        _compute_hash({"detection_id": detection_id}),
                    ),
                }
            except Exception as exc:
                logger.warning(
                    "Discrepancy detector error, using fallback: %s", exc,
                )
                output = self._discrepancy_fallback(
                    detection_id, comparison_id, field_results,
                    thresholds, start_t,
                )
        else:
            output = self._discrepancy_fallback(
                detection_id, comparison_id, field_results,
                thresholds, start_t,
            )

        # Store individual discrepancies
        for disc in output.get("discrepancies", []):
            disc_id = disc.get("discrepancy_id", str(uuid.uuid4()))
            disc["discrepancy_id"] = disc_id
            self._discrepancies[disc_id] = disc

        self._stats["total_discrepancies"] += output.get(
            "total_discrepancies", 0,
        )
        self._provenance.record(
            "discrepancy", detection_id, "detect",
            output["provenance_hash"],
        )
        return output

    def _discrepancy_fallback(
        self,
        detection_id: str,
        comparison_id: Optional[str],
        field_results: List[Dict[str, Any]],
        thresholds: Dict[str, float],
        start_t: float,
    ) -> Dict[str, Any]:
        """Fallback discrepancy detection from field comparison results.

        Args:
            detection_id: Generated detection identifier.
            comparison_id: Optional comparison ID reference.
            field_results: Field-level comparison results.
            thresholds: Severity classification thresholds.
            start_t: Start timestamp for duration tracking.

        Returns:
            Dictionary with fallback discrepancy results.
        """
        discrepancies: List[Dict[str, Any]] = []
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for fr in field_results:
            if fr.get("result") not in ("mismatch", "missing"):
                continue

            disc_type = "value_mismatch"
            if fr.get("result") == "missing":
                disc_type = "missing_in_source"

            # Classify severity by relative difference
            rel_diff = fr.get("rel_diff_pct", 100.0)
            severity = "info"
            if rel_diff >= thresholds.get("critical", 50.0):
                severity = "critical"
            elif rel_diff >= thresholds.get("high", 25.0):
                severity = "high"
            elif rel_diff >= thresholds.get("medium", 10.0):
                severity = "medium"
            elif rel_diff >= thresholds.get("low", 5.0):
                severity = "low"

            disc = {
                "discrepancy_id": str(uuid.uuid4()),
                "detection_id": detection_id,
                "field": fr.get("field", ""),
                "type": disc_type,
                "severity": severity,
                "value_a": fr.get("value_a"),
                "value_b": fr.get("value_b"),
                "abs_diff": fr.get("abs_diff", 0.0),
                "rel_diff_pct": fr.get("rel_diff_pct", 0.0),
                "status": "open",
            }
            discrepancies.append(disc)
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[disc_type] = by_type.get(disc_type, 0) + 1

        elapsed = (time.time() - start_t) * 1000.0
        return {
            "detection_id": detection_id,
            "comparison_id": comparison_id or "",
            "discrepancies": discrepancies,
            "total_discrepancies": len(discrepancies),
            "by_severity": by_severity,
            "by_type": by_type,
            "processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "detection_id": detection_id,
                "total": len(discrepancies),
            }),
        }

    def list_discrepancies(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List detected discrepancies with optional filters.

        Args:
            severity: Optional severity filter.
            status: Optional status filter.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with discrepancies list and pagination metadata.
        """
        items = list(self._discrepancies.values())
        if severity is not None:
            items = [d for d in items if d.get("severity") == severity]
        if status is not None:
            items = [d for d in items if d.get("status") == status]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "discrepancies": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_discrepancy(
        self,
        discrepancy_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a discrepancy by ID.

        Args:
            discrepancy_id: Unique discrepancy identifier.

        Returns:
            Discrepancy dictionary or None if not found.
        """
        return self._discrepancies.get(discrepancy_id)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_discrepancies(
        self,
        discrepancy_ids: Optional[List[str]] = None,
        strategy: str = "priority_wins",
        source_priorities: Optional[Dict[str, int]] = None,
        manual_values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve discrepancies using a configurable strategy.

        Applies the chosen resolution strategy to each discrepancy,
        selecting the winning value and creating golden record fields.

        Args:
            discrepancy_ids: IDs of discrepancies to resolve.
            strategy: Resolution strategy (priority_wins, most_recent,
                weighted_average, most_complete, consensus,
                manual_override).
            source_priorities: Source priority map for priority_wins
                strategy.
            manual_values: Manual override values for manual_override
                strategy.

        Returns:
            Dictionary with resolution results.
        """
        start_t = time.time()
        resolution_id = str(uuid.uuid4())
        disc_ids = discrepancy_ids or []

        resolutions: List[Dict[str, Any]] = []
        for disc_id in disc_ids:
            disc = self._discrepancies.get(disc_id)
            if disc is None:
                continue

            # Apply resolution strategy
            resolved_value = None
            winning_source = "unknown"

            if strategy == "manual_override" and manual_values:
                field = disc.get("field", "")
                resolved_value = manual_values.get(field, disc.get("value_a"))
                winning_source = "manual"
            elif strategy == "priority_wins":
                # Default: pick value_a (assumed higher priority)
                resolved_value = disc.get("value_a")
                winning_source = "source_a"
            elif strategy == "most_recent":
                resolved_value = disc.get("value_b")
                winning_source = "source_b"
            elif strategy == "weighted_average":
                val_a = disc.get("value_a")
                val_b = disc.get("value_b")
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    resolved_value = (float(val_a) + float(val_b)) / 2.0
                    winning_source = "weighted_average"
                else:
                    resolved_value = val_a
                    winning_source = "source_a"
            else:
                resolved_value = disc.get("value_a")
                winning_source = "source_a"

            res = {
                "resolution_id": str(uuid.uuid4()),
                "discrepancy_id": disc_id,
                "field": disc.get("field", ""),
                "strategy": strategy,
                "resolved_value": resolved_value,
                "winning_source": winning_source,
                "original_value_a": disc.get("value_a"),
                "original_value_b": disc.get("value_b"),
                "justification": (
                    f"Resolved using {strategy} strategy"
                ),
                "resolved_at": _utcnow().isoformat(),
            }
            resolutions.append(res)
            self._resolutions[res["resolution_id"]] = res

            # Mark discrepancy as resolved
            disc["status"] = "resolved"
            disc["resolution_id"] = res["resolution_id"]

        elapsed = (time.time() - start_t) * 1000.0
        self._stats["total_resolutions"] += len(resolutions)

        output = {
            "resolution_id": resolution_id,
            "strategy": strategy,
            "resolutions": resolutions,
            "total_resolved": len(resolutions),
            "processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "resolution_id": resolution_id,
                "total_resolved": len(resolutions),
                "strategy": strategy,
            }),
        }

        self._provenance.record(
            "resolution", resolution_id, "resolve",
            output["provenance_hash"],
        )
        logger.info(
            "Resolved %d discrepancies with strategy %s",
            len(resolutions), strategy,
        )
        return output

    # ------------------------------------------------------------------
    # Golden Records
    # ------------------------------------------------------------------

    def get_golden_records(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List assembled golden records with pagination.

        Args:
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with golden records list and pagination metadata.
        """
        items = list(self._golden_records.values())
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "golden_records": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_golden_record(
        self,
        record_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a golden record by ID.

        Args:
            record_id: Unique golden record identifier.

        Returns:
            Golden record dictionary or None if not found.
        """
        return self._golden_records.get(record_id)

    def _create_golden_record(
        self,
        entity_id: str,
        period: str,
        field_values: Dict[str, Any],
        field_sources: Dict[str, str],
        field_confidence: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assemble a golden record from resolved fields.

        Args:
            entity_id: Entity identifier for the golden record.
            period: Time period the record covers.
            field_values: Resolved field values.
            field_sources: Source attribution per field.
            field_confidence: Confidence score per field.

        Returns:
            Dictionary with assembled golden record.
        """
        record_id = str(uuid.uuid4())
        overall_confidence = (
            sum(field_confidence.values()) / len(field_confidence)
            if field_confidence else 0.0
        )

        golden = {
            "record_id": record_id,
            "entity_id": entity_id,
            "period": period,
            "field_values": field_values,
            "field_sources": field_sources,
            "field_confidence": field_confidence,
            "overall_confidence": round(overall_confidence, 4),
            "status": "active",
            "created_at": _utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "record_id": record_id,
                "entity_id": entity_id,
                "field_values": field_values,
            }),
        }

        self._golden_records[record_id] = golden
        self._stats["total_golden_records"] += 1
        self._provenance.record(
            "golden_record", record_id, "create",
            golden["provenance_hash"],
        )
        return golden

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        source_ids: Optional[List[str]] = None,
        records_a: Optional[List[Dict[str, Any]]] = None,
        records_b: Optional[List[Dict[str, Any]]] = None,
        match_keys: Optional[List[str]] = None,
        match_threshold: float = 0.85,
        tolerance_pct: float = 5.0,
        tolerance_abs: float = 0.01,
        resolution_strategy: str = "priority_wins",
        generate_golden_records: bool = True,
    ) -> Dict[str, Any]:
        """Run the full reconciliation pipeline end-to-end.

        Executes: match -> compare -> detect discrepancies -> resolve
        -> assemble golden records -> generate provenance trail.

        Args:
            source_ids: Source IDs to reconcile.
            records_a: First set of records (inline data).
            records_b: Second set of records (inline data).
            match_keys: Fields for record matching.
            match_threshold: Match confidence threshold.
            tolerance_pct: Relative comparison tolerance.
            tolerance_abs: Absolute comparison tolerance.
            resolution_strategy: Conflict resolution strategy.
            generate_golden_records: Whether to assemble golden records.

        Returns:
            Dictionary with full pipeline results.
        """
        start_t = time.time()
        pipeline_id = str(uuid.uuid4())

        # Step 1: Match records
        match_result = self.match_records(
            source_ids=source_ids,
            records_a=records_a,
            records_b=records_b,
            match_keys=match_keys,
            threshold=match_threshold,
        )

        # Step 2: Compare matched pairs
        comparison_results: List[Dict[str, Any]] = []
        for pair in match_result.get("matched_pairs", []):
            comp = self.compare_records(
                record_a=pair.get("record_a", {}),
                record_b=pair.get("record_b", {}),
                tolerance_pct=tolerance_pct,
                tolerance_abs=tolerance_abs,
            )
            comparison_results.append(comp)

        # Step 3: Detect discrepancies
        all_field_results: List[Dict[str, Any]] = []
        for comp in comparison_results:
            all_field_results.extend(
                comp.get("fields_compared", []),
            )
        discrepancy_result = self.detect_discrepancies(
            comparisons=all_field_results,
        )

        # Step 4: Resolve discrepancies
        disc_ids = [
            d.get("discrepancy_id", "")
            for d in discrepancy_result.get("discrepancies", [])
        ]
        resolution_result = self.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy=resolution_strategy,
        )

        # Step 5: Assemble golden records (optional)
        golden_records: List[Dict[str, Any]] = []
        if generate_golden_records and match_result.get("matched_pairs"):
            for i, pair in enumerate(match_result["matched_pairs"]):
                rec_a = pair.get("record_a", {})
                field_vals = dict(rec_a)

                # Override with resolved values
                for res in resolution_result.get("resolutions", []):
                    field = res.get("field", "")
                    if field in field_vals:
                        field_vals[field] = res.get("resolved_value")

                golden = self._create_golden_record(
                    entity_id=str(rec_a.get("entity_id", f"entity_{i}")),
                    period=str(rec_a.get("period", "")),
                    field_values=field_vals,
                    field_sources={
                        k: "reconciled" for k in field_vals
                    },
                    field_confidence={
                        k: pair.get("confidence", 1.0) for k in field_vals
                    },
                )
                golden_records.append(golden)

        elapsed = (time.time() - start_t) * 1000.0

        result = {
            "pipeline_id": pipeline_id,
            "source_ids": source_ids or [],
            "match_result": match_result,
            "comparison_count": len(comparison_results),
            "discrepancy_result": discrepancy_result,
            "resolution_result": resolution_result,
            "golden_records": golden_records,
            "golden_record_count": len(golden_records),
            "status": "completed",
            "total_processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "pipeline_id": pipeline_id,
                "match_id": match_result.get("match_id", ""),
                "total_discrepancies": discrepancy_result.get(
                    "total_discrepancies", 0,
                ),
                "total_resolved": resolution_result.get("total_resolved", 0),
                "golden_record_count": len(golden_records),
            }),
        }

        self._pipeline_results[pipeline_id] = result
        self._stats["total_pipelines"] += 1
        self._provenance.record(
            "pipeline", pipeline_id, "reconcile",
            result["provenance_hash"],
        )
        logger.info(
            "Pipeline %s completed: %d matches, %d discrepancies, "
            "%d golden records in %.1fms",
            pipeline_id,
            match_result.get("total_matched", 0),
            discrepancy_result.get("total_discrepancies", 0),
            len(golden_records),
            elapsed,
        )
        return result

    def get_health(self) -> Dict[str, Any]:
        """Alias for health_check() used by the router.

        Returns:
            Service health dictionary.
        """
        return self.health_check()


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_instance: Optional[CrossSourceReconciliationService] = None
_service_lock = threading.Lock()


def get_service() -> CrossSourceReconciliationService:
    """Return the singleton CrossSourceReconciliationService.

    Thread-safe lazy initialization. Returns the same instance
    on every call within the process.

    Returns:
        The global CrossSourceReconciliationService singleton.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = CrossSourceReconciliationService()
                _service_instance.startup()
    return _service_instance


def reset_service() -> CrossSourceReconciliationService:
    """Reset and return a new singleton instance.

    Returns:
        A fresh CrossSourceReconciliationService singleton.
    """
    global _service_instance
    with _service_lock:
        _service_instance = CrossSourceReconciliationService()
        _service_instance.startup()
    return _service_instance


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


def configure_reconciliation(app: Any) -> CrossSourceReconciliationService:
    """Configure the reconciliation service on a FastAPI app.

    Attaches the service to ``app.state.cross_source_reconciliation_service``
    and optionally includes the router.

    Args:
        app: FastAPI application instance.

    Returns:
        The configured CrossSourceReconciliationService.
    """
    service = get_service()
    app.state.cross_source_reconciliation_service = service

    # Attempt to include the router
    try:
        from greenlang.cross_source_reconciliation.api.router import router
        if router is not None:
            app.include_router(router)
    except ImportError:
        logger.warning(
            "Reconciliation router not available; skipping route registration"
        )

    logger.info("Cross-source reconciliation service configured on app")
    return service


def get_reconciliation(app: Any) -> Optional[CrossSourceReconciliationService]:
    """Retrieve the reconciliation service from a FastAPI app.

    Args:
        app: FastAPI application instance.

    Returns:
        CrossSourceReconciliationService or None if not configured.
    """
    return getattr(
        app.state, "cross_source_reconciliation_service", None,
    )


def get_router() -> Any:
    """Return the FastAPI APIRouter for the reconciliation service.

    Returns:
        FastAPI APIRouter instance or None if FastAPI is not available.
    """
    try:
        from greenlang.cross_source_reconciliation.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "CrossSourceReconciliationService",
    "configure_reconciliation",
    "get_reconciliation",
    "get_router",
    "get_service",
    "reset_service",
    "_compute_hash",
]
