# -*- coding: utf-8 -*-
"""
Reconciliation Pipeline Engine - AGENT-DATA-015 Cross-Source Reconciliation

Engine 7 of 7.  Orchestrates the full cross-source reconciliation workflow
by composing the six upstream engines (SourceRegistryEngine, MatchingEngine,
ComparisonEngine, DiscrepancyDetectorEngine, ResolutionEngine,
AuditTrailEngine) into a deterministic pipeline.

Pipeline stages:
    1. REGISTER  -- register and validate data sources
    2. ALIGN     -- align schemas across sources
    3. MATCH     -- match records across source pairs
    4. COMPARE   -- compare matched records field-by-field
    5. DETECT    -- detect and classify discrepancies
    6. RESOLVE   -- apply conflict resolution strategies
    7. GOLDEN    -- assemble golden records and generate audit trail

Zero-Hallucination: All calculations use deterministic Python arithmetic.
Record matching uses rule-based heuristics (exact key, fuzzy similarity,
temporal alignment, composite scoring).  No LLM calls for numeric
computations.  Every operation is traced through SHA-256 provenance chains.

Example:
    >>> from greenlang.cross_source_reconciliation.reconciliation_pipeline import (
    ...     ReconciliationPipelineEngine,
    ... )
    >>> engine = ReconciliationPipelineEngine()
    >>> report = engine.run_pipeline(
    ...     job_config={"job_id": "j1", "key_fields": ["id"]},
    ...     source_data={"erp": [{"id": "1", "value": 100}],
    ...                  "invoice": [{"id": "1", "value": 105}]},
    ... )
    >>> assert report["status"] in ("completed", "completed_with_warnings")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for sibling engines (metrics + engines)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation import metrics as _metrics_mod
    _METRICS_AVAILABLE = True
except ImportError:
    _metrics_mod = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.source_registry import (
        SourceRegistryEngine,
    )
    _SOURCE_REGISTRY_AVAILABLE = True
except ImportError:
    SourceRegistryEngine = None  # type: ignore[misc, assignment]
    _SOURCE_REGISTRY_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.matching import MatchingEngine
    _MATCHING_AVAILABLE = True
except ImportError:
    MatchingEngine = None  # type: ignore[misc, assignment]
    _MATCHING_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.comparison import (
        ComparisonEngine,
    )
    _COMPARISON_AVAILABLE = True
except ImportError:
    ComparisonEngine = None  # type: ignore[misc, assignment]
    _COMPARISON_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.discrepancy_detector import (
        DiscrepancyDetectorEngine,
    )
    _DISCREPANCY_AVAILABLE = True
except ImportError:
    DiscrepancyDetectorEngine = None  # type: ignore[misc, assignment]
    _DISCREPANCY_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.resolution import (
        ResolutionEngine,
    )
    _RESOLUTION_AVAILABLE = True
except ImportError:
    ResolutionEngine = None  # type: ignore[misc, assignment]
    _RESOLUTION_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.audit_trail import (
        AuditTrailEngine,
    )
    _AUDIT_AVAILABLE = True
except ImportError:
    AuditTrailEngine = None  # type: ignore[misc, assignment]
    _AUDIT_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.config import get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    get_config = None  # type: ignore[misc, assignment]
    _CONFIG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Local metric helper stubs (delegate when metrics module is present)
# ---------------------------------------------------------------------------


def _inc_jobs_processed(status: str) -> None:
    """Increment the jobs processed counter.

    Args:
        status: Job status (completed, failed, partial).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_jobs_processed(status)


def _inc_records_matched(strategy: str, count: int = 1) -> None:
    """Increment the records matched counter.

    Args:
        strategy: Matching strategy used.
        count: Number of records matched.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_records_matched(strategy, count)


def _inc_comparisons(result: str, count: int = 1) -> None:
    """Increment the comparisons counter.

    Args:
        result: Comparison result (match, mismatch).
        count: Number of comparisons.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_comparisons(result, count)


def _inc_discrepancies(
    discrepancy_type: str, severity: str, count: int = 1,
) -> None:
    """Increment the discrepancies counter.

    Args:
        discrepancy_type: Type of discrepancy.
        severity: Severity level.
        count: Number of discrepancies.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_discrepancies(discrepancy_type, severity, count)


def _inc_resolutions(strategy: str, count: int = 1) -> None:
    """Increment the resolutions counter.

    Args:
        strategy: Resolution strategy used.
        count: Number of resolutions.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_resolutions(strategy, count)


def _inc_golden_records(status: str, count: int = 1) -> None:
    """Increment the golden records counter.

    Args:
        status: Golden record status (created, merged).
        count: Number of golden records.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_golden_records(status, count)


def _observe_confidence(confidence: float) -> None:
    """Observe a match confidence score.

    Args:
        confidence: Confidence value (0.0-1.0).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_confidence(confidence)


def _observe_duration(duration: float) -> None:
    """Observe processing duration in seconds.

    Args:
        duration: Duration in seconds.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_duration(duration)


def _observe_magnitude(magnitude: float) -> None:
    """Observe discrepancy magnitude.

    Args:
        magnitude: Magnitude as percentage deviation.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_magnitude(magnitude)


def _set_active_jobs(count: int) -> None:
    """Set the active jobs gauge.

    Args:
        count: Active job count.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.set_active_jobs(count)


def _inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_errors(error_type)


# ---------------------------------------------------------------------------
# Pipeline stage enumeration
# ---------------------------------------------------------------------------

_STAGES = (
    "register", "align", "match", "compare",
    "detect", "resolve", "golden",
)


# ---------------------------------------------------------------------------
# Local data models
# ---------------------------------------------------------------------------


@dataclass
class PipelineStageResult:
    """Result of a single pipeline stage execution.

    Attributes:
        stage: Stage name (register, align, match, etc.).
        status: Execution status (completed, failed, skipped).
        duration_ms: Stage execution time in milliseconds.
        records_processed: Number of records processed in this stage.
        output_summary: Key output metrics for the stage.
        error: Error message if the stage failed.
    """

    stage: str = ""
    status: str = "pending"
    duration_ms: float = 0.0
    records_processed: int = 0
    output_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ReconciliationStats:
    """Aggregated reconciliation pipeline execution statistics.

    Attributes:
        total_runs: Total pipeline executions.
        total_sources_registered: Total sources registered across runs.
        total_records_matched: Total records matched across runs.
        total_comparisons: Total field comparisons performed.
        total_discrepancies: Total discrepancies detected.
        total_resolutions: Total resolutions applied.
        total_golden_records: Total golden records assembled.
        by_status: Run counts per completion status.
        by_match_strategy: Match counts per strategy.
        by_resolution_strategy: Resolution counts per strategy.
        by_discrepancy_type: Discrepancy counts per type.
        avg_match_confidence: Average match confidence across runs.
    """

    total_runs: int = 0
    total_sources_registered: int = 0
    total_records_matched: int = 0
    total_comparisons: int = 0
    total_discrepancies: int = 0
    total_resolutions: int = 0
    total_golden_records: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    by_match_strategy: Dict[str, int] = field(default_factory=dict)
    by_resolution_strategy: Dict[str, int] = field(default_factory=dict)
    by_discrepancy_type: Dict[str, int] = field(default_factory=dict)
    avg_match_confidence: float = 0.0


# ============================================================================
# ReconciliationPipelineEngine
# ============================================================================


class ReconciliationPipelineEngine:
    """Orchestrates the full cross-source reconciliation pipeline.

    Composes six upstream engines into a seven-stage sequential pipeline:
    register -> align -> match -> compare -> detect -> resolve -> golden.

    Strategy selection uses rule-based heuristics based on key field
    presence, data format consistency, temporal alignment, and field
    type analysis.  All decisions are deterministic.

    Attributes:
        _source_registry: Source registration engine instance.
        _matching: Record matching engine instance.
        _comparison: Field comparison engine instance.
        _discrepancy: Discrepancy detection engine instance.
        _resolution: Conflict resolution engine instance.
        _audit: Audit trail engine instance.
        _config: Cross-source reconciliation configuration.
        _provenance: SHA-256 provenance tracker.
        _statistics: Running pipeline statistics.

    Example:
        >>> engine = ReconciliationPipelineEngine()
        >>> report = engine.run_pipeline(
        ...     job_config={"job_id": "j1", "key_fields": ["id"]},
        ...     source_data={"src_a": [{"id": "1"}], "src_b": [{"id": "1"}]},
        ... )
        >>> assert report["status"] in ("completed", "completed_with_warnings")
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ReconciliationPipelineEngine with all sub-engines.

        Args:
            config: Optional CrossSourceReconciliationConfig override.
                Falls back to the singleton from ``get_config()``.
        """
        self._config = config
        if self._config is None and _CONFIG_AVAILABLE and get_config is not None:
            self._config = get_config()

        self._provenance: ProvenanceTracker = get_provenance_tracker()
        self._statistics = ReconciliationStats()

        # Initialize sub-engines (graceful when not available).
        # Each sub-engine has its own constructor signature -- we must
        # call each one correctly rather than passing config to all.
        self._source_registry = (
            SourceRegistryEngine()
            if _SOURCE_REGISTRY_AVAILABLE and SourceRegistryEngine is not None
            else None
        )
        self._matching = (
            MatchingEngine(config=self._config)
            if _MATCHING_AVAILABLE and MatchingEngine is not None
            else None
        )
        self._comparison = (
            ComparisonEngine()
            if _COMPARISON_AVAILABLE and ComparisonEngine is not None
            else None
        )
        self._discrepancy = (
            DiscrepancyDetectorEngine()
            if _DISCREPANCY_AVAILABLE and DiscrepancyDetectorEngine is not None
            else None
        )
        self._resolution = (
            ResolutionEngine()
            if _RESOLUTION_AVAILABLE and ResolutionEngine is not None
            else None
        )
        self._audit = (
            AuditTrailEngine()
            if _AUDIT_AVAILABLE and AuditTrailEngine is not None
            else None
        )

        available_engines = sum([
            _SOURCE_REGISTRY_AVAILABLE,
            _MATCHING_AVAILABLE,
            _COMPARISON_AVAILABLE,
            _DISCREPANCY_AVAILABLE,
            _RESOLUTION_AVAILABLE,
            _AUDIT_AVAILABLE,
        ])
        logger.info(
            "ReconciliationPipelineEngine initialized "
            "(%d/6 sub-engines available)",
            available_engines,
        )

    # ------------------------------------------------------------------
    # 1. run_pipeline - Full pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        job_config: Dict[str, Any],
        source_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Execute the full cross-source reconciliation pipeline.

        Runs all seven pipeline stages in order and returns a
        comprehensive reconciliation report with match results,
        discrepancies, resolutions, golden records, and provenance.

        Args:
            job_config: Reconciliation job configuration dict with keys:
                - ``job_id`` (str): Unique job identifier (auto-generated
                  if missing).
                - ``key_fields`` (list): Fields used for record matching.
                - ``match_strategy`` (str): Matching strategy
                  (exact, fuzzy, temporal, composite, auto).
                - ``match_threshold`` (float): Minimum confidence for
                  matches (0.0-1.0, default 0.8).
                - ``resolution_strategy`` (str): Conflict resolution
                  strategy (source_priority, most_recent, most_complete,
                  average, median, auto).
                - ``tolerance_rules`` (dict): Per-field tolerance rules.
                - ``source_priorities`` (dict): Source credibility scores.
                - ``field_types`` (dict): Per-field type overrides.
            source_data: Dict mapping source names to lists of records.
                At least 2 sources required, each with at least 1 record.

        Returns:
            Dict (ReconciliationReport) with keys:
                - job_id (str): Unique job identifier.
                - status (str): completed, completed_with_warnings, failed.
                - sources_registered (int): Number of sources registered.
                - total_records (int): Total input records across sources.
                - matches (list): Matched record pairs.
                - comparisons (list): Field-level comparisons.
                - discrepancies (list): Detected discrepancies.
                - resolutions (list): Applied resolutions.
                - golden_records (list): Assembled golden records.
                - stage_results (dict): Per-stage PipelineStageResult.
                - statistics (dict): Pipeline run statistics.
                - total_time_ms (float): Total pipeline time.
                - provenance_hash (str): SHA-256 chain hash.
                - error (str or None): Error message if failed.
        """
        job_id = job_config.get("job_id", str(uuid4()))
        pipeline_start = time.time()
        stage_results: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Pipeline %s starting: %d sources, strategy=%s",
            job_id[:8],
            len(source_data),
            job_config.get("match_strategy", "auto"),
        )

        _set_active_jobs(1)

        # Validate sources first
        validation = self.validate_sources(source_data)
        if not validation.get("valid", False):
            error_msg = validation.get("error", "Source validation failed")
            logger.error("Pipeline %s: %s", job_id[:8], error_msg)
            _inc_errors("validation")
            _inc_jobs_processed("failed")
            _set_active_jobs(0)
            return self._build_error_report(
                job_id, error_msg, pipeline_start, stage_results,
            )

        report: Dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "sources_registered": 0,
            "total_records": sum(len(v) for v in source_data.values()),
            "matches": [],
            "comparisons": [],
            "discrepancies": [],
            "resolutions": [],
            "golden_records": [],
            "stage_results": stage_results,
            "statistics": {},
            "total_time_ms": 0.0,
            "provenance_hash": "",
            "error": None,
        }

        try:
            # -- Stage 1: REGISTER --
            stage_1 = self._stage_register(
                job_id, source_data, job_config,
            )
            stage_results["register"] = self._stage_to_dict(stage_1)
            if stage_1.status == "failed":
                raise RuntimeError(
                    f"Stage REGISTER failed: {stage_1.error}"
                )
            source_ids = stage_1.output_summary.get("source_ids", [])
            source_credibilities = stage_1.output_summary.get(
                "credibilities", {},
            )
            report["sources_registered"] = len(source_ids)

            # -- Stage 2: ALIGN --
            stage_2 = self._stage_align(
                job_id, source_data, source_ids,
            )
            stage_results["align"] = self._stage_to_dict(stage_2)
            aligned_data = stage_2.output_summary.get(
                "aligned_data", source_data,
            )
            common_fields = stage_2.output_summary.get(
                "common_fields", [],
            )

            # -- Stage 3: MATCH --
            match_strategy = job_config.get("match_strategy", "auto")
            match_threshold = float(
                job_config.get("match_threshold", 0.8)
            )
            if match_strategy == "auto":
                match_strategy = self._select_match_strategy(
                    source_data,
                ).get("strategy", "exact")

            stage_3 = self._stage_match(
                job_id, aligned_data, source_ids,
                match_strategy, match_threshold,
            )
            stage_results["match"] = self._stage_to_dict(stage_3)
            matches = stage_3.output_summary.get("matches", [])
            report["matches"] = matches

            # -- Stage 4: COMPARE --
            tolerance_rules = job_config.get("tolerance_rules", {})
            field_types = job_config.get("field_types", {})
            if not field_types and matches:
                field_types = self._infer_field_types_from_data(
                    source_data,
                )

            stage_4 = self._stage_compare(
                job_id, matches, aligned_data,
                tolerance_rules, field_types,
            )
            stage_results["compare"] = self._stage_to_dict(stage_4)
            comparisons = stage_4.output_summary.get("comparisons", [])
            report["comparisons"] = comparisons

            # -- Stage 5: DETECT --
            stage_5 = self._stage_detect(
                job_id, comparisons, matches,
            )
            stage_results["detect"] = self._stage_to_dict(stage_5)
            discrepancies = stage_5.output_summary.get(
                "discrepancies", [],
            )
            report["discrepancies"] = discrepancies

            # -- Stage 6: RESOLVE --
            resolution_strategy = job_config.get(
                "resolution_strategy", "source_priority",
            )
            stage_6 = self._stage_resolve(
                job_id, discrepancies, source_credibilities,
                aligned_data, resolution_strategy,
            )
            stage_results["resolve"] = self._stage_to_dict(stage_6)
            resolutions = stage_6.output_summary.get("resolutions", [])
            report["resolutions"] = resolutions

            # -- Stage 7: GOLDEN RECORDS --
            stage_7 = self._stage_golden_records(
                job_id, aligned_data, resolutions,
                source_credibilities,
            )
            stage_results["golden"] = self._stage_to_dict(stage_7)
            golden_records = stage_7.output_summary.get(
                "golden_records", [],
            )
            report["golden_records"] = golden_records

            # Determine final status
            has_warnings = len(discrepancies) > 0
            report["status"] = (
                "completed_with_warnings" if has_warnings
                else "completed"
            )

        except Exception as exc:
            logger.error(
                "Pipeline %s failed: %s",
                job_id[:8], str(exc), exc_info=True,
            )
            report["status"] = "failed"
            report["error"] = str(exc)
            _inc_errors("pipeline")

        self._finalize_pipeline(
            report, pipeline_start, job_id,
        )
        return report

    # ------------------------------------------------------------------
    # 2. run_batch_pipeline - Batch processing
    # ------------------------------------------------------------------

    def run_batch_pipeline(
        self,
        jobs: List[Dict[str, Any]],
        source_data_map: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        """Execute the pipeline for multiple reconciliation jobs.

        Processes each job independently and returns all reports.

        Args:
            jobs: List of job configuration dicts (same format as
                ``run_pipeline`` job_config).
            source_data_map: Dict mapping job_id to source_data.
                If a job_id is not found, uses the first entry as
                a fallback.

        Returns:
            List of ReconciliationReport dicts, one per job.
        """
        batch_id = str(uuid4())
        batch_start = time.time()

        logger.info(
            "Batch %s starting: %d jobs",
            batch_id[:8], len(jobs),
        )

        reports: List[Dict[str, Any]] = []
        successful = 0
        failed = 0

        for idx, job_config in enumerate(jobs):
            job_id = job_config.get("job_id", str(uuid4()))
            source_data = source_data_map.get(job_id)

            if source_data is None:
                # Fall back to matching by index or first entry
                keys = list(source_data_map.keys())
                if idx < len(keys):
                    source_data = source_data_map[keys[idx]]
                elif keys:
                    source_data = source_data_map[keys[0]]
                else:
                    reports.append(self._build_error_report(
                        job_id, "No source data found", batch_start, {},
                    ))
                    failed += 1
                    continue

            try:
                report = self.run_pipeline(job_config, source_data)
                reports.append(report)

                if report["status"] in ("completed", "completed_with_warnings"):
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error(
                    "Batch %s: job '%s' raised %s: %s",
                    batch_id[:8], job_id[:8],
                    type(exc).__name__, str(exc),
                )
                reports.append(self._build_error_report(
                    job_id, str(exc), batch_start, {},
                ))
                failed += 1

        elapsed = time.time() - batch_start
        _observe_duration(elapsed)

        logger.info(
            "Batch %s completed: %d/%d successful, %.1fms",
            batch_id[:8], successful, len(jobs), elapsed * 1000.0,
        )

        return reports

    # ------------------------------------------------------------------
    # 3. get_statistics - Pipeline statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregated pipeline execution statistics.

        Returns:
            Dict with total_runs, record/match/discrepancy counts,
            breakdown by strategy and type, engine availability,
            and provenance chain length.
        """
        return {
            "total_runs": self._statistics.total_runs,
            "total_sources_registered": (
                self._statistics.total_sources_registered
            ),
            "total_records_matched": (
                self._statistics.total_records_matched
            ),
            "total_comparisons": self._statistics.total_comparisons,
            "total_discrepancies": self._statistics.total_discrepancies,
            "total_resolutions": self._statistics.total_resolutions,
            "total_golden_records": self._statistics.total_golden_records,
            "avg_match_confidence": (
                self._statistics.avg_match_confidence
            ),
            "by_status": dict(self._statistics.by_status),
            "by_match_strategy": dict(
                self._statistics.by_match_strategy
            ),
            "by_resolution_strategy": dict(
                self._statistics.by_resolution_strategy
            ),
            "by_discrepancy_type": dict(
                self._statistics.by_discrepancy_type
            ),
            "engine_availability": {
                "source_registry": _SOURCE_REGISTRY_AVAILABLE,
                "matching": _MATCHING_AVAILABLE,
                "comparison": _COMPARISON_AVAILABLE,
                "discrepancy_detector": _DISCREPANCY_AVAILABLE,
                "resolution": _RESOLUTION_AVAILABLE,
                "audit_trail": _AUDIT_AVAILABLE,
            },
            "provenance_chain_length": (
                self._provenance.get_chain_length()
            ),
        }

    # ------------------------------------------------------------------
    # 4. validate_sources - Source validation
    # ------------------------------------------------------------------

    def validate_sources(
        self,
        source_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Validate that source data meets pipeline requirements.

        Checks:
            - At least 2 sources are present.
            - Each source has at least 1 record.
            - All records have valid (non-empty string) field names.
            - No source name is empty.

        Args:
            source_data: Dict mapping source names to record lists.

        Returns:
            Dict with keys:
                - valid (bool): Whether validation passed.
                - error (str or None): Error message if invalid.
                - source_count (int): Number of sources.
                - record_counts (dict): Records per source.
                - warnings (list): Non-fatal warnings.
        """
        warnings: List[str] = []
        record_counts: Dict[str, int] = {}

        if not source_data:
            return {
                "valid": False,
                "error": "No source data provided",
                "source_count": 0,
                "record_counts": {},
                "warnings": [],
            }

        if len(source_data) < 2:
            return {
                "valid": False,
                "error": (
                    f"At least 2 sources required, got "
                    f"{len(source_data)}"
                ),
                "source_count": len(source_data),
                "record_counts": {},
                "warnings": [],
            }

        for source_name, records in source_data.items():
            if not source_name or not isinstance(source_name, str):
                return {
                    "valid": False,
                    "error": "Source name must be a non-empty string",
                    "source_count": len(source_data),
                    "record_counts": record_counts,
                    "warnings": warnings,
                }

            if not records:
                return {
                    "valid": False,
                    "error": (
                        f"Source '{source_name}' has no records"
                    ),
                    "source_count": len(source_data),
                    "record_counts": record_counts,
                    "warnings": warnings,
                }

            record_counts[source_name] = len(records)

            # Validate field names in first record
            if records and isinstance(records[0], dict):
                for field_name in records[0].keys():
                    if not field_name or not isinstance(field_name, str):
                        return {
                            "valid": False,
                            "error": (
                                f"Source '{source_name}' has invalid "
                                f"field name: {field_name!r}"
                            ),
                            "source_count": len(source_data),
                            "record_counts": record_counts,
                            "warnings": warnings,
                        }

        # Check for very uneven source sizes
        sizes = list(record_counts.values())
        if sizes:
            max_size = max(sizes)
            min_size = min(sizes)
            if max_size > 0 and min_size > 0:
                ratio = max_size / min_size
                if ratio > 10.0:
                    warnings.append(
                        f"Source sizes vary significantly "
                        f"(ratio {ratio:.1f}:1)"
                    )

        return {
            "valid": True,
            "error": None,
            "source_count": len(source_data),
            "record_counts": record_counts,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # 5. auto_configure - Auto-detect job configuration
    # ------------------------------------------------------------------

    def auto_configure(
        self,
        source_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Auto-detect a reconciliation job configuration from source data.

        Examines the source data to determine:
            - Key fields (common fields with unique values).
            - Field types (numeric, date, string, boolean).
            - Appropriate matching strategy.
            - Default tolerance rules.

        Args:
            source_data: Dict mapping source names to record lists.

        Returns:
            Dict (ReconciliationJobConfig) with auto-detected settings.
        """
        job_id = str(uuid4())

        # Gather all field names across sources
        all_fields: Dict[str, int] = {}
        for source_name, records in source_data.items():
            if records and isinstance(records[0], dict):
                for field_name in records[0].keys():
                    all_fields[field_name] = (
                        all_fields.get(field_name, 0) + 1
                    )

        num_sources = len(source_data)
        common_fields = [
            f for f, count in all_fields.items()
            if count >= num_sources
        ]

        # Detect key fields: common fields with high uniqueness
        key_fields = self._detect_key_fields(source_data, common_fields)

        # Infer field types
        field_types = self._infer_field_types_from_data(source_data)

        # Select match strategy
        strategy_result = self._select_match_strategy(source_data)
        match_strategy = strategy_result.get("strategy", "exact")

        # Build default tolerance rules for numeric fields
        tolerance_rules: Dict[str, Any] = {}
        for fname, ftype in field_types.items():
            if ftype == "numeric":
                tolerance_rules[fname] = {
                    "type": "relative",
                    "threshold": 0.05,
                }
            elif ftype == "date":
                tolerance_rules[fname] = {
                    "type": "temporal",
                    "threshold_days": 1,
                }

        return {
            "job_id": job_id,
            "key_fields": key_fields,
            "match_strategy": match_strategy,
            "match_threshold": 0.8,
            "resolution_strategy": "source_priority",
            "tolerance_rules": tolerance_rules,
            "field_types": field_types,
            "source_priorities": {},
            "common_fields": common_fields,
            "auto_configured": True,
        }

    # ------------------------------------------------------------------
    # 6. get_engine_health - Health check
    # ------------------------------------------------------------------

    def get_engine_health(self) -> Dict[str, Any]:
        """Return the health status of the pipeline and sub-engines.

        Returns:
            Dict with overall status, per-engine availability, and
            configuration summary.
        """
        engines = {
            "source_registry": _SOURCE_REGISTRY_AVAILABLE,
            "matching": _MATCHING_AVAILABLE,
            "comparison": _COMPARISON_AVAILABLE,
            "discrepancy_detector": _DISCREPANCY_AVAILABLE,
            "resolution": _RESOLUTION_AVAILABLE,
            "audit_trail": _AUDIT_AVAILABLE,
        }
        available = sum(engines.values())
        total = len(engines)

        if available == total:
            overall = "healthy"
        elif available >= 3:
            overall = "degraded"
        else:
            overall = "unhealthy"

        return {
            "status": overall,
            "engines_available": available,
            "engines_total": total,
            "engines": engines,
            "statistics": {
                "total_runs": self._statistics.total_runs,
                "total_golden_records": (
                    self._statistics.total_golden_records
                ),
            },
        }

    # ==================================================================
    # Private: Stage 1 - REGISTER
    # ==================================================================

    def _stage_register(
        self,
        job_id: str,
        source_data: Dict[str, List[Dict[str, Any]]],
        job_config: Dict[str, Any],
    ) -> PipelineStageResult:
        """Register and validate data sources.

        Registers each source with the source registry engine and
        computes initial credibility scores based on record counts,
        field completeness, and configured source priorities.

        Args:
            job_id: Unique job identifier.
            source_data: Dict mapping source names to record lists.
            job_config: Job configuration dict.

        Returns:
            PipelineStageResult with source_ids and credibilities.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="register")

        try:
            source_ids: List[str] = []
            credibilities: Dict[str, float] = {}
            configured_priorities = job_config.get(
                "source_priorities", {},
            )

            for source_name, records in source_data.items():
                source_id = source_name

                # Register with engine if available
                if self._source_registry is not None:
                    try:
                        reg_result = self._source_registry.register_source(
                            source_id=source_id,
                            source_name=source_name,
                            records=records,
                        )
                        if hasattr(reg_result, "source_id"):
                            source_id = reg_result.source_id
                    except Exception as reg_exc:
                        logger.warning(
                            "SourceRegistryEngine.register_source "
                            "failed for '%s': %s",
                            source_name, str(reg_exc),
                        )

                source_ids.append(source_id)

                # Compute credibility score
                credibility = self._compute_source_credibility(
                    source_name, records, configured_priorities,
                )
                credibilities[source_id] = credibility

            # Record provenance
            provenance_hash = self._compute_provenance(
                "register_sources",
                {"sources": list(source_data.keys())},
                {"source_ids": source_ids},
            )

            # Record audit event
            if self._audit is not None:
                try:
                    self._audit.record_event(
                        job_id=job_id,
                        event_type="source_registration",
                        details={
                            "source_ids": source_ids,
                            "credibilities": credibilities,
                        },
                    )
                except Exception:
                    pass

            result.status = "completed"
            result.records_processed = len(source_ids)
            result.output_summary = {
                "source_ids": source_ids,
                "credibilities": credibilities,
                "provenance_hash": provenance_hash,
            }

            self._statistics.total_sources_registered += len(source_ids)

            logger.info(
                "Stage REGISTER: %d sources registered, "
                "credibilities=%s",
                len(source_ids),
                {k: f"{v:.2f}" for k, v in credibilities.items()},
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("register")
            logger.error(
                "Stage REGISTER failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Stage 2 - ALIGN
    # ==================================================================

    def _stage_align(
        self,
        job_id: str,
        source_data: Dict[str, List[Dict[str, Any]]],
        source_ids: List[str],
    ) -> PipelineStageResult:
        """Align schemas across sources.

        Auto-discovers common columns, applies schema mappings, and
        normalizes values (units, currencies, dates) to ensure
        comparability.

        Args:
            job_id: Unique job identifier.
            source_data: Dict mapping source names to record lists.
            source_ids: Registered source identifiers.

        Returns:
            PipelineStageResult with aligned_data and common_fields.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="align")

        try:
            # Discover common fields across all sources
            field_sets: List[Set[str]] = []
            for source_name in source_ids:
                records = source_data.get(source_name, [])
                if records and isinstance(records[0], dict):
                    field_sets.append(set(records[0].keys()))
                else:
                    field_sets.append(set())

            if field_sets:
                common_fields = list(
                    set.intersection(*field_sets) if field_sets
                    else set()
                )
            else:
                common_fields = []

            # All fields across all sources (union)
            all_fields = list(
                set.union(*field_sets) if field_sets else set()
            )

            # Normalize values across sources
            aligned_data: Dict[str, List[Dict[str, Any]]] = {}
            total_records = 0
            for source_name in source_ids:
                records = source_data.get(source_name, [])
                normalized = []
                for record in records:
                    norm_record = self._normalize_record(record)
                    normalized.append(norm_record)
                aligned_data[source_name] = normalized
                total_records += len(normalized)

            # Record provenance
            provenance_hash = self._compute_provenance(
                "align_schemas",
                {"source_ids": source_ids, "all_fields": all_fields},
                {"common_fields": common_fields},
            )

            result.status = "completed"
            result.records_processed = total_records
            result.output_summary = {
                "aligned_data": aligned_data,
                "common_fields": sorted(common_fields),
                "all_fields": sorted(all_fields),
                "provenance_hash": provenance_hash,
            }

            logger.info(
                "Stage ALIGN: %d common fields, %d total fields, "
                "%d records aligned",
                len(common_fields), len(all_fields), total_records,
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("align")
            logger.error(
                "Stage ALIGN failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Stage 3 - MATCH
    # ==================================================================

    def _stage_match(
        self,
        job_id: str,
        source_data: Dict[str, List[Dict[str, Any]]],
        source_ids: List[str],
        strategy: str,
        threshold: float,
    ) -> PipelineStageResult:
        """Match records across source pairs.

        For each source pair, runs the matching engine to find
        corresponding records. Collects all match results and records
        match events in the audit trail.

        Args:
            job_id: Unique job identifier.
            source_data: Aligned source data.
            source_ids: Registered source identifiers.
            strategy: Matching strategy name.
            threshold: Minimum match confidence.

        Returns:
            PipelineStageResult with matches list.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="match")

        try:
            all_matches: List[Dict[str, Any]] = []
            total_matched = 0

            # Generate source pairs
            pairs = self._generate_source_pairs(source_ids)

            for src_a, src_b in pairs:
                records_a = source_data.get(src_a, [])
                records_b = source_data.get(src_b, [])

                if not records_a or not records_b:
                    continue

                # Use matching engine if available
                if self._matching is not None:
                    try:
                        match_result = self._matching.match_records(
                            source_a_id=src_a,
                            source_b_id=src_b,
                            records_a=records_a,
                            records_b=records_b,
                            strategy=strategy,
                            threshold=threshold,
                        )
                        pair_matches = (
                            getattr(match_result, "matches", [])
                            if hasattr(match_result, "matches")
                            else match_result
                            if isinstance(match_result, list)
                            else []
                        )
                        all_matches.extend(pair_matches)
                        total_matched += len(pair_matches)
                        continue
                    except Exception as match_exc:
                        logger.warning(
                            "MatchingEngine failed for %s vs %s: %s",
                            src_a, src_b, str(match_exc),
                        )

                # Fallback: simple key-based matching
                pair_matches = self._fallback_match(
                    src_a, src_b, records_a, records_b,
                    strategy, threshold,
                )
                all_matches.extend(pair_matches)
                total_matched += len(pair_matches)

            # Observe match confidences
            for match in all_matches:
                conf = match.get("confidence", 0.0)
                if isinstance(conf, (int, float)):
                    _observe_confidence(float(conf))

            _inc_records_matched(strategy, total_matched)

            # Record audit event
            if self._audit is not None:
                try:
                    self._audit.record_event(
                        job_id=job_id,
                        event_type="record_matching",
                        details={
                            "strategy": strategy,
                            "threshold": threshold,
                            "total_matches": total_matched,
                            "source_pairs": len(pairs),
                        },
                    )
                except Exception:
                    pass

            # Record provenance
            provenance_hash = self._compute_provenance(
                "match_records",
                {
                    "strategy": strategy,
                    "threshold": threshold,
                    "source_pairs": len(pairs),
                },
                {"total_matches": total_matched},
            )

            result.status = "completed"
            result.records_processed = total_matched
            result.output_summary = {
                "matches": all_matches,
                "total_matches": total_matched,
                "source_pairs": len(pairs),
                "strategy": strategy,
                "provenance_hash": provenance_hash,
            }

            # Update statistics
            self._statistics.total_records_matched += total_matched
            strategy_key = strategy
            self._statistics.by_match_strategy[strategy_key] = (
                self._statistics.by_match_strategy.get(strategy_key, 0)
                + total_matched
            )

            logger.info(
                "Stage MATCH: %d matches across %d pairs, "
                "strategy=%s, threshold=%.2f",
                total_matched, len(pairs), strategy, threshold,
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("matching")
            logger.error(
                "Stage MATCH failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Stage 4 - COMPARE
    # ==================================================================

    def _stage_compare(
        self,
        job_id: str,
        matches: List[Dict[str, Any]],
        source_data: Dict[str, List[Dict[str, Any]]],
        tolerance_rules: Dict[str, Any],
        field_types: Dict[str, str],
    ) -> PipelineStageResult:
        """Compare matched records field-by-field.

        For each matched pair, runs the comparison engine to evaluate
        every field. Collects all field-level comparisons.

        Args:
            job_id: Unique job identifier.
            matches: Matched record pairs from Stage 3.
            source_data: Aligned source data.
            tolerance_rules: Per-field tolerance configuration.
            field_types: Per-field type mappings.

        Returns:
            PipelineStageResult with comparisons list.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="compare")

        try:
            all_comparisons: List[Dict[str, Any]] = []
            match_count = 0
            mismatch_count = 0

            for match in matches:
                record_a = match.get("record_a", {})
                record_b = match.get("record_b", {})
                source_a = match.get("source_a", "")
                source_b = match.get("source_b", "")

                # Use comparison engine if available
                if self._comparison is not None:
                    try:
                        comp_result = self._comparison.compare_records(
                            record_a=record_a,
                            record_b=record_b,
                            source_a=source_a,
                            source_b=source_b,
                            tolerance_rules=tolerance_rules,
                            field_types=field_types,
                        )
                        field_comparisons = (
                            getattr(comp_result, "comparisons", [])
                            if hasattr(comp_result, "comparisons")
                            else comp_result
                            if isinstance(comp_result, list)
                            else []
                        )
                        all_comparisons.extend(field_comparisons)
                        for fc in field_comparisons:
                            fc_result = fc.get("result", "")
                            if fc_result == "match":
                                match_count += 1
                            else:
                                mismatch_count += 1
                        continue
                    except Exception as comp_exc:
                        logger.warning(
                            "ComparisonEngine failed: %s",
                            str(comp_exc),
                        )

                # Fallback: simple field comparison
                field_comps = self._fallback_compare(
                    record_a, record_b, source_a, source_b,
                    tolerance_rules, field_types,
                )
                all_comparisons.extend(field_comps)
                for fc in field_comps:
                    if fc.get("result") == "match":
                        match_count += 1
                    else:
                        mismatch_count += 1

            _inc_comparisons("match", match_count)
            _inc_comparisons("mismatch", mismatch_count)

            # Record audit event
            if self._audit is not None:
                try:
                    self._audit.record_event(
                        job_id=job_id,
                        event_type="field_comparison",
                        details={
                            "total_comparisons": len(all_comparisons),
                            "matches": match_count,
                            "mismatches": mismatch_count,
                        },
                    )
                except Exception:
                    pass

            # Record provenance
            provenance_hash = self._compute_provenance(
                "compare_fields",
                {"match_count": len(matches)},
                {
                    "total_comparisons": len(all_comparisons),
                    "matches": match_count,
                    "mismatches": mismatch_count,
                },
            )

            result.status = "completed"
            result.records_processed = len(all_comparisons)
            result.output_summary = {
                "comparisons": all_comparisons,
                "total_comparisons": len(all_comparisons),
                "matches": match_count,
                "mismatches": mismatch_count,
                "provenance_hash": provenance_hash,
            }

            self._statistics.total_comparisons += len(all_comparisons)

            logger.info(
                "Stage COMPARE: %d comparisons "
                "(%d match, %d mismatch)",
                len(all_comparisons), match_count, mismatch_count,
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("comparison")
            logger.error(
                "Stage COMPARE failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Stage 5 - DETECT
    # ==================================================================

    def _stage_detect(
        self,
        job_id: str,
        comparisons: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
    ) -> PipelineStageResult:
        """Detect and classify discrepancies from field comparisons.

        Runs the discrepancy detector on all comparisons to identify,
        classify, and prioritize conflicts between sources.

        Args:
            job_id: Unique job identifier.
            comparisons: Field-level comparisons from Stage 4.
            matches: Matched record pairs from Stage 3.

        Returns:
            PipelineStageResult with discrepancies list.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="detect")

        try:
            all_discrepancies: List[Dict[str, Any]] = []

            # Filter to mismatches only
            mismatches = [
                c for c in comparisons
                if c.get("result") != "match"
            ]

            if not mismatches:
                result.status = "completed"
                result.records_processed = 0
                result.output_summary = {
                    "discrepancies": [],
                    "total_discrepancies": 0,
                }
                logger.info("Stage DETECT: no mismatches to analyze")
                result.duration_ms = (time.time() - t0) * 1000.0
                return result

            # Use discrepancy detector engine if available
            if self._discrepancy is not None:
                try:
                    detect_result = (
                        self._discrepancy.detect_discrepancies(
                            comparisons=mismatches,
                            matches=matches,
                        )
                    )
                    all_discrepancies = (
                        getattr(detect_result, "discrepancies", [])
                        if hasattr(detect_result, "discrepancies")
                        else detect_result
                        if isinstance(detect_result, list)
                        else []
                    )
                except Exception as det_exc:
                    logger.warning(
                        "DiscrepancyDetectorEngine failed: %s",
                        str(det_exc),
                    )
                    all_discrepancies = []

            # Fallback: convert mismatches to discrepancies
            if not all_discrepancies:
                all_discrepancies = self._fallback_detect(mismatches)

            # Classify and prioritize
            for disc in all_discrepancies:
                disc_type = disc.get("type", "value_mismatch")
                severity = disc.get("severity", "medium")
                _inc_discrepancies(disc_type, severity)

                magnitude = disc.get("magnitude", 0.0)
                if isinstance(magnitude, (int, float)) and magnitude > 0:
                    _observe_magnitude(float(magnitude))

            # Record audit event
            if self._audit is not None:
                try:
                    self._audit.record_event(
                        job_id=job_id,
                        event_type="discrepancy_detection",
                        details={
                            "total_discrepancies": len(
                                all_discrepancies
                            ),
                            "from_mismatches": len(mismatches),
                        },
                    )
                except Exception:
                    pass

            # Record provenance
            provenance_hash = self._compute_provenance(
                "detect_discrepancies",
                {"mismatches": len(mismatches)},
                {"discrepancies": len(all_discrepancies)},
            )

            result.status = "completed"
            result.records_processed = len(all_discrepancies)
            result.output_summary = {
                "discrepancies": all_discrepancies,
                "total_discrepancies": len(all_discrepancies),
                "provenance_hash": provenance_hash,
            }

            # Update statistics
            self._statistics.total_discrepancies += len(
                all_discrepancies
            )
            for disc in all_discrepancies:
                dtype = disc.get("type", "unknown")
                self._statistics.by_discrepancy_type[dtype] = (
                    self._statistics.by_discrepancy_type.get(dtype, 0)
                    + 1
                )

            logger.info(
                "Stage DETECT: %d discrepancies from %d mismatches",
                len(all_discrepancies), len(mismatches),
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("detection")
            logger.error(
                "Stage DETECT failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Stage 6 - RESOLVE
    # ==================================================================

    def _stage_resolve(
        self,
        job_id: str,
        discrepancies: List[Dict[str, Any]],
        source_credibilities: Dict[str, float],
        source_data: Dict[str, List[Dict[str, Any]]],
        strategy: str,
    ) -> PipelineStageResult:
        """Resolve detected discrepancies using the configured strategy.

        For each discrepancy, applies the resolution strategy to
        determine the authoritative value. Records resolution events
        for audit.

        Args:
            job_id: Unique job identifier.
            discrepancies: Detected discrepancies from Stage 5.
            source_credibilities: Per-source credibility scores.
            source_data: Aligned source data.
            strategy: Resolution strategy name.

        Returns:
            PipelineStageResult with resolutions list.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="resolve")

        try:
            all_resolutions: List[Dict[str, Any]] = []

            if not discrepancies:
                result.status = "completed"
                result.records_processed = 0
                result.output_summary = {
                    "resolutions": [],
                    "total_resolutions": 0,
                }
                logger.info("Stage RESOLVE: no discrepancies to resolve")
                result.duration_ms = (time.time() - t0) * 1000.0
                return result

            for disc in discrepancies:
                # Use resolution engine if available
                if self._resolution is not None:
                    try:
                        res_result = self._resolution.resolve(
                            discrepancy=disc,
                            source_credibilities=source_credibilities,
                            strategy=strategy,
                        )
                        resolution = (
                            res_result
                            if isinstance(res_result, dict)
                            else getattr(res_result, "__dict__", {})
                        )
                        all_resolutions.append(resolution)
                        _inc_resolutions(strategy)
                        continue
                    except Exception as res_exc:
                        logger.warning(
                            "ResolutionEngine failed: %s",
                            str(res_exc),
                        )

                # Fallback: apply resolution heuristic
                resolution = self._fallback_resolve(
                    disc, source_credibilities, strategy,
                )
                all_resolutions.append(resolution)
                _inc_resolutions(strategy)

            # Record audit event
            if self._audit is not None:
                try:
                    self._audit.record_event(
                        job_id=job_id,
                        event_type="conflict_resolution",
                        details={
                            "strategy": strategy,
                            "total_resolutions": len(all_resolutions),
                        },
                    )
                except Exception:
                    pass

            # Record provenance
            provenance_hash = self._compute_provenance(
                "resolve_conflicts",
                {
                    "strategy": strategy,
                    "discrepancies": len(discrepancies),
                },
                {"resolutions": len(all_resolutions)},
            )

            result.status = "completed"
            result.records_processed = len(all_resolutions)
            result.output_summary = {
                "resolutions": all_resolutions,
                "total_resolutions": len(all_resolutions),
                "strategy": strategy,
                "provenance_hash": provenance_hash,
            }

            # Update statistics
            self._statistics.total_resolutions += len(all_resolutions)
            self._statistics.by_resolution_strategy[strategy] = (
                self._statistics.by_resolution_strategy.get(
                    strategy, 0
                )
                + len(all_resolutions)
            )

            logger.info(
                "Stage RESOLVE: %d resolutions, strategy=%s",
                len(all_resolutions), strategy,
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("resolution")
            logger.error(
                "Stage RESOLVE failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Stage 7 - GOLDEN RECORDS
    # ==================================================================

    def _stage_golden_records(
        self,
        job_id: str,
        source_data: Dict[str, List[Dict[str, Any]]],
        resolutions: List[Dict[str, Any]],
        source_credibilities: Dict[str, float],
    ) -> PipelineStageResult:
        """Assemble golden records from resolved data.

        Groups records by entity key and period, then assembles a
        single golden record for each group by applying resolutions
        and selecting the most authoritative values.

        Args:
            job_id: Unique job identifier.
            source_data: Aligned source data.
            resolutions: Applied resolutions from Stage 6.
            source_credibilities: Per-source credibility scores.

        Returns:
            PipelineStageResult with golden_records list.
        """
        t0 = time.time()
        result = PipelineStageResult(stage="golden")

        try:
            golden_records: List[Dict[str, Any]] = []

            # Build resolution index keyed by (field, entity_key)
            resolution_index: Dict[str, Dict[str, Any]] = {}
            for res in resolutions:
                field_name = res.get("field", "")
                entity_key = res.get("entity_key", "")
                idx_key = f"{entity_key}:{field_name}"
                resolution_index[idx_key] = res

            # Collect all unique records across sources
            # Group by entity key (first field values as composite key)
            entity_groups: Dict[str, List[Dict[str, Any]]] = {}
            for source_name, records in source_data.items():
                for record in records:
                    entity_key = self._compute_entity_key(record)
                    if entity_key not in entity_groups:
                        entity_groups[entity_key] = []
                    entity_groups[entity_key].append({
                        "source": source_name,
                        "record": record,
                    })

            # Assemble golden record for each entity group
            for entity_key, group in entity_groups.items():
                golden = self._assemble_golden_record(
                    entity_key, group, resolution_index,
                    source_credibilities,
                )
                golden_records.append(golden)

            _inc_golden_records("created", len(golden_records))

            # Record audit event
            if self._audit is not None:
                try:
                    self._audit.record_event(
                        job_id=job_id,
                        event_type="golden_record_assembly",
                        details={
                            "total_golden_records": len(golden_records),
                            "entity_groups": len(entity_groups),
                        },
                    )
                except Exception:
                    pass

            # Record provenance
            provenance_hash = self._compute_provenance(
                "assemble_golden_records",
                {
                    "entity_groups": len(entity_groups),
                    "resolutions": len(resolutions),
                },
                {"golden_records": len(golden_records)},
            )

            result.status = "completed"
            result.records_processed = len(golden_records)
            result.output_summary = {
                "golden_records": golden_records,
                "total_golden_records": len(golden_records),
                "provenance_hash": provenance_hash,
            }

            self._statistics.total_golden_records += len(golden_records)

            logger.info(
                "Stage GOLDEN: %d golden records from %d groups",
                len(golden_records), len(entity_groups),
            )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            _inc_errors("golden_record")
            logger.error(
                "Stage GOLDEN failed: %s", str(exc), exc_info=True,
            )

        result.duration_ms = (time.time() - t0) * 1000.0
        return result

    # ==================================================================
    # Private: Strategy selection
    # ==================================================================

    def _select_match_strategy(
        self,
        source_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        """Auto-select the best matching strategy based on data analysis.

        Decision logic:
            - If exact key fields present with consistent formats
              -> exact
            - If mixed formats or inconsistent naming -> fuzzy
            - If temporal fields with date mismatches -> temporal
            - If complex multi-field keys -> composite
            - Default -> exact

        Args:
            source_data: Dict mapping source names to record lists.

        Returns:
            Dict with 'strategy' key and 'reason' explanation.
        """
        # Collect field names and sample values
        all_field_sets: List[Set[str]] = []
        sample_records: List[Dict[str, Any]] = []
        for source_name, records in source_data.items():
            if records and isinstance(records[0], dict):
                all_field_sets.append(set(records[0].keys()))
                sample_records.append(records[0])

        if not all_field_sets:
            return {"strategy": "exact", "reason": "no data to analyze"}

        # Check field name consistency
        common = set.intersection(*all_field_sets)
        union = set.union(*all_field_sets)

        field_overlap = (
            len(common) / len(union) if union else 0.0
        )

        # Check for ID-like fields
        id_fields = [
            f for f in common
            if any(
                kw in f.lower()
                for kw in ("id", "key", "code", "number", "num")
            )
        ]

        # Check for date/temporal fields
        date_fields = [
            f for f in common
            if any(
                kw in f.lower()
                for kw in ("date", "time", "period", "year", "month")
            )
        ]

        # Decision tree
        if id_fields and field_overlap >= 0.8:
            return {
                "strategy": "exact",
                "reason": (
                    f"ID fields present ({id_fields}), "
                    f"high field overlap ({field_overlap:.0%})"
                ),
            }

        if date_fields and not id_fields:
            return {
                "strategy": "temporal",
                "reason": (
                    f"Temporal fields present ({date_fields}), "
                    f"no ID fields"
                ),
            }

        if field_overlap < 0.5:
            return {
                "strategy": "fuzzy",
                "reason": (
                    f"Low field overlap ({field_overlap:.0%}), "
                    f"schema differences"
                ),
            }

        if len(id_fields) > 1 or (id_fields and date_fields):
            return {
                "strategy": "composite",
                "reason": (
                    f"Multiple key types: ids={id_fields}, "
                    f"dates={date_fields}"
                ),
            }

        return {
            "strategy": "exact",
            "reason": "default (sufficient field overlap)",
        }

    # ==================================================================
    # Private: Field type inference
    # ==================================================================

    def _infer_field_types(
        self, records: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Detect field types from a list of records.

        Analyzes value patterns to classify each field as:
        numeric, date, boolean, or string.

        Args:
            records: List of record dicts to analyze.

        Returns:
            Dict mapping field names to type strings.
        """
        if not records:
            return {}

        field_types: Dict[str, str] = {}
        sample_size = min(len(records), 50)
        sample = records[:sample_size]

        # Collect all field names from sample
        all_fields: Set[str] = set()
        for record in sample:
            if isinstance(record, dict):
                all_fields.update(record.keys())

        for field_name in all_fields:
            values = [
                r.get(field_name) for r in sample
                if isinstance(r, dict) and r.get(field_name) is not None
            ]
            if not values:
                field_types[field_name] = "string"
                continue

            field_types[field_name] = self._classify_field_type(values)

        return field_types

    def _infer_field_types_from_data(
        self,
        source_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        """Infer field types across all sources.

        Combines field type inference from all sources, using majority
        vote when sources disagree on a field type.

        Args:
            source_data: Dict mapping source names to record lists.

        Returns:
            Dict mapping field names to type strings.
        """
        type_votes: Dict[str, Dict[str, int]] = {}

        for source_name, records in source_data.items():
            source_types = self._infer_field_types(records)
            for field_name, field_type in source_types.items():
                if field_name not in type_votes:
                    type_votes[field_name] = {}
                type_votes[field_name][field_type] = (
                    type_votes[field_name].get(field_type, 0) + 1
                )

        # Majority vote
        result: Dict[str, str] = {}
        for field_name, votes in type_votes.items():
            if votes:
                result[field_name] = max(votes, key=votes.get)  # type: ignore[arg-type]
            else:
                result[field_name] = "string"

        return result

    def _classify_field_type(
        self, values: List[Any],
    ) -> str:
        """Classify a field's type based on sample values.

        Args:
            values: Sample values for the field.

        Returns:
            Type string: 'numeric', 'date', 'boolean', or 'string'.
        """
        if not values:
            return "string"

        numeric_count = 0
        date_count = 0
        bool_count = 0

        for val in values:
            if isinstance(val, bool):
                bool_count += 1
            elif isinstance(val, (int, float)):
                numeric_count += 1
            elif isinstance(val, str):
                # Check if numeric string
                try:
                    float(val.replace(",", ""))
                    numeric_count += 1
                    continue
                except (ValueError, AttributeError):
                    pass

                # Check if date string
                if self._looks_like_date(val):
                    date_count += 1
                elif val.lower() in ("true", "false", "yes", "no"):
                    bool_count += 1

        total = len(values)
        threshold = 0.6

        if numeric_count / total >= threshold:
            return "numeric"
        if date_count / total >= threshold:
            return "date"
        if bool_count / total >= threshold:
            return "boolean"
        return "string"

    # ==================================================================
    # Private: Provenance
    # ==================================================================

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute a SHA-256 provenance hash and record it in the chain.

        Args:
            operation: Operation name for the provenance entry.
            input_data: Input data to hash.
            output_data: Output data to hash.

        Returns:
            SHA-256 chain hash string.
        """
        input_hash = self._provenance.build_hash(input_data)
        output_hash = self._provenance.build_hash(output_data)

        chain_hash = self._provenance.add_to_chain(
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={"operation": operation},
        )

        return chain_hash

    # ==================================================================
    # Private: Pipeline finalization
    # ==================================================================

    def _finalize_pipeline(
        self,
        report: Dict[str, Any],
        pipeline_start: float,
        job_id: str,
    ) -> None:
        """Finalize the pipeline by recording metrics and provenance.

        Args:
            report: The pipeline report dict to update.
            pipeline_start: Pipeline start timestamp.
            job_id: Unique job identifier.
        """
        elapsed = time.time() - pipeline_start
        report["total_time_ms"] = elapsed * 1000.0

        # Record final provenance
        provenance_hash = self._compute_provenance(
            "pipeline_complete",
            {"job_id": job_id, "status": report["status"]},
            {
                "matches": len(report.get("matches", [])),
                "discrepancies": len(report.get("discrepancies", [])),
                "golden_records": len(report.get("golden_records", [])),
            },
        )
        report["provenance_hash"] = provenance_hash

        # Build statistics summary
        report["statistics"] = {
            "sources_registered": report.get("sources_registered", 0),
            "total_records": report.get("total_records", 0),
            "total_matches": len(report.get("matches", [])),
            "total_comparisons": len(report.get("comparisons", [])),
            "total_discrepancies": len(
                report.get("discrepancies", [])
            ),
            "total_resolutions": len(report.get("resolutions", [])),
            "total_golden_records": len(
                report.get("golden_records", [])
            ),
        }

        # Update aggregate statistics
        self._statistics.total_runs += 1
        status = report["status"]
        self._statistics.by_status[status] = (
            self._statistics.by_status.get(status, 0) + 1
        )

        # Update average match confidence
        match_confs = [
            m.get("confidence", 0.0)
            for m in report.get("matches", [])
            if isinstance(m.get("confidence"), (int, float))
        ]
        if match_confs:
            n = self._statistics.total_runs
            old_avg = self._statistics.avg_match_confidence
            new_avg = _safe_mean(match_confs)
            self._statistics.avg_match_confidence = (
                (old_avg * (n - 1) + new_avg) / n if n > 0 else new_avg
            )

        # Record metrics
        _observe_duration(elapsed)
        _inc_jobs_processed(status)
        _set_active_jobs(0)

        logger.info(
            "Pipeline %s finalized: status=%s, "
            "%d matches, %d discrepancies, %d golden records, "
            "%.1fms",
            job_id[:8], status,
            len(report.get("matches", [])),
            len(report.get("discrepancies", [])),
            len(report.get("golden_records", [])),
            elapsed * 1000.0,
        )

    def _build_error_report(
        self,
        job_id: str,
        error: str,
        pipeline_start: float,
        stage_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build an error report when the pipeline fails early.

        Args:
            job_id: Unique job identifier.
            error: Error message.
            pipeline_start: Pipeline start timestamp.
            stage_results: Any completed stage results.

        Returns:
            Reconciliation report dict with failed status.
        """
        elapsed = time.time() - pipeline_start

        self._statistics.total_runs += 1
        self._statistics.by_status["failed"] = (
            self._statistics.by_status.get("failed", 0) + 1
        )

        return {
            "job_id": job_id,
            "status": "failed",
            "sources_registered": 0,
            "total_records": 0,
            "matches": [],
            "comparisons": [],
            "discrepancies": [],
            "resolutions": [],
            "golden_records": [],
            "stage_results": stage_results,
            "statistics": {},
            "total_time_ms": elapsed * 1000.0,
            "provenance_hash": "",
            "error": error,
        }

    def _stage_to_dict(
        self, stage_result: PipelineStageResult,
    ) -> Dict[str, Any]:
        """Convert a PipelineStageResult to a plain dict.

        Args:
            stage_result: The stage result dataclass.

        Returns:
            Dict representation of the stage result (excluding
            large output data).
        """
        return {
            "stage": stage_result.stage,
            "status": stage_result.status,
            "duration_ms": stage_result.duration_ms,
            "records_processed": stage_result.records_processed,
            "error": stage_result.error,
        }

    # ==================================================================
    # Private: Fallback implementations
    # ==================================================================

    def _fallback_match(
        self,
        src_a: str,
        src_b: str,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        strategy: str,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Fallback record matching when MatchingEngine is unavailable.

        Uses simple key-based matching on common fields.

        Args:
            src_a: Source A identifier.
            src_b: Source B identifier.
            records_a: Records from source A.
            records_b: Records from source B.
            strategy: Matching strategy (used for labeling).
            threshold: Minimum confidence threshold.

        Returns:
            List of match dicts.
        """
        matches: List[Dict[str, Any]] = []

        if not records_a or not records_b:
            return matches

        # Find common key-like fields
        fields_a = set(records_a[0].keys()) if records_a else set()
        fields_b = set(records_b[0].keys()) if records_b else set()
        common = fields_a & fields_b

        key_fields = [
            f for f in common
            if any(
                kw in f.lower()
                for kw in ("id", "key", "code", "number")
            )
        ]

        if not key_fields:
            key_fields = list(common)[:3]

        if not key_fields:
            return matches

        # Build index on source B
        b_index: Dict[str, Dict[str, Any]] = {}
        for rec_b in records_b:
            key = self._build_match_key(rec_b, key_fields)
            if key:
                b_index[key] = rec_b

        # Match source A against index
        for rec_a in records_a:
            key = self._build_match_key(rec_a, key_fields)
            if key and key in b_index:
                rec_b = b_index[key]

                # Compute simple confidence
                confidence = self._compute_match_confidence(
                    rec_a, rec_b, common,
                )

                if confidence >= threshold:
                    matches.append({
                        "match_id": str(uuid4()),
                        "source_a": src_a,
                        "source_b": src_b,
                        "record_a": rec_a,
                        "record_b": rec_b,
                        "key_fields": key_fields,
                        "key_value": key,
                        "confidence": confidence,
                        "strategy": strategy,
                    })

        return matches

    def _fallback_compare(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        source_a: str,
        source_b: str,
        tolerance_rules: Dict[str, Any],
        field_types: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Fallback field comparison when ComparisonEngine is unavailable.

        Compares each common field between two records.

        Args:
            record_a: Record from source A.
            record_b: Record from source B.
            source_a: Source A identifier.
            source_b: Source B identifier.
            tolerance_rules: Per-field tolerance rules.
            field_types: Per-field type mappings.

        Returns:
            List of field comparison dicts.
        """
        comparisons: List[Dict[str, Any]] = []
        common_fields = set(record_a.keys()) & set(record_b.keys())

        for field_name in sorted(common_fields):
            val_a = record_a.get(field_name)
            val_b = record_b.get(field_name)
            field_type = field_types.get(field_name, "string")
            tolerance = tolerance_rules.get(field_name, {})

            comparison = self._compare_field_values(
                field_name, val_a, val_b, source_a, source_b,
                field_type, tolerance,
            )
            comparisons.append(comparison)

        return comparisons

    def _fallback_detect(
        self, mismatches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fallback discrepancy detection from field mismatches.

        Converts each mismatch into a classified discrepancy.

        Args:
            mismatches: Field comparisons with result != 'match'.

        Returns:
            List of discrepancy dicts.
        """
        discrepancies: List[Dict[str, Any]] = []

        for mismatch in mismatches:
            disc_type = "value_mismatch"
            result = mismatch.get("result", "")

            if result == "missing_left" or result == "missing_right":
                disc_type = "missing_record"
            elif result == "type_mismatch":
                disc_type = "format_difference"

            magnitude = mismatch.get("deviation", 0.0)
            if isinstance(magnitude, (int, float)):
                magnitude = abs(float(magnitude))
            else:
                magnitude = 0.0

            # Classify severity
            severity = self._classify_severity(magnitude)

            discrepancies.append({
                "discrepancy_id": str(uuid4()),
                "type": disc_type,
                "severity": severity,
                "field": mismatch.get("field", ""),
                "source_a": mismatch.get("source_a", ""),
                "source_b": mismatch.get("source_b", ""),
                "value_a": mismatch.get("value_a"),
                "value_b": mismatch.get("value_b"),
                "magnitude": magnitude,
                "entity_key": mismatch.get("entity_key", ""),
            })

        return discrepancies

    def _fallback_resolve(
        self,
        discrepancy: Dict[str, Any],
        source_credibilities: Dict[str, float],
        strategy: str,
    ) -> Dict[str, Any]:
        """Fallback conflict resolution using heuristics.

        Applies the resolution strategy to determine which source
        value to accept.

        Args:
            discrepancy: The discrepancy to resolve.
            source_credibilities: Per-source credibility scores.
            strategy: Resolution strategy name.

        Returns:
            Resolution dict with resolved value and rationale.
        """
        source_a = discrepancy.get("source_a", "")
        source_b = discrepancy.get("source_b", "")
        value_a = discrepancy.get("value_a")
        value_b = discrepancy.get("value_b")

        resolved_value = value_a
        resolved_source = source_a
        rationale = ""

        if strategy == "source_priority":
            cred_a = source_credibilities.get(source_a, 0.5)
            cred_b = source_credibilities.get(source_b, 0.5)
            if cred_b > cred_a:
                resolved_value = value_b
                resolved_source = source_b
            rationale = (
                f"Source priority: {resolved_source} "
                f"(credibility {source_credibilities.get(resolved_source, 0.5):.2f})"
            )

        elif strategy == "most_recent":
            # Without timestamps, fall back to source priority
            resolved_value = value_a
            resolved_source = source_a
            rationale = "Most recent: defaulted to source A"

        elif strategy == "most_complete":
            # Prefer non-None, non-empty value
            if value_a is None or value_a == "":
                resolved_value = value_b
                resolved_source = source_b
            elif value_b is None or value_b == "":
                resolved_value = value_a
                resolved_source = source_a
            rationale = (
                f"Most complete: selected {resolved_source}"
            )

        elif strategy == "average":
            if (
                isinstance(value_a, (int, float))
                and isinstance(value_b, (int, float))
            ):
                resolved_value = (value_a + value_b) / 2.0
                resolved_source = "averaged"
                rationale = (
                    f"Average of {value_a} and {value_b}"
                )
            else:
                resolved_value = value_a
                resolved_source = source_a
                rationale = "Average: non-numeric, defaulted to A"

        elif strategy == "median":
            if (
                isinstance(value_a, (int, float))
                and isinstance(value_b, (int, float))
            ):
                resolved_value = (value_a + value_b) / 2.0
                resolved_source = "median"
                rationale = (
                    f"Median of {value_a} and {value_b}"
                )
            else:
                resolved_value = value_a
                resolved_source = source_a
                rationale = "Median: non-numeric, defaulted to A"

        else:
            rationale = f"Unknown strategy '{strategy}', defaulted to A"

        return {
            "resolution_id": str(uuid4()),
            "discrepancy_id": discrepancy.get("discrepancy_id", ""),
            "field": discrepancy.get("field", ""),
            "entity_key": discrepancy.get("entity_key", ""),
            "resolved_value": resolved_value,
            "resolved_source": resolved_source,
            "strategy": strategy,
            "rationale": rationale,
            "confidence": 0.8 if strategy != "average" else 0.7,
        }

    # ==================================================================
    # Private: Utility methods
    # ==================================================================

    def _generate_source_pairs(
        self, source_ids: List[str],
    ) -> List[Tuple[str, str]]:
        """Generate all unique source pairs for comparison.

        Args:
            source_ids: List of source identifiers.

        Returns:
            List of (source_a, source_b) tuples.
        """
        pairs: List[Tuple[str, str]] = []
        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                pairs.append((source_ids[i], source_ids[j]))
        return pairs

    def _compute_source_credibility(
        self,
        source_name: str,
        records: List[Dict[str, Any]],
        configured_priorities: Dict[str, float],
    ) -> float:
        """Compute a credibility score for a data source.

        Uses configured priority if available, otherwise computes
        from record completeness.

        Args:
            source_name: Source identifier.
            records: Records from the source.
            configured_priorities: Configured source priority scores.

        Returns:
            Credibility score (0.0-1.0).
        """
        # Use configured priority if available
        if source_name in configured_priorities:
            return float(
                max(0.0, min(1.0, configured_priorities[source_name]))
            )

        if not records:
            return 0.5

        # Compute from field completeness
        total_fields = 0
        non_null_fields = 0

        sample = records[:min(len(records), 100)]
        for record in sample:
            if isinstance(record, dict):
                for val in record.values():
                    total_fields += 1
                    if val is not None and val != "":
                        non_null_fields += 1

        if total_fields == 0:
            return 0.5

        completeness = non_null_fields / total_fields
        # Scale to 0.3-0.9 range
        return 0.3 + (completeness * 0.6)

    def _normalize_record(
        self, record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a record for consistent comparison.

        Strips whitespace from strings, normalizes numeric strings,
        and lowercases string values for comparison.

        Args:
            record: The record dict to normalize.

        Returns:
            Normalized record dict.
        """
        normalized: Dict[str, Any] = {}

        for key, value in record.items():
            if isinstance(value, str):
                value = value.strip()
            normalized[key] = value

        return normalized

    def _build_match_key(
        self,
        record: Dict[str, Any],
        key_fields: List[str],
    ) -> str:
        """Build a composite match key from key fields.

        Args:
            record: The record dict.
            key_fields: Fields to include in the key.

        Returns:
            Composite key string (pipe-delimited).
        """
        parts: List[str] = []
        for field_name in sorted(key_fields):
            val = record.get(field_name, "")
            if val is None:
                val = ""
            parts.append(str(val).strip().lower())

        return "|".join(parts)

    def _compute_match_confidence(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        common_fields: Set[str],
    ) -> float:
        """Compute a simple match confidence between two records.

        Calculates the fraction of common fields with matching values.

        Args:
            record_a: First record.
            record_b: Second record.
            common_fields: Set of fields present in both records.

        Returns:
            Confidence score (0.0-1.0).
        """
        if not common_fields:
            return 0.0

        matching = 0
        total = 0

        for field_name in common_fields:
            val_a = record_a.get(field_name)
            val_b = record_b.get(field_name)
            total += 1

            if val_a is None and val_b is None:
                matching += 1
            elif val_a is not None and val_b is not None:
                if str(val_a).strip().lower() == str(val_b).strip().lower():
                    matching += 1

        return matching / total if total > 0 else 0.0

    def _compare_field_values(
        self,
        field_name: str,
        val_a: Any,
        val_b: Any,
        source_a: str,
        source_b: str,
        field_type: str,
        tolerance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two field values and produce a comparison result.

        Args:
            field_name: Name of the field.
            val_a: Value from source A.
            val_b: Value from source B.
            source_a: Source A identifier.
            source_b: Source B identifier.
            field_type: Field type (numeric, date, string, boolean).
            tolerance: Tolerance rule for the field.

        Returns:
            Comparison result dict.
        """
        comparison: Dict[str, Any] = {
            "field": field_name,
            "source_a": source_a,
            "source_b": source_b,
            "value_a": val_a,
            "value_b": val_b,
            "field_type": field_type,
            "result": "match",
            "deviation": 0.0,
        }

        # Handle None values
        if val_a is None and val_b is None:
            comparison["result"] = "match"
            return comparison
        if val_a is None:
            comparison["result"] = "missing_left"
            return comparison
        if val_b is None:
            comparison["result"] = "missing_right"
            return comparison

        # Type-specific comparison
        if field_type == "numeric":
            comparison = self._compare_numeric(
                comparison, val_a, val_b, tolerance,
            )
        elif field_type == "date":
            comparison = self._compare_date(
                comparison, val_a, val_b, tolerance,
            )
        elif field_type == "boolean":
            comparison = self._compare_boolean(
                comparison, val_a, val_b,
            )
        else:
            comparison = self._compare_string(
                comparison, val_a, val_b,
            )

        return comparison

    def _compare_numeric(
        self,
        comparison: Dict[str, Any],
        val_a: Any,
        val_b: Any,
        tolerance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two numeric values with optional tolerance.

        Args:
            comparison: Base comparison dict to update.
            val_a: Numeric value from source A.
            val_b: Numeric value from source B.
            tolerance: Tolerance configuration.

        Returns:
            Updated comparison dict.
        """
        try:
            num_a = float(str(val_a).replace(",", ""))
            num_b = float(str(val_b).replace(",", ""))
        except (ValueError, TypeError):
            comparison["result"] = "type_mismatch"
            return comparison

        if num_a == num_b:
            comparison["result"] = "match"
            comparison["deviation"] = 0.0
            return comparison

        # Compute deviation
        abs_diff = abs(num_a - num_b)
        denominator = max(abs(num_a), abs(num_b), 1e-10)
        relative_deviation = abs_diff / denominator

        comparison["deviation"] = relative_deviation

        # Apply tolerance
        tol_type = tolerance.get("type", "relative")
        tol_threshold = float(tolerance.get("threshold", 0.0))

        if tol_type == "absolute" and abs_diff <= tol_threshold:
            comparison["result"] = "match"
        elif tol_type == "relative" and relative_deviation <= tol_threshold:
            comparison["result"] = "match"
        else:
            comparison["result"] = "mismatch"

        return comparison

    def _compare_date(
        self,
        comparison: Dict[str, Any],
        val_a: Any,
        val_b: Any,
        tolerance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two date values with optional tolerance.

        Args:
            comparison: Base comparison dict to update.
            val_a: Date value from source A.
            val_b: Date value from source B.
            tolerance: Tolerance configuration.

        Returns:
            Updated comparison dict.
        """
        str_a = str(val_a).strip()
        str_b = str(val_b).strip()

        if str_a == str_b:
            comparison["result"] = "match"
            return comparison

        # Try to parse dates for day-level comparison
        date_a = self._try_parse_date(str_a)
        date_b = self._try_parse_date(str_b)

        if date_a is not None and date_b is not None:
            diff_days = abs((date_a - date_b).days)
            threshold_days = int(
                tolerance.get("threshold_days", 0)
            )

            comparison["deviation"] = diff_days

            if diff_days <= threshold_days:
                comparison["result"] = "match"
            else:
                comparison["result"] = "mismatch"
        else:
            comparison["result"] = "mismatch"

        return comparison

    def _compare_boolean(
        self,
        comparison: Dict[str, Any],
        val_a: Any,
        val_b: Any,
    ) -> Dict[str, Any]:
        """Compare two boolean values.

        Args:
            comparison: Base comparison dict to update.
            val_a: Boolean value from source A.
            val_b: Boolean value from source B.

        Returns:
            Updated comparison dict.
        """
        bool_a = self._to_bool(val_a)
        bool_b = self._to_bool(val_b)

        if bool_a == bool_b:
            comparison["result"] = "match"
        else:
            comparison["result"] = "mismatch"
            comparison["deviation"] = 1.0

        return comparison

    def _compare_string(
        self,
        comparison: Dict[str, Any],
        val_a: Any,
        val_b: Any,
    ) -> Dict[str, Any]:
        """Compare two string values (case-insensitive).

        Args:
            comparison: Base comparison dict to update.
            val_a: String value from source A.
            val_b: String value from source B.

        Returns:
            Updated comparison dict.
        """
        str_a = str(val_a).strip().lower()
        str_b = str(val_b).strip().lower()

        if str_a == str_b:
            comparison["result"] = "match"
        else:
            comparison["result"] = "mismatch"

        return comparison

    def _compute_entity_key(
        self, record: Dict[str, Any],
    ) -> str:
        """Compute a stable entity key for grouping records.

        Uses all field values to produce a deterministic hash key.
        Prefers ID-like fields if available.

        Args:
            record: The record dict.

        Returns:
            Entity key string.
        """
        # Look for ID-like fields first
        id_fields = [
            f for f in record.keys()
            if any(
                kw in f.lower()
                for kw in ("id", "key", "code", "number")
            )
        ]

        if id_fields:
            parts = [
                str(record.get(f, "")).strip().lower()
                for f in sorted(id_fields)
            ]
            return "|".join(parts)

        # Fall back to hash of all values
        sorted_items = sorted(record.items())
        key_str = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]

    def _assemble_golden_record(
        self,
        entity_key: str,
        group: List[Dict[str, Any]],
        resolution_index: Dict[str, Dict[str, Any]],
        source_credibilities: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assemble a single golden record from a group of source records.

        For each field, uses the resolution if one exists, otherwise
        selects the value from the highest-credibility source.

        Args:
            entity_key: Entity identifier.
            group: List of dicts with 'source' and 'record' keys.
            resolution_index: Resolution lookup by field/entity key.
            source_credibilities: Per-source credibility scores.

        Returns:
            Golden record dict with source provenance.
        """
        golden: Dict[str, Any] = {
            "entity_key": entity_key,
            "fields": {},
            "source_provenance": {},
            "sources_contributing": [],
        }

        # Collect all fields from all source records
        all_fields: Set[str] = set()
        for item in group:
            record = item.get("record", {})
            all_fields.update(record.keys())

        contributing_sources: Set[str] = set()

        for field_name in sorted(all_fields):
            idx_key = f"{entity_key}:{field_name}"

            # Check resolution index first
            if idx_key in resolution_index:
                res = resolution_index[idx_key]
                golden["fields"][field_name] = res.get(
                    "resolved_value",
                )
                golden["source_provenance"][field_name] = res.get(
                    "resolved_source", "resolution",
                )
                contributing_sources.add(
                    res.get("resolved_source", "resolution")
                )
                continue

            # Select from highest-credibility source
            best_value = None
            best_source = ""
            best_credibility = -1.0

            for item in group:
                source = item.get("source", "")
                record = item.get("record", {})
                if field_name in record:
                    val = record[field_name]
                    cred = source_credibilities.get(source, 0.5)
                    if cred > best_credibility:
                        best_credibility = cred
                        best_value = val
                        best_source = source

            golden["fields"][field_name] = best_value
            golden["source_provenance"][field_name] = best_source
            if best_source:
                contributing_sources.add(best_source)

        golden["sources_contributing"] = sorted(contributing_sources)

        return golden

    def _detect_key_fields(
        self,
        source_data: Dict[str, List[Dict[str, Any]]],
        common_fields: List[str],
    ) -> List[str]:
        """Detect likely key fields from common fields.

        Prefers fields with 'id', 'key', 'code' in the name, and
        validates they have high uniqueness within each source.

        Args:
            source_data: Dict mapping source names to record lists.
            common_fields: Fields common to all sources.

        Returns:
            List of detected key field names.
        """
        id_candidates = [
            f for f in common_fields
            if any(
                kw in f.lower()
                for kw in ("id", "key", "code", "number", "num")
            )
        ]

        if id_candidates:
            # Validate uniqueness
            validated: List[str] = []
            for field_name in id_candidates:
                is_unique = True
                for records in source_data.values():
                    if not records:
                        continue
                    vals = [
                        r.get(field_name) for r in records
                        if isinstance(r, dict) and r.get(field_name) is not None
                    ]
                    if len(vals) != len(set(str(v) for v in vals)):
                        is_unique = False
                        break
                if is_unique:
                    validated.append(field_name)
            if validated:
                return validated
            return id_candidates[:2]

        # No ID-like fields; use first common field
        return common_fields[:1] if common_fields else []

    def _looks_like_date(self, value: str) -> bool:
        """Check if a string looks like a date value.

        Args:
            value: String to check.

        Returns:
            True if the string matches common date patterns.
        """
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}/\d{2}/\d{4}",
            r"\d{2}-\d{2}-\d{4}",
            r"\d{4}/\d{2}/\d{2}",
        ]
        for pattern in date_patterns:
            if re.match(pattern, value.strip()):
                return True
        return False

    def _try_parse_date(self, value: str) -> Optional[datetime]:
        """Attempt to parse a date string.

        Args:
            value: String to parse.

        Returns:
            Parsed datetime or None if parsing fails.
        """
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
        return None

    def _to_bool(self, value: Any) -> Optional[bool]:
        """Convert a value to boolean.

        Args:
            value: Value to convert.

        Returns:
            Boolean value or None if conversion fails.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "yes", "1", "on"):
                return True
            if lower in ("false", "no", "0", "off"):
                return False
        return None

    def _classify_severity(self, magnitude: float) -> str:
        """Classify discrepancy severity based on magnitude.

        Args:
            magnitude: Discrepancy magnitude (relative deviation).

        Returns:
            Severity string: critical, high, medium, low, or info.
        """
        if magnitude >= 0.50:
            return "critical"
        if magnitude >= 0.25:
            return "high"
        if magnitude >= 0.10:
            return "medium"
        if magnitude >= 0.05:
            return "low"
        return "info"


__all__ = ["ReconciliationPipelineEngine"]
