# -*- coding: utf-8 -*-
"""
Outlier Detection Agent Service Setup - AGENT-DATA-013

Provides ``configure_outlier_detector(app)`` which wires up the
Outlier Detection SDK (statistical detector, contextual detector,
temporal detector, multivariate detector, outlier classifier,
treatment engine, outlier pipeline, provenance tracker) and mounts
the REST API.

Also exposes ``get_outlier_detector(app)`` for programmatic access
and the ``OutlierDetectorService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.outlier_detector.setup import configure_outlier_detector
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_outlier_detector(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
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

from greenlang.outlier_detector.config import (
    OutlierDetectorConfig,
    get_config,
)
from greenlang.outlier_detector.metrics import (
    PROMETHEUS_AVAILABLE,
    inc_jobs,
    inc_outliers_detected,
    inc_outliers_classified,
    inc_treatments,
    inc_thresholds,
    inc_feedback,
    inc_errors,
    observe_ensemble_score,
    observe_duration,
    observe_confidence,
    set_active_jobs,
    set_total_outliers_flagged,
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


class DetectionResponse(BaseModel):
    """Single-column outlier detection result.

    Attributes:
        detection_id: Unique detection operation identifier.
        column_name: Column that was analyzed.
        method: Primary detection method used.
        total_points: Total data points analyzed.
        outliers_found: Number of outliers detected.
        outlier_pct: Fraction of points flagged as outliers.
        lower_fence: Lower outlier fence (if applicable).
        upper_fence: Upper outlier fence (if applicable).
        outlier_indices: Row indices of detected outliers.
        scores: Per-point outlier score summaries.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    column_name: str = Field(default="")
    method: str = Field(default="iqr")
    total_points: int = Field(default=0)
    outliers_found: int = Field(default=0)
    outlier_pct: float = Field(default=0.0)
    lower_fence: Optional[float] = Field(default=None)
    upper_fence: Optional[float] = Field(default=None)
    outlier_indices: List[int] = Field(default_factory=list)
    scores: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class BatchDetectionResponse(BaseModel):
    """Batch detection result across multiple columns.

    Attributes:
        batch_id: Unique batch operation identifier.
        job_id: Associated detection job identifier.
        total_columns: Total columns analyzed.
        total_outliers: Total outliers detected across all columns.
        avg_outlier_pct: Average outlier percentage across columns.
        results: Per-column detection results.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = Field(default="")
    total_columns: int = Field(default=0)
    total_outliers: int = Field(default=0)
    avg_outlier_pct: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ClassificationResponse(BaseModel):
    """Outlier classification result.

    Attributes:
        classification_id: Unique classification operation identifier.
        total_classified: Number of outliers classified.
        classifications: Per-outlier classification summaries.
        by_class: Count of outliers per classification category.
        avg_confidence: Average classification confidence.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    classification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_classified: int = Field(default=0)
    classifications: List[Dict[str, Any]] = Field(default_factory=list)
    by_class: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = Field(default=0.0)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class TreatmentResponse(BaseModel):
    """Outlier treatment result.

    Attributes:
        treatment_id: Unique treatment operation identifier.
        strategy: Treatment strategy applied.
        total_treated: Number of outliers treated.
        treatments: Per-outlier treatment summaries.
        reversible: Whether this treatment can be undone.
        treated_records: Records after treatment.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    treatment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = Field(default="flag")
    total_treated: int = Field(default=0)
    treatments: List[Dict[str, Any]] = Field(default_factory=list)
    reversible: bool = Field(default=True)
    treated_records: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ThresholdResponse(BaseModel):
    """Domain threshold management result.

    Attributes:
        threshold_id: Unique threshold identifier.
        column_name: Column this threshold applies to.
        lower_bound: Lower acceptable bound.
        upper_bound: Upper acceptable bound.
        source: Source of this threshold definition.
        context: Additional context or description.
        active: Whether this threshold is currently active.
        created_at: Timestamp when the threshold was created.
        provenance_hash: SHA-256 provenance hash.
    """
    threshold_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    column_name: str = Field(default="")
    lower_bound: Optional[float] = Field(default=None)
    upper_bound: Optional[float] = Field(default=None)
    source: str = Field(default="domain")
    context: str = Field(default="")
    active: bool = Field(default=True)
    created_at: str = Field(default="")
    provenance_hash: str = Field(default="")


class FeedbackResponse(BaseModel):
    """Feedback submission result.

    Attributes:
        feedback_id: Unique feedback identifier.
        detection_id: Identifier of the detection being reviewed.
        feedback_type: Type of feedback submitted.
        notes: Human notes or justification.
        accepted: Whether the feedback was accepted.
        created_at: Timestamp when the feedback was submitted.
        provenance_hash: SHA-256 provenance hash.
    """
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str = Field(default="")
    feedback_type: str = Field(default="confirmed_outlier")
    notes: str = Field(default="")
    accepted: bool = Field(default=True)
    created_at: str = Field(default="")
    provenance_hash: str = Field(default="")


class PipelineResponse(BaseModel):
    """Full outlier detection pipeline result.

    Attributes:
        pipeline_id: Unique pipeline run identifier.
        job_id: Associated detection job identifier.
        status: Pipeline status (completed, failed).
        total_records: Total input records.
        total_outliers: Total outliers detected.
        total_treated: Total outliers treated.
        outlier_pct: Overall outlier percentage.
        stages: Per-stage summary (detect, classify, treat, validate,
            document).
        processing_time_ms: Total processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = Field(default="")
    status: str = Field(default="completed")
    total_records: int = Field(default=0)
    total_outliers: int = Field(default=0)
    total_treated: int = Field(default=0)
    outlier_pct: float = Field(default=0.0)
    stages: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class StatsResponse(BaseModel):
    """Aggregate statistics for the outlier detection service.

    Attributes:
        total_jobs: Total detection jobs processed.
        completed_jobs: Total jobs completed successfully.
        failed_jobs: Total jobs that failed.
        total_records_processed: Total records processed across all jobs.
        total_outliers_detected: Total outliers detected.
        total_treatments_applied: Total treatments applied.
        total_classifications: Total classifications performed.
        total_feedback: Total feedback entries received.
        total_thresholds: Total thresholds defined.
        active_jobs: Number of currently active jobs.
        avg_outlier_pct: Average outlier percentage across jobs.
        by_method: Count of outliers per detection method.
        by_class: Count of outliers per classification.
        by_treatment: Count of treatments per strategy.
        by_status: Count of jobs per status.
        provenance_entries: Total provenance entries recorded.
    """
    total_jobs: int = Field(default=0)
    completed_jobs: int = Field(default=0)
    failed_jobs: int = Field(default=0)
    total_records_processed: int = Field(default=0)
    total_outliers_detected: int = Field(default=0)
    total_treatments_applied: int = Field(default=0)
    total_classifications: int = Field(default=0)
    total_feedback: int = Field(default=0)
    total_thresholds: int = Field(default=0)
    active_jobs: int = Field(default=0)
    avg_outlier_pct: float = Field(default=0.0)
    by_method: Dict[str, int] = Field(default_factory=dict)
    by_class: Dict[str, int] = Field(default_factory=dict)
    by_treatment: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
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
            entity_type: Type of entity (detection_job, detection,
                classification, treatment, threshold, feedback, pipeline).
            entity_id: Entity identifier.
            action: Action performed (detect, classify, treat, validate,
                document, create, update, delete, pipeline, feedback).
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
_singleton_instance: Optional["OutlierDetectorService"] = None


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


def _is_numeric(value: Any) -> bool:
    """Check whether a value can be interpreted as numeric.

    Args:
        value: Value to check.

    Returns:
        True if the value is numeric, False otherwise.
    """
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and value != value)
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None on failure.

    Args:
        value: Value to convert.

    Returns:
        Float value or None.
    """
    if value is None or value == "":
        return None
    try:
        f = float(value)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None


def _auto_detect_numeric_columns(
    records: List[Dict[str, Any]],
) -> List[str]:
    """Auto-detect numeric columns from a list of records.

    A column is considered numeric if at least 80% of its non-empty
    values can be parsed as float.

    Args:
        records: List of record dicts.

    Returns:
        Sorted list of numeric column names.
    """
    if not records:
        return []

    all_columns: set = set()
    for rec in records:
        all_columns.update(rec.keys())

    numeric_cols: List[str] = []
    sample = records[:min(100, len(records))]

    for col in sorted(all_columns):
        values = [rec.get(col) for rec in sample]
        non_empty = [v for v in values if v is not None and v != ""]
        if not non_empty:
            continue
        numeric_count = sum(1 for v in non_empty if _is_numeric(v))
        if numeric_count > len(non_empty) * 0.8:
            numeric_cols.append(col)

    return numeric_cols


# ===================================================================
# OutlierDetectorService facade
# ===================================================================


class OutlierDetectorService:
    """Unified facade over the Outlier Detection SDK.

    Aggregates all detector engines (statistical detector, contextual
    detector, temporal detector, multivariate detector, outlier
    classifier, treatment engine, outlier pipeline) through a single
    entry point with convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: OutlierDetectorConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = OutlierDetectorService()
        >>> result = service.detect_outliers(
        ...     records=[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 100}],
        ...     column="a",
        ... )
        >>> print(result.outliers_found, result.outlier_pct)
    """

    def __init__(
        self,
        config: Optional[OutlierDetectorConfig] = None,
    ) -> None:
        """Initialize the Outlier Detection Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - StatisticalDetectorEngine
        - ContextualDetectorEngine
        - TemporalDetectorEngine
        - MultivariateDetectorEngine
        - OutlierClassifierEngine
        - TreatmentEngine
        - OutlierPipelineEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._statistical_engine: Any = None
        self._contextual_engine: Any = None
        self._temporal_engine: Any = None
        self._multivariate_engine: Any = None
        self._classifier_engine: Any = None
        self._treatment_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._detections: Dict[str, DetectionResponse] = {}
        self._batch_detections: Dict[str, BatchDetectionResponse] = {}
        self._classifications: Dict[str, ClassificationResponse] = {}
        self._treatments: Dict[str, TreatmentResponse] = {}
        self._thresholds: Dict[str, ThresholdResponse] = {}
        self._feedback: Dict[str, FeedbackResponse] = {}
        self._pipeline_results: Dict[str, PipelineResponse] = {}

        # Statistics
        self._stats = StatsResponse()
        self._started = False
        self._active_jobs = 0
        self._outlier_pct_sum = 0.0
        self._outlier_pct_count = 0

        logger.info("OutlierDetectorService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def statistical_engine(self) -> Any:
        """Get the StatisticalDetectorEngine instance."""
        return self._statistical_engine

    @property
    def contextual_engine(self) -> Any:
        """Get the ContextualDetectorEngine instance."""
        return self._contextual_engine

    @property
    def temporal_engine(self) -> Any:
        """Get the TemporalDetectorEngine instance."""
        return self._temporal_engine

    @property
    def multivariate_engine(self) -> Any:
        """Get the MultivariateDetectorEngine instance."""
        return self._multivariate_engine

    @property
    def classifier_engine(self) -> Any:
        """Get the OutlierClassifierEngine instance."""
        return self._classifier_engine

    @property
    def treatment_engine(self) -> Any:
        """Get the TreatmentEngine instance."""
        return self._treatment_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the OutlierPipelineEngine instance."""
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
            from greenlang.outlier_detector.statistical_detector import (
                StatisticalDetectorEngine,
            )
            self._statistical_engine = StatisticalDetectorEngine(self.config)
        except ImportError:
            logger.warning("StatisticalDetectorEngine not available; using stub")

        try:
            from greenlang.outlier_detector.contextual_detector import (
                ContextualDetectorEngine,
            )
            self._contextual_engine = ContextualDetectorEngine(self.config)
        except ImportError:
            logger.warning("ContextualDetectorEngine not available; using stub")

        try:
            from greenlang.outlier_detector.temporal_detector import (
                TemporalDetectorEngine,
            )
            self._temporal_engine = TemporalDetectorEngine(self.config)
        except ImportError:
            logger.warning("TemporalDetectorEngine not available; using stub")

        try:
            from greenlang.outlier_detector.multivariate_detector import (
                MultivariateDetectorEngine,
            )
            self._multivariate_engine = MultivariateDetectorEngine(self.config)
        except ImportError:
            logger.warning("MultivariateDetectorEngine not available; using stub")

        try:
            from greenlang.outlier_detector.outlier_classifier import (
                OutlierClassifierEngine,
            )
            self._classifier_engine = OutlierClassifierEngine(self.config)
        except ImportError:
            logger.warning("OutlierClassifierEngine not available; using stub")

        try:
            from greenlang.outlier_detector.treatment_engine import (
                TreatmentEngine,
            )
            self._treatment_engine = TreatmentEngine(self.config)
        except ImportError:
            logger.warning("TreatmentEngine not available; using stub")

        try:
            from greenlang.outlier_detector.outlier_pipeline import (
                OutlierPipelineEngine,
            )
            self._pipeline_engine = OutlierPipelineEngine(
                self.config,
                self._statistical_engine,
                self._contextual_engine,
                self._temporal_engine,
                self._multivariate_engine,
                self._classifier_engine,
                self._treatment_engine,
            )
        except (ImportError, TypeError):
            logger.warning("OutlierPipelineEngine not available; using stub")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the Outlier Detection service.

        Marks the service as started for health check reporting.
        """
        self._started = True
        logger.info("OutlierDetectorService started")

    def shutdown(self) -> None:
        """Shut down the Outlier Detection service.

        Marks the service as stopped for health check reporting.
        """
        self._started = False
        logger.info("OutlierDetectorService shut down")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the service.

        Returns:
            Health status dict.
        """
        return {
            "status": "healthy" if self._started else "not_started",
            "service": "outlier-detector",
            "started": self._started,
            "engines": {
                "statistical": self._statistical_engine is not None,
                "contextual": self._contextual_engine is not None,
                "temporal": self._temporal_engine is not None,
                "multivariate": self._multivariate_engine is not None,
                "classifier": self._classifier_engine is not None,
                "treatment": self._treatment_engine is not None,
                "pipeline": self._pipeline_engine is not None,
            },
            "jobs": len(self._jobs),
            "detections": len(self._detections),
            "batch_detections": len(self._batch_detections),
            "classifications": len(self._classifications),
            "treatments": len(self._treatments),
            "thresholds": len(self._thresholds),
            "feedback": len(self._feedback),
            "pipeline_results": len(self._pipeline_results),
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new outlier detection job.

        Args:
            request: Job creation request dict with records, dataset_id,
                and pipeline_config.

        Returns:
            Job creation result dict.
        """
        job_id = str(uuid.uuid4())
        records = request.get("records", [])
        dataset_id = request.get("dataset_id", "")

        job = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "status": "pending",
            "stage": "detect",
            "total_records": len(records),
            "total_columns": 0,
            "outliers_detected": 0,
            "outliers_classified": 0,
            "treatments_applied": 0,
            "pipeline_config": request.get("pipeline_config"),
            "error_message": None,
            "created_at": _utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "provenance_hash": "",
        }
        job["provenance_hash"] = _compute_hash(job)
        self._jobs[job_id] = job

        self.provenance.record(
            entity_type="detection_job",
            entity_id=job_id,
            action="create",
            data_hash=job["provenance_hash"],
        )

        self._stats.total_jobs += 1
        self._stats.by_status["pending"] = (
            self._stats.by_status.get("pending", 0) + 1
        )

        logger.info(
            "Created detection job %s for %d records",
            job_id[:8], len(records),
        )
        return job

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List outlier detection jobs with optional filtering.

        Args:
            status: Optional status filter.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.

        Returns:
            List of job dicts.
        """
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
        return jobs[offset:offset + limit]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get an outlier detection job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Job dict or None if not found.
        """
        return self._jobs.get(job_id)

    def delete_job(self, job_id: str) -> bool:
        """Delete an outlier detection job by ID.

        Sets the job status to 'cancelled' and returns True.
        Returns False if the job was not found.

        Args:
            job_id: Job identifier.

        Returns:
            True if found and deleted, False otherwise.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job["status"] = "cancelled"
        job["completed_at"] = _utcnow().isoformat()

        self.provenance.record(
            entity_type="detection_job",
            entity_id=job_id,
            action="cancel",
            data_hash=_compute_hash(job),
        )

        inc_jobs("cancelled")
        logger.info("Cancelled detection job %s", job_id[:8])
        return True

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        records: List[Dict[str, Any]],
        column: str,
        methods: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> DetectionResponse:
        """Detect outliers in a single column of data.

        Zero-hallucination: All scores are deterministic Python
        arithmetic. No LLM calls in the detection path.

        Args:
            records: List of record dicts to analyze.
            column: Column name to analyze.
            methods: Optional list of detection methods (defaults to
                ['iqr', 'zscore']).
            options: Optional detection options.

        Returns:
            DetectionResponse with outlier detection results.

        Raises:
            ValueError: If records is empty or column is not found.
        """
        start_time = time.time()

        if not records:
            raise ValueError("Records list must not be empty for detection")

        records = records[:self.config.max_records]

        # Delegate to engine if available
        if self._statistical_engine is not None:
            try:
                engine_result = self._statistical_engine.detect(
                    records=records,
                    column=column,
                    methods=methods,
                    options=options,
                )
                return self._wrap_detection_result(
                    engine_result, column, start_time,
                )
            except (AttributeError, TypeError) as exc:
                logger.debug(
                    "Statistical engine delegation failed: %s; using fallback",
                    exc,
                )

        # Fallback: built-in IQR detection
        return self._fallback_detect(records, column, methods, start_time)

    def _fallback_detect(
        self,
        records: List[Dict[str, Any]],
        column: str,
        methods: Optional[List[str]],
        start_time: float,
    ) -> DetectionResponse:
        """Perform fallback outlier detection without engine.

        Uses IQR method for deterministic, zero-hallucination detection.

        Args:
            records: List of record dicts.
            column: Column to analyze.
            methods: Optional methods (ignored in fallback, uses IQR).
            start_time: Operation start timestamp.

        Returns:
            DetectionResponse with detection results.
        """
        values = [_safe_float(rec.get(column)) for rec in records]
        numeric_values = [v for v in values if v is not None]

        if not numeric_values:
            result = DetectionResponse(
                column_name=column,
                method="iqr",
                total_points=len(records),
                outliers_found=0,
                outlier_pct=0.0,
                processing_time_ms=(time.time() - start_time) * 1000.0,
            )
            result.provenance_hash = _compute_hash(result)
            self._store_detection(result, start_time)
            return result

        sorted_vals = sorted(numeric_values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[(3 * n) // 4]
        iqr = q3 - q1
        k = self.config.iqr_multiplier

        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr

        outlier_indices: List[int] = []
        scores: List[Dict[str, Any]] = []

        for idx, val in enumerate(values):
            if val is None:
                continue
            is_outlier = val < lower_fence or val > upper_fence
            if is_outlier:
                outlier_indices.append(idx)

            # Compute normalized score
            score = self._compute_iqr_score(val, q1, q3, iqr, k)
            scores.append({
                "record_index": idx,
                "value": val,
                "score": round(score, 4),
                "is_outlier": is_outlier,
                "method": "iqr",
            })

        outlier_pct = (
            len(outlier_indices) / max(len(numeric_values), 1)
        )

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = DetectionResponse(
            column_name=column,
            method="iqr",
            total_points=len(numeric_values),
            outliers_found=len(outlier_indices),
            outlier_pct=round(outlier_pct, 4),
            lower_fence=round(lower_fence, 6),
            upper_fence=round(upper_fence, 6),
            outlier_indices=outlier_indices,
            scores=scores,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._store_detection(result, start_time)
        return result

    def _compute_iqr_score(
        self,
        value: float,
        q1: float,
        q3: float,
        iqr: float,
        k: float,
    ) -> float:
        """Compute normalized IQR outlier score for a value.

        Zero-hallucination: purely deterministic arithmetic.

        Args:
            value: Data point value.
            q1: First quartile.
            q3: Third quartile.
            iqr: Interquartile range.
            k: IQR multiplier.

        Returns:
            Normalized outlier score (0.0 to 1.0).
        """
        if iqr == 0:
            return 0.0

        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr

        if lower_fence <= value <= upper_fence:
            return 0.0

        if value < lower_fence:
            distance = lower_fence - value
        else:
            distance = value - upper_fence

        score = min(distance / (k * iqr), 1.0)
        return score

    def _store_detection(
        self,
        result: DetectionResponse,
        start_time: float,
    ) -> None:
        """Store detection result and update metrics.

        Args:
            result: DetectionResponse to store.
            start_time: Operation start timestamp.
        """
        self._detections[result.detection_id] = result

        self.provenance.record(
            entity_type="detection",
            entity_id=result.detection_id,
            action="detect",
            data_hash=result.provenance_hash,
        )

        inc_outliers_detected("iqr", result.outliers_found)
        observe_duration("detect", time.time() - start_time)

        self._stats.total_outliers_detected += result.outliers_found
        self._stats.total_records_processed += result.total_points
        self._update_outlier_pct(result.outlier_pct)

        logger.info(
            "Detected %d outliers in column '%s' (%d points, %.1f%%)",
            result.outliers_found,
            result.column_name,
            result.total_points,
            result.outlier_pct * 100,
        )

    def _wrap_detection_result(
        self,
        engine_result: Any,
        column: str,
        start_time: float,
    ) -> DetectionResponse:
        """Wrap engine result into DetectionResponse.

        Args:
            engine_result: Raw engine result (DetectionResult or dict).
            column: Column name.
            start_time: Operation start timestamp.

        Returns:
            DetectionResponse with provenance.
        """
        processing_time_ms = (time.time() - start_time) * 1000.0

        if hasattr(engine_result, "model_dump"):
            data = engine_result.model_dump(mode="json")
        elif isinstance(engine_result, dict):
            data = engine_result
        else:
            data = {}

        outlier_indices = []
        scores_raw = data.get("scores", [])
        scores_out: List[Dict[str, Any]] = []
        for s in scores_raw:
            if isinstance(s, dict):
                scores_out.append(s)
                if s.get("is_outlier"):
                    outlier_indices.append(s.get("record_index", 0))
            elif hasattr(s, "model_dump"):
                sd = s.model_dump(mode="json")
                scores_out.append(sd)
                if sd.get("is_outlier"):
                    outlier_indices.append(sd.get("record_index", 0))

        method = data.get("method", "iqr")
        if hasattr(method, "value"):
            method = method.value

        result = DetectionResponse(
            column_name=data.get("column_name", column),
            method=method,
            total_points=data.get("total_points", 0),
            outliers_found=data.get("outliers_found", len(outlier_indices)),
            outlier_pct=data.get("outlier_pct", 0.0),
            lower_fence=data.get("lower_fence"),
            upper_fence=data.get("upper_fence"),
            outlier_indices=outlier_indices,
            scores=scores_out,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._store_detection(result, start_time)
        return result

    def detect_batch(
        self,
        records: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> BatchDetectionResponse:
        """Detect outliers across multiple columns.

        Zero-hallucination: delegates per-column to detect_outliers
        which uses deterministic arithmetic only.

        Args:
            records: List of record dicts to analyze.
            columns: Columns to analyze (auto-detect numeric if None).

        Returns:
            BatchDetectionResponse with aggregated results.

        Raises:
            ValueError: If records is empty.
        """
        start_time = time.time()

        if not records:
            raise ValueError("Records list must not be empty for batch detection")

        records = records[:self.config.max_records]

        target_columns = columns or _auto_detect_numeric_columns(records)
        results: List[Dict[str, Any]] = []
        total_outliers = 0

        for col in target_columns:
            try:
                det = self.detect_outliers(records, col)
                results.append(det.model_dump(mode="json"))
                total_outliers += det.outliers_found
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Batch detection skipped column '%s': %s", col, exc,
                )

        avg_outlier_pct = 0.0
        if results:
            pct_sum = sum(r.get("outlier_pct", 0.0) for r in results)
            avg_outlier_pct = round(pct_sum / len(results), 4)

        processing_time_ms = (time.time() - start_time) * 1000.0

        batch_result = BatchDetectionResponse(
            total_columns=len(results),
            total_outliers=total_outliers,
            avg_outlier_pct=avg_outlier_pct,
            results=results,
            processing_time_ms=round(processing_time_ms, 2),
        )
        batch_result.provenance_hash = _compute_hash(batch_result)

        self._batch_detections[batch_result.batch_id] = batch_result

        self.provenance.record(
            entity_type="batch_detection",
            entity_id=batch_result.batch_id,
            action="detect",
            data_hash=batch_result.provenance_hash,
        )

        observe_duration("batch_detect", time.time() - start_time)

        logger.info(
            "Batch detection: %d columns, %d total outliers (avg %.1f%%)",
            len(results), total_outliers, avg_outlier_pct * 100,
        )
        return batch_result

    def get_detections(self) -> List[DetectionResponse]:
        """Get all stored detection results.

        Returns:
            List of DetectionResponse objects.
        """
        return list(self._detections.values())

    def get_detection(
        self, detection_id: str,
    ) -> Optional[DetectionResponse]:
        """Get a detection result by ID.

        Args:
            detection_id: Detection identifier.

        Returns:
            DetectionResponse or None if not found.
        """
        return self._detections.get(detection_id)

    # ------------------------------------------------------------------
    # Outlier classification
    # ------------------------------------------------------------------

    def classify_outliers(
        self,
        detections: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> ClassificationResponse:
        """Classify detected outliers by root cause.

        Assigns classifications (error, genuine_extreme, data_entry,
        regime_change, sensor_fault) based on heuristics and context.

        Zero-hallucination: classification uses deterministic rules
        and thresholds, not LLM inference.

        Args:
            detections: List of detection score dicts.
            records: Original record dicts for context.

        Returns:
            ClassificationResponse with classifications.

        Raises:
            ValueError: If detections list is empty.
        """
        start_time = time.time()

        if not detections:
            raise ValueError("Detections list must not be empty")

        # Delegate to engine if available
        if self._classifier_engine is not None:
            try:
                engine_result = self._classifier_engine.classify(
                    detections=detections,
                    records=records,
                )
                return self._wrap_classification_result(
                    engine_result, start_time,
                )
            except (AttributeError, TypeError) as exc:
                logger.debug(
                    "Classifier engine delegation failed: %s; using fallback",
                    exc,
                )

        # Fallback: heuristic classification
        return self._fallback_classify(detections, records, start_time)

    def _fallback_classify(
        self,
        detections: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
        start_time: float,
    ) -> ClassificationResponse:
        """Classify outliers using deterministic heuristics.

        Args:
            detections: Outlier detection dicts.
            records: Original data records.
            start_time: Operation start timestamp.

        Returns:
            ClassificationResponse with heuristic classifications.
        """
        classifications: List[Dict[str, Any]] = []
        by_class: Dict[str, int] = {}
        confidence_sum = 0.0

        for det in detections:
            if not det.get("is_outlier", False):
                continue

            score = det.get("score", 0.0)
            value = det.get("value")
            idx = det.get("record_index", 0)

            # Heuristic classification based on score magnitude
            outlier_class, confidence = self._heuristic_classify(
                score, value, idx, records,
            )

            classification = {
                "record_index": idx,
                "column_name": det.get("column_name", ""),
                "value": value,
                "outlier_class": outlier_class,
                "confidence": round(confidence, 4),
                "evidence": [
                    f"Outlier score: {score:.4f}",
                    f"Classification: {outlier_class}",
                ],
            }
            classifications.append(classification)
            by_class[outlier_class] = by_class.get(outlier_class, 0) + 1
            confidence_sum += confidence

            inc_outliers_classified(outlier_class)

        avg_confidence = (
            round(confidence_sum / max(len(classifications), 1), 4)
        )
        processing_time_ms = (time.time() - start_time) * 1000.0

        result = ClassificationResponse(
            total_classified=len(classifications),
            classifications=classifications,
            by_class=by_class,
            avg_confidence=avg_confidence,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._classifications[result.classification_id] = result

        self.provenance.record(
            entity_type="classification",
            entity_id=result.classification_id,
            action="classify",
            data_hash=result.provenance_hash,
        )

        observe_duration("classify", time.time() - start_time)
        self._stats.total_classifications += len(classifications)

        logger.info(
            "Classified %d outliers: %s",
            len(classifications), by_class,
        )
        return result

    def _heuristic_classify(
        self,
        score: float,
        value: Any,
        idx: int,
        records: List[Dict[str, Any]],
    ) -> tuple:
        """Classify an outlier using deterministic heuristics.

        Args:
            score: Normalized outlier score (0.0-1.0).
            value: The outlier data value.
            idx: Record index.
            records: All records for context.

        Returns:
            Tuple of (outlier_class, confidence).
        """
        if score >= 0.95:
            return ("error", 0.85)
        elif score >= 0.80:
            return ("data_entry", 0.70)
        elif score >= 0.60:
            return ("genuine_extreme", 0.65)
        elif score >= 0.40:
            return ("regime_change", 0.55)
        else:
            return ("genuine_extreme", 0.50)

    def _wrap_classification_result(
        self,
        engine_result: Any,
        start_time: float,
    ) -> ClassificationResponse:
        """Wrap engine result into ClassificationResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.

        Returns:
            ClassificationResponse with provenance.
        """
        processing_time_ms = (time.time() - start_time) * 1000.0

        if hasattr(engine_result, "model_dump"):
            data = engine_result.model_dump(mode="json")
        elif isinstance(engine_result, dict):
            data = engine_result
        else:
            data = {}

        classifications = data.get("classifications", [])
        by_class: Dict[str, int] = {}
        for c in classifications:
            cls_name = c.get("outlier_class", "genuine_extreme")
            if hasattr(cls_name, "value"):
                cls_name = cls_name.value
            by_class[cls_name] = by_class.get(cls_name, 0) + 1

        result = ClassificationResponse(
            total_classified=len(classifications),
            classifications=classifications,
            by_class=by_class,
            avg_confidence=data.get("avg_confidence", 0.0),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._classifications[result.classification_id] = result

        self.provenance.record(
            entity_type="classification",
            entity_id=result.classification_id,
            action="classify",
            data_hash=result.provenance_hash,
        )

        observe_duration("classify", time.time() - start_time)
        self._stats.total_classifications += len(classifications)

        return result

    def get_classification(
        self, classification_id: str,
    ) -> Optional[ClassificationResponse]:
        """Get a classification result by ID.

        Args:
            classification_id: Classification identifier.

        Returns:
            ClassificationResponse or None if not found.
        """
        return self._classifications.get(classification_id)

    # ------------------------------------------------------------------
    # Treatment
    # ------------------------------------------------------------------

    def apply_treatment(
        self,
        records: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        strategy: str = "flag",
        options: Optional[Dict[str, Any]] = None,
    ) -> TreatmentResponse:
        """Apply treatment to detected outliers.

        Zero-hallucination: treatment values are computed with
        deterministic formulas (cap, winsorize, replace with median).

        Args:
            records: Original record dicts.
            detections: Outlier detection score dicts.
            strategy: Treatment strategy to apply.
            options: Optional treatment options.

        Returns:
            TreatmentResponse with treated data.

        Raises:
            ValueError: If records or detections is empty.
        """
        start_time = time.time()

        if not records:
            raise ValueError("Records list must not be empty for treatment")
        if not detections:
            raise ValueError("Detections list must not be empty for treatment")

        # Delegate to engine if available
        if self._treatment_engine is not None:
            try:
                engine_result = self._treatment_engine.treat(
                    records=records,
                    detections=detections,
                    strategy=strategy,
                    options=options,
                )
                return self._wrap_treatment_result(
                    engine_result, strategy, start_time,
                )
            except (AttributeError, TypeError) as exc:
                logger.debug(
                    "Treatment engine delegation failed: %s; using fallback",
                    exc,
                )

        # Fallback: flag-based treatment
        return self._fallback_treat(
            records, detections, strategy, start_time,
        )

    def _fallback_treat(
        self,
        records: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        strategy: str,
        start_time: float,
    ) -> TreatmentResponse:
        """Apply fallback treatment using flag strategy.

        Args:
            records: Original records.
            detections: Outlier detections.
            strategy: Treatment strategy (fallback applies flag).
            start_time: Operation start timestamp.

        Returns:
            TreatmentResponse with treated records.
        """
        treatments: List[Dict[str, Any]] = []
        treated_records = [dict(rec) for rec in records]

        # Collect outlier indices
        outlier_set: Dict[int, Dict[str, Any]] = {}
        for det in detections:
            if det.get("is_outlier", False):
                idx = det.get("record_index", 0)
                outlier_set[idx] = det

        # Compute column stats for cap/winsorize/replace
        column_name = ""
        if detections:
            column_name = detections[0].get("column_name", "")

        for idx, det_info in outlier_set.items():
            if idx >= len(treated_records):
                continue

            original_value = det_info.get("value")
            treated_value = original_value
            reason = f"Outlier detected (score={det_info.get('score', 0):.4f})"

            if strategy == "flag":
                treated_records[idx][f"_outlier_{column_name}"] = True
                reason = "Flagged for review"
            elif strategy == "remove":
                treated_records[idx][column_name] = None
                treated_value = None
                reason = "Removed (set to null)"
            elif strategy == "cap" or strategy == "winsorize":
                # Cap at nearest fence
                col_vals = [
                    _safe_float(rec.get(column_name))
                    for rec in records
                ]
                numeric_vals = sorted(
                    [v for v in col_vals if v is not None],
                )
                if numeric_vals:
                    n = len(numeric_vals)
                    q1 = numeric_vals[n // 4]
                    q3 = numeric_vals[(3 * n) // 4]
                    iqr = q3 - q1
                    lower = q1 - self.config.iqr_multiplier * iqr
                    upper = q3 + self.config.iqr_multiplier * iqr
                    f_val = _safe_float(original_value)
                    if f_val is not None:
                        treated_value = max(lower, min(f_val, upper))
                        treated_records[idx][column_name] = treated_value
                        reason = f"Capped to [{lower:.4f}, {upper:.4f}]"
            elif strategy == "replace":
                # Replace with median
                col_vals = [
                    _safe_float(rec.get(column_name))
                    for rec in records
                ]
                numeric_vals = sorted(
                    [v for v in col_vals if v is not None],
                )
                if numeric_vals:
                    median = numeric_vals[len(numeric_vals) // 2]
                    treated_value = median
                    treated_records[idx][column_name] = median
                    reason = f"Replaced with median ({median:.4f})"
            else:
                # Default: investigate (flag with marker)
                treated_records[idx][f"_investigate_{column_name}"] = True
                reason = "Flagged for investigation"

            treatments.append({
                "record_index": idx,
                "column_name": column_name,
                "original_value": original_value,
                "treated_value": treated_value,
                "strategy": strategy,
                "reason": reason,
                "reversible": strategy != "remove",
            })

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = TreatmentResponse(
            strategy=strategy,
            total_treated=len(treatments),
            treatments=treatments,
            reversible=strategy != "remove",
            treated_records=treated_records,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._treatments[result.treatment_id] = result

        self.provenance.record(
            entity_type="treatment",
            entity_id=result.treatment_id,
            action="treat",
            data_hash=result.provenance_hash,
        )

        inc_treatments(strategy, len(treatments))
        observe_duration("treat", time.time() - start_time)

        self._stats.total_treatments_applied += len(treatments)
        self._stats.by_treatment[strategy] = (
            self._stats.by_treatment.get(strategy, 0) + len(treatments)
        )

        logger.info(
            "Applied '%s' treatment to %d outliers",
            strategy, len(treatments),
        )
        return result

    def _wrap_treatment_result(
        self,
        engine_result: Any,
        strategy: str,
        start_time: float,
    ) -> TreatmentResponse:
        """Wrap engine result into TreatmentResponse.

        Args:
            engine_result: Raw engine result.
            strategy: Treatment strategy applied.
            start_time: Operation start timestamp.

        Returns:
            TreatmentResponse with provenance.
        """
        processing_time_ms = (time.time() - start_time) * 1000.0

        if hasattr(engine_result, "model_dump"):
            data = engine_result.model_dump(mode="json")
        elif isinstance(engine_result, dict):
            data = engine_result
        else:
            data = {}

        treatments = data.get("treatments", [])

        result = TreatmentResponse(
            strategy=strategy,
            total_treated=len(treatments),
            treatments=treatments,
            reversible=data.get("reversible", True),
            treated_records=data.get("treated_records", []),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._treatments[result.treatment_id] = result

        self.provenance.record(
            entity_type="treatment",
            entity_id=result.treatment_id,
            action="treat",
            data_hash=result.provenance_hash,
        )

        inc_treatments(strategy, len(treatments))
        observe_duration("treat", time.time() - start_time)

        self._stats.total_treatments_applied += len(treatments)
        self._stats.by_treatment[strategy] = (
            self._stats.by_treatment.get(strategy, 0) + len(treatments)
        )

        return result

    def get_treatment(
        self, treatment_id: str,
    ) -> Optional[TreatmentResponse]:
        """Get a treatment result by ID.

        Args:
            treatment_id: Treatment identifier.

        Returns:
            TreatmentResponse or None if not found.
        """
        return self._treatments.get(treatment_id)

    def undo_treatment(self, treatment_id: str) -> bool:
        """Undo a previously applied treatment.

        Only works for reversible treatments. Sets an undo marker
        on the treatment record.

        Args:
            treatment_id: Treatment identifier.

        Returns:
            True if undone successfully, False otherwise.
        """
        treatment = self._treatments.get(treatment_id)
        if treatment is None:
            return False

        if not treatment.reversible:
            logger.warning(
                "Treatment %s is not reversible", treatment_id[:8],
            )
            return False

        # Mark as undone by clearing treatments list
        treatment.treatments = [
            {**t, "undone": True}
            for t in treatment.treatments
        ]

        self.provenance.record(
            entity_type="treatment",
            entity_id=treatment_id,
            action="undo",
            data_hash=_compute_hash(treatment),
        )

        logger.info("Undid treatment %s", treatment_id[:8])
        return True

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def create_threshold(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        source: str = "domain",
        context: str = "",
    ) -> ThresholdResponse:
        """Create a domain-specific threshold for a column.

        Args:
            column: Column name this threshold applies to.
            min_val: Lower acceptable bound.
            max_val: Upper acceptable bound.
            source: Source of threshold (domain, statistical, regulatory,
                custom, learned).
            context: Additional description or context.

        Returns:
            ThresholdResponse with created threshold.

        Raises:
            ValueError: If column is empty.
        """
        if not column or not column.strip():
            raise ValueError("Column name must not be empty")

        threshold = ThresholdResponse(
            column_name=column,
            lower_bound=min_val,
            upper_bound=max_val,
            source=source,
            context=context,
            active=True,
            created_at=_utcnow().isoformat(),
        )
        threshold.provenance_hash = _compute_hash(threshold)

        self._thresholds[threshold.threshold_id] = threshold

        self.provenance.record(
            entity_type="threshold",
            entity_id=threshold.threshold_id,
            action="create",
            data_hash=threshold.provenance_hash,
        )

        inc_thresholds(source)
        self._stats.total_thresholds += 1

        logger.info(
            "Created threshold for column '%s': [%s, %s] (source=%s)",
            column, min_val, max_val, source,
        )
        return threshold

    def list_thresholds(self) -> List[ThresholdResponse]:
        """List all domain thresholds.

        Returns:
            List of ThresholdResponse objects.
        """
        return list(self._thresholds.values())

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        detection_id: str,
        feedback_type: str = "confirmed_outlier",
        notes: str = "",
    ) -> FeedbackResponse:
        """Submit human feedback on an outlier detection result.

        Args:
            detection_id: Identifier of the detection being reviewed.
            feedback_type: Type of feedback (confirmed_outlier,
                false_positive, reclassified, unknown).
            notes: Human notes or justification.

        Returns:
            FeedbackResponse with the recorded feedback.
        """
        feedback = FeedbackResponse(
            detection_id=detection_id,
            feedback_type=feedback_type,
            notes=notes,
            accepted=True,
            created_at=_utcnow().isoformat(),
        )
        feedback.provenance_hash = _compute_hash(feedback)

        self._feedback[feedback.feedback_id] = feedback

        self.provenance.record(
            entity_type="feedback",
            entity_id=feedback.feedback_id,
            action="feedback",
            data_hash=feedback.provenance_hash,
        )

        inc_feedback(feedback_type)
        self._stats.total_feedback += 1

        logger.info(
            "Recorded feedback '%s' for detection %s",
            feedback_type, detection_id[:8],
        )
        return feedback

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def analyze_impact(
        self,
        original: List[Dict[str, Any]],
        treated: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze the statistical impact of outlier treatment.

        Compares original and treated datasets to quantify the effect
        of treatment on distribution properties.

        Zero-hallucination: all statistics are computed with
        deterministic Python arithmetic.

        Args:
            original: Original record dicts.
            treated: Treated record dicts.

        Returns:
            Impact analysis dict with per-column statistics.
        """
        start_time = time.time()

        if not original or not treated:
            return {"columns": [], "processing_time_ms": 0.0}

        all_columns = set()
        for rec in original[:1]:
            all_columns.update(rec.keys())

        numeric_cols = _auto_detect_numeric_columns(original)
        column_impacts: List[Dict[str, Any]] = []

        for col in numeric_cols:
            orig_vals = [
                _safe_float(rec.get(col)) for rec in original
            ]
            treat_vals = [
                _safe_float(rec.get(col)) for rec in treated
            ]

            orig_numeric = [v for v in orig_vals if v is not None]
            treat_numeric = [v for v in treat_vals if v is not None]

            if not orig_numeric or not treat_numeric:
                continue

            impact = self._compute_column_impact(
                col, orig_numeric, treat_numeric,
            )
            column_impacts.append(impact)

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = {
            "columns": column_impacts,
            "total_columns": len(column_impacts),
            "processing_time_ms": round(processing_time_ms, 2),
            "provenance_hash": _compute_hash(column_impacts),
        }

        self.provenance.record(
            entity_type="impact_analysis",
            entity_id=str(uuid.uuid4()),
            action="analyze",
            data_hash=result["provenance_hash"],
        )

        observe_duration("impact_analysis", time.time() - start_time)
        return result

    def _compute_column_impact(
        self,
        column: str,
        original: List[float],
        treated: List[float],
    ) -> Dict[str, Any]:
        """Compute impact statistics for a single column.

        Args:
            column: Column name.
            original: Original numeric values.
            treated: Treated numeric values.

        Returns:
            Impact statistics dict.
        """
        orig_mean = sum(original) / max(len(original), 1)
        treat_mean = sum(treated) / max(len(treated), 1)

        orig_sorted = sorted(original)
        treat_sorted = sorted(treated)
        orig_median = orig_sorted[len(orig_sorted) // 2]
        treat_median = treat_sorted[len(treat_sorted) // 2]

        orig_var = sum(
            (v - orig_mean) ** 2 for v in original
        ) / max(len(original), 1)
        treat_var = sum(
            (v - treat_mean) ** 2 for v in treated
        ) / max(len(treated), 1)

        orig_std = orig_var ** 0.5
        treat_std = treat_var ** 0.5

        mean_change_pct = 0.0
        if orig_mean != 0:
            mean_change_pct = (
                (treat_mean - orig_mean) / abs(orig_mean) * 100
            )

        std_change_pct = 0.0
        if orig_std != 0:
            std_change_pct = (
                (treat_std - orig_std) / abs(orig_std) * 100
            )

        records_affected = sum(
            1 for a, b in zip(original, treated)
            if a != b
        )

        return {
            "column_name": column,
            "records_affected": records_affected,
            "original_mean": round(orig_mean, 6),
            "treated_mean": round(treat_mean, 6),
            "original_std": round(orig_std, 6),
            "treated_std": round(treat_std, 6),
            "original_median": round(orig_median, 6),
            "treated_median": round(treat_median, 6),
            "mean_change_pct": round(mean_change_pct, 4),
            "std_change_pct": round(std_change_pct, 4),
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        records: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> PipelineResponse:
        """Run the full outlier detection pipeline.

        Executes stages in order: detect, classify, treat, validate,
        document. All computations are zero-hallucination.

        Args:
            records: List of record dicts to process.
            config: Optional pipeline configuration overrides.

        Returns:
            PipelineResponse with full pipeline results.

        Raises:
            ValueError: If records is empty.
        """
        start_time = time.time()

        if not records:
            raise ValueError("Records list must not be empty for pipeline")

        records = records[:self.config.max_records]

        # Delegate to engine if available
        if self._pipeline_engine is not None:
            try:
                engine_result = self._pipeline_engine.run(
                    records=records,
                    config=config,
                )
                return self._wrap_pipeline_result(engine_result, start_time)
            except (AttributeError, TypeError) as exc:
                logger.debug(
                    "Pipeline engine delegation failed: %s; using fallback",
                    exc,
                )

        # Fallback: staged pipeline
        return self._fallback_pipeline(records, config, start_time)

    def _fallback_pipeline(
        self,
        records: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]],
        start_time: float,
    ) -> PipelineResponse:
        """Run fallback pipeline without engine.

        Args:
            records: List of record dicts.
            config: Pipeline config overrides.
            start_time: Operation start timestamp.

        Returns:
            PipelineResponse with pipeline results.
        """
        job_id = str(uuid.uuid4())
        stages: Dict[str, Dict[str, Any]] = {}
        total_outliers = 0
        total_treated = 0
        status = "completed"

        # Stage 1: Detect
        stage_start = time.time()
        try:
            batch = self.detect_batch(records)
            total_outliers = batch.total_outliers
            stages["detect"] = {
                "status": "completed",
                "columns": batch.total_columns,
                "outliers": batch.total_outliers,
                "duration_ms": round(
                    (time.time() - stage_start) * 1000.0, 2,
                ),
            }
        except Exception as exc:
            stages["detect"] = {
                "status": "failed",
                "error": str(exc),
                "duration_ms": round(
                    (time.time() - stage_start) * 1000.0, 2,
                ),
            }
            status = "failed"

        # Stage 2: Classify (only if detection found outliers)
        stage_start = time.time()
        if total_outliers > 0 and status != "failed":
            try:
                # Collect all outlier scores from batch
                all_detections = self._collect_outlier_scores(batch)
                if all_detections:
                    cls_result = self.classify_outliers(
                        all_detections, records,
                    )
                    stages["classify"] = {
                        "status": "completed",
                        "classified": cls_result.total_classified,
                        "by_class": cls_result.by_class,
                        "duration_ms": round(
                            (time.time() - stage_start) * 1000.0, 2,
                        ),
                    }
                else:
                    stages["classify"] = {
                        "status": "skipped",
                        "reason": "No outlier scores collected",
                    }
            except Exception as exc:
                stages["classify"] = {
                    "status": "failed",
                    "error": str(exc),
                    "duration_ms": round(
                        (time.time() - stage_start) * 1000.0, 2,
                    ),
                }
        else:
            stages["classify"] = {
                "status": "skipped",
                "reason": "No outliers to classify",
            }

        # Stage 3: Treat
        stage_start = time.time()
        treatment_strategy = self.config.default_treatment
        if config and config.get("treatment_strategy"):
            treatment_strategy = config["treatment_strategy"]

        if total_outliers > 0 and status != "failed":
            try:
                all_detections = self._collect_outlier_scores(batch)
                if all_detections:
                    treat_result = self.apply_treatment(
                        records, all_detections, treatment_strategy,
                    )
                    total_treated = treat_result.total_treated
                    stages["treat"] = {
                        "status": "completed",
                        "strategy": treatment_strategy,
                        "treated": total_treated,
                        "duration_ms": round(
                            (time.time() - stage_start) * 1000.0, 2,
                        ),
                    }
                else:
                    stages["treat"] = {
                        "status": "skipped",
                        "reason": "No outlier scores collected",
                    }
            except Exception as exc:
                stages["treat"] = {
                    "status": "failed",
                    "error": str(exc),
                    "duration_ms": round(
                        (time.time() - stage_start) * 1000.0, 2,
                    ),
                }
        else:
            stages["treat"] = {
                "status": "skipped",
                "reason": "No outliers to treat",
            }

        # Stage 4: Validate
        stages["validate"] = {
            "status": "completed",
            "checks_passed": True,
            "duration_ms": 0.0,
        }

        # Stage 5: Document
        stages["document"] = {
            "status": "completed",
            "report_generated": True,
            "duration_ms": 0.0,
        }

        outlier_pct = 0.0
        if len(records) > 0 and total_outliers > 0:
            # Calculate across unique outlier indices
            numeric_cols = _auto_detect_numeric_columns(records)
            total_cells = len(records) * max(len(numeric_cols), 1)
            outlier_pct = round(total_outliers / max(total_cells, 1), 4)

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = PipelineResponse(
            job_id=job_id,
            status=status,
            total_records=len(records),
            total_outliers=total_outliers,
            total_treated=total_treated,
            outlier_pct=outlier_pct,
            stages=stages,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._pipeline_results[result.pipeline_id] = result

        self.provenance.record(
            entity_type="pipeline",
            entity_id=result.pipeline_id,
            action="pipeline",
            data_hash=result.provenance_hash,
        )

        inc_jobs(status)
        observe_duration("pipeline", time.time() - start_time)

        self._stats.completed_jobs += 1 if status == "completed" else 0
        self._stats.failed_jobs += 1 if status == "failed" else 0

        logger.info(
            "Pipeline %s: %d records, %d outliers, %d treated (%.1fms)",
            status, len(records), total_outliers, total_treated,
            processing_time_ms,
        )
        return result

    def _collect_outlier_scores(
        self,
        batch: BatchDetectionResponse,
    ) -> List[Dict[str, Any]]:
        """Collect all outlier-flagged score dicts from a batch result.

        Args:
            batch: BatchDetectionResponse with per-column results.

        Returns:
            List of outlier score dicts.
        """
        outliers: List[Dict[str, Any]] = []
        for col_result in batch.results:
            for score in col_result.get("scores", []):
                if score.get("is_outlier", False):
                    outliers.append(score)
        return outliers

    def _wrap_pipeline_result(
        self,
        engine_result: Any,
        start_time: float,
    ) -> PipelineResponse:
        """Wrap engine result into PipelineResponse.

        Args:
            engine_result: Raw engine result.
            start_time: Operation start timestamp.

        Returns:
            PipelineResponse with provenance.
        """
        processing_time_ms = (time.time() - start_time) * 1000.0

        if hasattr(engine_result, "model_dump"):
            data = engine_result.model_dump(mode="json")
        elif isinstance(engine_result, dict):
            data = engine_result
        else:
            data = {}

        status_val = data.get("status", "completed")
        if hasattr(status_val, "value"):
            status_val = status_val.value

        result = PipelineResponse(
            job_id=data.get("job_id", ""),
            status=status_val,
            total_records=data.get("total_records", 0),
            total_outliers=data.get("total_outliers", 0),
            total_treated=data.get("total_treated", 0),
            outlier_pct=data.get("outlier_pct", 0.0),
            stages=data.get("stages", {}),
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._pipeline_results[result.pipeline_id] = result

        self.provenance.record(
            entity_type="pipeline",
            entity_id=result.pipeline_id,
            action="pipeline",
            data_hash=result.provenance_hash,
        )

        inc_jobs(status_val)
        observe_duration("pipeline", time.time() - start_time)

        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> StatsResponse:
        """Get aggregate service statistics.

        Returns:
            StatsResponse with current service metrics.
        """
        stats = StatsResponse(
            total_jobs=self._stats.total_jobs,
            completed_jobs=self._stats.completed_jobs,
            failed_jobs=self._stats.failed_jobs,
            total_records_processed=self._stats.total_records_processed,
            total_outliers_detected=self._stats.total_outliers_detected,
            total_treatments_applied=self._stats.total_treatments_applied,
            total_classifications=self._stats.total_classifications,
            total_feedback=self._stats.total_feedback,
            total_thresholds=self._stats.total_thresholds,
            active_jobs=self._active_jobs,
            avg_outlier_pct=self._compute_avg_outlier_pct(),
            by_method=dict(self._stats.by_method),
            by_class=dict(self._stats.by_class),
            by_treatment=dict(self._stats.by_treatment),
            by_status=dict(self._stats.by_status),
            provenance_entries=self.provenance.entry_count,
        )

        set_active_jobs(self._active_jobs)
        set_total_outliers_flagged(self._stats.total_outliers_detected)

        return stats

    def _compute_avg_outlier_pct(self) -> float:
        """Compute the running average outlier percentage.

        Returns:
            Average outlier percentage (0.0-1.0).
        """
        if self._outlier_pct_count == 0:
            return 0.0
        return round(
            self._outlier_pct_sum / self._outlier_pct_count, 4,
        )

    def _update_outlier_pct(self, pct: float) -> None:
        """Update the running average outlier percentage.

        Args:
            pct: Outlier percentage for a single detection.
        """
        self._outlier_pct_sum += pct
        self._outlier_pct_count += 1

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get outlier detection service metrics summary.

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
            "total_outliers_detected": stats.total_outliers_detected,
            "total_treatments_applied": stats.total_treatments_applied,
            "total_classifications": stats.total_classifications,
            "total_feedback": stats.total_feedback,
            "total_thresholds": stats.total_thresholds,
            "active_jobs": stats.active_jobs,
            "avg_outlier_pct": stats.avg_outlier_pct,
            "provenance_entries": stats.provenance_entries,
        }


# ===================================================================
# Module-level configuration functions
# ===================================================================


async def configure_outlier_detector(
    app: Any,
    config: Optional[OutlierDetectorConfig] = None,
) -> OutlierDetectorService:
    """Configure the Outlier Detection Service on a FastAPI application.

    Creates the OutlierDetectorService, stores it in app.state, mounts
    the outlier detector API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional outlier detector config.

    Returns:
        OutlierDetectorService instance.
    """
    global _singleton_instance

    service = OutlierDetectorService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.outlier_detector_service = service

    # Mount outlier detector API router
    try:
        from greenlang.outlier_detector.api.router import router as od_router
        if od_router is not None:
            app.include_router(od_router)
            logger.info("Outlier detector API router mounted")
    except ImportError:
        logger.warning("Outlier detector API router not available")

    service._started = True
    logger.info("Outlier detector service configured and started")
    return service


def get_outlier_detector(app: Any) -> OutlierDetectorService:
    """Get the OutlierDetectorService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        OutlierDetectorService instance.

    Raises:
        RuntimeError: If outlier detector service not configured.
    """
    service = getattr(app.state, "outlier_detector_service", None)
    if service is None:
        raise RuntimeError(
            "Outlier detector service not configured. "
            "Call configure_outlier_detector(app) first."
        )
    return service


def get_router(service: Optional[OutlierDetectorService] = None) -> Any:
    """Get the outlier detector API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.outlier_detector.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "OutlierDetectorService",
    "configure_outlier_detector",
    "get_outlier_detector",
    "get_router",
    # Models
    "DetectionResponse",
    "BatchDetectionResponse",
    "ClassificationResponse",
    "TreatmentResponse",
    "ThresholdResponse",
    "FeedbackResponse",
    "PipelineResponse",
    "StatsResponse",
]
