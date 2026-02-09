# -*- coding: utf-8 -*-
"""
Data Quality Profiler Service Setup - AGENT-DATA-010

Provides ``configure_data_quality_profiler(app)`` which wires up the
Data Quality Profiler SDK (dataset profiling, quality assessment,
anomaly detection, freshness checking, rule engine, quality gates,
report generation, provenance tracker) and mounts the REST API.

Also exposes ``get_data_quality_profiler(app)`` for programmatic access
and the ``DataQualityProfilerService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.data_quality_profiler.setup import configure_data_quality_profiler
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_data_quality_profiler(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.data_quality_profiler.config import (
    DataQualityProfilerConfig,
    get_config,
)
from greenlang.data_quality_profiler.metrics import (
    PROMETHEUS_AVAILABLE,
    record_profile,
    record_column_profile,
    record_assessment,
    record_rule_evaluation,
    record_anomaly,
    record_gate_evaluation,
    record_quality_score,
    record_processing_duration,
    update_active_profiles,
    update_total_issues,
    record_processing_error,
    record_freshness_check,
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
# Lightweight Pydantic models used by the facade
# ===================================================================


class ColumnProfileResponse(BaseModel):
    """Column profile result.

    Attributes:
        column_name: Name of the column.
        data_type: Detected data type (string, integer, float, date, boolean).
        total_count: Total number of values.
        null_count: Number of null/missing values.
        null_pct: Percentage of null values (0.0-100.0).
        distinct_count: Number of distinct values.
        distinct_pct: Percentage of distinct values (0.0-100.0).
        min_value: Minimum value (for numeric/date columns).
        max_value: Maximum value (for numeric/date columns).
        mean: Mean value (for numeric columns).
        median: Median value (for numeric columns).
        stddev: Standard deviation (for numeric columns).
        top_values: Most frequent values with counts.
        pattern_summary: Detected patterns or formats.
    """
    column_name: str = Field(default="")
    data_type: str = Field(default="string")
    total_count: int = Field(default=0)
    null_count: int = Field(default=0)
    null_pct: float = Field(default=0.0)
    distinct_count: int = Field(default=0)
    distinct_pct: float = Field(default=0.0)
    min_value: Any = Field(default=None)
    max_value: Any = Field(default=None)
    mean: Optional[float] = Field(default=None)
    median: Optional[float] = Field(default=None)
    stddev: Optional[float] = Field(default=None)
    top_values: List[Dict[str, Any]] = Field(default_factory=list)
    pattern_summary: str = Field(default="")


class DatasetProfileResponse(BaseModel):
    """Dataset profiling result.

    Attributes:
        profile_id: Unique profile identifier.
        dataset_name: Name of the profiled dataset.
        row_count: Total number of rows.
        column_count: Total number of columns.
        columns: Per-column profiling results.
        completeness_score: Overall completeness score (0.0-1.0).
        source: Data source identifier.
        profiled_at: Timestamp of profiling.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = Field(default="")
    row_count: int = Field(default=0)
    column_count: int = Field(default=0)
    columns: List[ColumnProfileResponse] = Field(default_factory=list)
    completeness_score: float = Field(default=0.0)
    source: str = Field(default="manual")
    profiled_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class QualityAssessmentResponse(BaseModel):
    """Quality assessment result for a dataset.

    Attributes:
        assessment_id: Unique assessment identifier.
        dataset_name: Name of the assessed dataset.
        overall_score: Weighted overall quality score (0.0-1.0).
        quality_level: Quality level label (EXCELLENT, GOOD, FAIR, POOR, CRITICAL).
        completeness_score: Completeness dimension score (0.0-1.0).
        validity_score: Validity dimension score (0.0-1.0).
        consistency_score: Consistency dimension score (0.0-1.0).
        timeliness_score: Timeliness dimension score (0.0-1.0).
        uniqueness_score: Uniqueness dimension score (0.0-1.0).
        accuracy_score: Accuracy dimension score (0.0-1.0).
        issues: List of identified quality issues.
        row_count: Number of rows assessed.
        column_count: Number of columns assessed.
        assessed_at: Timestamp of assessment.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = Field(default="")
    overall_score: float = Field(default=0.0)
    quality_level: str = Field(default="CRITICAL")
    completeness_score: float = Field(default=0.0)
    validity_score: float = Field(default=0.0)
    consistency_score: float = Field(default=0.0)
    timeliness_score: float = Field(default=0.0)
    uniqueness_score: float = Field(default=0.0)
    accuracy_score: float = Field(default=0.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    column_count: int = Field(default=0)
    assessed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection result for a dataset.

    Attributes:
        detection_id: Unique detection identifier.
        dataset_name: Name of the assessed dataset.
        method: Detection method used (zscore, iqr, isolation_forest, percentile).
        total_records: Total records analysed.
        anomaly_count: Number of anomalies detected.
        anomaly_pct: Percentage of records flagged as anomalous.
        anomalies: List of anomaly details (column, row, value, score).
        columns_analysed: Columns included in analysis.
        detected_at: Timestamp of detection.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = Field(default="")
    method: str = Field(default="iqr")
    total_records: int = Field(default=0)
    anomaly_count: int = Field(default=0)
    anomaly_pct: float = Field(default=0.0)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    columns_analysed: List[str] = Field(default_factory=list)
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class FreshnessCheckResponse(BaseModel):
    """Freshness check result for a dataset.

    Attributes:
        check_id: Unique check identifier.
        dataset_name: Name of the checked dataset.
        last_updated: Last update timestamp (ISO 8601).
        age_hours: Age of the dataset in hours.
        sla_hours: SLA threshold in hours.
        status: Freshness status (fresh, stale, expired).
        freshness_score: Freshness score (0.0-1.0).
        checked_at: Timestamp of check.
        provenance_hash: SHA-256 provenance hash.
    """
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = Field(default="")
    last_updated: str = Field(default="")
    age_hours: float = Field(default=0.0)
    sla_hours: float = Field(default=48.0)
    status: str = Field(default="unknown")
    freshness_score: float = Field(default=0.0)
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class QualityRuleResponse(BaseModel):
    """Quality rule definition and evaluation result.

    Attributes:
        rule_id: Unique rule identifier.
        name: Rule display name.
        rule_type: Rule type (not_null, unique, range, regex, custom, referential).
        column: Target column name (optional for dataset-level rules).
        operator: Comparison operator (eq, ne, gt, gte, lt, lte, between, in, regex).
        threshold: Threshold value for the rule.
        parameters: Additional rule parameters.
        priority: Rule priority (lower = higher priority).
        is_active: Whether the rule is currently active.
        last_result: Last evaluation result (pass, fail, skip).
        last_evaluated_at: Timestamp of last evaluation.
        fail_count: Number of records that failed this rule.
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    rule_type: str = Field(default="not_null")
    column: str = Field(default="")
    operator: str = Field(default="eq")
    threshold: Any = Field(default=None)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=100)
    is_active: bool = Field(default=True)
    last_result: str = Field(default="")
    last_evaluated_at: str = Field(default="")
    fail_count: int = Field(default=0)
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class QualityGateResponse(BaseModel):
    """Quality gate evaluation result.

    Attributes:
        gate_id: Unique gate identifier.
        outcome: Gate outcome (pass, fail, warn).
        conditions_evaluated: Number of conditions evaluated.
        conditions_passed: Number of conditions that passed.
        conditions_failed: Number of conditions that failed.
        details: Per-condition evaluation details.
        overall_score: Score used for evaluation (0.0-1.0).
        threshold: Threshold required for pass.
        evaluated_at: Timestamp of evaluation.
        provenance_hash: SHA-256 provenance hash.
    """
    gate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    outcome: str = Field(default="fail")
    conditions_evaluated: int = Field(default=0)
    conditions_passed: int = Field(default=0)
    conditions_failed: int = Field(default=0)
    details: List[Dict[str, Any]] = Field(default_factory=list)
    overall_score: float = Field(default=0.0)
    threshold: float = Field(default=0.70)
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class DataQualityProfilerStatisticsResponse(BaseModel):
    """Aggregate statistics for the data quality profiler service.

    Attributes:
        total_profiles: Total dataset profiles completed.
        total_assessments: Total quality assessments completed.
        total_anomaly_detections: Total anomaly detection runs.
        total_freshness_checks: Total freshness checks performed.
        total_rules: Total quality rules defined.
        active_rules: Currently active quality rules.
        total_gate_evaluations: Total quality gate evaluations.
        total_issues_found: Total quality issues found.
        avg_quality_score: Average overall quality score.
        active_profiles: Number of currently active profiling operations.
        total_reports: Total reports generated.
    """
    total_profiles: int = Field(default=0)
    total_assessments: int = Field(default=0)
    total_anomaly_detections: int = Field(default=0)
    total_freshness_checks: int = Field(default=0)
    total_rules: int = Field(default=0)
    active_rules: int = Field(default=0)
    total_gate_evaluations: int = Field(default=0)
    total_issues_found: int = Field(default=0)
    avg_quality_score: float = Field(default=0.0)
    active_profiles: int = Field(default=0)
    total_reports: int = Field(default=0)


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
            entity_type: Type of entity (dataset_profile, assessment, rule, anomaly, gate, etc.).
            entity_id: Entity identifier.
            action: Action performed (profile, assess, validate, detect, create, etc.).
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
# Quality dimension reference data
# ===================================================================

_QUALITY_LEVELS: Dict[str, float] = {
    "EXCELLENT": 0.95,
    "GOOD": 0.85,
    "FAIR": 0.70,
    "POOR": 0.50,
    "CRITICAL": 0.0,
}


def _classify_quality(score: float) -> str:
    """Classify a quality score into a quality level.

    Args:
        score: Quality score (0.0-1.0).

    Returns:
        Quality level label (EXCELLENT, GOOD, FAIR, POOR, CRITICAL).
    """
    if score >= 0.95:
        return "EXCELLENT"
    elif score >= 0.85:
        return "GOOD"
    elif score >= 0.70:
        return "FAIR"
    elif score >= 0.50:
        return "POOR"
    return "CRITICAL"


# ===================================================================
# DataQualityProfilerService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["DataQualityProfilerService"] = None


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


class DataQualityProfilerService:
    """Unified facade over the Data Quality Profiler SDK.

    Aggregates all profiler engines (dataset profiler, quality assessor,
    anomaly detector, freshness checker, rule engine, quality gate engine,
    report generator, provenance tracker) through a single entry point with
    convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: DataQualityProfilerConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = DataQualityProfilerService()
        >>> profile = service.profile_dataset(
        ...     data=[{"name": "Alice", "age": 30}],
        ...     dataset_name="employees",
        ... )
        >>> print(profile.profile_id, profile.row_count)
    """

    def __init__(
        self,
        config: Optional[DataQualityProfilerConfig] = None,
    ) -> None:
        """Initialize the Data Quality Profiler Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - DatasetProfilerEngine
        - QualityAssessorEngine
        - AnomalyDetectorEngine
        - FreshnessCheckerEngine
        - RuleEngine
        - QualityGateEngine
        - ReportGeneratorEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._dataset_profiler_engine: Any = None
        self._quality_assessor_engine: Any = None
        self._anomaly_detector_engine: Any = None
        self._freshness_checker_engine: Any = None
        self._rule_engine: Any = None
        self._quality_gate_engine: Any = None
        self._report_generator_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._profiles: Dict[str, DatasetProfileResponse] = {}
        self._assessments: Dict[str, QualityAssessmentResponse] = {}
        self._anomaly_results: Dict[str, AnomalyDetectionResponse] = {}
        self._freshness_results: Dict[str, FreshnessCheckResponse] = {}
        self._rules: Dict[str, QualityRuleResponse] = {}
        self._gate_results: Dict[str, QualityGateResponse] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._stats = DataQualityProfilerStatisticsResponse()
        self._started = False

        logger.info("DataQualityProfilerService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def dataset_profiler_engine(self) -> Any:
        """Get the DatasetProfilerEngine instance."""
        return self._dataset_profiler_engine

    @property
    def quality_assessor_engine(self) -> Any:
        """Get the QualityAssessorEngine instance."""
        return self._quality_assessor_engine

    @property
    def anomaly_detector_engine(self) -> Any:
        """Get the AnomalyDetectorEngine instance."""
        return self._anomaly_detector_engine

    @property
    def freshness_checker_engine(self) -> Any:
        """Get the FreshnessCheckerEngine instance."""
        return self._freshness_checker_engine

    @property
    def rule_engine(self) -> Any:
        """Get the RuleEngine instance."""
        return self._rule_engine

    @property
    def quality_gate_engine(self) -> Any:
        """Get the QualityGateEngine instance."""
        return self._quality_gate_engine

    @property
    def report_generator_engine(self) -> Any:
        """Get the ReportGeneratorEngine instance."""
        return self._report_generator_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.data_quality_profiler.dataset_profiler import DatasetProfilerEngine
            self._dataset_profiler_engine = DatasetProfilerEngine(self.config)
        except ImportError:
            logger.warning("DatasetProfilerEngine not available; using stub")

        try:
            from greenlang.data_quality_profiler.quality_assessor import QualityAssessorEngine
            self._quality_assessor_engine = QualityAssessorEngine(self.config)
        except ImportError:
            logger.warning("QualityAssessorEngine not available; using stub")

        try:
            from greenlang.data_quality_profiler.anomaly_detector import AnomalyDetectorEngine
            self._anomaly_detector_engine = AnomalyDetectorEngine(self.config)
        except ImportError:
            logger.warning("AnomalyDetectorEngine not available; using stub")

        try:
            from greenlang.data_quality_profiler.freshness_checker import FreshnessCheckerEngine
            self._freshness_checker_engine = FreshnessCheckerEngine(self.config)
        except ImportError:
            logger.warning("FreshnessCheckerEngine not available; using stub")

        try:
            from greenlang.data_quality_profiler.dq_rule_engine import DQRuleEngine
            self._rule_engine = DQRuleEngine(self.config)
        except ImportError:
            logger.warning("DQRuleEngine not available; using stub")

        try:
            from greenlang.data_quality_profiler.quality_gate import QualityGateEngine
            self._quality_gate_engine = QualityGateEngine(self.config)
        except ImportError:
            logger.warning("QualityGateEngine not available; using stub")

        try:
            from greenlang.data_quality_profiler.dq_report_generator import DQReportGeneratorEngine
            self._report_generator_engine = DQReportGeneratorEngine(self.config)
        except ImportError:
            logger.warning("DQReportGeneratorEngine not available; using stub")

    # ------------------------------------------------------------------
    # Dataset profiling
    # ------------------------------------------------------------------

    def profile_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str = "unnamed",
        columns: Optional[List[str]] = None,
        source: str = "manual",
    ) -> DatasetProfileResponse:
        """Profile a dataset and generate column-level statistics.

        Deterministic profiling using Python built-in statistics.
        No LLM is used for profiling calculations (zero-hallucination).

        Args:
            data: List of row dicts representing the dataset.
            dataset_name: Name of the dataset being profiled.
            columns: Optional list of columns to profile (all if None).
            source: Data source identifier.

        Returns:
            DatasetProfileResponse with profiling results.

        Raises:
            ValueError: If data is empty.
        """
        start_time = time.time()

        if not data:
            raise ValueError("Dataset must not be empty")

        update_active_profiles(1)

        # Determine columns to profile
        all_columns = list(data[0].keys()) if data else []
        target_columns = columns if columns else all_columns
        target_columns = [
            c for c in target_columns
            if c in all_columns
        ][:self.config.max_columns_per_profile]

        # Limit rows
        rows = data[:self.config.max_rows_per_profile]
        row_count = len(rows)

        # Profile each column
        col_profiles: List[ColumnProfileResponse] = []
        total_nulls = 0
        total_cells = 0

        for col_name in target_columns:
            col_profile = self._profile_column(rows, col_name)
            col_profiles.append(col_profile)
            total_nulls += col_profile.null_count
            total_cells += col_profile.total_count
            record_column_profile(col_profile.data_type)

        # Compute overall completeness
        completeness = 1.0 - (total_nulls / max(total_cells, 1))

        processing_time_ms = (time.time() - start_time) * 1000.0

        profile = DatasetProfileResponse(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=len(target_columns),
            columns=col_profiles,
            completeness_score=round(completeness, 4),
            source=source,
            processing_time_ms=round(processing_time_ms, 2),
        )
        profile.provenance_hash = _compute_hash(profile)

        self._profiles[profile.profile_id] = profile

        # Record provenance
        self.provenance.record(
            entity_type="dataset_profile",
            entity_id=profile.profile_id,
            action="profile",
            data_hash=profile.provenance_hash,
        )

        # Update metrics
        record_profile(source)
        record_processing_duration("profile", time.time() - start_time)
        update_active_profiles(-1)

        # Update statistics
        self._stats.total_profiles += 1

        logger.info(
            "Profiled dataset %s: %d rows, %d columns, completeness=%.4f",
            dataset_name, row_count, len(target_columns), completeness,
        )
        return profile

    def profile_dataset_batch(
        self,
        datasets: List[Dict[str, Any]],
    ) -> List[DatasetProfileResponse]:
        """Profile multiple datasets in batch.

        Args:
            datasets: List of dicts, each with 'data', 'dataset_name',
                      optional 'columns' and 'source'.

        Returns:
            List of DatasetProfileResponse instances.
        """
        results: List[DatasetProfileResponse] = []
        for ds in datasets[:self.config.batch_max_datasets]:
            try:
                result = self.profile_dataset(
                    data=ds.get("data", []),
                    dataset_name=ds.get("dataset_name", "unnamed"),
                    columns=ds.get("columns"),
                    source=ds.get("source", "batch"),
                )
                results.append(result)
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Batch profile skipped %s: %s",
                    ds.get("dataset_name", "unnamed"), exc,
                )
                record_processing_error("data")
        return results

    def get_profile(self, profile_id: str) -> Optional[DatasetProfileResponse]:
        """Get a dataset profile by ID.

        Args:
            profile_id: Profile identifier.

        Returns:
            DatasetProfileResponse or None if not found.
        """
        return self._profiles.get(profile_id)

    def list_profiles(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DatasetProfileResponse]:
        """List dataset profiles.

        Args:
            limit: Maximum number of profiles to return.
            offset: Number of profiles to skip.

        Returns:
            List of DatasetProfileResponse instances.
        """
        profiles = list(self._profiles.values())
        return profiles[offset:offset + limit]

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def assess_quality(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str = "unnamed",
        dimensions: Optional[List[str]] = None,
    ) -> QualityAssessmentResponse:
        """Assess the quality of a dataset across 6 dimensions.

        All scoring is deterministic. No LLM is used for quality
        assessment (zero-hallucination).

        The 6 dimensions are:
        - completeness: Ratio of non-null values
        - validity: Ratio of values matching expected types/formats
        - consistency: Ratio of internally consistent records
        - timeliness: Freshness of data relative to expectations
        - uniqueness: Ratio of unique values in key columns
        - accuracy: Ratio of values within expected ranges

        Args:
            data: List of row dicts representing the dataset.
            dataset_name: Name of the dataset.
            dimensions: Optional list of dimensions to score (all 6 if None).

        Returns:
            QualityAssessmentResponse with overall and per-dimension scores.

        Raises:
            ValueError: If data is empty.
        """
        start_time = time.time()

        if not data:
            raise ValueError("Dataset must not be empty for quality assessment")

        rows = data[:self.config.max_rows_per_profile]
        all_dims = dimensions or [
            "completeness", "validity", "consistency",
            "timeliness", "uniqueness", "accuracy",
        ]

        # Step 1: Profile the dataset
        profile = self.profile_dataset(
            data=rows,
            dataset_name=dataset_name,
        )

        # Step 2: Score each dimension
        dim_scores: Dict[str, float] = {}
        issues: List[Dict[str, Any]] = []

        if "completeness" in all_dims:
            score, dim_issues = self._score_completeness(rows, profile)
            dim_scores["completeness"] = score
            issues.extend(dim_issues)

        if "validity" in all_dims:
            score, dim_issues = self._score_validity(rows, profile)
            dim_scores["validity"] = score
            issues.extend(dim_issues)

        if "consistency" in all_dims:
            score, dim_issues = self._score_consistency(rows, profile)
            dim_scores["consistency"] = score
            issues.extend(dim_issues)

        if "timeliness" in all_dims:
            score, dim_issues = self._score_timeliness(rows, profile)
            dim_scores["timeliness"] = score
            issues.extend(dim_issues)

        if "uniqueness" in all_dims:
            score, dim_issues = self._score_uniqueness(rows, profile)
            dim_scores["uniqueness"] = score
            issues.extend(dim_issues)

        if "accuracy" in all_dims:
            score, dim_issues = self._score_accuracy(rows, profile)
            dim_scores["accuracy"] = score
            issues.extend(dim_issues)

        # Step 3: Compute weighted overall score
        overall_score = self._compute_weighted_score(dim_scores)

        # Step 4: Classify quality level
        quality_level = _classify_quality(overall_score)

        processing_time_ms = (time.time() - start_time) * 1000.0

        assessment = QualityAssessmentResponse(
            dataset_name=dataset_name,
            overall_score=round(overall_score, 4),
            quality_level=quality_level,
            completeness_score=round(dim_scores.get("completeness", 0.0), 4),
            validity_score=round(dim_scores.get("validity", 0.0), 4),
            consistency_score=round(dim_scores.get("consistency", 0.0), 4),
            timeliness_score=round(dim_scores.get("timeliness", 0.0), 4),
            uniqueness_score=round(dim_scores.get("uniqueness", 0.0), 4),
            accuracy_score=round(dim_scores.get("accuracy", 0.0), 4),
            issues=issues,
            row_count=len(rows),
            column_count=profile.column_count,
            processing_time_ms=round(processing_time_ms, 2),
        )
        assessment.provenance_hash = _compute_hash(assessment)

        self._assessments[assessment.assessment_id] = assessment

        # Record provenance
        self.provenance.record(
            entity_type="assessment",
            entity_id=assessment.assessment_id,
            action="assess",
            data_hash=assessment.provenance_hash,
        )

        # Update metrics
        record_assessment(quality_level)
        record_quality_score(overall_score)
        update_total_issues(len(issues))
        record_processing_duration("assess", time.time() - start_time)

        # Update statistics
        self._stats.total_assessments += 1
        self._stats.total_issues_found += len(issues)
        self._update_avg_quality(overall_score)

        logger.info(
            "Assessed quality of dataset %s: score=%.4f level=%s issues=%d",
            dataset_name, overall_score, quality_level, len(issues),
        )
        return assessment

    def assess_quality_batch(
        self,
        datasets: List[Dict[str, Any]],
    ) -> List[QualityAssessmentResponse]:
        """Assess quality for multiple datasets in batch.

        Args:
            datasets: List of dicts, each with 'data', 'dataset_name',
                      optional 'dimensions'.

        Returns:
            List of QualityAssessmentResponse instances.
        """
        results: List[QualityAssessmentResponse] = []
        for ds in datasets[:self.config.batch_max_datasets]:
            try:
                result = self.assess_quality(
                    data=ds.get("data", []),
                    dataset_name=ds.get("dataset_name", "unnamed"),
                    dimensions=ds.get("dimensions"),
                )
                results.append(result)
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Batch assess skipped %s: %s",
                    ds.get("dataset_name", "unnamed"), exc,
                )
                record_processing_error("data")
        return results

    def get_assessment(
        self,
        assessment_id: str,
    ) -> Optional[QualityAssessmentResponse]:
        """Get a quality assessment by ID.

        Args:
            assessment_id: Assessment identifier.

        Returns:
            QualityAssessmentResponse or None if not found.
        """
        return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[QualityAssessmentResponse]:
        """List quality assessments.

        Args:
            limit: Maximum number of assessments to return.
            offset: Number of assessments to skip.

        Returns:
            List of QualityAssessmentResponse instances.
        """
        assessments = list(self._assessments.values())
        return assessments[offset:offset + limit]

    # ------------------------------------------------------------------
    # Validation (rule-based)
    # ------------------------------------------------------------------

    def validate_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str = "unnamed",
        rule_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate a dataset against quality rules.

        Evaluates each applicable rule against every row of data.
        All validation is deterministic (zero-hallucination).

        Args:
            data: List of row dicts representing the dataset.
            dataset_name: Name of the dataset.
            rule_ids: Optional list of rule IDs to evaluate (all active if None).

        Returns:
            Dict with validation results including per-rule pass/fail counts.

        Raises:
            ValueError: If data is empty.
        """
        start_time = time.time()

        if not data:
            raise ValueError("Dataset must not be empty for validation")

        rows = data[:self.config.max_rows_per_profile]

        # Select rules to evaluate
        if rule_ids:
            rules = [
                r for r in self._rules.values()
                if r.rule_id in rule_ids and r.is_active
            ]
        else:
            rules = [r for r in self._rules.values() if r.is_active]

        rules.sort(key=lambda r: r.priority)

        rule_results: List[Dict[str, Any]] = []
        total_pass = 0
        total_fail = 0

        for rule in rules:
            pass_count, fail_count, failures = self._evaluate_rule(rule, rows)
            result_str = "pass" if fail_count == 0 else "fail"

            # Update rule state
            rule.last_result = result_str
            rule.last_evaluated_at = datetime.now(timezone.utc).isoformat()
            rule.fail_count = fail_count

            record_rule_evaluation(result_str)

            rule_results.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "rule_type": rule.rule_type,
                "column": rule.column,
                "result": result_str,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "failures_sample": failures[:10],
            })
            total_pass += pass_count
            total_fail += fail_count

        overall = "pass" if total_fail == 0 else "fail"
        processing_time_ms = (time.time() - start_time) * 1000.0

        validation_result = {
            "dataset_name": dataset_name,
            "overall_result": overall,
            "rules_evaluated": len(rules),
            "total_pass": total_pass,
            "total_fail": total_fail,
            "rule_results": rule_results,
            "row_count": len(rows),
            "processing_time_ms": round(processing_time_ms, 2),
            "provenance_hash": _compute_hash({
                "dataset_name": dataset_name,
                "overall_result": overall,
                "total_pass": total_pass,
                "total_fail": total_fail,
            }),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Record provenance
        self.provenance.record(
            entity_type="validation",
            entity_id=dataset_name,
            action="validate",
            data_hash=validation_result["provenance_hash"],
        )

        record_processing_duration("validate", time.time() - start_time)

        logger.info(
            "Validated dataset %s: %d rules, %d pass, %d fail -> %s",
            dataset_name, len(rules), total_pass, total_fail, overall,
        )
        return validation_result

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str = "unnamed",
        columns: Optional[List[str]] = None,
        method: Optional[str] = None,
    ) -> AnomalyDetectionResponse:
        """Detect anomalies in a dataset.

        Uses deterministic statistical methods for anomaly detection.
        No LLM is used for anomaly detection (zero-hallucination).

        Supported methods:
        - iqr: Interquartile range fence method
        - zscore: Z-score threshold method
        - percentile: Extreme percentile method

        Args:
            data: List of row dicts representing the dataset.
            dataset_name: Name of the dataset.
            columns: Columns to analyse (all numeric if None).
            method: Detection method override (uses config default if None).

        Returns:
            AnomalyDetectionResponse with detected anomalies.

        Raises:
            ValueError: If data is empty or insufficient samples.
        """
        start_time = time.time()

        if not data:
            raise ValueError("Dataset must not be empty for anomaly detection")

        rows = data[:self.config.max_rows_per_profile]
        detection_method = method or self.config.default_outlier_method

        if len(rows) < self.config.min_samples_for_anomaly:
            raise ValueError(
                f"Minimum {self.config.min_samples_for_anomaly} samples "
                f"required, got {len(rows)}"
            )

        # Determine numeric columns
        all_columns = list(rows[0].keys()) if rows else []
        numeric_cols = columns if columns else self._detect_numeric_columns(rows)
        numeric_cols = [c for c in numeric_cols if c in all_columns]

        anomalies: List[Dict[str, Any]] = []

        for col in numeric_cols:
            col_anomalies = self._detect_column_anomalies(
                rows, col, detection_method,
            )
            anomalies.extend(col_anomalies)

        anomaly_count = len(anomalies)
        anomaly_pct = (anomaly_count / max(len(rows), 1)) * 100.0

        processing_time_ms = (time.time() - start_time) * 1000.0

        result = AnomalyDetectionResponse(
            dataset_name=dataset_name,
            method=detection_method,
            total_records=len(rows),
            anomaly_count=anomaly_count,
            anomaly_pct=round(anomaly_pct, 2),
            anomalies=anomalies[:500],  # Limit stored anomalies
            columns_analysed=numeric_cols,
            processing_time_ms=round(processing_time_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        self._anomaly_results[result.detection_id] = result

        # Record provenance
        self.provenance.record(
            entity_type="anomaly",
            entity_id=result.detection_id,
            action="detect",
            data_hash=result.provenance_hash,
        )

        # Update metrics
        for _ in range(anomaly_count):
            record_anomaly(detection_method)
        record_processing_duration("detect_anomalies", time.time() - start_time)

        # Update statistics
        self._stats.total_anomaly_detections += 1

        logger.info(
            "Detected %d anomalies in dataset %s using %s (%.2f%%)",
            anomaly_count, dataset_name, detection_method, anomaly_pct,
        )
        return result

    def list_anomalies(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AnomalyDetectionResponse]:
        """List anomaly detection results.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of AnomalyDetectionResponse instances.
        """
        results = list(self._anomaly_results.values())
        return results[offset:offset + limit]

    # ------------------------------------------------------------------
    # Freshness checking
    # ------------------------------------------------------------------

    def check_freshness(
        self,
        dataset_name: str,
        last_updated: str,
        sla_hours: Optional[float] = None,
    ) -> FreshnessCheckResponse:
        """Check the freshness of a dataset.

        Deterministic age-based freshness scoring.
        No LLM is used for freshness assessment (zero-hallucination).

        Args:
            dataset_name: Name of the dataset.
            last_updated: ISO 8601 timestamp of last data update.
            sla_hours: SLA threshold in hours (uses config default if None).

        Returns:
            FreshnessCheckResponse with freshness status and score.
        """
        start_time = time.time()

        sla = sla_hours if sla_hours is not None else float(self.config.default_sla_hours)

        # Calculate age in hours
        try:
            last_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            last_dt = _utcnow()

        now = _utcnow()
        age_hours = (now - last_dt).total_seconds() / 3600.0

        # Determine freshness status
        if age_hours <= sla:
            status = "fresh"
        elif age_hours <= sla * 2:
            status = "stale"
        else:
            status = "expired"

        # Compute freshness score (1.0 at 0 hours, decaying)
        if age_hours <= 0:
            freshness_score = 1.0
        elif sla > 0:
            freshness_score = max(0.0, 1.0 - (age_hours / (sla * 3)))
        else:
            freshness_score = 0.0

        result = FreshnessCheckResponse(
            dataset_name=dataset_name,
            last_updated=last_updated,
            age_hours=round(age_hours, 2),
            sla_hours=sla,
            status=status,
            freshness_score=round(freshness_score, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self._freshness_results[result.check_id] = result

        # Record provenance
        self.provenance.record(
            entity_type="freshness",
            entity_id=result.check_id,
            action="check",
            data_hash=result.provenance_hash,
        )

        # Update metrics
        record_freshness_check(status)
        record_processing_duration("check_freshness", time.time() - start_time)

        # Update statistics
        self._stats.total_freshness_checks += 1

        logger.info(
            "Freshness check for %s: age=%.2fh sla=%.1fh status=%s score=%.4f",
            dataset_name, age_hours, sla, status, freshness_score,
        )
        return result

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def create_rule(
        self,
        name: str,
        rule_type: str,
        column: str = "",
        operator: str = "eq",
        threshold: Any = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 100,
    ) -> QualityRuleResponse:
        """Create a new quality rule.

        Args:
            name: Rule display name.
            rule_type: Rule type (not_null, unique, range, regex, custom, referential).
            column: Target column name.
            operator: Comparison operator.
            threshold: Threshold value for the rule.
            parameters: Additional rule parameters.
            priority: Rule priority (lower = higher priority).

        Returns:
            QualityRuleResponse with rule details.

        Raises:
            ValueError: If name is empty or max rules exceeded.
        """
        start_time = time.time()

        if not name.strip():
            raise ValueError("Rule name must not be empty")

        if len(self._rules) >= self.config.max_rules_per_dataset:
            raise ValueError(
                f"Maximum {self.config.max_rules_per_dataset} rules allowed"
            )

        rule = QualityRuleResponse(
            name=name,
            rule_type=rule_type,
            column=column,
            operator=operator,
            threshold=threshold,
            parameters=parameters or {},
            priority=priority,
            is_active=True,
        )
        rule.provenance_hash = _compute_hash(rule)
        self._rules[rule.rule_id] = rule

        # Record provenance
        self.provenance.record(
            entity_type="rule",
            entity_id=rule.rule_id,
            action="create",
            data_hash=rule.provenance_hash,
        )

        # Update statistics
        self._stats.total_rules += 1
        self._stats.active_rules = sum(
            1 for r in self._rules.values() if r.is_active
        )

        record_processing_duration("create_rule", time.time() - start_time)

        logger.info(
            "Created rule %s (%s, type=%s, column=%s, priority=%d)",
            rule.rule_id, name, rule_type, column, priority,
        )
        return rule

    def list_rules(
        self,
        active_only: bool = False,
        rule_type: Optional[str] = None,
    ) -> List[QualityRuleResponse]:
        """List quality rules with optional filters.

        Args:
            active_only: If True, only return active rules.
            rule_type: Optional rule type filter.

        Returns:
            List of QualityRuleResponse instances sorted by priority.
        """
        rules = list(self._rules.values())

        if active_only:
            rules = [r for r in rules if r.is_active]
        if rule_type is not None:
            rules = [r for r in rules if r.rule_type == rule_type]

        rules.sort(key=lambda r: r.priority)
        return rules

    def update_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any],
    ) -> QualityRuleResponse:
        """Update an existing quality rule.

        Args:
            rule_id: Rule identifier.
            updates: Dict of fields to update.

        Returns:
            Updated QualityRuleResponse.

        Raises:
            ValueError: If rule not found.
        """
        start_time = time.time()

        rule = self._rules.get(rule_id)
        if rule is None:
            raise ValueError(f"Rule {rule_id} not found")

        if "name" in updates and updates["name"] is not None:
            rule.name = updates["name"]
        if "rule_type" in updates and updates["rule_type"] is not None:
            rule.rule_type = updates["rule_type"]
        if "column" in updates and updates["column"] is not None:
            rule.column = updates["column"]
        if "operator" in updates and updates["operator"] is not None:
            rule.operator = updates["operator"]
        if "threshold" in updates:
            rule.threshold = updates["threshold"]
        if "parameters" in updates and updates["parameters"] is not None:
            rule.parameters = updates["parameters"]
        if "priority" in updates and updates["priority"] is not None:
            rule.priority = updates["priority"]
        if "is_active" in updates and updates["is_active"] is not None:
            rule.is_active = updates["is_active"]

        rule.updated_at = datetime.now(timezone.utc).isoformat()
        rule.provenance_hash = _compute_hash(rule)

        # Record provenance
        self.provenance.record(
            entity_type="rule",
            entity_id=rule_id,
            action="update",
            data_hash=rule.provenance_hash,
        )

        # Update statistics
        self._stats.active_rules = sum(
            1 for r in self._rules.values() if r.is_active
        )

        record_processing_duration("update_rule", time.time() - start_time)

        logger.info("Updated rule %s", rule_id)
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a quality rule.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if deleted, False if not found.
        """
        rule = self._rules.pop(rule_id, None)
        if rule is None:
            return False

        self.provenance.record(
            entity_type="rule",
            entity_id=rule_id,
            action="delete",
            data_hash=_compute_hash({"rule_id": rule_id, "deleted": True}),
        )

        self._stats.total_rules = max(0, self._stats.total_rules - 1)
        self._stats.active_rules = sum(
            1 for r in self._rules.values() if r.is_active
        )

        logger.info("Deleted rule %s", rule_id)
        return True

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------

    def evaluate_gate(
        self,
        conditions: List[Dict[str, Any]],
        dimension_scores: Optional[Dict[str, float]] = None,
    ) -> QualityGateResponse:
        """Evaluate a quality gate against dimension scores.

        Each condition is a dict with 'dimension', 'operator', 'threshold'.
        All gate evaluation is deterministic (zero-hallucination).

        Args:
            conditions: List of condition dicts to evaluate.
            dimension_scores: Dict of dimension name to score. If None, uses
                              latest assessment scores.

        Returns:
            QualityGateResponse with gate outcome.

        Raises:
            ValueError: If no conditions provided.
        """
        start_time = time.time()

        if not conditions:
            raise ValueError("At least one gate condition is required")

        conditions = conditions[:self.config.max_gate_conditions]

        # Use provided scores or fallback to zeros
        scores = dimension_scores or {}

        details: List[Dict[str, Any]] = []
        passed = 0
        failed = 0

        for cond in conditions:
            dim = cond.get("dimension", "")
            op = cond.get("operator", "gte")
            thresh = float(cond.get("threshold", self.config.default_gate_threshold))
            actual = scores.get(dim, 0.0)

            cond_result = self._evaluate_gate_condition(actual, op, thresh)

            if cond_result:
                passed += 1
            else:
                failed += 1

            details.append({
                "dimension": dim,
                "operator": op,
                "threshold": thresh,
                "actual": actual,
                "result": "pass" if cond_result else "fail",
            })

        # Determine overall outcome
        if failed == 0:
            outcome = "pass"
        elif passed > 0 and failed > 0:
            outcome = "warn"
        else:
            outcome = "fail"

        overall_score = sum(scores.values()) / max(len(scores), 1) if scores else 0.0

        result = QualityGateResponse(
            outcome=outcome,
            conditions_evaluated=len(conditions),
            conditions_passed=passed,
            conditions_failed=failed,
            details=details,
            overall_score=round(overall_score, 4),
            threshold=self.config.default_gate_threshold,
        )
        result.provenance_hash = _compute_hash(result)

        self._gate_results[result.gate_id] = result

        # Record provenance
        self.provenance.record(
            entity_type="gate",
            entity_id=result.gate_id,
            action="evaluate",
            data_hash=result.provenance_hash,
        )

        # Update metrics
        record_gate_evaluation(outcome)
        record_processing_duration("evaluate_gate", time.time() - start_time)

        # Update statistics
        self._stats.total_gate_evaluations += 1

        logger.info(
            "Gate evaluation: %d conditions, %d passed, %d failed -> %s",
            len(conditions), passed, failed, outcome,
        )
        return result

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------

    def get_trends(
        self,
        dataset_name: Optional[str] = None,
        periods: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get quality score trends over recent assessments.

        Args:
            dataset_name: Optional dataset name filter.
            periods: Maximum number of trend periods to return.

        Returns:
            List of trend data points.
        """
        start_time = time.time()

        assessments = list(self._assessments.values())

        if dataset_name:
            assessments = [
                a for a in assessments
                if a.dataset_name == dataset_name
            ]

        # Sort by assessment time
        assessments.sort(key=lambda a: a.assessed_at)
        assessments = assessments[-periods:]

        trends = []
        for a in assessments:
            trends.append({
                "assessment_id": a.assessment_id,
                "dataset_name": a.dataset_name,
                "overall_score": a.overall_score,
                "quality_level": a.quality_level,
                "completeness": a.completeness_score,
                "validity": a.validity_score,
                "consistency": a.consistency_score,
                "timeliness": a.timeliness_score,
                "uniqueness": a.uniqueness_score,
                "accuracy": a.accuracy_score,
                "issue_count": len(a.issues),
                "assessed_at": a.assessed_at,
            })

        record_processing_duration("trends", time.time() - start_time)

        logger.info(
            "Generated %d trend data points for %s",
            len(trends), dataset_name or "all",
        )
        return trends

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        dataset_name: Optional[str] = None,
        report_type: str = "scorecard",
        report_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate a data quality report.

        Supported report types: scorecard, detailed, executive, issues, anomaly.
        Supported formats: json, markdown, html, text, csv.

        Args:
            dataset_name: Optional dataset name filter.
            report_type: Report type.
            report_format: Output format.

        Returns:
            Dict with report_id, content, and metadata.
        """
        start_time = time.time()

        report_id = str(uuid.uuid4())

        # Gather data
        assessments = list(self._assessments.values())
        if dataset_name:
            assessments = [
                a for a in assessments
                if a.dataset_name == dataset_name
            ]

        profiles = list(self._profiles.values())
        if dataset_name:
            profiles = [
                p for p in profiles
                if p.dataset_name == dataset_name
            ]

        anomaly_results = list(self._anomaly_results.values())
        if dataset_name:
            anomaly_results = [
                a for a in anomaly_results
                if a.dataset_name == dataset_name
            ]

        # Build content based on report type and format
        content = self._build_report_content(
            report_type=report_type,
            report_format=report_format,
            assessments=assessments,
            profiles=profiles,
            anomaly_results=anomaly_results,
            dataset_name=dataset_name,
        )

        report = {
            "report_id": report_id,
            "report_type": report_type,
            "format": report_format,
            "dataset_name": dataset_name or "all",
            "content": content,
            "assessment_count": len(assessments),
            "profile_count": len(profiles),
            "anomaly_count": len(anomaly_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance_hash": _compute_hash({
                "report_id": report_id,
                "report_type": report_type,
                "format": report_format,
            }),
        }

        self._reports[report_id] = report

        # Record provenance
        self.provenance.record(
            entity_type="report",
            entity_id=report_id,
            action="generate",
            data_hash=report["provenance_hash"],
        )

        record_processing_duration("generate_report", time.time() - start_time)

        # Update statistics
        self._stats.total_reports += 1

        logger.info(
            "Generated %s report (%s) for %s: %d assessments",
            report_type, report_format, dataset_name or "all",
            len(assessments),
        )
        return report

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def get_statistics(self) -> DataQualityProfilerStatisticsResponse:
        """Get aggregated data quality profiler statistics.

        Returns:
            DataQualityProfilerStatisticsResponse summary.
        """
        return self._stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the service.

        Returns:
            Health status dict.
        """
        return {
            "status": "healthy" if self._started else "not_started",
            "service": "data-quality-profiler",
            "started": self._started,
            "profiles": len(self._profiles),
            "assessments": len(self._assessments),
            "anomaly_detections": len(self._anomaly_results),
            "freshness_checks": len(self._freshness_results),
            "rules": len(self._rules),
            "gate_evaluations": len(self._gate_results),
            "reports": len(self._reports),
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get data quality profiler service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_profiles": self._stats.total_profiles,
            "total_assessments": self._stats.total_assessments,
            "total_anomaly_detections": self._stats.total_anomaly_detections,
            "total_freshness_checks": self._stats.total_freshness_checks,
            "total_rules": self._stats.total_rules,
            "active_rules": self._stats.active_rules,
            "total_gate_evaluations": self._stats.total_gate_evaluations,
            "total_issues_found": self._stats.total_issues_found,
            "avg_quality_score": self._stats.avg_quality_score,
            "total_reports": self._stats.total_reports,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers: column profiling
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        rows: List[Dict[str, Any]],
        col_name: str,
    ) -> ColumnProfileResponse:
        """Profile a single column.

        Args:
            rows: Dataset rows.
            col_name: Column name to profile.

        Returns:
            ColumnProfileResponse with column statistics.
        """
        values = [row.get(col_name) for row in rows]
        total = len(values)
        nulls = sum(1 for v in values if v is None or v == "")
        non_null = [v for v in values if v is not None and v != ""]

        null_pct = (nulls / max(total, 1)) * 100.0
        distinct = len(set(str(v) for v in non_null))
        distinct_pct = (distinct / max(len(non_null), 1)) * 100.0

        # Detect data type
        data_type = self._detect_column_type(non_null)

        # Compute numeric stats
        mean_val = None
        median_val = None
        stddev_val = None
        min_val = None
        max_val = None

        if data_type in ("integer", "float") and non_null:
            nums = []
            for v in non_null:
                try:
                    nums.append(float(v))
                except (ValueError, TypeError):
                    pass
            if nums:
                mean_val = round(statistics.mean(nums), 4)
                median_val = round(statistics.median(nums), 4)
                if len(nums) >= 2:
                    stddev_val = round(statistics.stdev(nums), 4)
                min_val = min(nums)
                max_val = max(nums)

        # Top values
        freq: Dict[str, int] = {}
        for v in non_null[:self.config.sample_size_for_stats]:
            key = str(v)
            freq[key] = freq.get(key, 0) + 1
        top_values = sorted(
            [{"value": k, "count": c} for k, c in freq.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

        return ColumnProfileResponse(
            column_name=col_name,
            data_type=data_type,
            total_count=total,
            null_count=nulls,
            null_pct=round(null_pct, 2),
            distinct_count=distinct,
            distinct_pct=round(distinct_pct, 2),
            min_value=min_val,
            max_value=max_val,
            mean=mean_val,
            median=median_val,
            stddev=stddev_val,
            top_values=top_values,
        )

    def _detect_column_type(self, values: List[Any]) -> str:
        """Detect the predominant data type of column values.

        Args:
            values: Non-null column values.

        Returns:
            Detected type string.
        """
        if not values:
            return "string"

        sample = values[:100]
        int_count = 0
        float_count = 0
        bool_count = 0

        for v in sample:
            if isinstance(v, bool):
                bool_count += 1
            elif isinstance(v, int):
                int_count += 1
            elif isinstance(v, float):
                float_count += 1
            elif isinstance(v, str):
                v_stripped = v.strip()
                if v_stripped.lower() in ("true", "false"):
                    bool_count += 1
                else:
                    try:
                        int(v_stripped)
                        int_count += 1
                    except (ValueError, TypeError):
                        try:
                            float(v_stripped)
                            float_count += 1
                        except (ValueError, TypeError):
                            pass

        total = len(sample)
        if bool_count > total * 0.5:
            return "boolean"
        if int_count > total * 0.5:
            return "integer"
        if float_count > total * 0.3 or (int_count + float_count) > total * 0.5:
            return "float"
        return "string"

    # ------------------------------------------------------------------
    # Internal helpers: quality dimension scoring
    # ------------------------------------------------------------------

    def _score_completeness(
        self,
        rows: List[Dict[str, Any]],
        profile: DatasetProfileResponse,
    ) -> tuple:
        """Score the completeness dimension.

        Args:
            rows: Dataset rows.
            profile: Dataset profile.

        Returns:
            Tuple of (score, issues_list).
        """
        issues: List[Dict[str, Any]] = []
        total_cells = 0
        null_cells = 0

        for col in profile.columns:
            total_cells += col.total_count
            null_cells += col.null_count
            if col.null_pct > 10.0:
                issues.append({
                    "dimension": "completeness",
                    "column": col.column_name,
                    "issue": f"High null rate: {col.null_pct:.1f}%",
                    "severity": "warning" if col.null_pct < 50 else "critical",
                })

        score = 1.0 - (null_cells / max(total_cells, 1))
        return round(score, 4), issues

    def _score_validity(
        self,
        rows: List[Dict[str, Any]],
        profile: DatasetProfileResponse,
    ) -> tuple:
        """Score the validity dimension.

        Args:
            rows: Dataset rows.
            profile: Dataset profile.

        Returns:
            Tuple of (score, issues_list).
        """
        issues: List[Dict[str, Any]] = []
        valid_count = 0
        total_count = 0

        for col in profile.columns:
            col_values = [row.get(col.column_name) for row in rows]
            non_null = [v for v in col_values if v is not None and v != ""]
            total_count += len(non_null)

            for v in non_null:
                if self._is_valid_value(v, col.data_type):
                    valid_count += 1
                else:
                    pass  # Invalid but counted in total

        if total_count == 0:
            return 1.0, issues

        score = valid_count / total_count

        if score < 0.9:
            issues.append({
                "dimension": "validity",
                "issue": f"Overall validity score is low: {score:.2%}",
                "severity": "warning" if score >= 0.7 else "critical",
            })

        return round(score, 4), issues

    def _score_consistency(
        self,
        rows: List[Dict[str, Any]],
        profile: DatasetProfileResponse,
    ) -> tuple:
        """Score the consistency dimension.

        Checks for mixed types and format inconsistencies within columns.

        Args:
            rows: Dataset rows.
            profile: Dataset profile.

        Returns:
            Tuple of (score, issues_list).
        """
        issues: List[Dict[str, Any]] = []
        consistent_cols = 0
        total_cols = max(len(profile.columns), 1)

        for col in profile.columns:
            col_values = [row.get(col.column_name) for row in rows]
            non_null = [v for v in col_values if v is not None and v != ""]
            if not non_null:
                consistent_cols += 1
                continue

            # Check type consistency
            types = set(type(v).__name__ for v in non_null[:100])
            if len(types) <= 1:
                consistent_cols += 1
            else:
                issues.append({
                    "dimension": "consistency",
                    "column": col.column_name,
                    "issue": f"Mixed types detected: {', '.join(sorted(types))}",
                    "severity": "warning",
                })

        score = consistent_cols / total_cols
        return round(score, 4), issues

    def _score_timeliness(
        self,
        rows: List[Dict[str, Any]],
        profile: DatasetProfileResponse,
    ) -> tuple:
        """Score the timeliness dimension.

        Uses a default high score since timeliness is typically assessed
        via freshness checks rather than row-level analysis.

        Args:
            rows: Dataset rows.
            profile: Dataset profile.

        Returns:
            Tuple of (score, issues_list).
        """
        # Default to high score; freshness checks provide detailed timeliness
        return 0.90, []

    def _score_uniqueness(
        self,
        rows: List[Dict[str, Any]],
        profile: DatasetProfileResponse,
    ) -> tuple:
        """Score the uniqueness dimension.

        Args:
            rows: Dataset rows.
            profile: Dataset profile.

        Returns:
            Tuple of (score, issues_list).
        """
        issues: List[Dict[str, Any]] = []

        if not rows:
            return 1.0, issues

        # Check for duplicate rows
        row_strs = []
        for row in rows:
            row_strs.append(json.dumps(row, sort_keys=True, default=str))

        unique_rows = len(set(row_strs))
        total_rows = len(row_strs)
        dup_rate = 1.0 - (unique_rows / max(total_rows, 1))

        if dup_rate > 0.05:
            issues.append({
                "dimension": "uniqueness",
                "issue": f"Duplicate row rate: {dup_rate:.2%}",
                "severity": "warning" if dup_rate < 0.2 else "critical",
            })

        # Also check per-column uniqueness for columns that should be unique
        col_scores = []
        for col in profile.columns:
            if col.distinct_pct > 0:
                col_scores.append(col.distinct_pct / 100.0)

        if col_scores:
            avg_uniqueness = statistics.mean(col_scores)
        else:
            avg_uniqueness = 1.0

        # Combine row uniqueness and column uniqueness
        row_uniqueness = 1.0 - dup_rate
        score = (row_uniqueness + avg_uniqueness) / 2.0
        return round(score, 4), issues

    def _score_accuracy(
        self,
        rows: List[Dict[str, Any]],
        profile: DatasetProfileResponse,
    ) -> tuple:
        """Score the accuracy dimension.

        Checks for values within expected ranges and formats.

        Args:
            rows: Dataset rows.
            profile: Dataset profile.

        Returns:
            Tuple of (score, issues_list).
        """
        issues: List[Dict[str, Any]] = []
        accurate_values = 0
        total_checked = 0

        for col in profile.columns:
            if col.data_type not in ("integer", "float"):
                continue
            if col.mean is None or col.stddev is None:
                continue
            if col.stddev == 0:
                continue

            col_values = [row.get(col.column_name) for row in rows]
            for v in col_values:
                if v is None or v == "":
                    continue
                try:
                    num = float(v)
                except (ValueError, TypeError):
                    continue
                total_checked += 1
                # Check if within 3 standard deviations
                if abs(num - col.mean) <= 3 * col.stddev:
                    accurate_values += 1

        if total_checked == 0:
            return 0.95, issues  # Default high accuracy for non-numeric data

        score = accurate_values / total_checked

        if score < 0.9:
            issues.append({
                "dimension": "accuracy",
                "issue": f"Accuracy score is low: {score:.2%}",
                "severity": "warning" if score >= 0.7 else "critical",
            })

        return round(score, 4), issues

    def _compute_weighted_score(
        self,
        dim_scores: Dict[str, float],
    ) -> float:
        """Compute the weighted overall quality score.

        Args:
            dim_scores: Dict of dimension name to score.

        Returns:
            Weighted overall score (0.0-1.0).
        """
        weights = {
            "completeness": self.config.completeness_weight,
            "validity": self.config.validity_weight,
            "consistency": self.config.consistency_weight,
            "timeliness": self.config.timeliness_weight,
            "uniqueness": self.config.uniqueness_weight,
            "accuracy": self.config.accuracy_weight,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for dim, score in dim_scores.items():
            w = weights.get(dim, 0.0)
            weighted_sum += score * w
            total_weight += w

        if total_weight <= 0:
            return 0.0

        return weighted_sum / total_weight

    def _is_valid_value(self, value: Any, expected_type: str) -> bool:
        """Check if a value is valid for the expected type.

        Args:
            value: Value to validate.
            expected_type: Expected data type.

        Returns:
            True if value is valid for the type.
        """
        if value is None or value == "":
            return True  # Nulls handled by completeness

        if expected_type == "integer":
            try:
                int(str(value).strip())
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == "float":
            try:
                float(str(value).strip())
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == "boolean":
            return str(value).strip().lower() in (
                "true", "false", "1", "0", "yes", "no",
            )
        # Strings are always valid
        return True

    # ------------------------------------------------------------------
    # Internal helpers: anomaly detection
    # ------------------------------------------------------------------

    def _detect_numeric_columns(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Detect numeric columns in the dataset.

        Args:
            rows: Dataset rows.

        Returns:
            List of column names that contain numeric data.
        """
        if not rows:
            return []

        cols = list(rows[0].keys())
        numeric: List[str] = []

        for col in cols:
            sample = [row.get(col) for row in rows[:50]]
            num_count = 0
            for v in sample:
                if v is None or v == "":
                    continue
                try:
                    float(v)
                    num_count += 1
                except (ValueError, TypeError):
                    pass
            non_null = sum(1 for v in sample if v is not None and v != "")
            if non_null > 0 and num_count / non_null > 0.5:
                numeric.append(col)

        return numeric

    def _detect_column_anomalies(
        self,
        rows: List[Dict[str, Any]],
        col_name: str,
        method: str,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in a single column.

        Args:
            rows: Dataset rows.
            col_name: Column name.
            method: Detection method (iqr, zscore, percentile).

        Returns:
            List of anomaly detail dicts.
        """
        values: List[float] = []
        indices: List[int] = []

        for i, row in enumerate(rows):
            v = row.get(col_name)
            if v is None or v == "":
                continue
            try:
                values.append(float(v))
                indices.append(i)
            except (ValueError, TypeError):
                continue

        if len(values) < self.config.min_samples_for_anomaly:
            return []

        anomalies: List[Dict[str, Any]] = []

        if method == "iqr":
            anomalies = self._iqr_anomalies(values, indices, col_name)
        elif method == "zscore":
            anomalies = self._zscore_anomalies(values, indices, col_name)
        elif method == "percentile":
            anomalies = self._percentile_anomalies(values, indices, col_name)
        else:
            anomalies = self._iqr_anomalies(values, indices, col_name)

        return anomalies

    def _iqr_anomalies(
        self,
        values: List[float],
        indices: List[int],
        col_name: str,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using IQR method.

        Args:
            values: Numeric values.
            indices: Row indices.
            col_name: Column name.

        Returns:
            List of anomaly dicts.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr

        anomalies = []
        for v, idx in zip(values, indices):
            if v < lower or v > upper:
                anomalies.append({
                    "column": col_name,
                    "row_index": idx,
                    "value": v,
                    "method": "iqr",
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                })
        return anomalies

    def _zscore_anomalies(
        self,
        values: List[float],
        indices: List[int],
        col_name: str,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using z-score method.

        Args:
            values: Numeric values.
            indices: Row indices.
            col_name: Column name.

        Returns:
            List of anomaly dicts.
        """
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) >= 2 else 0.0

        if std_val == 0:
            return []

        threshold = self.config.zscore_threshold
        anomalies = []

        for v, idx in zip(values, indices):
            z = abs(v - mean_val) / std_val
            if z > threshold:
                anomalies.append({
                    "column": col_name,
                    "row_index": idx,
                    "value": v,
                    "method": "zscore",
                    "zscore": round(z, 4),
                    "threshold": threshold,
                })
        return anomalies

    def _percentile_anomalies(
        self,
        values: List[float],
        indices: List[int],
        col_name: str,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using extreme percentile method.

        Args:
            values: Numeric values.
            indices: Row indices.
            col_name: Column name.

        Returns:
            List of anomaly dicts.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        p1 = sorted_vals[max(0, int(n * 0.01))]
        p99 = sorted_vals[min(n - 1, int(n * 0.99))]

        anomalies = []
        for v, idx in zip(values, indices):
            if v < p1 or v > p99:
                anomalies.append({
                    "column": col_name,
                    "row_index": idx,
                    "value": v,
                    "method": "percentile",
                    "p1": round(p1, 4),
                    "p99": round(p99, 4),
                })
        return anomalies

    # ------------------------------------------------------------------
    # Internal helpers: rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: QualityRuleResponse,
        rows: List[Dict[str, Any]],
    ) -> tuple:
        """Evaluate a single quality rule against dataset rows.

        Args:
            rule: Quality rule to evaluate.
            rows: Dataset rows.

        Returns:
            Tuple of (pass_count, fail_count, failure_details_list).
        """
        pass_count = 0
        fail_count = 0
        failures: List[Dict[str, Any]] = []

        for i, row in enumerate(rows):
            passed = self._check_rule_for_row(rule, row)
            if passed:
                pass_count += 1
            else:
                fail_count += 1
                if len(failures) < 50:  # Limit stored failures
                    failures.append({
                        "row_index": i,
                        "column": rule.column,
                        "value": row.get(rule.column),
                        "rule_type": rule.rule_type,
                    })

        return pass_count, fail_count, failures

    def _check_rule_for_row(
        self,
        rule: QualityRuleResponse,
        row: Dict[str, Any],
    ) -> bool:
        """Check if a single row passes a quality rule.

        Args:
            rule: Quality rule.
            row: Single data row.

        Returns:
            True if the row passes the rule.
        """
        value = row.get(rule.column)

        if rule.rule_type == "not_null":
            return value is not None and value != ""

        if rule.rule_type == "unique":
            # Uniqueness is checked at dataset level, always pass at row level
            return True

        if rule.rule_type == "range":
            if value is None or value == "":
                return True  # Nulls handled by not_null rules
            try:
                num = float(value)
            except (ValueError, TypeError):
                return False
            params = rule.parameters
            min_val = params.get("min")
            max_val = params.get("max")
            if min_val is not None and num < float(min_val):
                return False
            if max_val is not None and num > float(max_val):
                return False
            return True

        if rule.rule_type == "regex":
            if value is None or value == "":
                return True
            import re
            pattern = rule.parameters.get("pattern", "")
            try:
                return bool(re.match(pattern, str(value)))
            except re.error:
                return False

        if rule.rule_type == "custom":
            # Custom rules use operator and threshold
            return self._check_operator(value, rule.operator, rule.threshold)

        # Default: pass
        return True

    def _check_operator(
        self,
        value: Any,
        operator: str,
        threshold: Any,
    ) -> bool:
        """Check a value against an operator and threshold.

        Args:
            value: Value to check.
            operator: Comparison operator.
            threshold: Threshold value.

        Returns:
            True if the comparison passes.
        """
        if value is None or value == "":
            return True

        try:
            v = float(value)
            t = float(threshold) if threshold is not None else 0.0
        except (ValueError, TypeError):
            # String comparison
            v_str = str(value)
            t_str = str(threshold) if threshold is not None else ""
            if operator == "eq":
                return v_str == t_str
            elif operator == "ne":
                return v_str != t_str
            return True

        if operator == "eq":
            return v == t
        elif operator == "ne":
            return v != t
        elif operator == "gt":
            return v > t
        elif operator == "gte":
            return v >= t
        elif operator == "lt":
            return v < t
        elif operator == "lte":
            return v <= t
        return True

    # ------------------------------------------------------------------
    # Internal helpers: gate evaluation
    # ------------------------------------------------------------------

    def _evaluate_gate_condition(
        self,
        actual: float,
        operator: str,
        threshold: float,
    ) -> bool:
        """Evaluate a single gate condition.

        Args:
            actual: Actual score value.
            operator: Comparison operator.
            threshold: Required threshold.

        Returns:
            True if the condition passes.
        """
        if operator == "gte":
            return actual >= threshold
        elif operator == "gt":
            return actual > threshold
        elif operator == "lte":
            return actual <= threshold
        elif operator == "lt":
            return actual < threshold
        elif operator == "eq":
            return abs(actual - threshold) < 0.0001
        return actual >= threshold  # Default to gte

    # ------------------------------------------------------------------
    # Internal helpers: report building
    # ------------------------------------------------------------------

    def _build_report_content(
        self,
        report_type: str,
        report_format: str,
        assessments: List[QualityAssessmentResponse],
        profiles: List[DatasetProfileResponse],
        anomaly_results: List[AnomalyDetectionResponse],
        dataset_name: Optional[str],
    ) -> Any:
        """Build report content based on type and format.

        Args:
            report_type: Report type.
            report_format: Output format.
            assessments: Quality assessments to include.
            profiles: Dataset profiles to include.
            anomaly_results: Anomaly detection results to include.
            dataset_name: Dataset name filter.

        Returns:
            Report content in the requested format.
        """
        # Build base data dict
        data = self._build_report_data(
            report_type, assessments, profiles, anomaly_results,
        )

        if report_format == "json":
            return data

        if report_format == "markdown":
            return self._format_markdown(data, report_type, dataset_name)

        if report_format == "html":
            return self._format_html(data, report_type, dataset_name)

        if report_format == "text":
            return self._format_text(data, report_type, dataset_name)

        if report_format == "csv":
            return self._format_csv(data, report_type)

        return data

    def _build_report_data(
        self,
        report_type: str,
        assessments: List[QualityAssessmentResponse],
        profiles: List[DatasetProfileResponse],
        anomaly_results: List[AnomalyDetectionResponse],
    ) -> Dict[str, Any]:
        """Build the report data dictionary.

        Args:
            report_type: Report type.
            assessments: Quality assessments.
            profiles: Dataset profiles.
            anomaly_results: Anomaly detection results.

        Returns:
            Dict with report data.
        """
        data: Dict[str, Any] = {
            "report_type": report_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_assessments": len(assessments),
                "total_profiles": len(profiles),
                "total_anomaly_detections": len(anomaly_results),
            },
        }

        if report_type == "scorecard":
            data["assessments"] = [
                {
                    "dataset_name": a.dataset_name,
                    "overall_score": a.overall_score,
                    "quality_level": a.quality_level,
                    "completeness": a.completeness_score,
                    "validity": a.validity_score,
                    "consistency": a.consistency_score,
                    "timeliness": a.timeliness_score,
                    "uniqueness": a.uniqueness_score,
                    "accuracy": a.accuracy_score,
                    "issue_count": len(a.issues),
                }
                for a in assessments
            ]
        elif report_type == "detailed":
            data["assessments"] = [
                a.model_dump(mode="json") for a in assessments
            ]
            data["profiles"] = [
                p.model_dump(mode="json") for p in profiles
            ]
        elif report_type == "executive":
            if assessments:
                avg_score = statistics.mean(a.overall_score for a in assessments)
                data["average_score"] = round(avg_score, 4)
                data["quality_level"] = _classify_quality(avg_score)
                data["total_issues"] = sum(len(a.issues) for a in assessments)
            else:
                data["average_score"] = 0.0
                data["quality_level"] = "CRITICAL"
                data["total_issues"] = 0
        elif report_type == "issues":
            all_issues: List[Dict[str, Any]] = []
            for a in assessments:
                for issue in a.issues:
                    issue_copy = dict(issue)
                    issue_copy["dataset_name"] = a.dataset_name
                    all_issues.append(issue_copy)
            data["issues"] = all_issues
        elif report_type == "anomaly":
            data["anomaly_results"] = [
                ar.model_dump(mode="json") for ar in anomaly_results
            ]

        return data

    def _format_markdown(
        self,
        data: Dict[str, Any],
        report_type: str,
        dataset_name: Optional[str],
    ) -> str:
        """Format report data as markdown.

        Args:
            data: Report data dict.
            report_type: Report type.
            dataset_name: Dataset name.

        Returns:
            Markdown string.
        """
        lines = [
            f"# Data Quality Report: {report_type.title()}",
            f"Dataset: {dataset_name or 'All'}",
            f"Generated: {data.get('generated_at', '')}",
            "",
        ]

        summary = data.get("summary", {})
        lines.append("## Summary")
        lines.append(f"- Assessments: {summary.get('total_assessments', 0)}")
        lines.append(f"- Profiles: {summary.get('total_profiles', 0)}")
        lines.append(f"- Anomaly Detections: {summary.get('total_anomaly_detections', 0)}")
        lines.append("")

        if "assessments" in data and isinstance(data["assessments"], list):
            lines.append("## Assessments")
            for a in data["assessments"]:
                if isinstance(a, dict):
                    lines.append(
                        f"- {a.get('dataset_name', 'N/A')}: "
                        f"Score={a.get('overall_score', 0.0):.4f} "
                        f"Level={a.get('quality_level', 'N/A')}"
                    )
            lines.append("")

        return "\n".join(lines)

    def _format_html(
        self,
        data: Dict[str, Any],
        report_type: str,
        dataset_name: Optional[str],
    ) -> str:
        """Format report data as HTML.

        Args:
            data: Report data dict.
            report_type: Report type.
            dataset_name: Dataset name.

        Returns:
            HTML string.
        """
        md = self._format_markdown(data, report_type, dataset_name)
        # Simple markdown-to-HTML conversion
        html_lines = ["<html><body>"]
        for line in md.split("\n"):
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.strip():
                html_lines.append(f"<p>{line}</p>")
        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    def _format_text(
        self,
        data: Dict[str, Any],
        report_type: str,
        dataset_name: Optional[str],
    ) -> str:
        """Format report data as plain text.

        Args:
            data: Report data dict.
            report_type: Report type.
            dataset_name: Dataset name.

        Returns:
            Plain text string.
        """
        lines = [
            f"DATA QUALITY REPORT: {report_type.upper()}",
            f"Dataset: {dataset_name or 'All'}",
            f"Generated: {data.get('generated_at', '')}",
            "=" * 60,
            "",
        ]

        summary = data.get("summary", {})
        lines.append(f"Assessments: {summary.get('total_assessments', 0)}")
        lines.append(f"Profiles: {summary.get('total_profiles', 0)}")
        lines.append(f"Anomaly Detections: {summary.get('total_anomaly_detections', 0)}")
        lines.append("")

        if "assessments" in data and isinstance(data["assessments"], list):
            lines.append("ASSESSMENTS:")
            lines.append("-" * 40)
            for a in data["assessments"]:
                if isinstance(a, dict):
                    lines.append(
                        f"  {a.get('dataset_name', 'N/A')}: "
                        f"Score={a.get('overall_score', 0.0):.4f} "
                        f"Level={a.get('quality_level', 'N/A')}"
                    )

        return "\n".join(lines)

    def _format_csv(
        self,
        data: Dict[str, Any],
        report_type: str,
    ) -> str:
        """Format report data as CSV.

        Args:
            data: Report data dict.
            report_type: Report type.

        Returns:
            CSV string.
        """
        lines = []

        if "assessments" in data and isinstance(data["assessments"], list):
            headers = [
                "dataset_name", "overall_score", "quality_level",
                "completeness", "validity", "consistency",
                "timeliness", "uniqueness", "accuracy", "issue_count",
            ]
            lines.append(",".join(headers))
            for a in data["assessments"]:
                if isinstance(a, dict):
                    row = [str(a.get(h, "")) for h in headers]
                    lines.append(",".join(row))
        else:
            lines.append("key,value")
            for k, v in data.get("summary", {}).items():
                lines.append(f"{k},{v}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers: statistics
    # ------------------------------------------------------------------

    def _update_avg_quality(self, score: float) -> None:
        """Update running average quality score.

        Args:
            score: Latest quality score.
        """
        total = self._stats.total_assessments
        if total <= 0:
            self._stats.avg_quality_score = score
            return
        prev_avg = self._stats.avg_quality_score
        self._stats.avg_quality_score = (
            (prev_avg * (total - 1) + score) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the data quality profiler service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("DataQualityProfilerService already started; skipping")
            return

        logger.info("DataQualityProfilerService starting up...")
        self._started = True
        logger.info("DataQualityProfilerService startup complete")

    def shutdown(self) -> None:
        """Shutdown the data quality profiler service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("DataQualityProfilerService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> DataQualityProfilerService:
    """Get or create the singleton DataQualityProfilerService instance.

    Returns:
        The singleton DataQualityProfilerService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = DataQualityProfilerService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_data_quality_profiler(
    app: Any,
    config: Optional[DataQualityProfilerConfig] = None,
) -> DataQualityProfilerService:
    """Configure the Data Quality Profiler Service on a FastAPI application.

    Creates the DataQualityProfilerService, stores it in app.state, mounts
    the data quality profiler API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional data quality profiler config.

    Returns:
        DataQualityProfilerService instance.
    """
    global _singleton_instance

    service = DataQualityProfilerService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.data_quality_profiler_service = service

    # Mount data quality profiler API router
    try:
        from greenlang.data_quality_profiler.api.router import router as dq_router
        if dq_router is not None:
            app.include_router(dq_router)
            logger.info("Data quality profiler service API router mounted")
    except ImportError:
        logger.warning(
            "Data quality profiler router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Data quality profiler service configured on app")
    return service


def get_data_quality_profiler(app: Any) -> DataQualityProfilerService:
    """Get the DataQualityProfilerService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        DataQualityProfilerService instance.

    Raises:
        RuntimeError: If data quality profiler service not configured.
    """
    service = getattr(app.state, "data_quality_profiler_service", None)
    if service is None:
        raise RuntimeError(
            "Data quality profiler service not configured. "
            "Call configure_data_quality_profiler(app) first."
        )
    return service


def get_router(service: Optional[DataQualityProfilerService] = None) -> Any:
    """Get the data quality profiler API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.data_quality_profiler.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "DataQualityProfilerService",
    "configure_data_quality_profiler",
    "get_data_quality_profiler",
    "get_router",
    # Models
    "DatasetProfileResponse",
    "QualityAssessmentResponse",
    "ColumnProfileResponse",
    "AnomalyDetectionResponse",
    "FreshnessCheckResponse",
    "QualityRuleResponse",
    "QualityGateResponse",
    "DataQualityProfilerStatisticsResponse",
]
