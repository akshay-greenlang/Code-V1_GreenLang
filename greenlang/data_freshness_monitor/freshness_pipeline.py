# -*- coding: utf-8 -*-
"""
Freshness Monitor Pipeline Engine - AGENT-DATA-016 Data Freshness Monitor

Engine 7 of 7.  Orchestrates the full data freshness monitoring workflow
by composing the six upstream engines (DatasetRegistryEngine,
SLADefinitionEngine, FreshnessCheckerEngine, StalenessDetectorEngine,
RefreshPredictorEngine, AlertManagerEngine) into a deterministic pipeline.

Pipeline stages:
    1. REGISTER   -- collect all registered datasets (from registry or input)
    2. CHECK      -- run FreshnessCheckerEngine.check_freshness on each dataset
    3. STALENESS  -- run StalenessDetectorEngine.detect_patterns on datasets
    4. PREDICT    -- run RefreshPredictorEngine.predict_next_refresh on datasets
    5. SLA_EVAL   -- run SLADefinitionEngine.evaluate_sla for violations
    6. ALERT      -- run AlertManagerEngine.create_and_send_alert for breaches
    7. REPORT     -- assemble MonitoringRun summary with stats

Zero-Hallucination: All calculations use deterministic Python arithmetic.
Freshness scoring uses time-delta comparisons, SLA evaluation uses
threshold rules, and staleness detection uses statistical pattern
analysis.  No LLM calls for numeric computations.  Every operation is
traced through SHA-256 provenance chains.

Example:
    >>> from greenlang.data_freshness_monitor.freshness_pipeline import (
    ...     FreshnessMonitorPipelineEngine,
    ... )
    >>> engine = FreshnessMonitorPipelineEngine()
    >>> run = engine.run_pipeline(
    ...     datasets=[{"dataset_id": "ds1", "name": "Scope1",
    ...                "last_updated": "2026-02-15T10:00:00Z"}],
    ... )
    >>> assert run.status.value in ("completed", "completed_with_warnings")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for sibling modules (provenance, metrics, config)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.provenance import (
        ProvenanceTracker,
        get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc, assignment]
    get_provenance_tracker = None  # type: ignore[misc, assignment]
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor import metrics as _metrics_mod
    _METRICS_AVAILABLE = True
except ImportError:
    _metrics_mod = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor.config import get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    get_config = None  # type: ignore[misc, assignment]
    _CONFIG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graceful imports for sibling engines
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.dataset_registry import (
        DatasetRegistryEngine,
    )
    _REGISTRY_AVAILABLE = True
except ImportError:
    DatasetRegistryEngine = None  # type: ignore[misc, assignment]
    _REGISTRY_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor.sla_definition import (
        SLADefinitionEngine,
    )
    _SLA_AVAILABLE = True
except ImportError:
    SLADefinitionEngine = None  # type: ignore[misc, assignment]
    _SLA_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor.freshness_checker import (
        FreshnessCheckerEngine,
    )
    _CHECKER_AVAILABLE = True
except ImportError:
    FreshnessCheckerEngine = None  # type: ignore[misc, assignment]
    _CHECKER_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor.staleness_detector import (
        StalenessDetectorEngine,
    )
    _STALENESS_AVAILABLE = True
except ImportError:
    StalenessDetectorEngine = None  # type: ignore[misc, assignment]
    _STALENESS_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor.refresh_predictor import (
        RefreshPredictorEngine,
    )
    _PREDICTOR_AVAILABLE = True
except ImportError:
    RefreshPredictorEngine = None  # type: ignore[misc, assignment]
    _PREDICTOR_AVAILABLE = False

try:
    from greenlang.data_freshness_monitor.alert_manager import (
        AlertManagerEngine,
    )
    _ALERT_AVAILABLE = True
except ImportError:
    AlertManagerEngine = None  # type: ignore[misc, assignment]
    _ALERT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graceful imports for models
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.models import (
        FreshnessAlert,
        FreshnessCheck,
        FreshnessReport,
        MonitoringRun,
        MonitoringStatus,
        RefreshPrediction,
        SLABreach,
        StalenessPattern,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False


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


def _compute_sha256(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Normalizes floats, sorts dictionary keys, and handles nested
    structures for reproducible hashing.

    Args:
        data: Data to hash (dict, list, str, number, or other).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Local metric helper stubs (delegate when metrics module is present)
# ---------------------------------------------------------------------------


def _inc_pipeline_runs(status: str) -> None:
    """Increment the pipeline runs counter.

    Args:
        status: Pipeline run status (completed, failed).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.inc_pipeline_runs(status)
        except AttributeError:
            pass


def _inc_datasets_checked(count: int = 1) -> None:
    """Increment the datasets checked counter.

    Args:
        count: Number of datasets checked.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.inc_datasets_checked(count)
        except AttributeError:
            pass


def _inc_breaches_found(severity: str, count: int = 1) -> None:
    """Increment the SLA breaches found counter.

    Args:
        severity: Breach severity level.
        count: Number of breaches.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.inc_breaches_found(severity, count)
        except AttributeError:
            pass


def _inc_alerts_sent(channel: str, count: int = 1) -> None:
    """Increment the alerts sent counter.

    Args:
        channel: Alert delivery channel.
        count: Number of alerts.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.inc_alerts_sent(channel, count)
        except AttributeError:
            pass


def _observe_pipeline_duration(duration: float) -> None:
    """Observe pipeline processing duration in seconds.

    Args:
        duration: Duration in seconds.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.observe_pipeline_duration(duration)
        except AttributeError:
            pass


def _inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.inc_errors(error_type)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Pipeline stage enumeration
# ---------------------------------------------------------------------------

_STAGES = (
    "register", "check", "staleness",
    "predict", "sla_eval", "alert", "report",
)


# ---------------------------------------------------------------------------
# Fallback data models (used when models.py is not yet available)
# ---------------------------------------------------------------------------

if not _MODELS_AVAILABLE:

    class MonitoringStatus(str, Enum):  # type: ignore[no-redef]
        """Monitoring run status."""
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        COMPLETED_WITH_WARNINGS = "completed_with_warnings"
        FAILED = "failed"

    @dataclass
    class FreshnessCheck:  # type: ignore[no-redef]
        """Result of a single dataset freshness check.

        Attributes:
            check_id: Unique check identifier.
            dataset_id: Dataset that was checked.
            checked_at: Timestamp of the check.
            age_hours: Hours since dataset last updated.
            freshness_tier: Tier classification (excellent/good/fair/poor/stale).
            freshness_score: Numeric score 0.0-1.0.
            status: Check status (fresh/stale/unknown).
            provenance_hash: SHA-256 hash for audit trail.
        """
        check_id: str = ""
        dataset_id: str = ""
        checked_at: str = ""
        age_hours: float = 0.0
        freshness_tier: str = "unknown"
        freshness_score: float = 0.0
        status: str = "unknown"
        provenance_hash: str = ""

    @dataclass
    class StalenessPattern:  # type: ignore[no-redef]
        """Detected staleness pattern for a dataset.

        Attributes:
            pattern_id: Unique pattern identifier.
            dataset_id: Dataset analysed.
            pattern_type: Type of pattern (periodic/degrading/irregular/none).
            avg_refresh_hours: Average hours between refreshes.
            stddev_refresh_hours: Standard deviation of refresh intervals.
            detected_at: Timestamp of detection.
            confidence: Confidence score 0.0-1.0.
        """
        pattern_id: str = ""
        dataset_id: str = ""
        pattern_type: str = "none"
        avg_refresh_hours: float = 0.0
        stddev_refresh_hours: float = 0.0
        detected_at: str = ""
        confidence: float = 0.0

    @dataclass
    class RefreshPrediction:  # type: ignore[no-redef]
        """Predicted next refresh for a dataset.

        Attributes:
            prediction_id: Unique prediction identifier.
            dataset_id: Dataset for which prediction is made.
            predicted_at: Timestamp when prediction was made.
            next_refresh_at: Predicted next refresh time (ISO format).
            confidence: Prediction confidence 0.0-1.0.
            method: Prediction method used.
        """
        prediction_id: str = ""
        dataset_id: str = ""
        predicted_at: str = ""
        next_refresh_at: str = ""
        confidence: float = 0.0
        method: str = "fallback"

    @dataclass
    class SLABreach:  # type: ignore[no-redef]
        """SLA breach record.

        Attributes:
            breach_id: Unique breach identifier.
            dataset_id: Dataset in breach.
            severity: Severity level (warning/critical).
            sla_hours: SLA threshold that was breached (hours).
            actual_hours: Actual age of the dataset (hours).
            breached_at: Timestamp of breach detection.
            resolved: Whether breach has been resolved.
        """
        breach_id: str = ""
        dataset_id: str = ""
        severity: str = "warning"
        sla_hours: float = 0.0
        actual_hours: float = 0.0
        breached_at: str = ""
        resolved: bool = False

    @dataclass
    class FreshnessAlert:  # type: ignore[no-redef]
        """Alert generated for a freshness breach.

        Attributes:
            alert_id: Unique alert identifier.
            breach_id: Associated breach identifier.
            dataset_id: Dataset referenced.
            severity: Alert severity (warning/critical).
            channel: Delivery channel.
            sent_at: Timestamp when alert was sent.
            acknowledged: Whether alert has been acknowledged.
        """
        alert_id: str = ""
        breach_id: str = ""
        dataset_id: str = ""
        severity: str = "warning"
        channel: str = "default"
        sent_at: str = ""
        acknowledged: bool = False

    @dataclass
    class MonitoringRun:  # type: ignore[no-redef]
        """Record of a complete pipeline monitoring run.

        Attributes:
            id: Unique run identifier (MR-{uuid_hex[:12]}).
            started_at: Run start timestamp (ISO format).
            completed_at: Run completion timestamp (ISO format).
            status: Run status.
            datasets_checked: Count of datasets checked.
            breaches_found: Count of SLA breaches detected.
            alerts_sent: Count of alerts dispatched.
            stage_results: Per-stage results summary.
            provenance_hash: SHA-256 hash of the full run results.
            error: Error message if run failed.
        """
        id: str = ""
        started_at: str = ""
        completed_at: str = ""
        status: Any = None  # MonitoringStatus
        datasets_checked: int = 0
        breaches_found: int = 0
        alerts_sent: int = 0
        stage_results: Dict[str, Any] = field(default_factory=dict)
        provenance_hash: str = ""
        error: Optional[str] = None

    @dataclass
    class FreshnessReport:  # type: ignore[no-redef]
        """Freshness monitoring report.

        Attributes:
            report_id: Unique report identifier.
            report_type: Type of report (pipeline/ghg_protocol/csrd_esrs).
            generated_at: Report generation timestamp.
            summary: Summary dict with overall statistics.
            checks: List of FreshnessCheck results included.
            breaches: List of SLABreach records included.
            alerts: List of FreshnessAlert records included.
            compliance_status: Overall compliance status.
            provenance_hash: SHA-256 hash of the report data.
        """
        report_id: str = ""
        report_type: str = "pipeline"
        generated_at: str = ""
        summary: Dict[str, Any] = field(default_factory=dict)
        checks: List[Any] = field(default_factory=list)
        breaches: List[Any] = field(default_factory=list)
        alerts: List[Any] = field(default_factory=list)
        compliance_status: str = "unknown"
        provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Local data models
# ---------------------------------------------------------------------------


@dataclass
class PipelineStageResult:
    """Result of a single pipeline stage execution.

    Attributes:
        stage: Stage name (register, check, staleness, etc.).
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
class PipelineStatistics:
    """Aggregated pipeline execution statistics.

    Attributes:
        total_runs: Total pipeline executions.
        total_datasets_checked: Total datasets checked across all runs.
        total_breaches_found: Total SLA breaches detected across runs.
        total_alerts_sent: Total alerts dispatched across runs.
        total_predictions: Total refresh predictions generated.
        total_staleness_patterns: Total staleness patterns detected.
        by_status: Run counts per completion status.
        by_severity: Breach counts per severity level.
        by_freshness_tier: Dataset counts per freshness tier.
        avg_freshness_score: Running average freshness score.
    """

    total_runs: int = 0
    total_datasets_checked: int = 0
    total_breaches_found: int = 0
    total_alerts_sent: int = 0
    total_predictions: int = 0
    total_staleness_patterns: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_freshness_tier: Dict[str, int] = field(default_factory=dict)
    avg_freshness_score: float = 0.0


# ============================================================================
# FreshnessMonitorPipelineEngine
# ============================================================================


class FreshnessMonitorPipelineEngine:
    """Orchestrates the full data freshness monitoring pipeline.

    Composes six upstream engines into a seven-stage sequential pipeline:
    register -> check -> staleness -> predict -> sla_eval -> alert -> report.

    All decisions are deterministic. Freshness scoring uses time-delta
    comparisons against configurable tier thresholds. SLA evaluation
    checks dataset age against warning/critical thresholds. Staleness
    detection analyses refresh history for recurring patterns.

    Each sub-engine parameter is optional. When None, the pipeline
    creates its own fallback instance. Only FreshnessCheckerEngine
    accepts a config parameter; all other engines use no-arg constructors.

    Attributes:
        _dataset_registry: Dataset registration engine instance.
        _sla_definition: SLA definition and evaluation engine instance.
        _freshness_checker: Freshness checking engine instance.
        _staleness_detector: Staleness pattern detection engine instance.
        _refresh_predictor: Refresh prediction engine instance.
        _alert_manager: Alert generation and dispatch engine instance.
        _config: Data freshness monitor configuration.
        _provenance: SHA-256 provenance tracker.
        _statistics: Running pipeline statistics.
        _pipeline_history: List of completed MonitoringRun records.
        _run_count: Total number of pipeline runs executed.
        _lock: Thread-safety lock for shared state.

    Example:
        >>> engine = FreshnessMonitorPipelineEngine()
        >>> run = engine.run_pipeline(
        ...     datasets=[{"dataset_id": "ds1", "name": "Emissions",
        ...                "last_updated": "2026-02-15T10:00:00Z"}],
        ... )
        >>> assert run.status in (
        ...     MonitoringStatus.COMPLETED,
        ...     MonitoringStatus.COMPLETED_WITH_WARNINGS,
        ... )
    """

    def __init__(
        self,
        dataset_registry: Optional[Any] = None,
        sla_definition: Optional[Any] = None,
        freshness_checker: Optional[Any] = None,
        staleness_detector: Optional[Any] = None,
        refresh_predictor: Optional[Any] = None,
        alert_manager: Optional[Any] = None,
    ) -> None:
        """Initialize FreshnessMonitorPipelineEngine with all sub-engines.

        Each sub-engine parameter is optional. If None, the pipeline
        attempts to import and instantiate the engine from its sibling
        module. Only FreshnessCheckerEngine receives a config parameter.

        Args:
            dataset_registry: Optional DatasetRegistryEngine instance.
            sla_definition: Optional SLADefinitionEngine instance.
            freshness_checker: Optional FreshnessCheckerEngine instance.
            staleness_detector: Optional StalenessDetectorEngine instance.
            refresh_predictor: Optional RefreshPredictorEngine instance.
            alert_manager: Optional AlertManagerEngine instance.
        """
        # Load configuration
        self._config: Optional[Any] = None
        if _CONFIG_AVAILABLE and get_config is not None:
            try:
                self._config = get_config()
            except Exception:
                pass

        # Initialize provenance tracker
        if _PROVENANCE_AVAILABLE and get_provenance_tracker is not None:
            self._provenance: Any = get_provenance_tracker()
        else:
            self._provenance = _FallbackProvenanceTracker()

        self._statistics = PipelineStatistics()
        self._pipeline_history: List[MonitoringRun] = []
        self._run_count: int = 0
        self._lock = threading.Lock()

        # Initialize sub-engines with fallback creation
        self._dataset_registry = self._init_registry(dataset_registry)
        self._sla_definition = self._init_sla(sla_definition)
        self._freshness_checker = self._init_checker(freshness_checker)
        self._staleness_detector = self._init_staleness(staleness_detector)
        self._refresh_predictor = self._init_predictor(refresh_predictor)
        self._alert_manager = self._init_alert(alert_manager)

        available_engines = sum([
            self._dataset_registry is not None,
            self._sla_definition is not None,
            self._freshness_checker is not None,
            self._staleness_detector is not None,
            self._refresh_predictor is not None,
            self._alert_manager is not None,
        ])
        logger.info(
            "FreshnessMonitorPipelineEngine initialized "
            "(%d/6 sub-engines available)",
            available_engines,
        )

    # ------------------------------------------------------------------
    # Sub-engine initialization helpers
    # ------------------------------------------------------------------

    def _init_registry(self, provided: Optional[Any]) -> Optional[Any]:
        """Initialize the dataset registry engine.

        Args:
            provided: Externally provided engine instance, or None.

        Returns:
            Engine instance or None if unavailable.
        """
        if provided is not None:
            return provided
        if _REGISTRY_AVAILABLE and DatasetRegistryEngine is not None:
            try:
                return DatasetRegistryEngine()
            except Exception as exc:
                logger.warning(
                    "Failed to create DatasetRegistryEngine: %s", exc,
                )
        return None

    def _init_sla(self, provided: Optional[Any]) -> Optional[Any]:
        """Initialize the SLA definition engine.

        Args:
            provided: Externally provided engine instance, or None.

        Returns:
            Engine instance or None if unavailable.
        """
        if provided is not None:
            return provided
        if _SLA_AVAILABLE and SLADefinitionEngine is not None:
            try:
                return SLADefinitionEngine()
            except Exception as exc:
                logger.warning(
                    "Failed to create SLADefinitionEngine: %s", exc,
                )
        return None

    def _init_checker(self, provided: Optional[Any]) -> Optional[Any]:
        """Initialize the freshness checker engine.

        FreshnessCheckerEngine is the only sub-engine that accepts a
        config parameter.

        Args:
            provided: Externally provided engine instance, or None.

        Returns:
            Engine instance or None if unavailable.
        """
        if provided is not None:
            return provided
        if _CHECKER_AVAILABLE and FreshnessCheckerEngine is not None:
            try:
                return FreshnessCheckerEngine(config=self._config)
            except Exception as exc:
                logger.warning(
                    "Failed to create FreshnessCheckerEngine: %s", exc,
                )
        return None

    def _init_staleness(self, provided: Optional[Any]) -> Optional[Any]:
        """Initialize the staleness detector engine.

        Args:
            provided: Externally provided engine instance, or None.

        Returns:
            Engine instance or None if unavailable.
        """
        if provided is not None:
            return provided
        if _STALENESS_AVAILABLE and StalenessDetectorEngine is not None:
            try:
                return StalenessDetectorEngine()
            except Exception as exc:
                logger.warning(
                    "Failed to create StalenessDetectorEngine: %s", exc,
                )
        return None

    def _init_predictor(self, provided: Optional[Any]) -> Optional[Any]:
        """Initialize the refresh predictor engine.

        Args:
            provided: Externally provided engine instance, or None.

        Returns:
            Engine instance or None if unavailable.
        """
        if provided is not None:
            return provided
        if _PREDICTOR_AVAILABLE and RefreshPredictorEngine is not None:
            try:
                return RefreshPredictorEngine()
            except Exception as exc:
                logger.warning(
                    "Failed to create RefreshPredictorEngine: %s", exc,
                )
        return None

    def _init_alert(self, provided: Optional[Any]) -> Optional[Any]:
        """Initialize the alert manager engine.

        Args:
            provided: Externally provided engine instance, or None.

        Returns:
            Engine instance or None if unavailable.
        """
        if provided is not None:
            return provided
        if _ALERT_AVAILABLE and AlertManagerEngine is not None:
            try:
                return AlertManagerEngine()
            except Exception as exc:
                logger.warning(
                    "Failed to create AlertManagerEngine: %s", exc,
                )
        return None

    # ==================================================================
    # 1. run_pipeline - Full pipeline orchestration
    # ==================================================================

    def run_pipeline(
        self,
        datasets: Optional[List[dict]] = None,
    ) -> MonitoringRun:
        """Execute the full data freshness monitoring pipeline.

        Runs all seven pipeline stages in order and returns a
        MonitoringRun record with comprehensive results including
        freshness checks, SLA breaches, alerts, and provenance.

        When ``datasets`` is None, the engine attempts to retrieve
        all registered datasets from the dataset registry engine.
        When provided, the list of dataset dicts is used directly.

        Each dataset dict should contain at minimum:
            - ``dataset_id`` (str): Unique dataset identifier.
            - ``name`` (str): Human-readable dataset name.
            - ``last_updated`` (str): ISO 8601 timestamp of last update.

        Args:
            datasets: Optional list of dataset dicts to monitor. When
                None, queries the DatasetRegistryEngine for all
                registered datasets.

        Returns:
            MonitoringRun with id, status, timestamps, counts, stage
            results, and provenance hash.
        """
        run_id = f"MR-{uuid4().hex[:12]}"
        started_at = _utcnow()
        pipeline_start = time.time()
        stage_results: Dict[str, Dict[str, Any]] = {}

        logger.info("Pipeline %s starting", run_id)

        # -- Stage 1: REGISTER --
        if datasets is None:
            datasets = self._stage_register()
        if not datasets:
            datasets = []

        logger.info(
            "Pipeline %s: %d datasets to monitor",
            run_id, len(datasets),
        )

        checks: List[FreshnessCheck] = []
        staleness_patterns: List[StalenessPattern] = []
        predictions: List[RefreshPrediction] = []
        breaches: List[SLABreach] = []
        alerts: List[FreshnessAlert] = []

        try:
            # -- Stage 2: CHECK FRESHNESS --
            stage_t0 = time.time()
            checks = self.run_check_stage(datasets)
            stage_results["check"] = {
                "status": "completed",
                "records_processed": len(checks),
                "duration_ms": (time.time() - stage_t0) * 1000.0,
            }
            _inc_datasets_checked(len(checks))

            # -- Stage 3: DETECT STALENESS --
            stage_t0 = time.time()
            refresh_histories = self._build_refresh_histories(datasets)
            staleness_patterns = self.run_staleness_stage(
                datasets, refresh_histories,
            )
            stage_results["staleness"] = {
                "status": "completed",
                "records_processed": len(staleness_patterns),
                "duration_ms": (time.time() - stage_t0) * 1000.0,
            }

            # -- Stage 4: PREDICT REFRESHES --
            stage_t0 = time.time()
            predictions = self.run_prediction_stage(
                datasets, refresh_histories,
            )
            stage_results["predict"] = {
                "status": "completed",
                "records_processed": len(predictions),
                "duration_ms": (time.time() - stage_t0) * 1000.0,
            }

            # -- Stage 5: EVALUATE SLAs --
            stage_t0 = time.time()
            sla_map = self._build_sla_map(datasets)
            breaches = self.run_sla_evaluation_stage(checks, sla_map)
            stage_results["sla_eval"] = {
                "status": "completed",
                "records_processed": len(breaches),
                "duration_ms": (time.time() - stage_t0) * 1000.0,
            }
            for breach in breaches:
                _inc_breaches_found(
                    getattr(breach, "severity", "warning"),
                )

            # -- Stage 6: GENERATE ALERTS --
            stage_t0 = time.time()
            escalation_policies = self._build_escalation_policies(datasets)
            alerts = self.run_alert_stage(breaches, escalation_policies)
            stage_results["alert"] = {
                "status": "completed",
                "records_processed": len(alerts),
                "duration_ms": (time.time() - stage_t0) * 1000.0,
            }
            for alert in alerts:
                _inc_alerts_sent(getattr(alert, "channel", "default"))

            # -- Stage 7: PRODUCE REPORT --
            status = self._determine_run_status(checks, breaches)

        except Exception as exc:
            logger.error(
                "Pipeline %s failed: %s",
                run_id, str(exc), exc_info=True,
            )
            status = MonitoringStatus.FAILED
            stage_results["error"] = {
                "status": "failed",
                "error": str(exc),
                "duration_ms": 0.0,
            }
            _inc_errors("pipeline")

        # Assemble MonitoringRun
        completed_at = _utcnow()
        elapsed = time.time() - pipeline_start

        provenance_hash = self._compute_provenance(
            "pipeline_complete",
            {
                "run_id": run_id,
                "datasets_checked": len(checks),
            },
            {
                "breaches_found": len(breaches),
                "alerts_sent": len(alerts),
                "status": status.value if hasattr(status, "value") else str(status),
            },
        )

        run = MonitoringRun(
            id=run_id,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            status=status,
            datasets_checked=len(checks),
            breaches_found=len(breaches),
            alerts_sent=len(alerts),
            stage_results=stage_results,
            provenance_hash=provenance_hash,
            error=None if status != MonitoringStatus.FAILED else stage_results.get(
                "error", {},
            ).get("error"),
        )

        # Finalize
        self._finalize_run(run, elapsed, checks, breaches, alerts)
        _observe_pipeline_duration(elapsed)

        status_val = status.value if hasattr(status, "value") else str(status)
        _inc_pipeline_runs(status_val)

        logger.info(
            "Pipeline %s completed: status=%s, %d checked, "
            "%d breaches, %d alerts, %.1fms",
            run_id, status_val,
            len(checks), len(breaches), len(alerts),
            elapsed * 1000.0,
        )

        return run

    # ==================================================================
    # 2. run_check_stage
    # ==================================================================

    def run_check_stage(
        self,
        datasets: List[dict],
    ) -> List[FreshnessCheck]:
        """Run freshness checks on all provided datasets.

        Delegates to FreshnessCheckerEngine.check_freshness when
        available, otherwise uses a fallback implementation that
        computes age from the ``last_updated`` field and classifies
        freshness based on configurable tier thresholds.

        Args:
            datasets: List of dataset dicts with at least
                ``dataset_id`` and ``last_updated`` fields.

        Returns:
            List of FreshnessCheck results, one per dataset.
        """
        checks: List[FreshnessCheck] = []

        for ds in datasets:
            dataset_id = ds.get("dataset_id", str(uuid4()))
            try:
                if self._freshness_checker is not None:
                    result = self._freshness_checker.check_freshness(ds)
                    if result is not None:
                        checks.append(result)
                        continue
            except Exception as exc:
                logger.warning(
                    "FreshnessCheckerEngine.check_freshness failed "
                    "for '%s': %s", dataset_id, exc,
                )

            # Fallback implementation
            check = self._fallback_check_freshness(ds)
            checks.append(check)

        return checks

    # ==================================================================
    # 3. run_staleness_stage
    # ==================================================================

    def run_staleness_stage(
        self,
        datasets: List[dict],
        refresh_histories: Dict[str, List[datetime]],
    ) -> List[StalenessPattern]:
        """Detect staleness patterns across datasets with refresh history.

        Delegates to StalenessDetectorEngine.detect_patterns when
        available, otherwise uses a fallback that computes basic
        refresh interval statistics.

        Args:
            datasets: List of dataset dicts.
            refresh_histories: Dict mapping dataset_id to a list of
                datetime timestamps representing historical refresh events.

        Returns:
            List of StalenessPattern results for datasets that have
            sufficient refresh history.
        """
        patterns: List[StalenessPattern] = []

        for ds in datasets:
            dataset_id = ds.get("dataset_id", "")
            history = refresh_histories.get(dataset_id, [])
            if len(history) < 2:
                continue

            try:
                if self._staleness_detector is not None:
                    result = self._staleness_detector.detect_patterns(
                        ds, history,
                    )
                    if result is not None:
                        patterns.append(result)
                        continue
            except Exception as exc:
                logger.warning(
                    "StalenessDetectorEngine.detect_patterns failed "
                    "for '%s': %s", dataset_id, exc,
                )

            # Fallback implementation
            pattern = self._fallback_detect_staleness(ds, history)
            patterns.append(pattern)

        return patterns

    # ==================================================================
    # 4. run_prediction_stage
    # ==================================================================

    def run_prediction_stage(
        self,
        datasets: List[dict],
        refresh_histories: Dict[str, List[datetime]],
    ) -> List[RefreshPrediction]:
        """Predict next refresh for datasets with sufficient history.

        Delegates to RefreshPredictorEngine.predict_next_refresh when
        available, otherwise uses a fallback that estimates based on
        the mean refresh interval.

        Args:
            datasets: List of dataset dicts.
            refresh_histories: Dict mapping dataset_id to datetime lists.

        Returns:
            List of RefreshPrediction results for eligible datasets.
        """
        predictions: List[RefreshPrediction] = []
        min_samples = 3

        for ds in datasets:
            dataset_id = ds.get("dataset_id", "")
            history = refresh_histories.get(dataset_id, [])
            if len(history) < min_samples:
                continue

            try:
                if self._refresh_predictor is not None:
                    result = self._refresh_predictor.predict_next_refresh(
                        ds, history,
                    )
                    if result is not None:
                        predictions.append(result)
                        continue
            except Exception as exc:
                logger.warning(
                    "RefreshPredictorEngine.predict_next_refresh failed "
                    "for '%s': %s", dataset_id, exc,
                )

            # Fallback implementation
            prediction = self._fallback_predict_refresh(ds, history)
            predictions.append(prediction)

        return predictions

    # ==================================================================
    # 5. run_sla_evaluation_stage
    # ==================================================================

    def run_sla_evaluation_stage(
        self,
        checks: List[FreshnessCheck],
        sla_map: Dict[str, dict],
    ) -> List[SLABreach]:
        """Evaluate SLA compliance for all freshness check results.

        For each check, looks up the dataset's SLA from sla_map and
        evaluates whether the dataset age exceeds warning or critical
        thresholds. Delegates to SLADefinitionEngine.evaluate_sla when
        available, otherwise uses fallback threshold comparison.

        When breaches are detected and the alert manager is available,
        records the breach for subsequent alerting.

        Args:
            checks: List of FreshnessCheck results from the check stage.
            sla_map: Dict mapping dataset_id to SLA config dicts with
                ``warning_hours`` and ``critical_hours`` keys.

        Returns:
            List of SLABreach records for all violated thresholds.
        """
        breaches: List[SLABreach] = []

        for check in checks:
            dataset_id = getattr(check, "dataset_id", "")
            age_hours = getattr(check, "age_hours", 0.0)
            sla = sla_map.get(dataset_id, self._default_sla())

            try:
                if self._sla_definition is not None:
                    result = self._sla_definition.evaluate_sla(check, sla)
                    if result is not None:
                        if isinstance(result, list):
                            breaches.extend(result)
                        else:
                            breaches.append(result)
                        # Record breaches with alert manager
                        self._record_breaches_with_alert_manager(
                            result if isinstance(result, list) else [result],
                        )
                        continue
            except Exception as exc:
                logger.warning(
                    "SLADefinitionEngine.evaluate_sla failed "
                    "for '%s': %s", dataset_id, exc,
                )

            # Fallback implementation
            breach_list = self._fallback_evaluate_sla(
                check, sla, dataset_id, age_hours,
            )
            breaches.extend(breach_list)
            self._record_breaches_with_alert_manager(breach_list)

        return breaches

    # ==================================================================
    # 6. run_alert_stage
    # ==================================================================

    def run_alert_stage(
        self,
        breaches: List[SLABreach],
        escalation_policies: Dict[str, dict],
    ) -> List[FreshnessAlert]:
        """Generate and send alerts for SLA breaches.

        Delegates to AlertManagerEngine.create_and_send_alert when
        available, otherwise creates fallback alert records.

        Args:
            breaches: List of SLABreach records from the SLA stage.
            escalation_policies: Dict mapping dataset_id to escalation
                policy dicts with channel/interval configurations.

        Returns:
            List of FreshnessAlert records for dispatched alerts.
        """
        alerts: List[FreshnessAlert] = []

        for breach in breaches:
            dataset_id = getattr(breach, "dataset_id", "")
            breach_id = getattr(breach, "breach_id", "")
            severity = getattr(breach, "severity", "warning")
            policy = escalation_policies.get(dataset_id, {})

            try:
                if self._alert_manager is not None:
                    result = self._alert_manager.create_and_send_alert(
                        breach, policy,
                    )
                    if result is not None:
                        alerts.append(result)
                        continue
            except Exception as exc:
                logger.warning(
                    "AlertManagerEngine.create_and_send_alert failed "
                    "for breach '%s': %s", breach_id, exc,
                )

            # Fallback implementation
            alert = self._fallback_create_alert(
                breach, dataset_id, breach_id, severity,
                policy,
            )
            alerts.append(alert)

        return alerts

    # ==================================================================
    # 7. run_report_stage
    # ==================================================================

    def run_report_stage(
        self,
        run: MonitoringRun,
        checks: List[FreshnessCheck],
        breaches: List[SLABreach],
        alerts: List[FreshnessAlert],
    ) -> FreshnessReport:
        """Assemble a comprehensive freshness monitoring report.

        Combines the monitoring run metadata with detailed check,
        breach, and alert records into a FreshnessReport with summary
        statistics and provenance.

        Args:
            run: The MonitoringRun record for the current pipeline run.
            checks: List of FreshnessCheck results.
            breaches: List of SLABreach records.
            alerts: List of FreshnessAlert records.

        Returns:
            FreshnessReport with full details and provenance hash.
        """
        now = _utcnow()

        # Compute tier distribution
        tier_counts: Dict[str, int] = {}
        scores: List[float] = []
        for check in checks:
            tier = getattr(check, "freshness_tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            score = getattr(check, "freshness_score", 0.0)
            scores.append(score)

        summary = {
            "run_id": getattr(run, "id", ""),
            "datasets_checked": len(checks),
            "breaches_found": len(breaches),
            "alerts_sent": len(alerts),
            "avg_freshness_score": _safe_mean(scores),
            "tier_distribution": tier_counts,
            "warning_breaches": sum(
                1 for b in breaches
                if getattr(b, "severity", "") == "warning"
            ),
            "critical_breaches": sum(
                1 for b in breaches
                if getattr(b, "severity", "") == "critical"
            ),
        }

        provenance_hash = self._compute_provenance(
            "generate_report",
            {"run_id": getattr(run, "id", ""), "checks": len(checks)},
            summary,
        )

        compliance = "compliant"
        if any(
            getattr(b, "severity", "") == "critical"
            for b in breaches
        ):
            compliance = "non_compliant"
        elif breaches:
            compliance = "at_risk"

        return FreshnessReport(
            report_id=f"FR-{uuid4().hex[:12]}",
            report_type="pipeline",
            generated_at=now.isoformat(),
            summary=summary,
            checks=list(checks),
            breaches=list(breaches),
            alerts=list(alerts),
            compliance_status=compliance,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # 8. generate_compliance_report
    # ==================================================================

    def generate_compliance_report(
        self,
        checks: List[FreshnessCheck],
        breaches: List[SLABreach],
        report_type: str = "general",
    ) -> FreshnessReport:
        """Generate a compliance-focused freshness report.

        Supports multiple report types including general, GHG Protocol,
        and CSRD/ESRS. Adds compliance-specific metadata to the report
        summary.

        Args:
            checks: List of FreshnessCheck results to include.
            breaches: List of SLABreach records to include.
            report_type: Report type string. Supported values:
                ``general``, ``ghg_protocol``, ``csrd_esrs``.

        Returns:
            FreshnessReport with compliance-specific summary.
        """
        now = _utcnow()

        scores = [
            getattr(c, "freshness_score", 0.0) for c in checks
        ]
        stale_count = sum(
            1 for c in checks
            if getattr(c, "status", "") == "stale"
        )
        fresh_count = sum(
            1 for c in checks
            if getattr(c, "status", "") == "fresh"
        )

        compliance_status = "compliant"
        if any(
            getattr(b, "severity", "") == "critical"
            for b in breaches
        ):
            compliance_status = "non_compliant"
        elif breaches:
            compliance_status = "at_risk"

        summary = {
            "report_type": report_type,
            "total_datasets": len(checks),
            "fresh_datasets": fresh_count,
            "stale_datasets": stale_count,
            "avg_freshness_score": _safe_mean(scores),
            "total_breaches": len(breaches),
            "warning_breaches": sum(
                1 for b in breaches
                if getattr(b, "severity", "") == "warning"
            ),
            "critical_breaches": sum(
                1 for b in breaches
                if getattr(b, "severity", "") == "critical"
            ),
            "compliance_status": compliance_status,
            "generated_at": now.isoformat(),
        }

        provenance_hash = self._compute_provenance(
            f"compliance_report_{report_type}",
            {"datasets": len(checks), "breaches": len(breaches)},
            summary,
        )

        return FreshnessReport(
            report_id=f"FR-{uuid4().hex[:12]}",
            report_type=report_type,
            generated_at=now.isoformat(),
            summary=summary,
            checks=list(checks),
            breaches=list(breaches),
            alerts=[],
            compliance_status=compliance_status,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # 9. generate_ghg_protocol_report
    # ==================================================================

    def generate_ghg_protocol_report(
        self,
        checks: List[FreshnessCheck],
        breaches: List[SLABreach],
    ) -> dict:
        """Generate a GHG Protocol-focused data freshness report.

        Evaluates data freshness against GHG Protocol data quality
        requirements. The GHG Protocol Corporate Standard requires
        that activity data be "the most recent and representative
        data available." This report assesses whether monitored
        datasets meet that standard.

        Args:
            checks: List of FreshnessCheck results.
            breaches: List of SLABreach records.

        Returns:
            Dict with GHG Protocol-specific freshness assessment
            including scope coverage, data quality tier, and
            recommendations.
        """
        now = _utcnow()

        scores = [
            getattr(c, "freshness_score", 0.0) for c in checks
        ]
        avg_score = _safe_mean(scores)

        # GHG Protocol data quality tiers
        if avg_score >= 0.8:
            quality_tier = "high"
        elif avg_score >= 0.5:
            quality_tier = "medium"
        else:
            quality_tier = "low"

        critical_count = sum(
            1 for b in breaches
            if getattr(b, "severity", "") == "critical"
        )

        recommendations: List[str] = []
        if critical_count > 0:
            recommendations.append(
                "Critical data freshness breaches detected. "
                "Update emission factor datasets immediately."
            )
        if avg_score < 0.5:
            recommendations.append(
                "Overall data freshness is below acceptable levels. "
                "Review data collection schedules."
            )
        if not checks:
            recommendations.append(
                "No datasets monitored. Register emission data "
                "sources for freshness tracking."
            )

        report = {
            "framework": "ghg_protocol",
            "version": "corporate_standard_2024",
            "generated_at": now.isoformat(),
            "total_datasets": len(checks),
            "avg_freshness_score": avg_score,
            "data_quality_tier": quality_tier,
            "total_breaches": len(breaches),
            "critical_breaches": critical_count,
            "compliant": critical_count == 0 and avg_score >= 0.5,
            "recommendations": recommendations,
            "provenance_hash": self._compute_provenance(
                "ghg_protocol_report",
                {"datasets": len(checks)},
                {"quality_tier": quality_tier, "avg_score": avg_score},
            ),
        }

        logger.info(
            "GHG Protocol report: quality=%s, score=%.2f, "
            "compliant=%s",
            quality_tier, avg_score, report["compliant"],
        )

        return report

    # ==================================================================
    # 10. generate_csrd_esrs_report
    # ==================================================================

    def generate_csrd_esrs_report(
        self,
        checks: List[FreshnessCheck],
        breaches: List[SLABreach],
    ) -> dict:
        """Generate a CSRD/ESRS-focused data freshness report.

        Evaluates data freshness against European Sustainability
        Reporting Standards (ESRS) requirements under the Corporate
        Sustainability Reporting Directive (CSRD). ESRS requires
        entities to report on the timeliness and reliability of
        environmental data.

        Args:
            checks: List of FreshnessCheck results.
            breaches: List of SLABreach records.

        Returns:
            Dict with CSRD/ESRS-specific freshness assessment
            including ESRS disclosure references, data timeliness
            scores, and audit readiness status.
        """
        now = _utcnow()

        scores = [
            getattr(c, "freshness_score", 0.0) for c in checks
        ]
        avg_score = _safe_mean(scores)

        stale_datasets = [
            getattr(c, "dataset_id", "")
            for c in checks
            if getattr(c, "status", "") == "stale"
        ]

        critical_count = sum(
            1 for b in breaches
            if getattr(b, "severity", "") == "critical"
        )

        # ESRS audit readiness
        if avg_score >= 0.7 and critical_count == 0:
            audit_readiness = "ready"
        elif avg_score >= 0.4:
            audit_readiness = "partial"
        else:
            audit_readiness = "not_ready"

        findings: List[str] = []
        if stale_datasets:
            findings.append(
                f"{len(stale_datasets)} dataset(s) exceed acceptable "
                f"timeliness thresholds: {', '.join(stale_datasets[:5])}"
            )
        if critical_count > 0:
            findings.append(
                f"{critical_count} critical SLA breach(es) require "
                "immediate remediation for ESRS compliance."
            )

        report = {
            "framework": "csrd_esrs",
            "version": "ESRS_2024",
            "generated_at": now.isoformat(),
            "esrs_references": [
                "ESRS E1 - Climate Change",
                "ESRS 2 - General Disclosures (BP-2, DC-Q)",
            ],
            "total_datasets": len(checks),
            "avg_freshness_score": avg_score,
            "stale_datasets": stale_datasets,
            "total_breaches": len(breaches),
            "critical_breaches": critical_count,
            "audit_readiness": audit_readiness,
            "findings": findings,
            "compliant": audit_readiness == "ready",
            "provenance_hash": self._compute_provenance(
                "csrd_esrs_report",
                {"datasets": len(checks)},
                {"audit_readiness": audit_readiness, "avg_score": avg_score},
            ),
        }

        logger.info(
            "CSRD/ESRS report: audit_readiness=%s, score=%.2f, "
            "compliant=%s",
            audit_readiness, avg_score, report["compliant"],
        )

        return report

    # ==================================================================
    # 11. get_pipeline_history
    # ==================================================================

    def get_pipeline_history(self) -> List[MonitoringRun]:
        """Return the history of all pipeline monitoring runs.

        Returns a copy of the internal pipeline history list, ordered
        from oldest to newest.

        Returns:
            List of MonitoringRun records from all completed runs.
        """
        with self._lock:
            return list(self._pipeline_history)

    # ==================================================================
    # 12. get_statistics
    # ==================================================================

    def get_statistics(self) -> dict:
        """Return aggregated pipeline execution statistics.

        Returns:
            Dict with total_runs, dataset/breach/alert counts,
            breakdowns by status/severity/tier, engine availability,
            and provenance chain length.
        """
        with self._lock:
            return {
                "total_runs": self._statistics.total_runs,
                "total_datasets_checked": (
                    self._statistics.total_datasets_checked
                ),
                "total_breaches_found": (
                    self._statistics.total_breaches_found
                ),
                "total_alerts_sent": self._statistics.total_alerts_sent,
                "total_predictions": self._statistics.total_predictions,
                "total_staleness_patterns": (
                    self._statistics.total_staleness_patterns
                ),
                "avg_freshness_score": (
                    self._statistics.avg_freshness_score
                ),
                "by_status": dict(self._statistics.by_status),
                "by_severity": dict(self._statistics.by_severity),
                "by_freshness_tier": dict(
                    self._statistics.by_freshness_tier
                ),
                "engine_availability": {
                    "dataset_registry": self._dataset_registry is not None,
                    "sla_definition": self._sla_definition is not None,
                    "freshness_checker": self._freshness_checker is not None,
                    "staleness_detector": (
                        self._staleness_detector is not None
                    ),
                    "refresh_predictor": (
                        self._refresh_predictor is not None
                    ),
                    "alert_manager": self._alert_manager is not None,
                },
                "provenance_chain_length": (
                    self._provenance.get_chain_length()
                    if hasattr(self._provenance, "get_chain_length")
                    else 0
                ),
                "pipeline_history_length": len(self._pipeline_history),
            }

    # ==================================================================
    # 13. reset
    # ==================================================================

    def reset(self) -> None:
        """Reset the pipeline engine to its initial state.

        Clears all pipeline history, resets statistics counters, and
        resets the provenance tracker. Sub-engine instances are
        preserved.
        """
        with self._lock:
            self._pipeline_history.clear()
            self._run_count = 0
            self._statistics = PipelineStatistics()

        if hasattr(self._provenance, "reset"):
            self._provenance.reset()

        logger.info("FreshnessMonitorPipelineEngine reset to initial state")

    # ==================================================================
    # Private: Stage 1 - REGISTER (collect datasets)
    # ==================================================================

    def _stage_register(self) -> List[dict]:
        """Collect all registered datasets from the registry engine.

        When the DatasetRegistryEngine is available, queries it for
        all active datasets. Otherwise returns an empty list.

        Returns:
            List of dataset dicts from the registry.
        """
        if self._dataset_registry is not None:
            try:
                result = self._dataset_registry.list_datasets()
                if isinstance(result, list):
                    return result
                if hasattr(result, "datasets"):
                    return result.datasets
            except Exception as exc:
                logger.warning(
                    "DatasetRegistryEngine.list_datasets failed: %s", exc,
                )
        return []

    # ==================================================================
    # Private: Fallback implementations
    # ==================================================================

    def _fallback_check_freshness(
        self,
        dataset: dict,
    ) -> FreshnessCheck:
        """Fallback freshness check using time-delta computation.

        Computes dataset age from ``last_updated`` and classifies
        into freshness tiers based on config thresholds or defaults.

        Args:
            dataset: Dataset dict with ``dataset_id`` and
                ``last_updated`` fields.

        Returns:
            FreshnessCheck with computed age, tier, and score.
        """
        dataset_id = dataset.get("dataset_id", str(uuid4()))
        last_updated_str = dataset.get("last_updated", "")
        now = _utcnow()

        age_hours = self._compute_age_hours(last_updated_str, now)
        tier = self._classify_freshness_tier(age_hours)
        score = self._compute_freshness_score(age_hours)
        status = "fresh" if tier in ("excellent", "good", "fair") else "stale"

        provenance_hash = self._compute_provenance(
            "check_freshness",
            {"dataset_id": dataset_id, "last_updated": last_updated_str},
            {"age_hours": age_hours, "tier": tier, "score": score},
        )

        return FreshnessCheck(
            check_id=f"FC-{uuid4().hex[:12]}",
            dataset_id=dataset_id,
            checked_at=now.isoformat(),
            age_hours=age_hours,
            freshness_tier=tier,
            freshness_score=score,
            status=status,
            provenance_hash=provenance_hash,
        )

    def _fallback_detect_staleness(
        self,
        dataset: dict,
        history: List[datetime],
    ) -> StalenessPattern:
        """Fallback staleness detection using refresh interval stats.

        Computes mean and standard deviation of refresh intervals.
        Classifies the pattern type based on coefficient of variation.

        Args:
            dataset: Dataset dict.
            history: Sorted list of refresh datetimes.

        Returns:
            StalenessPattern with computed statistics.
        """
        dataset_id = dataset.get("dataset_id", "")
        sorted_history = sorted(history)

        intervals_hours: List[float] = []
        for i in range(1, len(sorted_history)):
            delta = sorted_history[i] - sorted_history[i - 1]
            intervals_hours.append(delta.total_seconds() / 3600.0)

        avg_hours = _safe_mean(intervals_hours)
        stddev_hours = 0.0
        if len(intervals_hours) >= 2 and avg_hours > 0:
            variance = sum(
                (x - avg_hours) ** 2 for x in intervals_hours
            ) / len(intervals_hours)
            stddev_hours = variance ** 0.5

        # Classify pattern type by coefficient of variation
        cv = stddev_hours / avg_hours if avg_hours > 0 else 0.0
        if cv < 0.2:
            pattern_type = "periodic"
        elif cv < 0.5:
            pattern_type = "semi_regular"
        elif cv < 1.0:
            pattern_type = "irregular"
        else:
            pattern_type = "degrading"

        confidence = max(0.0, min(1.0, 1.0 - cv))

        return StalenessPattern(
            pattern_id=f"SP-{uuid4().hex[:12]}",
            dataset_id=dataset_id,
            pattern_type=pattern_type,
            avg_refresh_hours=round(avg_hours, 4),
            stddev_refresh_hours=round(stddev_hours, 4),
            detected_at=_utcnow().isoformat(),
            confidence=round(confidence, 4),
        )

    def _fallback_predict_refresh(
        self,
        dataset: dict,
        history: List[datetime],
    ) -> RefreshPrediction:
        """Fallback refresh prediction using mean interval.

        Estimates the next refresh time by adding the mean refresh
        interval to the most recent refresh timestamp.

        Args:
            dataset: Dataset dict.
            history: Sorted list of refresh datetimes.

        Returns:
            RefreshPrediction with estimated next refresh time.
        """
        dataset_id = dataset.get("dataset_id", "")
        sorted_history = sorted(history)

        intervals_hours: List[float] = []
        for i in range(1, len(sorted_history)):
            delta = sorted_history[i] - sorted_history[i - 1]
            intervals_hours.append(delta.total_seconds() / 3600.0)

        avg_hours = _safe_mean(intervals_hours)
        last_refresh = sorted_history[-1]
        predicted_next = last_refresh + timedelta(hours=avg_hours)

        # Confidence based on sample size and regularity
        n = len(intervals_hours)
        base_confidence = min(1.0, n / 10.0)
        if avg_hours > 0 and n >= 2:
            variance = sum(
                (x - avg_hours) ** 2 for x in intervals_hours
            ) / n
            cv = (variance ** 0.5) / avg_hours
            regularity_factor = max(0.0, 1.0 - cv)
        else:
            regularity_factor = 0.5

        confidence = round(base_confidence * regularity_factor, 4)

        return RefreshPrediction(
            prediction_id=f"RP-{uuid4().hex[:12]}",
            dataset_id=dataset_id,
            predicted_at=_utcnow().isoformat(),
            next_refresh_at=predicted_next.isoformat(),
            confidence=confidence,
            method="mean_interval",
        )

    def _fallback_evaluate_sla(
        self,
        check: FreshnessCheck,
        sla: dict,
        dataset_id: str,
        age_hours: float,
    ) -> List[SLABreach]:
        """Fallback SLA evaluation using threshold comparison.

        Compares dataset age against warning and critical thresholds.

        Args:
            check: FreshnessCheck result for the dataset.
            sla: SLA config dict with ``warning_hours`` and
                ``critical_hours`` keys.
            dataset_id: Dataset identifier.
            age_hours: Current dataset age in hours.

        Returns:
            List of SLABreach records (0, 1, or 2 entries).
        """
        breaches: List[SLABreach] = []
        warning_hours = sla.get("warning_hours", 24.0)
        critical_hours = sla.get("critical_hours", 72.0)
        now = _utcnow()

        if age_hours >= critical_hours:
            breaches.append(SLABreach(
                breach_id=f"SB-{uuid4().hex[:12]}",
                dataset_id=dataset_id,
                severity="critical",
                sla_hours=critical_hours,
                actual_hours=round(age_hours, 4),
                breached_at=now.isoformat(),
                resolved=False,
            ))
        elif age_hours >= warning_hours:
            breaches.append(SLABreach(
                breach_id=f"SB-{uuid4().hex[:12]}",
                dataset_id=dataset_id,
                severity="warning",
                sla_hours=warning_hours,
                actual_hours=round(age_hours, 4),
                breached_at=now.isoformat(),
                resolved=False,
            ))

        return breaches

    def _fallback_create_alert(
        self,
        breach: SLABreach,
        dataset_id: str,
        breach_id: str,
        severity: str,
        policy: dict,
    ) -> FreshnessAlert:
        """Fallback alert creation when AlertManagerEngine is unavailable.

        Creates a FreshnessAlert record without actually dispatching
        to any channel.

        Args:
            breach: SLABreach that triggered this alert.
            dataset_id: Dataset identifier.
            breach_id: Breach identifier.
            severity: Alert severity level.
            policy: Escalation policy dict for the dataset.

        Returns:
            FreshnessAlert record.
        """
        channel = policy.get("channel", "default")

        return FreshnessAlert(
            alert_id=f"FA-{uuid4().hex[:12]}",
            breach_id=breach_id,
            dataset_id=dataset_id,
            severity=severity,
            channel=channel,
            sent_at=_utcnow().isoformat(),
            acknowledged=False,
        )

    # ==================================================================
    # Private: Helper methods
    # ==================================================================

    def _record_breaches_with_alert_manager(
        self,
        breaches: List[SLABreach],
    ) -> None:
        """Record breaches with the alert manager for later alerting.

        Args:
            breaches: List of SLABreach records to record.
        """
        if self._alert_manager is None:
            return
        for breach in breaches:
            try:
                self._alert_manager.record_breach(breach)
            except Exception as exc:
                logger.debug(
                    "AlertManagerEngine.record_breach failed: %s", exc,
                )

    def _compute_age_hours(
        self,
        last_updated_str: str,
        now: datetime,
    ) -> float:
        """Compute dataset age in hours from last_updated timestamp.

        Attempts to parse ISO 8601 format. Falls back to 0.0 on
        parse failure.

        Args:
            last_updated_str: ISO 8601 timestamp string.
            now: Current UTC datetime.

        Returns:
            Age in hours, or 0.0 if parsing fails.
        """
        if not last_updated_str:
            return 0.0

        try:
            last_updated = datetime.fromisoformat(
                last_updated_str.replace("Z", "+00:00"),
            )
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)
            delta = now - last_updated
            return max(0.0, delta.total_seconds() / 3600.0)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse last_updated '%s': %s",
                last_updated_str, exc,
            )
            return 0.0

    def _classify_freshness_tier(self, age_hours: float) -> str:
        """Classify a dataset into a freshness tier based on age.

        Uses config thresholds when available, otherwise applies
        default thresholds: excellent <1h, good <6h, fair <24h,
        poor <72h, stale >=72h.

        Args:
            age_hours: Dataset age in hours.

        Returns:
            Freshness tier string: excellent, good, fair, poor, or stale.
        """
        excellent = 1.0
        good = 6.0
        fair = 24.0
        poor = 72.0

        if self._config is not None:
            excellent = getattr(
                self._config, "freshness_excellent_hours", excellent,
            )
            good = getattr(
                self._config, "freshness_good_hours", good,
            )
            fair = getattr(
                self._config, "freshness_fair_hours", fair,
            )
            poor = getattr(
                self._config, "freshness_poor_hours", poor,
            )

        if age_hours < excellent:
            return "excellent"
        if age_hours < good:
            return "good"
        if age_hours < fair:
            return "fair"
        if age_hours < poor:
            return "poor"
        return "stale"

    def _compute_freshness_score(self, age_hours: float) -> float:
        """Compute a numeric freshness score from dataset age.

        Uses an exponential decay function: score = exp(-age / half_life).
        Default half-life is 24 hours (configurable via fair threshold).

        Args:
            age_hours: Dataset age in hours.

        Returns:
            Freshness score between 0.0 and 1.0.
        """
        import math

        half_life = 24.0
        if self._config is not None:
            half_life = getattr(
                self._config, "freshness_fair_hours", half_life,
            )

        if age_hours <= 0.0:
            return 1.0

        score = math.exp(-0.693 * age_hours / half_life)
        return round(max(0.0, min(1.0, score)), 4)

    def _default_sla(self) -> dict:
        """Return the default SLA configuration.

        Uses config values when available, otherwise applies defaults
        of 24h warning and 72h critical.

        Returns:
            Dict with warning_hours and critical_hours keys.
        """
        warning = 24.0
        critical = 72.0

        if self._config is not None:
            warning = getattr(
                self._config, "default_sla_warning_hours", warning,
            )
            critical = getattr(
                self._config, "default_sla_critical_hours", critical,
            )

        return {
            "warning_hours": warning,
            "critical_hours": critical,
        }

    def _build_sla_map(
        self,
        datasets: List[dict],
    ) -> Dict[str, dict]:
        """Build a mapping from dataset_id to SLA configuration.

        Checks each dataset for inline SLA overrides, queries the
        SLADefinitionEngine when available, and falls back to the
        default SLA config.

        Args:
            datasets: List of dataset dicts.

        Returns:
            Dict mapping dataset_id to SLA config dicts.
        """
        sla_map: Dict[str, dict] = {}
        default = self._default_sla()

        for ds in datasets:
            dataset_id = ds.get("dataset_id", "")

            # Check inline SLA overrides
            if "sla" in ds and isinstance(ds["sla"], dict):
                sla_map[dataset_id] = ds["sla"]
                continue

            # Try SLA engine
            if self._sla_definition is not None:
                try:
                    sla = self._sla_definition.get_sla(dataset_id)
                    if sla is not None:
                        if isinstance(sla, dict):
                            sla_map[dataset_id] = sla
                        elif hasattr(sla, "warning_hours"):
                            sla_map[dataset_id] = {
                                "warning_hours": sla.warning_hours,
                                "critical_hours": getattr(
                                    sla, "critical_hours", default["critical_hours"],
                                ),
                            }
                        continue
                except Exception:
                    pass

            sla_map[dataset_id] = default

        return sla_map

    def _build_refresh_histories(
        self,
        datasets: List[dict],
    ) -> Dict[str, List[datetime]]:
        """Build refresh history maps from dataset metadata.

        Extracts ``refresh_history`` from each dataset dict. If
        timestamps are strings, parses them to datetime objects.

        Args:
            datasets: List of dataset dicts, optionally containing
                ``refresh_history`` as a list of ISO timestamps.

        Returns:
            Dict mapping dataset_id to sorted lists of datetimes.
        """
        histories: Dict[str, List[datetime]] = {}

        for ds in datasets:
            dataset_id = ds.get("dataset_id", "")
            raw_history = ds.get("refresh_history", [])

            if not raw_history:
                continue

            parsed: List[datetime] = []
            for entry in raw_history:
                if isinstance(entry, datetime):
                    parsed.append(entry)
                elif isinstance(entry, str):
                    try:
                        dt = datetime.fromisoformat(
                            entry.replace("Z", "+00:00"),
                        )
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        parsed.append(dt)
                    except (ValueError, TypeError):
                        pass

            if parsed:
                histories[dataset_id] = sorted(parsed)

        return histories

    def _build_escalation_policies(
        self,
        datasets: List[dict],
    ) -> Dict[str, dict]:
        """Build escalation policy map from dataset metadata.

        Extracts ``escalation_policy`` from each dataset dict.

        Args:
            datasets: List of dataset dicts.

        Returns:
            Dict mapping dataset_id to escalation policy dicts.
        """
        policies: Dict[str, dict] = {}

        for ds in datasets:
            dataset_id = ds.get("dataset_id", "")
            policy = ds.get("escalation_policy", {})
            if isinstance(policy, dict):
                policies[dataset_id] = policy

        return policies

    def _determine_run_status(
        self,
        checks: List[FreshnessCheck],
        breaches: List[SLABreach],
    ) -> MonitoringStatus:
        """Determine the overall monitoring run status.

        Args:
            checks: List of FreshnessCheck results.
            breaches: List of SLABreach records.

        Returns:
            MonitoringStatus enum value.
        """
        if not checks:
            return MonitoringStatus.COMPLETED

        if any(
            getattr(b, "severity", "") == "critical"
            for b in breaches
        ):
            return MonitoringStatus.COMPLETED_WITH_WARNINGS

        if breaches:
            return MonitoringStatus.COMPLETED_WITH_WARNINGS

        return MonitoringStatus.COMPLETED

    def _finalize_run(
        self,
        run: MonitoringRun,
        elapsed: float,
        checks: List[FreshnessCheck],
        breaches: List[SLABreach],
        alerts: List[FreshnessAlert],
    ) -> None:
        """Finalize a pipeline run by updating statistics and history.

        Args:
            run: The MonitoringRun to finalize.
            elapsed: Total pipeline elapsed time in seconds.
            checks: FreshnessCheck results from the run.
            breaches: SLABreach records from the run.
            alerts: FreshnessAlert records from the run.
        """
        with self._lock:
            # Append to history
            self._pipeline_history.append(run)
            self._run_count += 1

            # Update aggregate statistics
            self._statistics.total_runs += 1
            self._statistics.total_datasets_checked += len(checks)
            self._statistics.total_breaches_found += len(breaches)
            self._statistics.total_alerts_sent += len(alerts)

            # Status tracking
            status_key = (
                run.status.value
                if hasattr(run.status, "value")
                else str(run.status)
            )
            self._statistics.by_status[status_key] = (
                self._statistics.by_status.get(status_key, 0) + 1
            )

            # Severity tracking
            for breach in breaches:
                sev = getattr(breach, "severity", "unknown")
                self._statistics.by_severity[sev] = (
                    self._statistics.by_severity.get(sev, 0) + 1
                )

            # Tier tracking
            for check in checks:
                tier = getattr(check, "freshness_tier", "unknown")
                self._statistics.by_freshness_tier[tier] = (
                    self._statistics.by_freshness_tier.get(tier, 0) + 1
                )

            # Update running average freshness score
            scores = [
                getattr(c, "freshness_score", 0.0) for c in checks
            ]
            if scores:
                run_avg = _safe_mean(scores)
                n = self._statistics.total_runs
                old_avg = self._statistics.avg_freshness_score
                self._statistics.avg_freshness_score = (
                    (old_avg * (n - 1) + run_avg) / n
                    if n > 0
                    else run_avg
                )

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

        Delegates to the provenance tracker when available, otherwise
        computes a standalone SHA-256 hash.

        Args:
            operation: Operation name for the provenance entry.
            input_data: Input data to hash.
            output_data: Output data to hash.

        Returns:
            SHA-256 chain hash string.
        """
        input_hash = _compute_sha256(input_data)
        output_hash = _compute_sha256(output_data)

        if hasattr(self._provenance, "add_to_chain"):
            try:
                return self._provenance.add_to_chain(
                    operation=operation,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    metadata={"operation": operation},
                )
            except Exception:
                pass

        # Standalone hash when provenance tracker is unavailable
        combined = json.dumps(
            {
                "operation": operation,
                "input": input_hash,
                "output": output_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fallback provenance tracker (minimal, used when provenance.py absent)
# ---------------------------------------------------------------------------


class _FallbackProvenanceTracker:
    """Minimal provenance tracker used when the real tracker is unavailable.

    Provides the same interface as ProvenanceTracker but stores only
    the chain hash and entry count in memory.

    Attributes:
        _chain_length: Number of entries recorded.
        _last_hash: Most recent chain hash.
    """

    GENESIS_HASH = hashlib.sha256(
        b"greenlang-data-freshness-monitor-genesis"
    ).hexdigest()

    def __init__(self) -> None:
        """Initialize the fallback tracker with genesis hash."""
        self._chain_length: int = 0
        self._last_hash: str = self.GENESIS_HASH
        self._lock = threading.Lock()

    def add_to_chain(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a chain link and return the new chain hash.

        Args:
            operation: Operation name.
            input_hash: SHA-256 hash of input.
            output_hash: SHA-256 hash of output.
            metadata: Optional metadata dict.

        Returns:
            New chain hash.
        """
        combined = json.dumps(
            {
                "previous": self._last_hash,
                "input": input_hash,
                "output": output_hash,
                "operation": operation,
            },
            sort_keys=True,
        )
        new_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        with self._lock:
            self._last_hash = new_hash
            self._chain_length += 1

        return new_hash

    def get_chain_length(self) -> int:
        """Return the number of entries in the chain.

        Returns:
            Chain entry count.
        """
        with self._lock:
            return self._chain_length

    def get_current_hash(self) -> str:
        """Return the most recent chain hash.

        Returns:
            Current chain hash string.
        """
        with self._lock:
            return self._last_hash

    def reset(self) -> None:
        """Reset the tracker to genesis state."""
        with self._lock:
            self._chain_length = 0
            self._last_hash = self.GENESIS_HASH

    def build_hash(self, data: Any) -> str:
        """Compute a SHA-256 hash of arbitrary data.

        Args:
            data: Data to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return _compute_sha256(data)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "FreshnessMonitorPipelineEngine",
    "MonitoringRun",
    "MonitoringStatus",
    "FreshnessCheck",
    "FreshnessReport",
    "SLABreach",
    "FreshnessAlert",
    "StalenessPattern",
    "RefreshPrediction",
    "PipelineStageResult",
    "PipelineStatistics",
]
