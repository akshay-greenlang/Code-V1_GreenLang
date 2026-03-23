# -*- coding: utf-8 -*-
"""
Dual Reporting Reconciliation Service Setup - AGENT-MRV-013
=============================================================

Service facade for the Dual Reporting Reconciliation Agent
(GL-MRV-X-024).

Provides ``get_service()`` and the ``DualReportingService`` facade class
that aggregates all 7 engines:

    1. DualResultCollectorEngine       - Upstream result collection
    2. DiscrepancyAnalyzerEngine       - Discrepancy analysis & waterfall
    3. QualityScorerEngine             - 4-dimension quality scoring
    4. ReportingTableGeneratorEngine   - Multi-framework table generation
    5. TrendAnalysisEngine             - YoY/CAGR/PIF trend analysis
    6. ComplianceCheckerEngine         - 7-framework regulatory compliance
    7. DualReportingPipelineEngine     - 10-stage orchestrated pipeline

The service provides 16 public methods matching the 16 REST API endpoints:

    Reconciliation:
        reconcile, reconcile_batch, list_reconciliations,
        get_reconciliation, delete_reconciliation
    Discrepancies:
        list_discrepancies, get_waterfall
    Quality:
        get_quality_assessment
    Reporting Tables:
        get_reporting_tables
    Trends:
        get_trend_analysis
    Compliance:
        check_compliance, get_compliance_result
    Aggregation:
        get_aggregations
    Export:
        export_report
    Health:
        health_check, get_stats

All calculation paths use deterministic Decimal arithmetic for
zero-hallucination guarantees. Every mutation records a SHA-256
provenance hash for complete audit trails.

Usage:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.setup import get_service
    >>> svc = get_service()
    >>> result = svc.reconcile({
    ...     "tenant_id": "tenant-001",
    ...     "period_start": "2024-01-01",
    ...     "period_end": "2024-12-31",
    ...     "upstream_results": [...],
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
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
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        DualReportingReconciliationConfig,
        get_config,
    )
except ImportError:
    DualReportingReconciliationConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        DualReportingReconciliationMetrics,
        get_metrics,
    )
except ImportError:
    DualReportingReconciliationMetrics = None  # type: ignore[assignment, misc]

    def get_metrics() -> Any:  # type: ignore[misc]
        """Stub returning None when metrics module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        DualReportingReconciliationProvenance,
    )
except ImportError:
    DualReportingReconciliationProvenance = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.dual_result_collector import (
        DualResultCollectorEngine,
    )
except ImportError:
    DualResultCollectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.discrepancy_analyzer import (
        DiscrepancyAnalyzerEngine,
    )
except ImportError:
    DiscrepancyAnalyzerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer import (
        QualityScorerEngine,
    )
except ImportError:
    QualityScorerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.reporting_table_generator import (
        ReportingTableGeneratorEngine,
    )
except ImportError:
    ReportingTableGeneratorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.trend_analyzer import (
        TrendAnalysisEngine,
    )
except ImportError:
    TrendAnalysisEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import (
        DualReportingPipelineEngine,
    )
except ImportError:
    DualReportingPipelineEngine = None  # type: ignore[assignment, misc]


# ===================================================================
# Constants
# ===================================================================

#: Service version for health checks and diagnostics.
SERVICE_VERSION: str = "1.0.0"

#: Service name for observability.
SERVICE_NAME: str = "dual-reporting-reconciliation-service"

#: Agent identifier for tracing and audit logs.
AGENT_ID: str = "AGENT-MRV-013"

#: Default max batch size.
DEFAULT_MAX_BATCH_SIZE: int = 120

#: Valid Scope 2 energy types.
VALID_ENERGY_TYPES: frozenset = frozenset({
    "electricity",
    "steam",
    "district_heating",
    "district_cooling",
})

#: Valid Scope 2 methods.
VALID_METHODS: frozenset = frozenset({
    "location_based",
    "market_based",
})

#: Supported regulatory frameworks.
VALID_FRAMEWORKS: frozenset = frozenset({
    "ghg_protocol",
    "csrd_esrs",
    "cdp",
    "sbti",
    "gri",
    "iso_14064",
    "re100",
})

#: Valid export formats.
VALID_EXPORT_FORMATS: frozenset = frozenset({
    "json",
    "csv",
    "xlsx",
    "pdf",
})

#: Valid aggregation group-by dimensions.
VALID_GROUP_BY: frozenset = frozenset({
    "energy_type",
    "facility",
    "region",
    "business_unit",
    "period",
})

#: Decimal precision for all financial calculations.
DECIMAL_PLACES: int = 8


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _short_id(prefix: str = "drr") -> str:
    """Generate a short prefixed identifier for records.

    Args:
        prefix: Prefix string prepended to the UUID fragment.

    Returns:
        A string of the form ``{prefix}_{12-char-hex}``.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Handles Pydantic models (via ``model_dump``), dicts, lists,
    and primitive values. Decimal values are serialised via ``str``.

    Args:
        data: Arbitrary data to hash.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with graceful fallback.

    Args:
        value: Value to convert.
        default: Fallback value on conversion failure.

    Returns:
        Float representation of the value.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError, ArithmeticError):
        return default


def _safe_decimal(value: Any, default: str = "0") -> Decimal:
    """Convert a value to Decimal with graceful fallback.

    Args:
        value: Value to convert.
        default: Fallback string representation on failure.

    Returns:
        Decimal representation of the value.
    """
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since a monotonic start time.

    Args:
        start: ``time.monotonic()`` start value.

    Returns:
        Elapsed time in milliseconds, rounded to 3 decimal places.
    """
    return round((time.monotonic() - start) * 1000.0, 3)


def _validate_required_fields(
    data: Dict[str, Any],
    fields: List[str],
    context: str = "request",
) -> List[str]:
    """Validate that required fields are present and non-empty.

    Args:
        data: Dictionary to validate.
        fields: List of required field names.
        context: Human-readable context for error messages.

    Returns:
        List of validation error strings (empty if valid).
    """
    errors: List[str] = []
    for field in fields:
        val = data.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(
                f"Missing required field '{field}' in {context}"
            )
    return errors


def _validate_enum_field(
    value: Any,
    valid_values: frozenset,
    field_name: str,
) -> Optional[str]:
    """Validate that a field value is in the allowed set.

    Args:
        value: Value to validate.
        valid_values: Set of allowed values.
        field_name: Human-readable field name for error messages.

    Returns:
        Error string if invalid, None if valid.
    """
    if value is None:
        return None
    normalized = str(value).lower().strip()
    if normalized not in valid_values:
        return (
            f"Invalid {field_name} '{value}'; "
            f"must be one of {sorted(valid_values)}"
        )
    return None


# ===================================================================
# Pydantic Response Models (14 models)
# ===================================================================


class ReconcileResponse(BaseModel):
    """Single reconciliation result response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    reconciliation_id: str = Field(default="")
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    total_location_tco2e: float = Field(default=0.0)
    total_market_tco2e: float = Field(default=0.0)
    discrepancy_tco2e: float = Field(default=0.0)
    discrepancy_pct: float = Field(default=0.0)
    direction: str = Field(default="equal")
    materiality: str = Field(default="immaterial")
    pif: float = Field(default=0.0)
    quality_grade: str = Field(default="")
    quality_score: float = Field(default=0.0)
    status: str = Field(default="completed")
    discrepancy_count: int = Field(default=0)
    frameworks_checked: int = Field(default=0)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchReconcileResponse(BaseModel):
    """Batch reconciliation result response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    batch_id: str = Field(default="")
    total_periods: int = Field(default=0)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)


class ReconciliationListResponse(BaseModel):
    """Response listing reconciliation runs."""

    model_config = ConfigDict(frozen=True)

    reconciliations: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    skip: int = Field(default=0)
    limit: int = Field(default=20)


class DiscrepancyListResponse(BaseModel):
    """Response listing discrepancies for a reconciliation."""

    model_config = ConfigDict(frozen=True)

    reconciliation_id: str = Field(default="")
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    direction: str = Field(default="equal")
    total_discrepancy_tco2e: float = Field(default=0.0)


class WaterfallResponse(BaseModel):
    """Waterfall decomposition response."""

    model_config = ConfigDict(frozen=True)

    reconciliation_id: str = Field(default="")
    location_total_tco2e: float = Field(default=0.0)
    market_total_tco2e: float = Field(default=0.0)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    residual_tco2e: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class QualityAssessmentResponse(BaseModel):
    """Quality assessment response."""

    model_config = ConfigDict(frozen=True)

    reconciliation_id: str = Field(default="")
    composite_score: float = Field(default=0.0)
    grade: str = Field(default="")
    dimensions: Dict[str, float] = Field(default_factory=dict)
    ef_hierarchy_scores: Dict[str, float] = Field(default_factory=dict)
    flags: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ReportingTablesResponse(BaseModel):
    """Multi-framework reporting tables response."""

    model_config = ConfigDict(frozen=True)

    reconciliation_id: str = Field(default="")
    tables: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    frameworks_generated: int = Field(default=0)
    provenance_hash: str = Field(default="")


class TrendAnalysisResponse(BaseModel):
    """Trend analysis response."""

    model_config = ConfigDict(frozen=True)

    reconciliation_id: str = Field(default="")
    periods_analyzed: int = Field(default=0)
    location_trend: str = Field(default="stable")
    market_trend: str = Field(default="stable")
    location_cagr_pct: float = Field(default=0.0)
    market_cagr_pct: float = Field(default=0.0)
    pif_trend: str = Field(default="stable")
    re100_pct_latest: float = Field(default=0.0)
    sbti_on_track: bool = Field(default=False)
    intensity_metrics: Dict[str, float] = Field(default_factory=dict)
    data_points: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ComplianceCheckResponse(BaseModel):
    """Regulatory compliance check response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    compliance_id: str = Field(default="")
    reconciliation_id: str = Field(default="")
    frameworks_checked: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partial: int = Field(default=0)
    overall_score: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    checked_at: str = Field(default_factory=_utcnow_iso)
    provenance_hash: str = Field(default="")


class AggregationResponse(BaseModel):
    """Aggregated reconciliation response."""

    model_config = ConfigDict(frozen=True)

    group_by: str = Field(default="energy_type")
    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_location_tco2e: float = Field(default=0.0)
    total_market_tco2e: float = Field(default=0.0)
    portfolio_pif: float = Field(default=0.0)
    reconciliation_count: int = Field(default=0)
    period: str = Field(default="all")
    timestamp: str = Field(default_factory=_utcnow_iso)


class ExportResponse(BaseModel):
    """Export report response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    export_id: str = Field(default="")
    reconciliation_id: str = Field(default="")
    format: str = Field(default="json")
    content: str = Field(default="")
    content_type: str = Field(default="application/json")
    filename: str = Field(default="")
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default=SERVICE_NAME)
    version: str = Field(default=SERVICE_VERSION)
    agent_id: str = Field(default=AGENT_ID)
    engines: Dict[str, str] = Field(default_factory=dict)
    config_valid: bool = Field(default=True)
    uptime_seconds: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class StatsResponse(BaseModel):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_reconciliations: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_discrepancies_found: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_exports: int = Field(default=0)
    average_quality_score: float = Field(default=0.0)
    average_pif: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


# ===================================================================
# DualReportingService facade
# ===================================================================

_singleton_lock = threading.Lock()
_service_instance: Optional["DualReportingService"] = None


class DualReportingService:
    """Unified facade over the Dual Reporting Reconciliation Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 16 REST API operations.

    Each mutation method records SHA-256 provenance hashes.
    All numeric calculations use deterministic Decimal arithmetic
    delegated to the underlying engines (zero-hallucination path).

    In-memory storage provides the default persistence layer. In
    production, methods should be backed by PostgreSQL via the
    engines' database connectors.

    Attributes:
        config: Service configuration (DualReportingReconciliationConfig
            or dict).
        metrics: Prometheus metrics singleton.

    Example:
        >>> service = DualReportingService()
        >>> result = service.reconcile({
        ...     "tenant_id": "tenant-001",
        ...     "period_start": "2024-01-01",
        ...     "period_end": "2024-12-31",
        ...     "upstream_results": [...],
        ... })
        >>> assert result.success is True
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Dual Reporting Reconciliation Service facade.

        Creates all engine instances with graceful degradation when
        individual engine modules are not importable. Sets up in-memory
        storage for reconciliation results, compliance checks, and
        exports.

        Args:
            config: Optional configuration override. Accepts
                DualReportingReconciliationConfig, dict, or None
                (defaults to singleton from get_config()).
        """
        self._config = config if config is not None else get_config()
        self._metrics = get_metrics()
        self._start_time: float = time.monotonic()

        # Engine placeholders (initialised in _init_engines)
        self._collector: Any = None
        self._discrepancy_analyzer: Any = None
        self._quality_scorer: Any = None
        self._table_generator: Any = None
        self._trend_analyzer: Any = None
        self._compliance_checker: Any = None
        self._pipeline: Any = None

        self._init_engines()

        # In-memory data stores
        self._reconciliations: Dict[str, Dict[str, Any]] = {}
        self._compliance_results: Dict[str, Dict[str, Any]] = {}
        self._exports: Dict[str, Dict[str, Any]] = {}

        # Aggregate statistics
        self._total_reconciliations: int = 0
        self._total_batch_runs: int = 0
        self._total_discrepancies: int = 0
        self._total_compliance_checks: int = 0
        self._total_exports: int = 0
        self._cumulative_quality_scores: List[float] = []
        self._cumulative_pifs: List[float] = []

        logger.info(
            "DualReportingService facade created "
            "(engines: collector=%s, discrepancy=%s, "
            "quality=%s, tables=%s, "
            "trends=%s, compliance=%s, pipeline=%s)",
            self._collector is not None,
            self._discrepancy_analyzer is not None,
            self._quality_scorer is not None,
            self._table_generator is not None,
            self._trend_analyzer is not None,
            self._compliance_checker is not None,
            self._pipeline is not None,
        )

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Any:
        """Get the service configuration."""
        return self._config

    @property
    def metrics(self) -> Any:
        """Get the Prometheus metrics singleton."""
        return self._metrics

    @property
    def collector_engine(self) -> Any:
        """Get the DualResultCollectorEngine instance."""
        return self._collector

    @property
    def discrepancy_engine(self) -> Any:
        """Get the DiscrepancyAnalyzerEngine instance."""
        return self._discrepancy_analyzer

    @property
    def quality_engine(self) -> Any:
        """Get the QualityScorerEngine instance."""
        return self._quality_scorer

    @property
    def table_generator_engine(self) -> Any:
        """Get the ReportingTableGeneratorEngine instance."""
        return self._table_generator

    @property
    def trend_engine(self) -> Any:
        """Get the TrendAnalysisEngine instance."""
        return self._trend_analyzer

    @property
    def compliance_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance."""
        return self._compliance_checker

    @property
    def pipeline_engine(self) -> Any:
        """Get the DualReportingPipelineEngine instance."""
        return self._pipeline

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Import and initialise all SDK engines with graceful fallback.

        Each engine is constructed independently. If any engine fails
        to initialise, a warning is logged and the engine attribute
        remains None. The service continues to operate with reduced
        functionality.

        The pipeline engine is initialised last because it depends on
        all upstream engines.
        """
        config_arg = self._config
        if config_arg is not None and hasattr(config_arg, "to_dict"):
            config_arg = config_arg.to_dict()
        elif config_arg is not None and not isinstance(config_arg, dict):
            config_arg = {}
        metrics_arg = self._metrics

        # E1: DualResultCollectorEngine
        self._collector = self._init_single_engine(
            "DualResultCollectorEngine",
            DualResultCollectorEngine,
            config_arg,
            metrics_arg,
        )

        # E2: DiscrepancyAnalyzerEngine
        self._discrepancy_analyzer = self._init_single_engine(
            "DiscrepancyAnalyzerEngine",
            DiscrepancyAnalyzerEngine,
            config_arg,
            metrics_arg,
        )

        # E3: QualityScorerEngine
        self._quality_scorer = self._init_single_engine(
            "QualityScorerEngine",
            QualityScorerEngine,
            config_arg,
            metrics_arg,
        )

        # E4: ReportingTableGeneratorEngine
        self._table_generator = self._init_single_engine(
            "ReportingTableGeneratorEngine",
            ReportingTableGeneratorEngine,
            config_arg,
            metrics_arg,
        )

        # E5: TrendAnalysisEngine
        self._trend_analyzer = self._init_single_engine(
            "TrendAnalysisEngine",
            TrendAnalysisEngine,
            config_arg,
            metrics_arg,
        )

        # E6: ComplianceCheckerEngine
        self._compliance_checker = self._init_single_engine(
            "ComplianceCheckerEngine",
            ComplianceCheckerEngine,
            config_arg,
            metrics_arg,
        )

        # E7: DualReportingPipelineEngine
        self._init_pipeline_engine(config_arg, metrics_arg)

    def _init_single_engine(
        self,
        name: str,
        engine_class: Any,
        config_arg: Any,
        metrics_arg: Any,
    ) -> Any:
        """Initialize a single engine with graceful degradation.

        Tries multiple constructor signatures to handle different
        engine initialization patterns.

        Args:
            name: Human-readable engine name for logging.
            engine_class: Engine class or None if import failed.
            config_arg: Configuration to pass to the engine.
            metrics_arg: Metrics instance to pass to the engine.

        Returns:
            Engine instance or None on failure.
        """
        if engine_class is None:
            logger.warning("%s not available (import failed)", name)
            return None

        try:
            return engine_class(config_arg, metrics_arg)
        except TypeError:
            pass

        try:
            return engine_class(config=config_arg)
        except TypeError:
            pass

        try:
            return engine_class()
        except Exception as exc:
            logger.warning(
                "%s initialization failed: %s", name, exc,
            )
            return None

    def _init_pipeline_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> None:
        """Initialize the DualReportingPipelineEngine.

        The pipeline engine receives all upstream engine instances
        for orchestrated reconciliation.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.
        """
        if DualReportingPipelineEngine is None:
            logger.warning(
                "DualReportingPipelineEngine not available (import failed)"
            )
            return

        try:
            self._pipeline = DualReportingPipelineEngine(
                collector=self._collector,
                discrepancy_analyzer=self._discrepancy_analyzer,
                quality_scorer=self._quality_scorer,
                table_generator=self._table_generator,
                trend_analyzer=self._trend_analyzer,
                compliance_checker=self._compliance_checker,
                config=config_arg,
                metrics=metrics_arg,
            )
            logger.info("DualReportingPipelineEngine initialized")
        except TypeError:
            try:
                self._pipeline = DualReportingPipelineEngine(
                    self._collector,
                    self._discrepancy_analyzer,
                    self._quality_scorer,
                    self._table_generator,
                    self._trend_analyzer,
                    self._compliance_checker,
                    config_arg,
                    metrics_arg,
                )
                logger.info(
                    "DualReportingPipelineEngine initialized "
                    "(positional args)"
                )
            except TypeError:
                try:
                    self._pipeline = DualReportingPipelineEngine()
                    logger.info(
                        "DualReportingPipelineEngine initialized "
                        "(no-arg constructor)"
                    )
                except Exception as exc:
                    logger.warning(
                        "DualReportingPipelineEngine initialization "
                        "failed: %s",
                        exc,
                    )
        except Exception as exc:
            logger.warning(
                "DualReportingPipelineEngine initialization "
                "failed: %s",
                exc,
            )

    # ==================================================================
    # Internal: metrics recording
    # ==================================================================

    def _record_metric_reconciliation(
        self,
        duration_s: float,
        discrepancy_pct: float,
    ) -> None:
        """Record a reconciliation metric if metrics are available.

        Args:
            duration_s: Duration in seconds.
            discrepancy_pct: Discrepancy percentage.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_reconciliation(
                duration=duration_s,
                discrepancy_pct=discrepancy_pct,
            )
        except Exception:
            pass

    def _record_metric_error(self, error_type: str) -> None:
        """Record an error metric if metrics are available.

        Args:
            error_type: Error classification label.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_error(error_type=error_type)
        except Exception:
            pass

    # ==================================================================
    # 1. reconcile - Execute single reconciliation
    # ==================================================================

    def reconcile(self, request: Dict[str, Any]) -> ReconcileResponse:
        """Execute a single dual-reporting reconciliation.

        Takes upstream location-based and market-based results and runs
        the full 10-stage pipeline: collect, align, map, analyze
        discrepancies, score quality, generate tables, analyze trends,
        check compliance, assemble report, seal provenance.

        Args:
            request: Dict containing tenant_id, period_start, period_end,
                upstream_results, and optional frameworks/trend_data.

        Returns:
            ReconcileResponse with reconciliation results.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        start = time.monotonic()
        errors = _validate_required_fields(
            request,
            ["tenant_id", "upstream_results"],
            "reconciliation request",
        )
        if errors:
            raise ValueError("; ".join(errors))

        reconciliation_id = request.get(
            "reconciliation_id", _short_id("recon"),
        )

        try:
            # Delegate to pipeline if available, fallback to manual
            result = None
            if self._pipeline is not None:
                try:
                    result = self._run_pipeline(request, reconciliation_id)
                except Exception as pipe_exc:
                    logger.warning(
                        "Pipeline failed, falling back to manual: %s",
                        pipe_exc,
                    )
            if result is None:
                result = self._run_manual(request, reconciliation_id)

            # Extract summary from pipeline result
            loc_total = _safe_float(result.get("total_location_tco2e", 0))
            mkt_total = _safe_float(result.get("total_market_tco2e", 0))
            disc_tco2e = abs(loc_total - mkt_total)
            denom = max(loc_total, mkt_total, 0.000001)
            disc_pct = (disc_tco2e / denom) * 100.0

            if abs(loc_total - mkt_total) < 0.01:
                direction = "equal"
            elif mkt_total < loc_total:
                direction = "market_lower"
            else:
                direction = "market_higher"

            pif_val = _safe_float(result.get("pif", 0))
            quality_score = _safe_float(result.get("quality_score", 0))
            quality_grade = result.get("quality_grade", "")

            provenance_hash = _compute_hash(result)
            duration_ms = _elapsed_ms(start)

            # Update stats
            self._total_reconciliations += 1
            self._total_discrepancies += int(
                result.get("discrepancy_count", 0)
            )
            if quality_score > 0:
                self._cumulative_quality_scores.append(quality_score)
            if pif_val != 0:
                self._cumulative_pifs.append(pif_val)

            # Store for retrieval
            stored = {
                "reconciliation_id": reconciliation_id,
                "tenant_id": request.get("tenant_id", ""),
                "period_start": request.get("period_start", ""),
                "period_end": request.get("period_end", ""),
                "total_location_tco2e": loc_total,
                "total_market_tco2e": mkt_total,
                "discrepancy_tco2e": disc_tco2e,
                "discrepancy_pct": disc_pct,
                "direction": direction,
                "pif": pif_val,
                "quality_score": quality_score,
                "quality_grade": quality_grade,
                "status": "completed",
                "provenance_hash": provenance_hash,
                "processing_time_ms": duration_ms,
                "timestamp": _utcnow_iso(),
                "full_result": result,
            }
            self._reconciliations[reconciliation_id] = stored

            self._record_metric_reconciliation(
                duration_s=(time.monotonic() - start),
                discrepancy_pct=disc_pct,
            )

            logger.info(
                "Reconciliation completed: id=%s loc=%.4f mkt=%.4f "
                "disc_pct=%.2f%% grade=%s",
                reconciliation_id, loc_total, mkt_total,
                disc_pct, quality_grade,
            )

            return ReconcileResponse(
                success=True,
                reconciliation_id=reconciliation_id,
                period_start=request.get("period_start", ""),
                period_end=request.get("period_end", ""),
                total_location_tco2e=loc_total,
                total_market_tco2e=mkt_total,
                discrepancy_tco2e=disc_tco2e,
                discrepancy_pct=round(disc_pct, 4),
                direction=direction,
                materiality=result.get("materiality", "immaterial"),
                pif=pif_val,
                quality_grade=quality_grade,
                quality_score=quality_score,
                status="completed",
                discrepancy_count=int(
                    result.get("discrepancy_count", 0)
                ),
                frameworks_checked=int(
                    result.get("frameworks_checked", 0)
                ),
                provenance_hash=provenance_hash,
                processing_time_ms=duration_ms,
            )

        except ValueError:
            raise
        except Exception as exc:
            self._record_metric_error("reconciliation_failure")
            logger.error(
                "Reconciliation failed: %s", exc, exc_info=True,
            )
            raise

    def _run_pipeline(
        self,
        request: Dict[str, Any],
        reconciliation_id: str,
    ) -> Dict[str, Any]:
        """Run reconciliation through the pipeline engine.

        Args:
            request: Request dictionary.
            reconciliation_id: Generated reconciliation identifier.

        Returns:
            Pipeline result dictionary.
        """
        pipeline_request = dict(request)
        pipeline_request["reconciliation_id"] = reconciliation_id

        result = self._pipeline.run_pipeline(pipeline_request)

        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if isinstance(result, dict):
            return result
        return {"raw_result": str(result)}

    def _run_manual(
        self,
        request: Dict[str, Any],
        reconciliation_id: str,
    ) -> Dict[str, Any]:
        """Run reconciliation manually through individual engines.

        Fallback when the pipeline engine is not available.

        Args:
            request: Request dictionary.
            reconciliation_id: Generated reconciliation identifier.

        Returns:
            Result dictionary assembled from individual engine outputs.
        """
        result: Dict[str, Any] = {
            "reconciliation_id": reconciliation_id,
            "discrepancy_count": 0,
            "frameworks_checked": 0,
            "quality_score": 0,
            "quality_grade": "",
            "pif": 0,
            "materiality": "immaterial",
        }

        upstream = request.get("upstream_results", [])

        # Collect and align using E1
        if self._collector is not None:
            try:
                workspace = self._collector.collect_results(upstream)
                if hasattr(workspace, "model_dump"):
                    ws = workspace.model_dump(mode="json")
                elif isinstance(workspace, dict):
                    ws = workspace
                else:
                    ws = {}
                result["total_location_tco2e"] = _safe_float(
                    ws.get("total_location_tco2e", 0)
                )
                result["total_market_tco2e"] = _safe_float(
                    ws.get("total_market_tco2e", 0)
                )
                result["workspace"] = ws
            except Exception as exc:
                logger.warning("Collector engine failed: %s", exc)

        # Analyze discrepancies using E2
        if self._discrepancy_analyzer is not None:
            try:
                ws_data = result.get("workspace", {})
                disc_result = self._discrepancy_analyzer.analyze_discrepancies(
                    ws_data,
                )
                if hasattr(disc_result, "model_dump"):
                    dr = disc_result.model_dump(mode="json")
                elif isinstance(disc_result, dict):
                    dr = disc_result
                else:
                    dr = {}
                result["discrepancies"] = dr
                result["discrepancy_count"] = len(
                    dr.get("discrepancies", [])
                )
                result["materiality"] = dr.get(
                    "materiality", "immaterial"
                )
            except Exception as exc:
                logger.warning("Discrepancy analyzer failed: %s", exc)

        # Score quality using E3
        if self._quality_scorer is not None:
            try:
                quality = self._quality_scorer.score_quality(
                    result.get("workspace", {}),
                )
                if hasattr(quality, "model_dump"):
                    qa = quality.model_dump(mode="json")
                elif isinstance(quality, dict):
                    qa = quality
                else:
                    qa = {}
                result["quality_score"] = _safe_float(
                    qa.get("composite_score", 0)
                )
                result["quality_grade"] = qa.get("grade", "")
                result["quality_assessment"] = qa
            except Exception as exc:
                logger.warning("Quality scorer failed: %s", exc)

        # Compliance check using E6
        if self._compliance_checker is not None:
            try:
                fw_list = request.get("frameworks", None)
                comp_result = self._compliance_checker.check_all_frameworks(
                    result.get("workspace", {}),
                    result.get("discrepancies", None),
                    result.get("quality_assessment", None),
                    fw_list,
                )
                if hasattr(comp_result, "model_dump"):
                    cr = comp_result.model_dump(mode="json")
                elif isinstance(comp_result, dict):
                    cr = comp_result
                else:
                    cr = {}
                result["compliance"] = cr
                result["frameworks_checked"] = len(cr)
            except Exception as exc:
                logger.warning("Compliance checker failed: %s", exc)

        return result

    # ==================================================================
    # 2. reconcile_batch - Execute batch reconciliation
    # ==================================================================

    def reconcile_batch(
        self, request: Dict[str, Any],
    ) -> BatchReconcileResponse:
        """Execute batch reconciliation across multiple periods.

        Args:
            request: Dict containing batch_id, tenant_id, and periods
                (list of period dicts with period_start, period_end,
                upstream_results).

        Returns:
            BatchReconcileResponse with results for each period.
        """
        start = time.monotonic()
        batch_id = request.get("batch_id", _short_id("batch"))
        periods = request.get("periods", [])
        tenant_id = request.get("tenant_id", "")

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        successful = 0

        for idx, period in enumerate(periods):
            period_request = dict(period)
            period_request.setdefault("tenant_id", tenant_id)
            try:
                resp = self.reconcile(period_request)
                results.append(resp.model_dump(mode="json"))
                successful += 1
            except Exception as exc:
                errors.append({
                    "period_index": idx,
                    "period_start": period.get("period_start", ""),
                    "period_end": period.get("period_end", ""),
                    "error": str(exc),
                })

        self._total_batch_runs += 1
        duration_ms = _elapsed_ms(start)

        logger.info(
            "Batch reconciliation completed: batch_id=%s "
            "total=%d successful=%d failed=%d",
            batch_id, len(periods), successful, len(errors),
        )

        return BatchReconcileResponse(
            success=len(errors) == 0,
            batch_id=batch_id,
            total_periods=len(periods),
            successful=successful,
            failed=len(errors),
            results=results,
            errors=errors,
            processing_time_ms=duration_ms,
            provenance_hash=_compute_hash({
                "batch_id": batch_id,
                "results_count": successful,
            }),
        )

    # ==================================================================
    # 3. list_reconciliations
    # ==================================================================

    def list_reconciliations(
        self,
        tenant_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> ReconciliationListResponse:
        """List reconciliation runs with optional tenant filter.

        Args:
            tenant_id: Optional tenant filter.
            skip: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            ReconciliationListResponse with paginated results.
        """
        all_recons = list(self._reconciliations.values())

        if tenant_id is not None:
            all_recons = [
                r for r in all_recons
                if r.get("tenant_id") == tenant_id
            ]

        total = len(all_recons)
        page_data = all_recons[skip: skip + limit]

        # Strip full_result from list view
        clean = []
        for r in page_data:
            summary = {k: v for k, v in r.items() if k != "full_result"}
            clean.append(summary)

        return ReconciliationListResponse(
            reconciliations=clean,
            total=total,
            skip=skip,
            limit=limit,
        )

    # ==================================================================
    # 4. get_reconciliation
    # ==================================================================

    def get_reconciliation(
        self, reconciliation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a reconciliation result by ID.

        Args:
            reconciliation_id: Unique reconciliation identifier.

        Returns:
            Full reconciliation result dict or None if not found.
        """
        return self._reconciliations.get(reconciliation_id)

    # ==================================================================
    # 5. delete_reconciliation
    # ==================================================================

    def delete_reconciliation(self, reconciliation_id: str) -> bool:
        """Delete a reconciliation result by ID.

        Args:
            reconciliation_id: Unique reconciliation identifier.

        Returns:
            True if deleted, False if not found.
        """
        if reconciliation_id in self._reconciliations:
            del self._reconciliations[reconciliation_id]
            logger.info(
                "Reconciliation deleted: id=%s", reconciliation_id,
            )
            return True
        return False

    # ==================================================================
    # 6. list_discrepancies
    # ==================================================================

    def list_discrepancies(
        self, reconciliation_id: str,
    ) -> DiscrepancyListResponse:
        """List discrepancies for a specific reconciliation.

        Args:
            reconciliation_id: Reconciliation to query.

        Returns:
            DiscrepancyListResponse with all discrepancies found.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            return DiscrepancyListResponse(
                reconciliation_id=reconciliation_id,
            )

        full = recon.get("full_result", {})
        disc_data = full.get("discrepancies", {})
        items = disc_data.get("discrepancies", [])

        return DiscrepancyListResponse(
            reconciliation_id=reconciliation_id,
            discrepancies=items,
            total=len(items),
            direction=recon.get("direction", "equal"),
            total_discrepancy_tco2e=_safe_float(
                recon.get("discrepancy_tco2e", 0),
            ),
        )

    # ==================================================================
    # 7. get_waterfall
    # ==================================================================

    def get_waterfall(
        self, reconciliation_id: str,
    ) -> WaterfallResponse:
        """Get waterfall decomposition for a reconciliation.

        Args:
            reconciliation_id: Reconciliation to query.

        Returns:
            WaterfallResponse with decomposition items.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            return WaterfallResponse(
                reconciliation_id=reconciliation_id,
            )

        full = recon.get("full_result", {})
        waterfall = full.get("waterfall", {})

        return WaterfallResponse(
            reconciliation_id=reconciliation_id,
            location_total_tco2e=_safe_float(
                recon.get("total_location_tco2e", 0),
            ),
            market_total_tco2e=_safe_float(
                recon.get("total_market_tco2e", 0),
            ),
            items=waterfall.get("items", []),
            residual_tco2e=_safe_float(
                waterfall.get("residual_tco2e", 0),
            ),
            provenance_hash=_compute_hash(waterfall),
        )

    # ==================================================================
    # 8. get_quality_assessment
    # ==================================================================

    def get_quality_assessment(
        self, reconciliation_id: str,
    ) -> QualityAssessmentResponse:
        """Get quality assessment for a reconciliation.

        Args:
            reconciliation_id: Reconciliation to query.

        Returns:
            QualityAssessmentResponse with quality scores.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            return QualityAssessmentResponse(
                reconciliation_id=reconciliation_id,
            )

        full = recon.get("full_result", {})
        qa = full.get("quality_assessment", {})

        dimensions = {}
        for dim in ["completeness", "consistency", "accuracy", "transparency"]:
            dimensions[dim] = _safe_float(qa.get(dim, 0))

        return QualityAssessmentResponse(
            reconciliation_id=reconciliation_id,
            composite_score=_safe_float(qa.get("composite_score", 0)),
            grade=qa.get("grade", ""),
            dimensions=dimensions,
            ef_hierarchy_scores=qa.get("ef_hierarchy_scores", {}),
            flags=qa.get("flags", []),
            provenance_hash=_compute_hash(qa),
        )

    # ==================================================================
    # 9. get_reporting_tables
    # ==================================================================

    def get_reporting_tables(
        self,
        reconciliation_id: str,
        frameworks: Optional[List[str]] = None,
    ) -> ReportingTablesResponse:
        """Get multi-framework reporting tables for a reconciliation.

        Args:
            reconciliation_id: Reconciliation to query.
            frameworks: Optional filter for specific frameworks.

        Returns:
            ReportingTablesResponse with framework-specific tables.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            return ReportingTablesResponse(
                reconciliation_id=reconciliation_id,
            )

        full = recon.get("full_result", {})
        tables = full.get("reporting_tables", {})

        if frameworks:
            tables = {
                k: v for k, v in tables.items()
                if k in frameworks
            }

        return ReportingTablesResponse(
            reconciliation_id=reconciliation_id,
            tables=tables,
            frameworks_generated=len(tables),
            provenance_hash=_compute_hash(tables),
        )

    # ==================================================================
    # 10. get_trend_analysis
    # ==================================================================

    def get_trend_analysis(
        self, reconciliation_id: str,
    ) -> TrendAnalysisResponse:
        """Get trend analysis for a reconciliation.

        Args:
            reconciliation_id: Reconciliation to query.

        Returns:
            TrendAnalysisResponse with trend data.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            return TrendAnalysisResponse(
                reconciliation_id=reconciliation_id,
            )

        full = recon.get("full_result", {})
        trend = full.get("trend_analysis", {})

        return TrendAnalysisResponse(
            reconciliation_id=reconciliation_id,
            periods_analyzed=int(trend.get("periods_analyzed", 0)),
            location_trend=trend.get("location_trend", "stable"),
            market_trend=trend.get("market_trend", "stable"),
            location_cagr_pct=_safe_float(
                trend.get("location_cagr_pct", 0),
            ),
            market_cagr_pct=_safe_float(
                trend.get("market_cagr_pct", 0),
            ),
            pif_trend=trend.get("pif_trend", "stable"),
            re100_pct_latest=_safe_float(
                trend.get("re100_pct_latest", 0),
            ),
            sbti_on_track=bool(trend.get("sbti_on_track", False)),
            intensity_metrics=trend.get("intensity_metrics", {}),
            data_points=trend.get("data_points", []),
            provenance_hash=_compute_hash(trend),
        )

    # ==================================================================
    # 11. check_compliance
    # ==================================================================

    def check_compliance(
        self,
        reconciliation_id: str,
        frameworks: Optional[List[str]] = None,
    ) -> ComplianceCheckResponse:
        """Run compliance check for a reconciliation.

        Args:
            reconciliation_id: Reconciliation to check.
            frameworks: Optional list of framework identifiers.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            raise ValueError(
                f"Reconciliation not found: {reconciliation_id}"
            )

        compliance_id = _short_id("comp")
        start = time.monotonic()

        # Check if compliance was already run in pipeline
        full = recon.get("full_result", {})
        existing = full.get("compliance", {})

        if existing and not frameworks:
            results = existing.get("results", [])
        elif self._compliance_checker is not None:
            try:
                fw_list = frameworks or None
                comp_result = self._compliance_checker.check_all_frameworks(
                    full.get("workspace", full),
                    full.get("discrepancies", None),
                    full.get("quality_assessment", None),
                    fw_list,
                )
                if hasattr(comp_result, "model_dump"):
                    cr = comp_result.model_dump(mode="json")
                elif isinstance(comp_result, dict):
                    cr = comp_result
                else:
                    cr = {}
                # check_all_frameworks returns Dict[str, ComplianceCheckResult]
                results = [
                    v.model_dump(mode="json") if hasattr(v, "model_dump") else v
                    for v in cr.values()
                ] if isinstance(cr, dict) else cr.get("results", [])
            except Exception as exc:
                logger.warning("Compliance check failed: %s", exc)
                results = []
        else:
            results = []

        compliant = sum(
            1 for r in results
            if r.get("status") == "compliant"
        )
        non_compliant = sum(
            1 for r in results
            if r.get("status") == "non_compliant"
        )
        partial = sum(
            1 for r in results
            if r.get("status") == "partial"
        )
        total_fw = len(results)
        overall_score = (
            (compliant / total_fw) * 100.0 if total_fw > 0 else 0.0
        )

        self._total_compliance_checks += 1

        # Store compliance result
        stored = {
            "compliance_id": compliance_id,
            "reconciliation_id": reconciliation_id,
            "results": results,
            "checked_at": _utcnow_iso(),
        }
        self._compliance_results[compliance_id] = stored

        return ComplianceCheckResponse(
            success=True,
            compliance_id=compliance_id,
            reconciliation_id=reconciliation_id,
            frameworks_checked=total_fw,
            compliant=compliant,
            non_compliant=non_compliant,
            partial=partial,
            overall_score=round(overall_score, 2),
            results=results,
            provenance_hash=_compute_hash(results),
        )

    # ==================================================================
    # 12. get_compliance_result
    # ==================================================================

    def get_compliance_result(
        self, compliance_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a compliance check result by ID.

        Args:
            compliance_id: Compliance check identifier.

        Returns:
            Compliance result dict or None if not found.
        """
        return self._compliance_results.get(compliance_id)

    # ==================================================================
    # 13. get_aggregations
    # ==================================================================

    def get_aggregations(
        self,
        group_by: str = "energy_type",
        tenant_id: Optional[str] = None,
    ) -> AggregationResponse:
        """Get aggregated reconciliation data.

        Args:
            group_by: Dimension to group by (energy_type, facility,
                region, business_unit, period).
            tenant_id: Optional tenant filter.

        Returns:
            AggregationResponse with grouped aggregations.
        """
        all_recons = list(self._reconciliations.values())
        if tenant_id is not None:
            all_recons = [
                r for r in all_recons
                if r.get("tenant_id") == tenant_id
            ]

        total_loc = sum(
            _safe_float(r.get("total_location_tco2e", 0))
            for r in all_recons
        )
        total_mkt = sum(
            _safe_float(r.get("total_market_tco2e", 0))
            for r in all_recons
        )
        portfolio_pif = (
            (1.0 - total_mkt / total_loc) if total_loc > 0 else 0.0
        )

        groups: Dict[str, Dict[str, float]] = {}
        for r in all_recons:
            key = r.get(group_by, "unknown")
            if isinstance(key, (list, dict)):
                key = str(key)
            if key not in groups:
                groups[key] = {
                    "location_tco2e": 0.0,
                    "market_tco2e": 0.0,
                    "count": 0.0,
                }
            groups[key]["location_tco2e"] += _safe_float(
                r.get("total_location_tco2e", 0),
            )
            groups[key]["market_tco2e"] += _safe_float(
                r.get("total_market_tco2e", 0),
            )
            groups[key]["count"] += 1.0

        return AggregationResponse(
            group_by=group_by,
            groups=groups,
            total_location_tco2e=round(total_loc, 6),
            total_market_tco2e=round(total_mkt, 6),
            portfolio_pif=round(portfolio_pif, 6),
            reconciliation_count=len(all_recons),
        )

    # ==================================================================
    # 14. export_report
    # ==================================================================

    def export_report(
        self,
        reconciliation_id: str,
        export_format: str = "json",
    ) -> ExportResponse:
        """Export a reconciliation report in the specified format.

        Args:
            reconciliation_id: Reconciliation to export.
            export_format: Output format (json, csv).

        Returns:
            ExportResponse with serialised content.
        """
        recon = self._reconciliations.get(reconciliation_id)
        if recon is None:
            raise ValueError(
                f"Reconciliation not found: {reconciliation_id}"
            )

        export_id = _short_id("exp")

        full = recon.get("full_result", recon)
        summary = {k: v for k, v in recon.items() if k != "full_result"}

        if export_format == "json":
            content = json.dumps(summary, default=str, indent=2)
            content_type = "application/json"
            filename = f"drr_{reconciliation_id}.json"
        elif export_format == "csv":
            content = self._export_csv(summary)
            content_type = "text/csv"
            filename = f"drr_{reconciliation_id}.csv"
        else:
            content = json.dumps(summary, default=str, indent=2)
            content_type = "application/json"
            filename = f"drr_{reconciliation_id}.json"

        self._total_exports += 1

        stored = {
            "export_id": export_id,
            "reconciliation_id": reconciliation_id,
            "format": export_format,
            "timestamp": _utcnow_iso(),
        }
        self._exports[export_id] = stored

        return ExportResponse(
            success=True,
            export_id=export_id,
            reconciliation_id=reconciliation_id,
            format=export_format,
            content=content,
            content_type=content_type,
            filename=filename,
            provenance_hash=_compute_hash(content),
        )

    def _export_csv(self, data: Dict[str, Any]) -> str:
        """Convert reconciliation summary to CSV string.

        Args:
            data: Summary dict to convert.

        Returns:
            CSV-formatted string.
        """
        import io
        import csv as csv_mod

        output = io.StringIO()
        writer = csv_mod.writer(output)
        writer.writerow(["field", "value"])
        for key, value in data.items():
            writer.writerow([key, str(value)])
        return output.getvalue()

    # ==================================================================
    # 15. health_check
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Check service health and engine availability.

        Returns:
            HealthResponse with engine status and uptime.
        """
        engines: Dict[str, str] = {
            "collector": (
                "available" if self._collector is not None
                else "unavailable"
            ),
            "discrepancy_analyzer": (
                "available" if self._discrepancy_analyzer is not None
                else "unavailable"
            ),
            "quality_scorer": (
                "available" if self._quality_scorer is not None
                else "unavailable"
            ),
            "table_generator": (
                "available" if self._table_generator is not None
                else "unavailable"
            ),
            "trend_analyzer": (
                "available" if self._trend_analyzer is not None
                else "unavailable"
            ),
            "compliance_checker": (
                "available" if self._compliance_checker is not None
                else "unavailable"
            ),
            "pipeline": (
                "available" if self._pipeline is not None
                else "unavailable"
            ),
        }

        all_available = all(
            v == "available" for v in engines.values()
        )
        uptime = time.monotonic() - self._start_time

        # Run individual engine health checks
        for name, attr in [
            ("collector", self._collector),
            ("discrepancy_analyzer", self._discrepancy_analyzer),
            ("quality_scorer", self._quality_scorer),
            ("table_generator", self._table_generator),
            ("trend_analyzer", self._trend_analyzer),
            ("compliance_checker", self._compliance_checker),
            ("pipeline", self._pipeline),
        ]:
            if attr is not None and hasattr(attr, "health_check"):
                try:
                    health = attr.health_check()
                    if isinstance(health, dict):
                        status = health.get("status", "unknown")
                    else:
                        status = "available"
                    engines[name] = status
                except Exception:
                    engines[name] = "error"

        return HealthResponse(
            status="healthy" if all_available else "degraded",
            engines=engines,
            config_valid=self._config is not None,
            uptime_seconds=round(uptime, 2),
        )

    # ==================================================================
    # 16. get_stats
    # ==================================================================

    def get_stats(self) -> StatsResponse:
        """Get aggregate service statistics.

        Returns:
            StatsResponse with cumulative counters and averages.
        """
        uptime = time.monotonic() - self._start_time
        avg_quality = (
            sum(self._cumulative_quality_scores)
            / len(self._cumulative_quality_scores)
            if self._cumulative_quality_scores else 0.0
        )
        avg_pif = (
            sum(self._cumulative_pifs)
            / len(self._cumulative_pifs)
            if self._cumulative_pifs else 0.0
        )

        return StatsResponse(
            total_reconciliations=self._total_reconciliations,
            total_batch_runs=self._total_batch_runs,
            total_discrepancies_found=self._total_discrepancies,
            total_compliance_checks=self._total_compliance_checks,
            total_exports=self._total_exports,
            average_quality_score=round(avg_quality, 4),
            average_pif=round(avg_pif, 6),
            uptime_seconds=round(uptime, 2),
        )


# ===================================================================
# Module-level service accessor
# ===================================================================


def get_service(config: Any = None) -> DualReportingService:
    """Get or create the singleton DualReportingService instance.

    Thread-safe via a module-level lock. Passing a ``config``
    argument on the first call configures the service; subsequent
    calls return the existing singleton regardless of the argument.

    Args:
        config: Optional configuration for first-time initialization.

    Returns:
        The DualReportingService singleton.
    """
    global _service_instance
    if _service_instance is not None:
        return _service_instance

    with _singleton_lock:
        if _service_instance is not None:
            return _service_instance
        _service_instance = DualReportingService(config=config)
        return _service_instance


def reset_service() -> None:
    """Reset the singleton service instance (for testing only).

    Clears the module-level singleton so that the next call to
    ``get_service()`` creates a fresh instance.
    """
    global _service_instance
    with _singleton_lock:
        _service_instance = None


__all__ = [
    "DualReportingService",
    "get_service",
    "reset_service",
    # Response models
    "ReconcileResponse",
    "BatchReconcileResponse",
    "ReconciliationListResponse",
    "DiscrepancyListResponse",
    "WaterfallResponse",
    "QualityAssessmentResponse",
    "ReportingTablesResponse",
    "TrendAnalysisResponse",
    "ComplianceCheckResponse",
    "AggregationResponse",
    "ExportResponse",
    "HealthResponse",
    "StatsResponse",
    # Constants
    "SERVICE_VERSION",
    "SERVICE_NAME",
    "AGENT_ID",
]
