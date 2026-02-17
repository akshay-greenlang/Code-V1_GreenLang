# -*- coding: utf-8 -*-
"""
GL-DATA-X-015: GreenLang Missing Value Imputer Agent SDK
=========================================================

This package provides intelligent missing value imputation for GreenLang
sustainability datasets. It supports:

- Missingness analysis (MCAR/MAR/MNAR classification, pattern detection,
  per-column statistics, correlation matrices)
- Statistical imputation (mean, median, mode, hot deck)
- ML-based imputation (KNN, random forest, gradient boosting, MICE,
  matrix factorization, regression)
- Time-series imputation (linear interpolation, spline interpolation,
  seasonal decomposition, LOCF, NOCB)
- Rule-based imputation (conditional rules, lookup tables, regulatory
  defaults)
- Imputation validation (KS test, chi-square, plausibility range,
  distribution preservation, cross-validation)
- End-to-end imputation pipeline orchestration (analyze, strategize,
  impute, validate, document)
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_MVI_ env prefix

Key Components:
    - config: MissingValueImputerConfig with GL_MVI_ env prefix
    - missingness_analyzer: Missingness analysis engine
    - statistical_imputer: Statistical imputation engine
    - ml_imputer: ML-based imputation engine
    - rule_based_imputer: Rule-based imputation engine
    - time_series_imputer: Time-series imputation engine
    - validation_engine: Imputation validation engine
    - imputation_pipeline: End-to-end pipeline engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: MissingValueImputerService facade

Example:
    >>> from greenlang.missing_value_imputer import MissingValueImputerService
    >>> service = MissingValueImputerService()
    >>> result = service.analyze_missingness(
    ...     records=[{"a": 1, "b": None}, {"a": None, "b": 2}],
    ... )
    >>> print(result.columns_with_missing, result.overall_missing_pct)
    2 0.5

Agent ID: GL-DATA-X-015
Agent Name: Missing Value Imputer Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-015"
__agent_name__ = "Missing Value Imputer Agent"

# SDK availability flag
MISSING_VALUE_IMPUTER_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration (4 items)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.config import (
    MissingValueImputerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance (1 item)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics (25 items)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    mvi_jobs_processed_total,
    mvi_values_imputed_total,
    mvi_analyses_completed_total,
    mvi_validations_passed_total,
    mvi_rules_evaluated_total,
    mvi_strategies_selected_total,
    mvi_processing_errors_total,
    mvi_confidence_score,
    mvi_processing_duration_seconds,
    mvi_completeness_improvement,
    mvi_active_jobs,
    mvi_total_missing_detected,
    # Helper functions
    inc_jobs,
    inc_values_imputed,
    inc_analyses,
    inc_validations,
    inc_rules_evaluated,
    inc_strategies_selected,
    inc_errors,
    observe_confidence,
    observe_duration,
    observe_completeness_improvement,
    set_active_jobs,
    set_total_missing_detected,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback (7)
# ---------------------------------------------------------------------------
try:
    from greenlang.missing_value_imputer.missingness_analyzer import (
        MissingnessAnalyzerEngine,
    )
except ImportError:
    MissingnessAnalyzerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.statistical_imputer import (
        StatisticalImputerEngine,
    )
except ImportError:
    StatisticalImputerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.ml_imputer import MLImputerEngine
except ImportError:
    MLImputerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.rule_based_imputer import (
        RuleBasedImputerEngine,
    )
except ImportError:
    RuleBasedImputerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.time_series_imputer import (
        TimeSeriesImputerEngine,
    )
except ImportError:
    TimeSeriesImputerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.validation_engine import (
        ValidationEngine,
    )
except ImportError:
    ValidationEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.imputation_pipeline import (
        ImputationPipelineEngine,
    )
except ImportError:
    ImputationPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from data_quality_profiler (3 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_quality_profiler.models import (
        QualityDimension as L1QualityDimension,
        DataType as L1DataType,
    )
    QualityDimension = L1QualityDimension
    DataType = L1DataType
except ImportError:
    QualityDimension = None  # type: ignore[assignment, misc]
    DataType = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.completeness_analyzer import (
        CompletenessAnalyzer as L1CompletenessAnalyzer,
    )
    CompletenessAnalyzer = L1CompletenessAnalyzer
except ImportError:
    CompletenessAnalyzer = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Enumerations (12 items)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.models import (
    MissingnessType,
    ImputationStrategy,
    ImputationStatus,
    ConfidenceLevel,
    DataColumnType,
    ValidationMethod,
    RuleConditionType,
    RulePriority,
    ReportFormat,
    PipelineStage,
    PatternType,
    TimeSeriesFrequency,
)

# ---------------------------------------------------------------------------
# SDK data models (20 items)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.models import (
    MissingnessPattern,
    ColumnAnalysis,
    MissingnessReport,
    ImputationRule,
    RuleCondition,
    LookupTable,
    LookupEntry,
    ImputedValue,
    ImputationResult,
    ImputationBatch,
    ValidationResult,
    ValidationReport,
    ImputationTemplate,
    PipelineConfig,
    PipelineResult,
    ImputationJobConfig,
    ImputationStatistics,
    TimeSeriesConfig,
    MLModelConfig,
    StrategySelection,
)

# ---------------------------------------------------------------------------
# Request models (8 items)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.models import (
    CreateJobRequest,
    AnalyzeMissingnessRequest,
    ImputeValuesRequest,
    BatchImputeRequest,
    ValidateRequest,
    CreateRuleRequest,
    CreateTemplateRequest,
    RunPipelineRequest,
)

# ---------------------------------------------------------------------------
# Service setup facade and response models (4 + 8 items)
# ---------------------------------------------------------------------------
from greenlang.missing_value_imputer.setup import (
    MissingValueImputerService,
    configure_missing_value_imputer,
    get_missing_value_imputer,
    get_router,
    # Response models
    AnalysisResponse,
    ImputationResponse,
    BatchImputationResponse,
    ValidationResponse,
    RuleResponse,
    TemplateResponse,
    PipelineResponse,
    StatsResponse,
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
try:
    from greenlang.missing_value_imputer.api.router import router
except ImportError:
    router = None  # type: ignore[assignment]

__all__ = [
    # -------------------------------------------------------------------------
    # Version and identity
    # -------------------------------------------------------------------------
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # -------------------------------------------------------------------------
    # SDK flag
    # -------------------------------------------------------------------------
    "MISSING_VALUE_IMPUTER_SDK_AVAILABLE",
    # -------------------------------------------------------------------------
    # Configuration (4)
    # -------------------------------------------------------------------------
    "MissingValueImputerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -------------------------------------------------------------------------
    # Provenance (1)
    # -------------------------------------------------------------------------
    "ProvenanceTracker",
    # -------------------------------------------------------------------------
    # Metrics flag
    # -------------------------------------------------------------------------
    "PROMETHEUS_AVAILABLE",
    # -------------------------------------------------------------------------
    # Metric objects (12)
    # -------------------------------------------------------------------------
    "mvi_jobs_processed_total",
    "mvi_values_imputed_total",
    "mvi_analyses_completed_total",
    "mvi_validations_passed_total",
    "mvi_rules_evaluated_total",
    "mvi_strategies_selected_total",
    "mvi_processing_errors_total",
    "mvi_confidence_score",
    "mvi_processing_duration_seconds",
    "mvi_completeness_improvement",
    "mvi_active_jobs",
    "mvi_total_missing_detected",
    # -------------------------------------------------------------------------
    # Metric helper functions (12)
    # -------------------------------------------------------------------------
    "inc_jobs",
    "inc_values_imputed",
    "inc_analyses",
    "inc_validations",
    "inc_rules_evaluated",
    "inc_strategies_selected",
    "inc_errors",
    "observe_confidence",
    "observe_duration",
    "observe_completeness_improvement",
    "set_active_jobs",
    "set_total_missing_detected",
    # -------------------------------------------------------------------------
    # Core engines (Layer 2) (7)
    # -------------------------------------------------------------------------
    "MissingnessAnalyzerEngine",
    "StatisticalImputerEngine",
    "MLImputerEngine",
    "RuleBasedImputerEngine",
    "TimeSeriesImputerEngine",
    "ValidationEngine",
    "ImputationPipelineEngine",
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (3)
    # -------------------------------------------------------------------------
    "QualityDimension",
    "DataType",
    "CompletenessAnalyzer",
    # -------------------------------------------------------------------------
    # Enumerations (12)
    # -------------------------------------------------------------------------
    "MissingnessType",
    "ImputationStrategy",
    "ImputationStatus",
    "ConfidenceLevel",
    "DataColumnType",
    "ValidationMethod",
    "RuleConditionType",
    "RulePriority",
    "ReportFormat",
    "PipelineStage",
    "PatternType",
    "TimeSeriesFrequency",
    # -------------------------------------------------------------------------
    # SDK data models (20)
    # -------------------------------------------------------------------------
    "MissingnessPattern",
    "ColumnAnalysis",
    "MissingnessReport",
    "ImputationRule",
    "RuleCondition",
    "LookupTable",
    "LookupEntry",
    "ImputedValue",
    "ImputationResult",
    "ImputationBatch",
    "ValidationResult",
    "ValidationReport",
    "ImputationTemplate",
    "PipelineConfig",
    "PipelineResult",
    "ImputationJobConfig",
    "ImputationStatistics",
    "TimeSeriesConfig",
    "MLModelConfig",
    "StrategySelection",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "CreateJobRequest",
    "AnalyzeMissingnessRequest",
    "ImputeValuesRequest",
    "BatchImputeRequest",
    "ValidateRequest",
    "CreateRuleRequest",
    "CreateTemplateRequest",
    "RunPipelineRequest",
    # -------------------------------------------------------------------------
    # Service setup facade (4)
    # -------------------------------------------------------------------------
    "MissingValueImputerService",
    "configure_missing_value_imputer",
    "get_missing_value_imputer",
    "get_router",
    # -------------------------------------------------------------------------
    # Response models (8)
    # -------------------------------------------------------------------------
    "AnalysisResponse",
    "ImputationResponse",
    "BatchImputationResponse",
    "ValidationResponse",
    "RuleResponse",
    "TemplateResponse",
    "PipelineResponse",
    "StatsResponse",
    # -------------------------------------------------------------------------
    # Router
    # -------------------------------------------------------------------------
    "router",
]
