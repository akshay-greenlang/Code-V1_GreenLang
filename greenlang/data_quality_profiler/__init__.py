# -*- coding: utf-8 -*-
"""
GL-DATA-X-013: GreenLang Data Quality Profiler Agent SDK
=========================================================

This package provides dataset profiling, quality assessment,
anomaly detection, freshness checking, rule-based validation,
quality gates, trend analysis, report generation, and provenance
tracking SDK for the GreenLang framework. It supports:

- Dataset profiling with column-level statistics (types, nulls, cardinality)
- Quality assessment across 6 dimensions (completeness, validity,
  consistency, timeliness, uniqueness, accuracy)
- Quality level classification (EXCELLENT, GOOD, FAIR, POOR, CRITICAL)
- Anomaly detection (IQR, z-score, percentile methods)
- Freshness/timeliness SLA checking
- Rule-based validation (not_null, unique, range, regex, custom)
- Quality gate evaluation with pass/fail/warn outcomes
- Quality trend analysis over time
- Report generation (JSON, Markdown, HTML, text, CSV)
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_DQ_ env prefix

Key Components:
    - config: DataQualityProfilerConfig with GL_DQ_ env prefix
    - dataset_profiler: Dataset profiling engine
    - quality_assessor: Quality assessment engine
    - anomaly_detector: Anomaly detection engine
    - freshness_checker: Freshness checking engine
    - dq_rule_engine: Rule-based validation engine
    - quality_gate: Quality gate evaluation engine
    - dq_report_generator: Report generation engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: DataQualityProfilerService facade

Example:
    >>> from greenlang.data_quality_profiler import DataQualityProfilerService
    >>> service = DataQualityProfilerService()
    >>> profile = service.profile_dataset(
    ...     data=[{"name": "Alice", "age": 30, "email": "alice@example.com"}],
    ...     dataset_name="employees",
    ... )
    >>> print(profile.row_count, profile.completeness_score)
    1 1.0

Agent ID: GL-DATA-X-013
Agent Name: Data Quality Profiler Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-013"
__agent_name__ = "Data Quality Profiler Agent"

# SDK availability flag
DATA_QUALITY_PROFILER_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.data_quality_profiler.config import (
    DataQualityProfilerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.data_quality_profiler.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.data_quality_profiler.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    dq_datasets_profiled_total,
    dq_columns_profiled_total,
    dq_assessments_completed_total,
    dq_rules_evaluated_total,
    dq_anomalies_detected_total,
    dq_gates_evaluated_total,
    dq_overall_quality_score,
    dq_processing_duration_seconds,
    dq_active_profiles,
    dq_total_issues_found,
    dq_processing_errors_total,
    dq_freshness_checks_total,
    # Helper functions
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

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.data_quality_profiler.dataset_profiler import DatasetProfilerEngine
except ImportError:
    DatasetProfilerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.quality_assessor import QualityAssessorEngine
except ImportError:
    QualityAssessorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.anomaly_detector import AnomalyDetectorEngine
except ImportError:
    AnomalyDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.freshness_checker import FreshnessCheckerEngine
except ImportError:
    FreshnessCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.dq_rule_engine import DQRuleEngine
except ImportError:
    DQRuleEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.quality_gate import QualityGateEngine
except ImportError:
    QualityGateEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.dq_report_generator import DQReportGeneratorEngine
except ImportError:
    DQReportGeneratorEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.data_quality_profiler.setup import (
    DataQualityProfilerService,
    configure_data_quality_profiler,
    get_data_quality_profiler,
    get_router,
    # Models
    DatasetProfileResponse,
    QualityAssessmentResponse,
    ColumnProfileResponse,
    AnomalyDetectionResponse,
    FreshnessCheckResponse,
    QualityRuleResponse,
    QualityGateResponse,
    DataQualityProfilerStatisticsResponse,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "DATA_QUALITY_PROFILER_SDK_AVAILABLE",
    # Configuration
    "DataQualityProfilerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "dq_datasets_profiled_total",
    "dq_columns_profiled_total",
    "dq_assessments_completed_total",
    "dq_rules_evaluated_total",
    "dq_anomalies_detected_total",
    "dq_gates_evaluated_total",
    "dq_overall_quality_score",
    "dq_processing_duration_seconds",
    "dq_active_profiles",
    "dq_total_issues_found",
    "dq_processing_errors_total",
    "dq_freshness_checks_total",
    # Metric helper functions
    "record_profile",
    "record_column_profile",
    "record_assessment",
    "record_rule_evaluation",
    "record_anomaly",
    "record_gate_evaluation",
    "record_quality_score",
    "record_processing_duration",
    "update_active_profiles",
    "update_total_issues",
    "record_processing_error",
    "record_freshness_check",
    # Core engines (Layer 2)
    "DatasetProfilerEngine",
    "QualityAssessorEngine",
    "AnomalyDetectorEngine",
    "FreshnessCheckerEngine",
    "DQRuleEngine",
    "QualityGateEngine",
    "DQReportGeneratorEngine",
    # Service setup facade
    "DataQualityProfilerService",
    "configure_data_quality_profiler",
    "get_data_quality_profiler",
    "get_router",
    # Response models
    "DatasetProfileResponse",
    "QualityAssessmentResponse",
    "ColumnProfileResponse",
    "AnomalyDetectionResponse",
    "FreshnessCheckResponse",
    "QualityRuleResponse",
    "QualityGateResponse",
    "DataQualityProfilerStatisticsResponse",
]
