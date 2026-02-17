# -*- coding: utf-8 -*-
"""
GL-DATA-X-022: GreenLang Validation Rule Engine SDK
=====================================================

This package provides validation rule registration, rule composition,
rule evaluation, conflict detection, rule pack management, validation
reporting, and end-to-end pipeline orchestration SDK for the GreenLang
framework. It supports:

- Rule registry with multi-type support (range_check, format_validation,
  cross_field, regex, lookup, threshold, conditional, temporal,
  referential, custom, regulatory, completeness, uniqueness, consistency)
  and severity classification (critical, error, warning, info, debug)
- Rule composition with compound rules (AND/OR/NOT) supporting
  configurable nesting depth and short-circuit evaluation
- Rule evaluation with deterministic execution, pass/warn/fail
  thresholds, batch processing, and per-rule provenance tracking
- Conflict detection for contradictions, overlaps, subsumptions,
  redundancies, circular dependencies, priority ambiguities, scope
  collisions, and temporal conflicts between rules within a rule set
- Rule packs for versioned bundles of rule sets aligned to regulatory
  frameworks (GHG Protocol, CSRD/ESRS, TCFD, CDP, SBTi, ISO 14064,
  EUDR, custom) with import/export and apply operations
- Validation reporting with JSON, HTML, PDF, CSV, Markdown, and XML
  output formats for compliance audit trails and stakeholder
  communication
- End-to-end pipeline orchestration chaining all 7 engines with
  configurable stages and short-circuit on failure
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics with gl_vre_ prefix for observability
- FastAPI REST API with 20 endpoints at /api/v1/validation-rules
- Thread-safe configuration with GL_VRE_ env prefix

Key Components:
    - config: ValidationRuleEngineConfig with GL_VRE_ env prefix
    - rule_registry: Rule registration, lookup, and lifecycle engine
    - rule_composer: Compound rule composition engine (AND/OR/NOT)
    - rule_evaluator: Rule evaluation and scoring engine
    - conflict_detector: Rule conflict detection and analysis engine
    - rule_pack: Rule pack management engine (import/export/apply)
    - validation_reporter: Validation report generation engine
    - validation_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_vre_ prefix
    - api: FastAPI HTTP service with 20 endpoints
    - setup: ValidationRuleEngineService facade

Example:
    >>> from greenlang.validation_rule_engine import ValidationRuleEngineService
    >>> service = ValidationRuleEngineService()
    >>> result = service.register_rule(
    ...     name="co2e_range_check",
    ...     rule_type="range_check",
    ...     severity="error",
    ...     field="co2e",
    ...     min_value=0.0,
    ...     max_value=1_000_000.0,
    ... )
    >>> print(result.rule_id, result.status)
    co2e_range_check active

Agent ID: GL-DATA-X-022
Agent Name: Validation Rule Engine
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-022"
__agent_name__ = "Validation Rule Engine"

# SDK availability flag
VALIDATION_RULE_ENGINE_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.validation_rule_engine.config import (
    ValidationRuleEngineConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.validation_rule_engine.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.validation_rule_engine.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    vre_rules_registered_total,
    vre_rule_sets_created_total,
    vre_evaluations_total,
    vre_evaluation_failures_total,
    vre_conflicts_detected_total,
    vre_reports_generated_total,
    vre_rules_per_set,
    vre_evaluation_duration_seconds,
    vre_processing_duration_seconds,
    vre_active_rules,
    vre_active_rule_sets,
    vre_pass_rate,
    # Helper functions
    record_rule_registered,
    record_rule_set_created,
    record_evaluation,
    record_evaluation_failure,
    record_conflict_detected,
    record_report_generated,
    observe_rules_per_set,
    observe_evaluation_duration,
    observe_processing_duration,
    set_active_rules,
    set_active_rule_sets,
    set_pass_rate,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.validation_rule_engine.rule_registry import RuleRegistryEngine
except ImportError:
    RuleRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.rule_composer import RuleComposerEngine
except ImportError:
    RuleComposerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.rule_evaluator import RuleEvaluatorEngine
except ImportError:
    RuleEvaluatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.conflict_detector import ConflictDetectorEngine
except ImportError:
    ConflictDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.rule_pack import RulePackEngine
except ImportError:
    RulePackEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.validation_reporter import ValidationReporterEngine
except ImportError:
    ValidationReporterEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.validation_pipeline import ValidationPipelineEngine
except ImportError:
    ValidationPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.validation_rule_engine.setup import (
    ValidationRuleEngineService,
    configure_validation_rule_engine,
    get_validation_rule_engine,
    get_router,
    # Models
    RuleResponse,
    RuleSetResponse,
    CompoundRuleResponse,
    EvaluationResponse,
    BatchEvaluationResponse,
    ConflictReportResponse,
    ValidationReportResponse,
    RulePackResponse,
    PipelineResultResponse,
    ValidationStatisticsResponse,
)

# Backwards-compatible aliases for pre-existing __init__.py references
ConflictDetectionResponse = ConflictReportResponse
PackApplyResponse = RulePackResponse
ReportResponse = ValidationReportResponse
ValidationRuleStatisticsResponse = ValidationStatisticsResponse

# ---------------------------------------------------------------------------
# Layer 1 re-exports from data_quality_profiler
# ---------------------------------------------------------------------------
try:
    from greenlang.data_quality_profiler.quality_rule_engine import (
        QualityRuleEngine,
    )
except ImportError:
    QualityRuleEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.validity_checker import (
        ValidityChecker,
    )
except ImportError:
    ValidityChecker = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.models import (
        QualityDimension,
        RuleType,
    )
except ImportError:
    QualityDimension = None  # type: ignore[assignment, misc]
    RuleType = None  # type: ignore[assignment, misc]

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "VALIDATION_RULE_ENGINE_SDK_AVAILABLE",
    # Configuration
    "ValidationRuleEngineConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "vre_rules_registered_total",
    "vre_rule_sets_created_total",
    "vre_evaluations_total",
    "vre_evaluation_failures_total",
    "vre_conflicts_detected_total",
    "vre_reports_generated_total",
    "vre_rules_per_set",
    "vre_evaluation_duration_seconds",
    "vre_processing_duration_seconds",
    "vre_active_rules",
    "vre_active_rule_sets",
    "vre_pass_rate",
    # Metric helper functions
    "record_rule_registered",
    "record_rule_set_created",
    "record_evaluation",
    "record_evaluation_failure",
    "record_conflict_detected",
    "record_report_generated",
    "observe_rules_per_set",
    "observe_evaluation_duration",
    "observe_processing_duration",
    "set_active_rules",
    "set_active_rule_sets",
    "set_pass_rate",
    # Core engines (Layer 2)
    "RuleRegistryEngine",
    "RuleComposerEngine",
    "RuleEvaluatorEngine",
    "ConflictDetectorEngine",
    "RulePackEngine",
    "ValidationReporterEngine",
    "ValidationPipelineEngine",
    # Service setup facade
    "ValidationRuleEngineService",
    "configure_validation_rule_engine",
    "get_validation_rule_engine",
    "get_router",
    # Response models
    "RuleResponse",
    "RuleSetResponse",
    "CompoundRuleResponse",
    "EvaluationResponse",
    "BatchEvaluationResponse",
    "ConflictReportResponse",
    "ValidationReportResponse",
    "RulePackResponse",
    "PipelineResultResponse",
    "ValidationStatisticsResponse",
    # Backwards-compatible aliases
    "ConflictDetectionResponse",
    "PackApplyResponse",
    "ReportResponse",
    "ValidationRuleStatisticsResponse",
    # Layer 1 re-exports (data_quality_profiler)
    "QualityRuleEngine",
    "ValidityChecker",
    "QualityDimension",
    "RuleType",
]
