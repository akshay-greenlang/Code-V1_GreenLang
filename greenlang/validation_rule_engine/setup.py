# -*- coding: utf-8 -*-
"""
Validation Rule Engine Service Setup - AGENT-DATA-019

Provides ``configure_validation_rule_engine(app)`` which wires up the
Validation Rule Engine Agent SDK (rule registry, rule composer, rule
evaluator, conflict detector, rule pack, validation reporter, validation
pipeline, provenance tracker) and mounts the REST API.

Also exposes ``get_validation_rule_engine()`` for programmatic access
and the ``ValidationRuleEngineService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.validation_rule_engine.setup import configure_validation_rule_engine
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_validation_rule_engine(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
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

from greenlang.validation_rule_engine.config import (
    ValidationRuleEngineConfig,
    get_config,
)
from greenlang.validation_rule_engine.metrics import (
    PROMETHEUS_AVAILABLE,
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
from greenlang.validation_rule_engine.provenance import ProvenanceTracker

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
# Optional engine imports (graceful degradation)
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


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class RuleResponse(BaseModel):
    """Validation rule registration / retrieval response.

    Attributes:
        rule_id: Unique validation rule identifier (UUID4).
        name: Human-readable rule name.
        rule_type: Classification of the rule (completeness, range,
            format, uniqueness, custom, freshness, cross_field,
            conditional, statistical, referential).
        column: Target column or field name for the rule.
        operator: Comparison operator (equals, not_equals, greater_than,
            less_than, greater_equal, less_equal, between, matches,
            contains, in_set, not_in_set, is_null).
        threshold: Primary threshold value for comparison.
        parameters: Additional rule-specific parameters (min, max,
            pattern, reference_dataset, etc.).
        severity: Rule severity level (critical, high, medium, low).
        status: Rule lifecycle status (draft, active, deprecated,
            archived).
        version: SemVer version string for rule versioning.
        description: Human-readable description of the rule purpose.
        tags: Key-value labels for filtering and categorization.
        metadata: Additional unstructured metadata.
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC last-update timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    rule_type: str = Field(default="range")
    column: str = Field(default="")
    operator: str = Field(default="between")
    threshold: Optional[Any] = Field(default=None)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    severity: str = Field(default="medium")
    status: str = Field(default="draft")
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class RuleSetResponse(BaseModel):
    """Rule set creation / retrieval response.

    Attributes:
        set_id: Unique rule set identifier (UUID4).
        name: Human-readable rule set name.
        description: Description of the rule set purpose.
        version: SemVer version string for rule set versioning.
        status: Rule set lifecycle status (draft, active, deprecated,
            archived).
        rule_count: Number of rules contained in this set.
        sla_thresholds: Dynamic pass/warn/fail thresholds for this set,
            e.g. ``{"pass": 0.95, "warn": 0.80}``.
        parent_set_id: Optional parent rule set ID for inheritance.
        tags: Key-value labels for filtering and categorization.
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC last-update timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    set_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: str = Field(default="")
    version: str = Field(default="1.0.0")
    status: str = Field(default="draft")
    rule_count: int = Field(default=0)
    sla_thresholds: Dict[str, float] = Field(default_factory=dict)
    parent_set_id: Optional[str] = Field(default=None)
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class CompoundRuleResponse(BaseModel):
    """Compound rule composition response.

    Attributes:
        compound_id: Unique compound rule identifier (UUID4).
        name: Human-readable compound rule name.
        operator: Logical operator used for composition (AND, OR, NOT).
        child_rule_ids: Ordered list of child rule IDs composing this
            compound rule.
        description: Human-readable description of the compound rule.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    compound_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    operator: str = Field(default="AND")
    child_rule_ids: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    provenance_hash: str = Field(default="")


class EvaluationResponse(BaseModel):
    """Rule evaluation execution result response.

    Attributes:
        evaluation_id: Unique evaluation run identifier (UUID4).
        rule_set_id: ID of the rule set that was evaluated.
        dataset_name: Name or identifier of the dataset evaluated.
        total_rules: Total number of rules evaluated in this run.
        passed: Number of rules that passed.
        failed: Number of rules that failed.
        warned: Number of rules that produced warnings.
        pass_rate: Overall pass rate as a float (0.0 to 1.0).
        sla_result: SLA threshold evaluation outcome (pass, warn, fail).
        per_rule_results: List of per-rule evaluation detail dicts,
            each containing rule_id, result, actual_value, expected_value,
            severity, and duration_ms.
        duration_ms: Total wall-clock evaluation time in milliseconds.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_set_id: str = Field(default="")
    dataset_name: str = Field(default="")
    total_rules: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    warned: int = Field(default=0)
    pass_rate: float = Field(default=0.0)
    sla_result: str = Field(default="pass")
    per_rule_results: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation across multiple datasets response.

    Attributes:
        batch_id: Unique batch evaluation identifier (UUID4).
        datasets_evaluated: Number of datasets processed in the batch.
        overall_pass_rate: Aggregate pass rate across all datasets (0.0
            to 1.0).
        per_dataset_results: List of per-dataset evaluation summaries,
            each containing dataset_name, pass_rate, total_rules, passed,
            failed, warned, and sla_result.
        duration_ms: Total wall-clock batch time in milliseconds.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    datasets_evaluated: int = Field(default=0)
    overall_pass_rate: float = Field(default=0.0)
    per_dataset_results: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ConflictReportResponse(BaseModel):
    """Rule conflict detection result response.

    Attributes:
        conflict_id: Unique conflict report identifier (UUID4).
        total_conflicts: Total number of conflicts detected.
        conflicts: List of individual conflict dicts, each containing
            conflict_type, rule_ids, column, description, severity,
            and suggested_resolution.
        severity_distribution: Count of conflicts by severity level,
            e.g. ``{"critical": 1, "high": 3, "medium": 5, "low": 2}``.
        recommendations: List of human-readable remediation suggestions.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_conflicts: int = Field(default=0)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    severity_distribution: Dict[str, int] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ValidationReportResponse(BaseModel):
    """Validation report generation result response.

    Attributes:
        report_id: Unique report identifier (UUID4).
        report_type: Type of validation report (evaluation_summary,
            compliance_report, conflict_analysis, rule_coverage,
            audit_trail, quality_scorecard, trend_analysis,
            exception_report).
        format: Output format (json, html, pdf, csv, markdown, xml,
            text).
        content: The rendered report content as a string.
        report_hash: SHA-256 hash of the report content for integrity.
        generated_at: ISO-8601 UTC generation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = Field(default="evaluation_summary")
    format: str = Field(default="json")
    content: str = Field(default="")
    report_hash: str = Field(default="")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class RulePackResponse(BaseModel):
    """Regulatory rule pack application / listing response.

    Attributes:
        pack_name: Name of the regulatory rule pack (ghg_protocol,
            csrd_esrs, eudr, soc2, custom).
        pack_type: Classification of the pack framework.
        version: SemVer version of the rule pack.
        rules_count: Number of rules included in the pack.
        description: Human-readable description of the rule pack.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    pack_name: str = Field(default="")
    pack_type: str = Field(default="custom")
    version: str = Field(default="1.0.0")
    rules_count: int = Field(default=0)
    description: str = Field(default="")
    provenance_hash: str = Field(default="")


class PipelineResultResponse(BaseModel):
    """End-to-end validation pipeline execution result response.

    Attributes:
        pipeline_id: Unique pipeline run identifier (UUID4).
        stages_completed: Number of pipeline stages that completed
            successfully.
        evaluation_summary: Summary dict with pass_rate, total_rules,
            passed, failed, warned.
        conflicts_found: Number of rule conflicts detected during the
            pipeline run.
        report_id: ID of the validation report generated, or None if
            report generation was skipped.
        duration_ms: Total wall-clock pipeline time in milliseconds.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stages_completed: int = Field(default=0)
    evaluation_summary: Dict[str, Any] = Field(default_factory=dict)
    conflicts_found: int = Field(default=0)
    report_id: Optional[str] = Field(default=None)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ValidationStatisticsResponse(BaseModel):
    """Aggregate statistics for the validation rule engine service.

    Attributes:
        total_rules: Total number of registered validation rules.
        total_rule_sets: Total number of rule sets in the registry.
        total_evaluations: Total number of evaluation runs performed.
        total_conflicts: Total number of conflicts detected.
        avg_pass_rate: Average pass rate across all evaluations (0.0
            to 1.0).
        rules_by_type: Rule count broken down by rule type.
        rules_by_severity: Rule count broken down by severity level.
    """

    model_config = {"extra": "forbid"}

    total_rules: int = Field(default=0)
    total_rule_sets: int = Field(default=0)
    total_evaluations: int = Field(default=0)
    total_conflicts: int = Field(default=0)
    avg_pass_rate: float = Field(default=0.0)
    rules_by_type: Dict[str, int] = Field(default_factory=dict)
    rules_by_severity: Dict[str, int] = Field(default_factory=dict)


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
# ValidationRuleEngineService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["ValidationRuleEngineService"] = None


class ValidationRuleEngineService:
    """Unified facade over the Validation Rule Engine Agent SDK.

    Aggregates all seven validation engines (rule registry, rule composer,
    rule evaluator, conflict detector, rule pack, validation reporter,
    validation pipeline) through a single entry point with convenience
    methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: ValidationRuleEngineConfig instance.
        provenance: ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = ValidationRuleEngineService()
        >>> result = service.register_rule(
        ...     name="emission_factor_range",
        ...     rule_type="range",
        ...     column="emission_factor",
        ...     operator="between",
        ...     threshold={"min": 0.0, "max": 100.0},
        ...     severity="high",
        ... )
        >>> print(result.rule_id, result.status)
    """

    def __init__(
        self,
        config: Optional[ValidationRuleEngineConfig] = None,
    ) -> None:
        """Initialize the Validation Rule Engine Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - RuleRegistryEngine
        - RuleComposerEngine
        - RuleEvaluatorEngine
        - ConflictDetectorEngine
        - RulePackEngine
        - ValidationReporterEngine
        - ValidationPipelineEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = ProvenanceTracker(
            genesis_hash=self.config.genesis_hash,
        )

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._rule_registry_engine: Any = None
        self._rule_composer_engine: Any = None
        self._rule_evaluator_engine: Any = None
        self._conflict_detector_engine: Any = None
        self._rule_pack_engine: Any = None
        self._validation_reporter_engine: Any = None
        self._validation_pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._rules: Dict[str, RuleResponse] = {}
        self._rule_sets: Dict[str, RuleSetResponse] = {}
        self._compound_rules: Dict[str, CompoundRuleResponse] = {}
        self._evaluations: Dict[str, EvaluationResponse] = {}
        self._batch_evaluations: Dict[str, BatchEvaluationResponse] = {}
        self._conflicts: Dict[str, ConflictReportResponse] = {}
        self._reports: Dict[str, ValidationReportResponse] = {}
        self._packs: Dict[str, RulePackResponse] = {}
        self._pipeline_results: Dict[str, PipelineResultResponse] = {}

        # Statistics
        self._stats = ValidationStatisticsResponse()
        self._total_pass_sum: float = 0.0
        self._started = False

        logger.info("ValidationRuleEngineService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def rule_registry_engine(self) -> Any:
        """Get the RuleRegistryEngine instance."""
        return self._rule_registry_engine

    @property
    def rule_composer_engine(self) -> Any:
        """Get the RuleComposerEngine instance."""
        return self._rule_composer_engine

    @property
    def rule_evaluator_engine(self) -> Any:
        """Get the RuleEvaluatorEngine instance."""
        return self._rule_evaluator_engine

    @property
    def conflict_detector_engine(self) -> Any:
        """Get the ConflictDetectorEngine instance."""
        return self._conflict_detector_engine

    @property
    def rule_pack_engine(self) -> Any:
        """Get the RulePackEngine instance."""
        return self._rule_pack_engine

    @property
    def validation_reporter_engine(self) -> Any:
        """Get the ValidationReporterEngine instance."""
        return self._validation_reporter_engine

    @property
    def validation_pipeline_engine(self) -> Any:
        """Get the ValidationPipelineEngine instance."""
        return self._validation_pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are wired together using dependency injection: the
        registry is shared across composer, evaluator, conflict detector,
        and pipeline. The shared ProvenanceTracker is injected into all
        engines for unified audit trails.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        # E1: RuleRegistryEngine(provenance)
        if RuleRegistryEngine is not None:
            try:
                self._rule_registry_engine = RuleRegistryEngine(
                    provenance=self.provenance,
                )
                logger.info("RuleRegistryEngine initialized")
            except Exception as exc:
                logger.warning("RuleRegistryEngine init failed: %s", exc)
        else:
            logger.warning("RuleRegistryEngine not available; using stub")

        # E2: RuleComposerEngine(registry, provenance, max_nesting_depth, max_rules_per_set)
        if RuleComposerEngine is not None:
            try:
                self._rule_composer_engine = RuleComposerEngine(
                    registry=self._rule_registry_engine,
                    provenance=self.provenance,
                    max_nesting_depth=self.config.max_compound_depth,
                    max_rules_per_set=self.config.max_rules_per_set,
                )
                logger.info("RuleComposerEngine initialized")
            except Exception as exc:
                logger.warning("RuleComposerEngine init failed: %s", exc)
        else:
            logger.warning("RuleComposerEngine not available; using stub")

        # E3: RuleEvaluatorEngine(registry, composer, provenance)
        if RuleEvaluatorEngine is not None:
            try:
                self._rule_evaluator_engine = RuleEvaluatorEngine(
                    registry=self._rule_registry_engine,
                    composer=self._rule_composer_engine,
                    provenance=self.provenance,
                )
                logger.info("RuleEvaluatorEngine initialized")
            except Exception as exc:
                logger.warning("RuleEvaluatorEngine init failed: %s", exc)
        else:
            logger.warning("RuleEvaluatorEngine not available; using stub")

        # E4: ConflictDetectorEngine(registry, provenance)
        if ConflictDetectorEngine is not None:
            try:
                self._conflict_detector_engine = ConflictDetectorEngine(
                    registry=self._rule_registry_engine,
                    provenance=self.provenance,
                )
                logger.info("ConflictDetectorEngine initialized")
            except Exception as exc:
                logger.warning("ConflictDetectorEngine init failed: %s", exc)
        else:
            logger.warning(
                "ConflictDetectorEngine not available; using stub"
            )

        # E5: RulePackEngine (not yet implemented -- graceful stub)
        if RulePackEngine is not None:
            try:
                self._rule_pack_engine = RulePackEngine(
                    provenance=self.provenance,
                )
                logger.info("RulePackEngine initialized")
            except Exception as exc:
                logger.warning("RulePackEngine init failed: %s", exc)
        else:
            logger.warning("RulePackEngine not available; using stub")

        # E6: ValidationReporterEngine(provenance)
        if ValidationReporterEngine is not None:
            try:
                self._validation_reporter_engine = ValidationReporterEngine(
                    provenance=self.provenance,
                )
                logger.info("ValidationReporterEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ValidationReporterEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ValidationReporterEngine not available; using stub"
            )

        # E7: ValidationPipelineEngine(registry, composer, evaluator,
        #     conflict_detector, rule_pack, reporter, provenance)
        if ValidationPipelineEngine is not None:
            try:
                self._validation_pipeline_engine = ValidationPipelineEngine(
                    registry=self._rule_registry_engine,
                    composer=self._rule_composer_engine,
                    evaluator=self._rule_evaluator_engine,
                    conflict_detector=self._conflict_detector_engine,
                    rule_pack=self._rule_pack_engine,
                    reporter=self._validation_reporter_engine,
                    provenance=self.provenance,
                )
                logger.info("ValidationPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ValidationPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ValidationPipelineEngine not available; using stub"
            )

    # ==================================================================
    # Rule operations (delegate to RuleRegistryEngine)
    # ==================================================================

    def register_rule(
        self,
        name: str,
        rule_type: str,
        column: str = "",
        operator: str = "between",
        threshold: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None,
        severity: str = "medium",
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuleResponse:
        """Register a new validation rule in the rule registry.

        Delegates to the RuleRegistryEngine for registration. All
        operations are deterministic. No LLM is used for registration
        logic (zero-hallucination).

        Args:
            name: Human-readable rule name (must not be empty).
            rule_type: Classification of the rule (completeness, range,
                format, uniqueness, custom, freshness, cross_field,
                conditional, statistical, referential).
            column: Target column or field name for the rule.
            operator: Comparison operator for evaluation.
            threshold: Primary threshold value or dict with min/max.
            parameters: Additional rule-specific parameters.
            severity: Rule severity level (critical, high, medium, low).
            description: Human-readable description.
            tags: Key-value labels for filtering.
            metadata: Additional unstructured metadata.

        Returns:
            RuleResponse with registered rule details.

        Raises:
            ValueError: If name or rule_type are empty.
        """
        t0 = time.perf_counter()

        if not name:
            raise ValueError("name must not be empty")
        if not rule_type:
            raise ValueError("rule_type must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._rule_registry_engine is not None:
                engine_result = self._rule_registry_engine.register_rule(
                    name=name,
                    rule_type=rule_type,
                    column=column,
                    operator=operator,
                    threshold=threshold,
                    parameters=parameters,
                    severity=severity,
                    description=description,
                    tags=tags,
                    metadata=metadata,
                )

            # Build response
            rule_id = (
                engine_result.get("rule_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = RuleResponse(
                rule_id=rule_id,
                name=name,
                rule_type=rule_type,
                column=column,
                operator=operator,
                threshold=threshold,
                parameters=parameters or {},
                severity=severity,
                status=engine_result.get("status", "draft") if engine_result else "draft",
                version=engine_result.get("version", "1.0.0") if engine_result else "1.0.0",
                description=description,
                tags=tags or {},
                metadata=metadata or {},
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
                updated_at=engine_result.get("updated_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._rules[response.rule_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="validation_rule",
                entity_id=response.rule_id,
                action="rule_registered",
                metadata={
                    "name": name,
                    "rule_type": rule_type,
                    "column": column,
                    "severity": severity,
                },
            )

            # Record metrics
            record_rule_registered(rule_type, severity)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("rule_register", elapsed)

            # Update statistics
            self._stats.total_rules += 1
            type_counts = self._stats.rules_by_type
            type_counts[rule_type] = type_counts.get(rule_type, 0) + 1
            sev_counts = self._stats.rules_by_severity
            sev_counts[severity] = sev_counts.get(severity, 0) + 1

            # Update gauge
            set_active_rules(self._stats.total_rules)

            logger.info(
                "Registered rule %s: name=%s type=%s column=%s severity=%s",
                response.rule_id,
                name,
                rule_type,
                column,
                severity,
            )
            return response

        except Exception as exc:
            logger.error("register_rule failed: %s", exc, exc_info=True)
            raise

    def search_rules(
        self,
        rule_type: Optional[str] = None,
        severity: Optional[str] = None,
        column: Optional[str] = None,
        status: Optional[str] = None,
        tag_key: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[RuleResponse]:
        """Search validation rules with optional filtering and pagination.

        All filters are applied with AND logic.

        Args:
            rule_type: Filter by exact rule type.
            severity: Filter by exact severity level.
            column: Filter by exact column name.
            status: Filter by exact lifecycle status.
            tag_key: Filter by tag key presence.
            query: Substring search in name and description.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of RuleResponse instances matching the filters.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_registry_engine is not None:
                try:
                    engine_results = self._rule_registry_engine.search_rules(
                        rule_type=rule_type,
                        severity=severity,
                        column=column,
                        status=status,
                        tag_key=tag_key,
                        query=query,
                        limit=limit,
                        offset=offset,
                    )
                    results = [
                        self._dict_to_rule_response(rec)
                        for rec in engine_results
                    ]
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("rule_search", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            rules = list(self._rules.values())
            filtered = self._filter_rules(
                rules, rule_type, severity, column, status,
                tag_key, query,
            )
            paginated = filtered[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("rule_search", elapsed)
            return paginated

        except Exception as exc:
            logger.error("search_rules failed: %s", exc, exc_info=True)
            raise

    def get_rule(self, rule_id: str) -> Optional[RuleResponse]:
        """Get a validation rule by its unique identifier.

        Args:
            rule_id: Rule identifier (UUID4 string).

        Returns:
            RuleResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_registry_engine is not None:
                try:
                    engine_result = self._rule_registry_engine.get_rule(
                        rule_id,
                    )
                    if engine_result is not None:
                        resp = self._dict_to_rule_response(engine_result)
                        elapsed = time.perf_counter() - t0
                        observe_processing_duration("rule_get", elapsed)
                        return resp
                    return None
                except (AttributeError, TypeError, KeyError):
                    pass

            # Fallback to in-memory store
            result = self._rules.get(rule_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("rule_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_rule failed: %s", exc, exc_info=True)
            raise

    def update_rule(
        self,
        rule_id: str,
        name: Optional[str] = None,
        column: Optional[str] = None,
        operator: Optional[str] = None,
        threshold: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RuleResponse]:
        """Update mutable fields of an existing validation rule.

        Only fields explicitly provided (non-None) are updated. The
        rule_type is immutable after registration. The version is
        auto-bumped: breaking changes increment major, additive
        changes increment minor, cosmetic changes increment patch.

        Args:
            rule_id: Rule identifier to update.
            name: Updated rule name, or None to leave unchanged.
            column: Updated column name, or None to leave unchanged.
            operator: Updated operator, or None to leave unchanged.
            threshold: Updated threshold, or None to leave unchanged.
            parameters: Updated parameters, or None to leave unchanged.
            severity: Updated severity, or None to leave unchanged.
            status: Updated lifecycle status, or None to leave unchanged.
            description: Updated description, or None to leave unchanged.
            tags: Updated tags, or None to leave unchanged.
            metadata: Updated metadata, or None to leave unchanged.

        Returns:
            Updated RuleResponse or None if rule not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._rule_registry_engine is not None:
                try:
                    # Build kwargs only for non-None fields.
                    # Note: the engine does not support updating 'name'
                    # (name is immutable at the engine level). The facade
                    # applies a name update to the in-memory cache only.
                    update_kwargs: Dict[str, Any] = {}
                    if column is not None:
                        update_kwargs["column"] = column
                    if operator is not None:
                        update_kwargs["operator"] = operator
                    if threshold is not None:
                        update_kwargs["threshold"] = threshold
                    if parameters is not None:
                        update_kwargs["parameters"] = parameters
                    if severity is not None:
                        update_kwargs["severity"] = severity
                    if status is not None:
                        update_kwargs["status"] = status
                    if description is not None:
                        update_kwargs["description"] = description
                    if tags is not None:
                        update_kwargs["tags"] = tags
                    if metadata is not None:
                        update_kwargs["metadata"] = metadata

                    if update_kwargs:
                        engine_result = self._rule_registry_engine.update_rule(
                            rule_id, **update_kwargs,
                        )
                except KeyError:
                    return None

            if engine_result is not None:
                response = self._dict_to_rule_response(engine_result)
                # Apply name update at facade level if requested
                if name is not None:
                    response.name = name
                response.provenance_hash = _compute_hash(response)
                self._rules[response.rule_id] = response
            else:
                # Fallback to in-memory store
                cached = self._rules.get(rule_id)
                if cached is None:
                    return None
                if name is not None:
                    cached.name = name
                if column is not None:
                    cached.column = column
                if operator is not None:
                    cached.operator = operator
                if threshold is not None:
                    cached.threshold = threshold
                if parameters is not None:
                    cached.parameters = parameters
                if severity is not None:
                    cached.severity = severity
                if status is not None:
                    cached.status = status
                if description is not None:
                    cached.description = description
                if tags is not None:
                    cached.tags = tags
                if metadata is not None:
                    cached.metadata = metadata
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
                response = cached

            # Record provenance
            self.provenance.record(
                entity_type="validation_rule",
                entity_id=rule_id,
                action="rule_updated",
                metadata={
                    "name": name,
                    "severity": severity,
                    "status": status,
                },
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("rule_update", elapsed)

            logger.info(
                "Updated rule %s: name=%s severity=%s status=%s",
                rule_id,
                name,
                severity,
                status,
            )
            return response

        except Exception as exc:
            logger.error("update_rule failed: %s", exc, exc_info=True)
            raise

    def delete_rule(self, rule_id: str) -> bool:
        """Soft-delete a validation rule by setting its status to archived.

        Args:
            rule_id: Rule identifier to delete.

        Returns:
            True if the rule was found and archived, False if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_registry_engine is not None:
                try:
                    self._rule_registry_engine.delete_rule(rule_id)
                except (KeyError, ValueError):
                    return False

            # Update in-memory cache
            cached = self._rules.get(rule_id)
            if cached is not None:
                cached.status = "archived"
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
            elif self._rule_registry_engine is None:
                return False

            # Record provenance
            self.provenance.record(
                entity_type="validation_rule",
                entity_id=rule_id,
                action="rule_deleted",
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("rule_delete", elapsed)

            logger.info("Soft-deleted (archived) rule %s", rule_id)
            return True

        except Exception as exc:
            logger.error("delete_rule failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Rule set operations (delegate to RuleComposerEngine)
    # ==================================================================

    def create_rule_set(
        self,
        name: str,
        description: str = "",
        rule_ids: Optional[List[str]] = None,
        sla_thresholds: Optional[Dict[str, float]] = None,
        parent_set_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> RuleSetResponse:
        """Create a new rule set (named collection of rules).

        Delegates to the RuleComposerEngine for rule set creation. All
        operations are deterministic (zero-hallucination).

        Args:
            name: Human-readable rule set name (must not be empty).
            description: Human-readable description of the rule set.
            rule_ids: Optional list of rule IDs to include in the set.
            sla_thresholds: Dynamic pass/warn thresholds, e.g.
                ``{"pass": 0.95, "warn": 0.80}``.
            parent_set_id: Optional parent rule set ID for inheritance.
            tags: Key-value labels for categorization.

        Returns:
            RuleSetResponse with created rule set details.

        Raises:
            ValueError: If name is empty.
        """
        t0 = time.perf_counter()

        if not name:
            raise ValueError("name must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._rule_composer_engine is not None:
                # Engine expects tags as List[str]; convert
                # Dict[str,str] -> list of keys
                engine_tags: Optional[List[str]] = None
                if tags:
                    engine_tags = list(tags.keys())
                engine_result = self._rule_composer_engine.create_rule_set(
                    name=name,
                    description=description,
                    rule_ids=rule_ids or [],
                    sla_thresholds=sla_thresholds,
                    tags=engine_tags,
                )

            # Build response
            set_id = (
                engine_result.get("set_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()
            rule_count = len(rule_ids) if rule_ids else 0

            if engine_result is not None:
                rule_count = engine_result.get("rule_count", rule_count)

            default_sla = {
                "pass": self.config.default_pass_threshold,
                "warn": self.config.default_warn_threshold,
            }

            # Engine may return tags as list; convert back to dict
            raw_tags = engine_result.get("tags", []) if engine_result else []
            resp_tags = tags or {}
            if isinstance(raw_tags, list) and not resp_tags:
                resp_tags = {t: "" for t in raw_tags}

            response = RuleSetResponse(
                set_id=set_id,
                name=name,
                description=description,
                version=engine_result.get("version", "1.0.0") if engine_result else "1.0.0",
                status=engine_result.get("status", "draft") if engine_result else "draft",
                rule_count=rule_count,
                sla_thresholds=sla_thresholds or default_sla,
                parent_set_id=parent_set_id,
                tags=resp_tags,
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
                updated_at=engine_result.get("updated_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._rule_sets[response.set_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="rule_set",
                entity_id=response.set_id,
                action="rule_set_created",
                metadata={
                    "name": name,
                    "rule_count": rule_count,
                    "parent_set_id": parent_set_id,
                },
            )

            # Record metrics
            record_rule_set_created("custom")
            observe_rules_per_set(rule_count)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("set_create", elapsed)

            # Update statistics
            self._stats.total_rule_sets += 1
            set_active_rule_sets(self._stats.total_rule_sets)

            logger.info(
                "Created rule set %s: name=%s rules=%d parent=%s",
                response.set_id,
                name,
                rule_count,
                parent_set_id,
            )
            return response

        except Exception as exc:
            logger.error("create_rule_set failed: %s", exc, exc_info=True)
            raise

    def list_rule_sets(
        self,
        status: Optional[str] = None,
        tag_key: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[RuleSetResponse]:
        """List rule sets with optional filtering and pagination.

        Args:
            status: Filter by exact lifecycle status.
            tag_key: Filter by tag key presence.
            query: Substring search in name and description.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of RuleSetResponse instances matching the filters.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_composer_engine is not None:
                try:
                    # Engine accepts tags (List[str]), limit, offset
                    engine_tags: Optional[List[str]] = None
                    if tag_key:
                        engine_tags = [tag_key]
                    engine_results = self._rule_composer_engine.list_rule_sets(
                        tags=engine_tags,
                        limit=limit,
                        offset=offset,
                    )
                    results = [
                        self._dict_to_rule_set_response(rec)
                        for rec in engine_results
                    ]
                    # Apply additional facade-level filters
                    if status is not None:
                        results = [r for r in results if r.status == status]
                    if query is not None:
                        q = query.lower()
                        results = [
                            r for r in results
                            if q in r.name.lower()
                            or q in r.description.lower()
                        ]
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("set_list", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            sets = list(self._rule_sets.values())
            if status is not None:
                sets = [s for s in sets if s.status == status]
            if tag_key is not None:
                sets = [s for s in sets if tag_key in s.tags]
            if query is not None:
                q = query.lower()
                sets = [
                    s for s in sets
                    if q in s.name.lower() or q in s.description.lower()
                ]
            paginated = sets[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("set_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_rule_sets failed: %s", exc, exc_info=True)
            raise

    def get_rule_set(self, set_id: str) -> Optional[RuleSetResponse]:
        """Get a rule set by its unique identifier.

        Args:
            set_id: Rule set identifier (UUID4 string).

        Returns:
            RuleSetResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_composer_engine is not None:
                try:
                    engine_result = self._rule_composer_engine.get_rule_set(
                        set_id,
                    )
                    if engine_result is not None:
                        resp = self._dict_to_rule_set_response(engine_result)
                        elapsed = time.perf_counter() - t0
                        observe_processing_duration("set_get", elapsed)
                        return resp
                    return None
                except (AttributeError, TypeError, KeyError):
                    pass

            # Fallback to in-memory store
            result = self._rule_sets.get(set_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("set_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_rule_set failed: %s", exc, exc_info=True)
            raise

    def update_rule_set(
        self,
        set_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        sla_thresholds: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        rule_ids: Optional[List[str]] = None,
    ) -> Optional[RuleSetResponse]:
        """Update mutable fields of an existing rule set.

        Only fields explicitly provided (non-None) are updated.

        Args:
            set_id: Rule set identifier to update.
            name: Updated name, or None to leave unchanged.
            description: Updated description, or None to leave unchanged.
            status: Updated lifecycle status, or None to leave unchanged.
            sla_thresholds: Updated SLA thresholds, or None to leave
                unchanged.
            tags: Updated tags, or None to leave unchanged.
            rule_ids: Updated list of rule IDs, or None to leave
                unchanged.

        Returns:
            Updated RuleSetResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._rule_composer_engine is not None:
                try:
                    # Engine uses **kwargs; build dict of non-None fields.
                    # Supported: name, description, tags, sla_thresholds,
                    # status. Tags must be List[str] for the engine.
                    update_kwargs: Dict[str, Any] = {}
                    if name is not None:
                        update_kwargs["name"] = name
                    if description is not None:
                        update_kwargs["description"] = description
                    if status is not None:
                        update_kwargs["status"] = status
                    if sla_thresholds is not None:
                        update_kwargs["sla_thresholds"] = sla_thresholds
                    if tags is not None:
                        update_kwargs["tags"] = list(tags.keys())

                    if update_kwargs:
                        engine_result = self._rule_composer_engine.update_rule_set(
                            set_id, **update_kwargs,
                        )
                except (KeyError, ValueError):
                    return None

            if engine_result is not None:
                response = self._dict_to_rule_set_response(engine_result)
                response.provenance_hash = _compute_hash(response)
                self._rule_sets[response.set_id] = response
            else:
                # Fallback to in-memory store
                cached = self._rule_sets.get(set_id)
                if cached is None:
                    return None
                if name is not None:
                    cached.name = name
                if description is not None:
                    cached.description = description
                if status is not None:
                    cached.status = status
                if sla_thresholds is not None:
                    cached.sla_thresholds = sla_thresholds
                if tags is not None:
                    cached.tags = tags
                if rule_ids is not None:
                    cached.rule_count = len(rule_ids)
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
                response = cached

            # Record provenance
            self.provenance.record(
                entity_type="rule_set",
                entity_id=set_id,
                action="rule_set_updated",
                metadata={
                    "name": name,
                    "status": status,
                },
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("set_update", elapsed)

            logger.info(
                "Updated rule set %s: name=%s status=%s",
                set_id,
                name,
                status,
            )
            return response

        except Exception as exc:
            logger.error("update_rule_set failed: %s", exc, exc_info=True)
            raise

    def delete_rule_set(self, set_id: str) -> bool:
        """Soft-delete a rule set by setting its status to archived.

        Args:
            set_id: Rule set identifier to delete.

        Returns:
            True if the rule set was found and archived, False if not
            found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_composer_engine is not None:
                try:
                    self._rule_composer_engine.delete_rule_set(set_id)
                except (KeyError, ValueError):
                    return False

            # Update in-memory cache
            cached = self._rule_sets.get(set_id)
            if cached is not None:
                cached.status = "archived"
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
            elif self._rule_composer_engine is None:
                return False

            # Record provenance
            self.provenance.record(
                entity_type="rule_set",
                entity_id=set_id,
                action="rule_set_deactivated",
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("set_delete", elapsed)

            logger.info("Soft-deleted (archived) rule set %s", set_id)
            return True

        except Exception as exc:
            logger.error("delete_rule_set failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Evaluation operations (delegate to RuleEvaluatorEngine)
    # ==================================================================

    def evaluate_rules(
        self,
        rule_set_id: str,
        dataset_name: str = "",
        data: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResponse:
        """Evaluate a rule set against a dataset.

        Executes all rules in the specified rule set against the provided
        data. All evaluation logic is deterministic (zero-hallucination):
        comparisons, range checks, regex matching, and aggregations are
        performed using standard Python operators only.

        Args:
            rule_set_id: ID of the rule set to evaluate.
            dataset_name: Name or identifier of the dataset being
                evaluated.
            data: List of record dicts to evaluate against. Each dict
                represents one row of data.
            parameters: Additional evaluation parameters.

        Returns:
            EvaluationResponse with per-rule results and aggregate
            pass rate.

        Raises:
            ValueError: If rule_set_id is empty.
        """
        t0 = time.perf_counter()

        if not rule_set_id:
            raise ValueError("rule_set_id must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if (
                self._rule_evaluator_engine is not None
                and self._rule_composer_engine is not None
            ):
                # Engine expects the full rule_set dict, not just an ID
                rule_set_dict = self._rule_composer_engine.get_rule_set(
                    rule_set_id,
                )
                if rule_set_dict is not None:
                    engine_result = self._rule_evaluator_engine.evaluate_rule_set(
                        rule_set=rule_set_dict,
                        data=data or [],
                        context=parameters,
                    )

            # Build response
            evaluation_id = (
                engine_result.get("evaluation_id", _new_uuid())
                if engine_result else _new_uuid()
            )

            total_rules = engine_result.get("total_rules", 0) if engine_result else 0
            passed = engine_result.get("passed", 0) if engine_result else 0
            failed = engine_result.get("failed", 0) if engine_result else 0
            warned = engine_result.get("warned", 0) if engine_result else 0
            pass_rate = engine_result.get("pass_rate", 0.0) if engine_result else 0.0
            sla_result = engine_result.get("sla_result", "pass") if engine_result else "pass"
            per_rule = engine_result.get("per_rule_results", []) if engine_result else []

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            response = EvaluationResponse(
                evaluation_id=evaluation_id,
                rule_set_id=rule_set_id,
                dataset_name=dataset_name,
                total_rules=total_rules,
                passed=passed,
                failed=failed,
                warned=warned,
                pass_rate=pass_rate,
                sla_result=sla_result,
                per_rule_results=per_rule,
                duration_ms=round(elapsed_ms, 2),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._evaluations[response.evaluation_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="evaluation",
                entity_id=response.evaluation_id,
                action="evaluation_completed",
                metadata={
                    "rule_set_id": rule_set_id,
                    "dataset_name": dataset_name,
                    "total_rules": total_rules,
                    "passed": passed,
                    "failed": failed,
                    "pass_rate": pass_rate,
                    "sla_result": sla_result,
                },
            )

            # Record metrics
            record_evaluation(sla_result, "rule_set")
            if failed > 0:
                record_evaluation_failure("high")
            elapsed = time.perf_counter() - t0
            observe_evaluation_duration("rule_set", elapsed)

            # Update statistics
            self._stats.total_evaluations += 1
            self._total_pass_sum += pass_rate
            if self._stats.total_evaluations > 0:
                self._stats.avg_pass_rate = (
                    self._total_pass_sum / self._stats.total_evaluations
                )
                set_pass_rate(self._stats.avg_pass_rate)

            logger.info(
                "Evaluated rule set %s on %s: total=%d passed=%d failed=%d "
                "warned=%d pass_rate=%.2f sla=%s elapsed=%.1fms",
                rule_set_id,
                dataset_name,
                total_rules,
                passed,
                failed,
                warned,
                pass_rate,
                sla_result,
                elapsed_ms,
            )
            return response

        except Exception as exc:
            logger.error("evaluate_rules failed: %s", exc, exc_info=True)
            raise

    def evaluate_batch(
        self,
        rule_set_id: str,
        datasets: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> BatchEvaluationResponse:
        """Batch evaluate a rule set across multiple datasets.

        Each dataset dict must contain ``name`` (str) and ``data``
        (list of record dicts) keys. All evaluation logic is
        deterministic (zero-hallucination).

        Args:
            rule_set_id: ID of the rule set to evaluate.
            datasets: List of dataset dicts, each with ``name`` and
                ``data`` keys.
            parameters: Additional evaluation parameters shared across
                all datasets.

        Returns:
            BatchEvaluationResponse with per-dataset results and
            aggregate pass rate.

        Raises:
            ValueError: If rule_set_id is empty.
        """
        t0 = time.perf_counter()

        if not rule_set_id:
            raise ValueError("rule_set_id must not be empty")

        try:
            dataset_list = datasets or []

            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if (
                self._rule_evaluator_engine is not None
                and self._rule_composer_engine is not None
            ):
                try:
                    # Engine expects datasets as Dict[str, List[Dict]]
                    # and a full rule_set dict
                    rule_set_dict = self._rule_composer_engine.get_rule_set(
                        rule_set_id,
                    )
                    if rule_set_dict is not None:
                        engine_datasets: Dict[str, List[Dict[str, Any]]] = {}
                        for ds in dataset_list:
                            ds_name = ds.get("name", "unknown")
                            engine_datasets[ds_name] = ds.get("data", [])
                        engine_result = self._rule_evaluator_engine.evaluate_batch(
                            datasets=engine_datasets,
                            rule_set=rule_set_dict,
                            context=parameters,
                        )
                except (AttributeError, TypeError):
                    pass

            if engine_result is not None:
                batch_id = engine_result.get("batch_id", _new_uuid())
                datasets_evaluated = engine_result.get("datasets_evaluated", 0)
                overall_pass_rate = engine_result.get("overall_pass_rate", 0.0)
                per_dataset = engine_result.get("per_dataset_results", [])
            else:
                # Fallback: run individual evaluations
                batch_id = _new_uuid()
                per_dataset = []
                total_pass = 0.0

                for ds_entry in dataset_list:
                    ds_name = ds_entry.get("name", "unknown")
                    ds_data = ds_entry.get("data", [])
                    try:
                        eval_result = self.evaluate_rules(
                            rule_set_id=rule_set_id,
                            dataset_name=ds_name,
                            data=ds_data,
                            parameters=parameters,
                        )
                        per_dataset.append({
                            "dataset_name": ds_name,
                            "pass_rate": eval_result.pass_rate,
                            "total_rules": eval_result.total_rules,
                            "passed": eval_result.passed,
                            "failed": eval_result.failed,
                            "warned": eval_result.warned,
                            "sla_result": eval_result.sla_result,
                        })
                        total_pass += eval_result.pass_rate
                    except Exception as exc:
                        per_dataset.append({
                            "dataset_name": ds_name,
                            "error": str(exc),
                            "pass_rate": 0.0,
                        })

                datasets_evaluated = len(dataset_list)
                overall_pass_rate = (
                    total_pass / datasets_evaluated
                    if datasets_evaluated > 0 else 0.0
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            response = BatchEvaluationResponse(
                batch_id=batch_id,
                datasets_evaluated=datasets_evaluated,
                overall_pass_rate=round(overall_pass_rate, 4),
                per_dataset_results=per_dataset,
                duration_ms=round(elapsed_ms, 2),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._batch_evaluations[response.batch_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="evaluation",
                entity_id=response.batch_id,
                action="batch_evaluation_completed",
                metadata={
                    "rule_set_id": rule_set_id,
                    "datasets_evaluated": datasets_evaluated,
                    "overall_pass_rate": round(overall_pass_rate, 4),
                },
            )

            # Record metrics
            elapsed = time.perf_counter() - t0
            observe_evaluation_duration("batch", elapsed)

            logger.info(
                "Batch evaluation %s: datasets=%d overall_pass_rate=%.2f "
                "elapsed=%.1fms",
                response.batch_id,
                datasets_evaluated,
                overall_pass_rate,
                elapsed_ms,
            )
            return response

        except Exception as exc:
            logger.error("evaluate_batch failed: %s", exc, exc_info=True)
            raise

    def get_evaluation(
        self,
        evaluation_id: str,
    ) -> Optional[EvaluationResponse]:
        """Get an evaluation result by its identifier.

        Checks both single evaluation and batch evaluation caches.

        Args:
            evaluation_id: Evaluation identifier (UUID4 string).

        Returns:
            EvaluationResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            result = self._evaluations.get(evaluation_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("evaluation_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_evaluation failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Conflict detection (delegate to ConflictDetectorEngine)
    # ==================================================================

    def detect_conflicts(
        self,
        rule_set_id: Optional[str] = None,
        rule_ids: Optional[List[str]] = None,
    ) -> ConflictReportResponse:
        """Detect conflicts among validation rules.

        Analyzes rules for contradictions, overlaps, subsumptions, and
        redundancies. All conflict detection logic is deterministic
        (zero-hallucination): range overlap analysis, regex pattern
        comparison, and severity consistency checks use standard
        comparison operators only.

        Args:
            rule_set_id: Optional rule set ID to scope the analysis.
            rule_ids: Optional list of rule IDs to analyze. If both
                rule_set_id and rule_ids are provided, rule_set_id
                takes precedence.

        Returns:
            ConflictReportResponse with detected conflicts and
            remediation suggestions.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._conflict_detector_engine is not None:
                engine_result = self._conflict_detector_engine.detect_conflicts(
                    rule_set_id=rule_set_id,
                    rule_ids=rule_ids or [],
                )

            # Build response
            conflict_id = (
                engine_result.get("conflict_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            total_conflicts = (
                engine_result.get("total_conflicts", 0)
                if engine_result else 0
            )
            conflicts = (
                engine_result.get("conflicts", [])
                if engine_result else []
            )
            severity_dist = (
                engine_result.get("severity_distribution", {})
                if engine_result else {}
            )
            recommendations = (
                engine_result.get("recommendations", [])
                if engine_result else []
            )

            response = ConflictReportResponse(
                conflict_id=conflict_id,
                total_conflicts=total_conflicts,
                conflicts=conflicts,
                severity_distribution=severity_dist,
                recommendations=recommendations,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._conflicts[response.conflict_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="conflict",
                entity_id=response.conflict_id,
                action="conflict_detected",
                metadata={
                    "rule_set_id": rule_set_id,
                    "total_conflicts": total_conflicts,
                },
            )

            # Record metrics
            for conflict in conflicts:
                c_type = conflict.get("conflict_type", "unknown")
                record_conflict_detected(c_type)

            elapsed = time.perf_counter() - t0
            observe_processing_duration("conflict_detect", elapsed)

            # Update statistics
            self._stats.total_conflicts += total_conflicts

            logger.info(
                "Conflict detection %s: conflicts=%d severity=%s",
                response.conflict_id,
                total_conflicts,
                severity_dist,
            )
            return response

        except Exception as exc:
            logger.error("detect_conflicts failed: %s", exc, exc_info=True)
            raise

    def list_conflicts(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConflictReportResponse]:
        """List previously detected conflict reports.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of ConflictReportResponse instances.
        """
        t0 = time.perf_counter()

        try:
            conflicts = list(self._conflicts.values())
            paginated = conflicts[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("conflict_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_conflicts failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Rule pack operations (delegate to RulePackEngine)
    # ==================================================================

    def apply_pack(
        self,
        pack_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> RulePackResponse:
        """Apply a regulatory rule pack to the rule registry.

        Loads a pre-built rule pack (ghg_protocol, csrd_esrs, eudr,
        soc2, or custom) and registers all its rules in the registry.
        All rule definitions are deterministic (zero-hallucination):
        thresholds, ranges, and patterns come from regulatory references,
        not LLM generation.

        Args:
            pack_name: Name of the rule pack to apply (ghg_protocol,
                csrd_esrs, eudr, soc2, custom).
            parameters: Optional pack-specific parameters (sector
                overrides, region filters, etc.).

        Returns:
            RulePackResponse with pack application details.

        Raises:
            ValueError: If pack_name is empty.
        """
        t0 = time.perf_counter()

        if not pack_name:
            raise ValueError("pack_name must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._rule_pack_engine is not None:
                engine_result = self._rule_pack_engine.apply_pack(
                    pack_name=pack_name,
                    parameters=parameters or {},
                )

            # Build response
            pack_type = (
                engine_result.get("pack_type", pack_name)
                if engine_result else pack_name
            )
            version = (
                engine_result.get("version", "1.0.0")
                if engine_result else "1.0.0"
            )
            rules_count = (
                engine_result.get("rules_count", 0)
                if engine_result else 0
            )
            description = (
                engine_result.get("description", f"Regulatory rule pack: {pack_name}")
                if engine_result
                else f"Regulatory rule pack: {pack_name}"
            )

            response = RulePackResponse(
                pack_name=pack_name,
                pack_type=pack_type,
                version=version,
                rules_count=rules_count,
                description=description,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._packs[pack_name] = response

            # Record provenance
            self.provenance.record(
                entity_type="rule_pack",
                entity_id=pack_name,
                action="rule_pack_applied",
                metadata={
                    "pack_name": pack_name,
                    "pack_type": pack_type,
                    "version": version,
                    "rules_count": rules_count,
                },
            )

            # Record metrics
            record_rule_set_created(pack_type)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("pack_import", elapsed)

            logger.info(
                "Applied rule pack %s: type=%s version=%s rules=%d",
                pack_name,
                pack_type,
                version,
                rules_count,
            )
            return response

        except Exception as exc:
            logger.error("apply_pack failed: %s", exc, exc_info=True)
            raise

    def list_packs(self) -> List[RulePackResponse]:
        """List all available regulatory rule packs.

        Returns available packs from the RulePackEngine (if loaded)
        plus any packs that have been applied in this session.

        Returns:
            List of RulePackResponse instances.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._rule_pack_engine is not None:
                try:
                    engine_results = self._rule_pack_engine.list_packs()
                    results = [
                        self._dict_to_rule_pack_response(rec)
                        for rec in engine_results
                    ]
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("pack_list", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store
            packs = list(self._packs.values())
            elapsed = time.perf_counter() - t0
            observe_processing_duration("pack_list", elapsed)
            return packs

        except Exception as exc:
            logger.error("list_packs failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Report generation (delegate to ValidationReporterEngine)
    # ==================================================================

    def generate_report(
        self,
        report_type: str = "evaluation_summary",
        report_format: str = "json",
        evaluation_id: Optional[str] = None,
        rule_set_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ValidationReportResponse:
        """Generate a validation report in the specified format.

        Delegates to the ValidationReporterEngine. All report generation
        logic is deterministic (zero-hallucination).

        Args:
            report_type: Type of validation report (evaluation_summary,
                compliance_report, conflict_analysis, rule_coverage,
                audit_trail, quality_scorecard, trend_analysis,
                exception_report).
            report_format: Output format (json, html, pdf, csv,
                markdown, xml, text).
            evaluation_id: Optional evaluation ID to include in the
                report.
            rule_set_id: Optional rule set ID to scope the report.
            parameters: Report generation parameters.

        Returns:
            ValidationReportResponse with generated report content.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._validation_reporter_engine is not None:
                engine_result = self._validation_reporter_engine.generate_report(
                    report_type=report_type,
                    report_format=report_format,
                    evaluation_id=evaluation_id,
                    rule_set_id=rule_set_id,
                    parameters=parameters or {},
                )

            # Build response
            report_id = (
                engine_result.get("report_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            content = (
                engine_result.get("content", "")
                if engine_result else ""
            )
            report_hash = (
                engine_result.get("report_hash", "")
                if engine_result else ""
            )
            if not report_hash and content:
                report_hash = hashlib.sha256(content.encode()).hexdigest()

            response = ValidationReportResponse(
                report_id=report_id,
                report_type=report_type,
                format=report_format,
                content=content,
                report_hash=report_hash,
                generated_at=_utcnow_iso(),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._reports[response.report_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="report",
                entity_id=response.report_id,
                action="report_generated",
                metadata={
                    "report_type": report_type,
                    "format": report_format,
                    "evaluation_id": evaluation_id,
                    "rule_set_id": rule_set_id,
                },
            )

            # Record metrics
            record_report_generated(report_type, report_format)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("report_generate", elapsed)

            logger.info(
                "Generated report %s: type=%s format=%s",
                response.report_id,
                report_type,
                report_format,
            )
            return response

        except Exception as exc:
            logger.error("generate_report failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Pipeline orchestration (delegate to ValidationPipelineEngine)
    # ==================================================================

    def run_pipeline(
        self,
        rule_set_id: str = "",
        dataset_name: str = "",
        data: Optional[List[Dict[str, Any]]] = None,
        detect_conflicts: bool = True,
        generate_report: bool = False,
        report_format: str = "json",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> PipelineResultResponse:
        """Run the end-to-end validation pipeline.

        Orchestrates all stages: evaluate rules, detect conflicts,
        and optionally generate a validation report.

        Args:
            rule_set_id: ID of the rule set to evaluate.
            dataset_name: Name of the dataset to evaluate.
            data: List of record dicts to evaluate against.
            detect_conflicts: Whether to run conflict detection.
            generate_report: Whether to generate a validation report.
            report_format: Format for the generated report.
            parameters: Additional pipeline parameters.

        Returns:
            PipelineResultResponse with overall pipeline results.
        """
        t0 = time.perf_counter()

        try:
            pipeline_id = _new_uuid()
            stages_completed = 0
            evaluation_summary: Dict[str, Any] = {}
            conflicts_found = 0
            report_id: Optional[str] = None
            errors: List[str] = []

            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._validation_pipeline_engine is not None:
                try:
                    engine_result = self._validation_pipeline_engine.run_pipeline(
                        rule_set_id=rule_set_id,
                        dataset_name=dataset_name,
                        data=data or [],
                        detect_conflicts=detect_conflicts,
                        generate_report=generate_report,
                        report_format=report_format,
                        parameters=parameters or {},
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    logger.warning(
                        "Pipeline engine execution failed: %s", exc,
                    )

            if engine_result is not None:
                pipeline_id = engine_result.get("pipeline_id", pipeline_id)
                stages_completed = engine_result.get("stages_completed", 0)
                evaluation_summary = engine_result.get(
                    "evaluation_summary", {},
                )
                conflicts_found = engine_result.get("conflicts_found", 0)
                report_id = engine_result.get("report_id")
                errors = engine_result.get("errors", [])
            else:
                # Fallback: run stages individually
                # Stage 1: Evaluate rules
                if rule_set_id:
                    try:
                        eval_resp = self.evaluate_rules(
                            rule_set_id=rule_set_id,
                            dataset_name=dataset_name,
                            data=data,
                            parameters=parameters,
                        )
                        evaluation_summary = {
                            "pass_rate": eval_resp.pass_rate,
                            "total_rules": eval_resp.total_rules,
                            "passed": eval_resp.passed,
                            "failed": eval_resp.failed,
                            "warned": eval_resp.warned,
                            "sla_result": eval_resp.sla_result,
                        }
                        stages_completed += 1
                    except Exception as exc:
                        errors.append(f"evaluation: {exc}")

                # Stage 2: Detect conflicts
                if detect_conflicts and rule_set_id:
                    try:
                        conflict_resp = self.detect_conflicts(
                            rule_set_id=rule_set_id,
                        )
                        conflicts_found = conflict_resp.total_conflicts
                        stages_completed += 1
                    except Exception as exc:
                        errors.append(f"conflict_detection: {exc}")

                # Stage 3: Generate report
                if generate_report:
                    try:
                        report_resp = self.generate_report(
                            report_type="evaluation_summary",
                            report_format=report_format,
                            rule_set_id=rule_set_id,
                        )
                        report_id = report_resp.report_id
                        stages_completed += 1
                    except Exception as exc:
                        errors.append(f"report: {exc}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            response = PipelineResultResponse(
                pipeline_id=pipeline_id,
                stages_completed=stages_completed,
                evaluation_summary=evaluation_summary,
                conflicts_found=conflicts_found,
                report_id=report_id,
                duration_ms=round(elapsed_ms, 2),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._pipeline_results[response.pipeline_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="evaluation",
                entity_id=response.pipeline_id,
                action="evaluation_completed",
                metadata={
                    "rule_set_id": rule_set_id,
                    "stages_completed": stages_completed,
                    "conflicts_found": conflicts_found,
                    "duration_ms": round(elapsed_ms, 2),
                    "errors": errors,
                },
            )

            # Record metrics
            elapsed = time.perf_counter() - t0
            observe_evaluation_duration("full_pipeline", elapsed)

            logger.info(
                "Pipeline %s completed: stages=%d pass_rate=%.2f "
                "conflicts=%d report=%s elapsed=%.1fms errors=%d",
                response.pipeline_id,
                stages_completed,
                evaluation_summary.get("pass_rate", 0.0),
                conflicts_found,
                report_id,
                elapsed_ms,
                len(errors),
            )
            return response

        except Exception as exc:
            logger.error("run_pipeline failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Statistics and health
    # ==================================================================

    def get_health(self) -> Dict[str, Any]:
        """Perform a health check on the validation rule engine service.

        Returns a dictionary with health status for each engine and
        the overall service.

        Returns:
            Dictionary with health check results including:
            - ``status``: Overall service status (healthy, degraded,
                unhealthy).
            - ``engines``: Per-engine availability status.
            - ``started``: Whether the service has been started.
            - ``statistics``: Summary statistics.
            - ``provenance_chain_valid``: Whether the provenance chain
                is intact.
            - ``timestamp``: ISO-8601 UTC timestamp of the check.
        """
        t0 = time.perf_counter()

        engines: Dict[str, str] = {
            "rule_registry": (
                "available"
                if self._rule_registry_engine is not None
                else "unavailable"
            ),
            "rule_composer": (
                "available"
                if self._rule_composer_engine is not None
                else "unavailable"
            ),
            "rule_evaluator": (
                "available"
                if self._rule_evaluator_engine is not None
                else "unavailable"
            ),
            "conflict_detector": (
                "available"
                if self._conflict_detector_engine is not None
                else "unavailable"
            ),
            "rule_pack": (
                "available"
                if self._rule_pack_engine is not None
                else "unavailable"
            ),
            "validation_reporter": (
                "available"
                if self._validation_reporter_engine is not None
                else "unavailable"
            ),
            "validation_pipeline": (
                "available"
                if self._validation_pipeline_engine is not None
                else "unavailable"
            ),
        }

        available_count = sum(
            1 for status in engines.values() if status == "available"
        )
        total_engines = len(engines)

        if available_count == total_engines:
            overall_status = "healthy"
        elif available_count >= 4:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Verify provenance chain
        chain_valid = self.provenance.verify_chain()

        result = {
            "status": overall_status,
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_engines,
            "started": self._started,
            "statistics": {
                "total_rules": self._stats.total_rules,
                "total_rule_sets": self._stats.total_rule_sets,
                "total_evaluations": self._stats.total_evaluations,
                "total_conflicts": self._stats.total_conflicts,
                "avg_pass_rate": round(self._stats.avg_pass_rate, 4),
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

        elapsed = time.perf_counter() - t0
        observe_processing_duration("health_check", elapsed)

        logger.info(
            "Health check: status=%s engines=%d/%d chain_valid=%s",
            overall_status,
            available_count,
            total_engines,
            chain_valid,
        )
        return result

    def get_statistics(self) -> ValidationStatisticsResponse:
        """Get aggregate statistics for the validation rule engine service.

        Enriches statistics from engine-level data when available.

        Returns:
            ValidationStatisticsResponse with current statistics.
        """
        t0 = time.perf_counter()

        # Enrich from engine statistics where available
        if self._rule_registry_engine is not None:
            try:
                reg_stats = self._rule_registry_engine.get_statistics()
                self._stats.total_rules = reg_stats.get(
                    "total_rules", self._stats.total_rules,
                )
            except (AttributeError, Exception):
                pass

        if self._rule_composer_engine is not None:
            try:
                comp_stats = self._rule_composer_engine.get_statistics()
                self._stats.total_rule_sets = comp_stats.get(
                    "total_rule_sets", self._stats.total_rule_sets,
                )
            except (AttributeError, Exception):
                pass

        if self._rule_evaluator_engine is not None:
            try:
                eval_stats = self._rule_evaluator_engine.get_statistics()
                self._stats.total_evaluations = eval_stats.get(
                    "total_evaluations", self._stats.total_evaluations,
                )
                self._stats.avg_pass_rate = eval_stats.get(
                    "avg_pass_rate", self._stats.avg_pass_rate,
                )
            except (AttributeError, Exception):
                pass

        if self._conflict_detector_engine is not None:
            try:
                conf_stats = self._conflict_detector_engine.get_statistics()
                self._stats.total_conflicts = conf_stats.get(
                    "total_conflicts", self._stats.total_conflicts,
                )
            except (AttributeError, Exception):
                pass

        elapsed = time.perf_counter() - t0
        observe_processing_duration("statistics", elapsed)

        logger.debug(
            "Statistics: rules=%d rule_sets=%d evaluations=%d "
            "conflicts=%d avg_pass_rate=%.2f",
            self._stats.total_rules,
            self._stats.total_rule_sets,
            self._stats.total_evaluations,
            self._stats.total_conflicts,
            self._stats.avg_pass_rate,
        )
        return self._stats

    # ==================================================================
    # Provenance and metrics access
    # ==================================================================

    def get_provenance(self) -> ProvenanceTracker:
        """Get the provenance tracker instance.

        Returns:
            ProvenanceTracker instance used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics as a dictionary.

        Returns:
            Dictionary of metric names to current values.
        """
        return {
            "total_rules": self._stats.total_rules,
            "total_rule_sets": self._stats.total_rule_sets,
            "total_evaluations": self._stats.total_evaluations,
            "total_conflicts": self._stats.total_conflicts,
            "avg_pass_rate": self._stats.avg_pass_rate,
            "rules_by_type": dict(self._stats.rules_by_type),
            "rules_by_severity": dict(self._stats.rules_by_severity),
            "provenance_entries": self.provenance.entry_count,
            "provenance_chain_valid": self.provenance.verify_chain(),
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def startup(self) -> None:
        """Start the validation rule engine service.

        Safe to call multiple times. Resets Prometheus gauges to baseline
        values on first call.
        """
        if self._started:
            logger.debug(
                "ValidationRuleEngineService already started; skipping"
            )
            return

        logger.info("ValidationRuleEngineService starting up...")
        self._started = True
        set_active_rules(0)
        set_active_rule_sets(0)
        set_pass_rate(0.0)
        logger.info("ValidationRuleEngineService startup complete")

    def shutdown(self) -> None:
        """Shutdown the validation rule engine service and release resources."""
        if not self._started:
            return

        self._started = False
        set_active_rules(0)
        set_active_rule_sets(0)
        set_pass_rate(0.0)
        logger.info("ValidationRuleEngineService shut down")

    # ==================================================================
    # Internal helpers: dict -> response model conversion
    # ==================================================================

    def _dict_to_rule_response(
        self,
        rec: Dict[str, Any],
    ) -> RuleResponse:
        """Convert a raw engine dict to RuleResponse.

        Handles engine format differences: the RuleRegistryEngine may
        store tags as a list of strings rather than a dict. When tags
        is a list, it is converted to a dict with each tag as a key
        mapped to an empty string.

        Args:
            rec: Dictionary from the RuleRegistryEngine.

        Returns:
            RuleResponse model.
        """
        raw_tags = rec.get("tags", {})
        if isinstance(raw_tags, list):
            raw_tags = {t: "" for t in raw_tags}
        elif not isinstance(raw_tags, dict):
            raw_tags = {}

        raw_metadata = rec.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        return RuleResponse(
            rule_id=rec.get("rule_id", rec.get("id", "")),
            name=rec.get("name", ""),
            rule_type=rec.get("rule_type", "range"),
            column=rec.get("column", ""),
            operator=rec.get("operator", "between"),
            threshold=rec.get("threshold"),
            parameters=rec.get("parameters", {}),
            severity=rec.get("severity", "medium"),
            status=rec.get("status", "draft"),
            version=rec.get("version", "1.0.0"),
            description=rec.get("description", ""),
            tags=raw_tags,
            metadata=raw_metadata,
            created_at=rec.get("created_at", ""),
            updated_at=rec.get("updated_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_rule_set_response(
        self,
        rec: Dict[str, Any],
    ) -> RuleSetResponse:
        """Convert a raw engine dict to RuleSetResponse.

        Handles engine format differences: tags may be a list of
        strings from the RuleComposerEngine. Converts to dict.

        Args:
            rec: Dictionary from the RuleComposerEngine.

        Returns:
            RuleSetResponse model.
        """
        raw_tags = rec.get("tags", {})
        if isinstance(raw_tags, list):
            raw_tags = {t: "" for t in raw_tags}
        elif not isinstance(raw_tags, dict):
            raw_tags = {}

        # rule_count: engine may provide rule_ids list instead
        rule_count = rec.get("rule_count", 0)
        if rule_count == 0:
            rule_ids_list = rec.get("rule_ids", [])
            if rule_ids_list:
                rule_count = len(rule_ids_list)

        return RuleSetResponse(
            set_id=rec.get("set_id", rec.get("id", "")),
            name=rec.get("name", ""),
            description=rec.get("description", ""),
            version=rec.get("version", "1.0.0"),
            status=rec.get("status", "draft"),
            rule_count=rule_count,
            sla_thresholds=rec.get("sla_thresholds", {}),
            parent_set_id=rec.get("parent_set_id"),
            tags=raw_tags,
            created_at=rec.get("created_at", ""),
            updated_at=rec.get("updated_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_rule_pack_response(
        self,
        rec: Dict[str, Any],
    ) -> RulePackResponse:
        """Convert a raw engine dict to RulePackResponse.

        Args:
            rec: Dictionary from the RulePackEngine.

        Returns:
            RulePackResponse model.
        """
        return RulePackResponse(
            pack_name=rec.get("pack_name", rec.get("name", "")),
            pack_type=rec.get("pack_type", "custom"),
            version=rec.get("version", "1.0.0"),
            rules_count=rec.get("rules_count", 0),
            description=rec.get("description", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    # ==================================================================
    # Internal helpers: filtering
    # ==================================================================

    def _filter_rules(
        self,
        rules: List[RuleResponse],
        rule_type: Optional[str],
        severity: Optional[str],
        column: Optional[str],
        status: Optional[str],
        tag_key: Optional[str],
        query: Optional[str],
    ) -> List[RuleResponse]:
        """Filter rule response list by multiple criteria.

        Args:
            rules: List of RuleResponse instances.
            rule_type: Exact rule type filter.
            severity: Exact severity filter.
            column: Exact column name filter.
            status: Exact status filter.
            tag_key: Tag key presence filter.
            query: Substring search in name and description.

        Returns:
            Filtered list of RuleResponse instances.
        """
        result = rules
        if rule_type is not None:
            result = [r for r in result if r.rule_type == rule_type]
        if severity is not None:
            result = [r for r in result if r.severity == severity]
        if column is not None:
            result = [r for r in result if r.column == column]
        if status is not None:
            result = [r for r in result if r.status == status]
        if tag_key is not None:
            result = [r for r in result if tag_key in r.tags]
        if query is not None:
            q = query.lower()
            result = [
                r for r in result
                if q in r.name.lower() or q in r.description.lower()
            ]
        return result


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> ValidationRuleEngineService:
    """Get or create the singleton ValidationRuleEngineService instance.

    Returns:
        The singleton ValidationRuleEngineService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ValidationRuleEngineService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_validation_rule_engine(
    app: Any,
    config: Optional[ValidationRuleEngineConfig] = None,
) -> ValidationRuleEngineService:
    """Configure the Validation Rule Engine Service on a FastAPI application.

    Creates the ValidationRuleEngineService, stores it in app.state, mounts
    the validation rule engine API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional validation rule engine config.

    Returns:
        ValidationRuleEngineService instance.
    """
    global _singleton_instance

    service = ValidationRuleEngineService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.validation_rule_engine_service = service

    # Mount validation rule engine API router
    router = get_router()
    if router is not None:
        app.include_router(router)
        logger.info("Validation rule engine API router mounted")
    else:
        logger.warning(
            "Validation rule engine router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Validation rule engine service configured on app")
    return service


def get_validation_rule_engine() -> ValidationRuleEngineService:
    """Get the singleton ValidationRuleEngineService instance.

    Returns:
        ValidationRuleEngineService singleton instance.

    Raises:
        RuntimeError: If validation rule engine service not configured.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ValidationRuleEngineService()
    return _singleton_instance


def get_router(
    service: Optional[ValidationRuleEngineService] = None,
) -> Any:
    """Get the validation rule engine API router.

    Creates a FastAPI APIRouter with all 20 endpoints for the Validation
    Rule Engine service at prefix ``/api/v1/validation-rules``.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from fastapi import APIRouter, HTTPException, Query
        from fastapi.responses import JSONResponse
    except ImportError:
        return None

    router = APIRouter(
        prefix="/api/v1/validation-rules",
        tags=["validation-rules"],
    )

    def _svc() -> ValidationRuleEngineService:
        """Get the singleton service for route handlers."""
        return get_validation_rule_engine()

    # ------------------------------------------------------------------
    # 1. POST /rules - Register a new validation rule
    # ------------------------------------------------------------------
    @router.post("/rules", response_model=RuleResponse, status_code=201)
    async def post_register_rule(
        request: Dict[str, Any],
    ) -> RuleResponse:
        """Register a new validation rule in the rule registry."""
        try:
            return _svc().register_rule(
                name=request.get("name", ""),
                rule_type=request.get("rule_type", "range"),
                column=request.get("column", ""),
                operator=request.get("operator", "between"),
                threshold=request.get("threshold"),
                parameters=request.get("parameters"),
                severity=request.get("severity", "medium"),
                description=request.get("description", ""),
                tags=request.get("tags"),
                metadata=request.get("metadata"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. GET /rules - List / search validation rules
    # ------------------------------------------------------------------
    @router.get("/rules", response_model=List[RuleResponse])
    async def get_search_rules(
        rule_type: Optional[str] = Query(None),
        severity: Optional[str] = Query(None),
        column: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        tag_key: Optional[str] = Query(None),
        query: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[RuleResponse]:
        """Search validation rules with optional filtering."""
        return _svc().search_rules(
            rule_type=rule_type,
            severity=severity,
            column=column,
            status=status,
            tag_key=tag_key,
            query=query,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 3. GET /rules/{rule_id} - Get a validation rule by ID
    # ------------------------------------------------------------------
    @router.get("/rules/{rule_id}", response_model=RuleResponse)
    async def get_rule_by_id(rule_id: str) -> RuleResponse:
        """Get a validation rule by its unique identifier."""
        result = _svc().get_rule(rule_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        return result

    # ------------------------------------------------------------------
    # 4. PUT /rules/{rule_id} - Update a validation rule
    # ------------------------------------------------------------------
    @router.put("/rules/{rule_id}", response_model=RuleResponse)
    async def put_update_rule(
        rule_id: str,
        request: Dict[str, Any],
    ) -> RuleResponse:
        """Update mutable fields of an existing validation rule."""
        result = _svc().update_rule(
            rule_id=rule_id,
            name=request.get("name"),
            column=request.get("column"),
            operator=request.get("operator"),
            threshold=request.get("threshold"),
            parameters=request.get("parameters"),
            severity=request.get("severity"),
            status=request.get("status"),
            description=request.get("description"),
            tags=request.get("tags"),
            metadata=request.get("metadata"),
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        return result

    # ------------------------------------------------------------------
    # 5. DELETE /rules/{rule_id} - Soft-delete a validation rule
    # ------------------------------------------------------------------
    @router.delete("/rules/{rule_id}", status_code=204)
    async def delete_rule_by_id(rule_id: str) -> None:
        """Soft-delete a validation rule by archiving it."""
        deleted = _svc().delete_rule(rule_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Rule not found")

    # ------------------------------------------------------------------
    # 6. POST /rule-sets - Create a rule set
    # ------------------------------------------------------------------
    @router.post(
        "/rule-sets",
        response_model=RuleSetResponse,
        status_code=201,
    )
    async def post_create_rule_set(
        request: Dict[str, Any],
    ) -> RuleSetResponse:
        """Create a new rule set (named collection of rules)."""
        try:
            return _svc().create_rule_set(
                name=request.get("name", ""),
                description=request.get("description", ""),
                rule_ids=request.get("rule_ids"),
                sla_thresholds=request.get("sla_thresholds"),
                parent_set_id=request.get("parent_set_id"),
                tags=request.get("tags"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. GET /rule-sets - List rule sets
    # ------------------------------------------------------------------
    @router.get("/rule-sets", response_model=List[RuleSetResponse])
    async def get_list_rule_sets(
        status: Optional[str] = Query(None),
        tag_key: Optional[str] = Query(None),
        query: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[RuleSetResponse]:
        """List rule sets with optional filtering and pagination."""
        return _svc().list_rule_sets(
            status=status,
            tag_key=tag_key,
            query=query,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 8. GET /rule-sets/{set_id} - Get rule set details
    # ------------------------------------------------------------------
    @router.get(
        "/rule-sets/{set_id}",
        response_model=RuleSetResponse,
    )
    async def get_rule_set_by_id(set_id: str) -> RuleSetResponse:
        """Get a rule set by its unique identifier."""
        result = _svc().get_rule_set(set_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Rule set not found",
            )
        return result

    # ------------------------------------------------------------------
    # 9. PUT /rule-sets/{set_id} - Update a rule set
    # ------------------------------------------------------------------
    @router.put(
        "/rule-sets/{set_id}",
        response_model=RuleSetResponse,
    )
    async def put_update_rule_set(
        set_id: str,
        request: Dict[str, Any],
    ) -> RuleSetResponse:
        """Update mutable fields of an existing rule set."""
        result = _svc().update_rule_set(
            set_id=set_id,
            name=request.get("name"),
            description=request.get("description"),
            status=request.get("status"),
            sla_thresholds=request.get("sla_thresholds"),
            tags=request.get("tags"),
            rule_ids=request.get("rule_ids"),
        )
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Rule set not found",
            )
        return result

    # ------------------------------------------------------------------
    # 10. DELETE /rule-sets/{set_id} - Soft-delete a rule set
    # ------------------------------------------------------------------
    @router.delete("/rule-sets/{set_id}", status_code=204)
    async def delete_rule_set_by_id(set_id: str) -> None:
        """Soft-delete a rule set by archiving it."""
        deleted = _svc().delete_rule_set(set_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Rule set not found",
            )

    # ------------------------------------------------------------------
    # 11. POST /evaluate - Evaluate rules against data
    # ------------------------------------------------------------------
    @router.post("/evaluate", response_model=EvaluationResponse)
    async def post_evaluate_rules(
        request: Dict[str, Any],
    ) -> EvaluationResponse:
        """Evaluate a rule set against a dataset."""
        try:
            return _svc().evaluate_rules(
                rule_set_id=request.get("rule_set_id", ""),
                dataset_name=request.get("dataset_name", ""),
                data=request.get("data"),
                parameters=request.get("parameters"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. POST /evaluate/batch - Batch evaluate across datasets
    # ------------------------------------------------------------------
    @router.post(
        "/evaluate/batch",
        response_model=BatchEvaluationResponse,
    )
    async def post_evaluate_batch(
        request: Dict[str, Any],
    ) -> BatchEvaluationResponse:
        """Batch evaluate a rule set across multiple datasets."""
        try:
            return _svc().evaluate_batch(
                rule_set_id=request.get("rule_set_id", ""),
                datasets=request.get("datasets"),
                parameters=request.get("parameters"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. GET /evaluations/{eval_id} - Get evaluation results
    # ------------------------------------------------------------------
    @router.get(
        "/evaluations/{eval_id}",
        response_model=EvaluationResponse,
    )
    async def get_evaluation_by_id(eval_id: str) -> EvaluationResponse:
        """Get an evaluation result by its identifier."""
        result = _svc().get_evaluation(eval_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Evaluation not found",
            )
        return result

    # ------------------------------------------------------------------
    # 14. POST /conflicts/detect - Detect rule conflicts
    # ------------------------------------------------------------------
    @router.post(
        "/conflicts/detect",
        response_model=ConflictReportResponse,
    )
    async def post_detect_conflicts(
        request: Dict[str, Any],
    ) -> ConflictReportResponse:
        """Detect contradictory, overlapping, or redundant rules."""
        return _svc().detect_conflicts(
            rule_set_id=request.get("rule_set_id"),
            rule_ids=request.get("rule_ids"),
        )

    # ------------------------------------------------------------------
    # 15. GET /conflicts - List detected conflicts
    # ------------------------------------------------------------------
    @router.get(
        "/conflicts",
        response_model=List[ConflictReportResponse],
    )
    async def get_list_conflicts(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[ConflictReportResponse]:
        """List previously detected conflict reports."""
        return _svc().list_conflicts(
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 16. POST /packs/{pack_name}/apply - Apply a regulatory rule pack
    # ------------------------------------------------------------------
    @router.post(
        "/packs/{pack_name}/apply",
        response_model=RulePackResponse,
        status_code=201,
    )
    async def post_apply_pack(
        pack_name: str,
        request: Dict[str, Any],
    ) -> RulePackResponse:
        """Apply a regulatory rule pack to the rule registry."""
        try:
            return _svc().apply_pack(
                pack_name=pack_name,
                parameters=request.get("parameters"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. GET /packs - List available rule packs
    # ------------------------------------------------------------------
    @router.get("/packs", response_model=List[RulePackResponse])
    async def get_list_packs() -> List[RulePackResponse]:
        """List all available regulatory rule packs."""
        return _svc().list_packs()

    # ------------------------------------------------------------------
    # 18. POST /reports - Generate validation report
    # ------------------------------------------------------------------
    @router.post(
        "/reports",
        response_model=ValidationReportResponse,
        status_code=201,
    )
    async def post_generate_report(
        request: Dict[str, Any],
    ) -> ValidationReportResponse:
        """Generate a validation report in the specified format."""
        return _svc().generate_report(
            report_type=request.get("report_type", "evaluation_summary"),
            report_format=request.get("format", "json"),
            evaluation_id=request.get("evaluation_id"),
            rule_set_id=request.get("rule_set_id"),
            parameters=request.get("parameters"),
        )

    # ------------------------------------------------------------------
    # 19. POST /pipeline - Run validation pipeline
    # ------------------------------------------------------------------
    @router.post(
        "/pipeline",
        response_model=PipelineResultResponse,
    )
    async def post_run_pipeline(
        request: Dict[str, Any],
    ) -> PipelineResultResponse:
        """Run the full end-to-end validation pipeline."""
        return _svc().run_pipeline(
            rule_set_id=request.get("rule_set_id", ""),
            dataset_name=request.get("dataset_name", ""),
            data=request.get("data"),
            detect_conflicts=request.get("detect_conflicts", True),
            generate_report=request.get("generate_report", False),
            report_format=request.get("report_format", "json"),
            parameters=request.get("parameters"),
        )

    # ------------------------------------------------------------------
    # 20. GET /health - Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def get_health_check() -> Dict[str, Any]:
        """Perform a health check on the validation rule engine service."""
        return _svc().get_health()

    return router


# ===================================================================
# HealthResponse (lightweight model)
# ===================================================================


class HealthResponse(BaseModel):
    """Health check response for the VRE service."""

    model_config = {"extra": "forbid"}

    status: str = Field(default="healthy")
    service: str = Field(default="validation-rule-engine")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    timestamp: str = Field(default="")


# ===================================================================
# Response-model aliases (used by tests / __init__.py)
# ===================================================================

ConflictDetectionResponse = ConflictReportResponse
PackApplyResponse = RulePackResponse
ReportResponse = ValidationReportResponse
ValidationRuleStatisticsResponse = ValidationStatisticsResponse


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service class
    "ValidationRuleEngineService",
    # FastAPI integration
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
    "ConflictDetectionResponse",
    "ValidationReportResponse",
    "ReportResponse",
    "RulePackResponse",
    "PackApplyResponse",
    "PipelineResultResponse",
    "ValidationStatisticsResponse",
    "ValidationRuleStatisticsResponse",
    "HealthResponse",
]
