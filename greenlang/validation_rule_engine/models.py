# -*- coding: utf-8 -*-
"""
Validation Rule Engine Service Data Models - AGENT-DATA-019

Pydantic v2 data models for the Validation Rule Engine SDK. Attempts to
re-export Layer 1 enumerations, engines, and models from the Data Quality
Profiler (QualityRuleEngine, ValidityChecker, QualityDimension, RuleType),
and defines all SDK models for validation rule management, rule sets,
compound rules, rule packs, versioning, evaluation, conflict detection,
reporting, templates, dependencies, SLA thresholds, and audit trails.

Re-exported Layer 1 sources (best-effort, with fallback stubs):
    - greenlang.data_quality_profiler.quality_rule_engine: QualityRuleEngine
    - greenlang.data_quality_profiler.validity_checker: ValidityChecker
    - greenlang.data_quality_profiler.models: QualityDimension, RuleType

New enumerations (12):
    - ValidationRuleType, RuleOperator, RuleSeverity, RuleStatus,
      CompoundOperator, EvaluationResult, ConflictType, RulePackType,
      ReportType, ReportFormat, VersionBumpType, SLALevel

New SDK models (14):
    - ValidationRule, RuleSet, RuleSetMember, CompoundRule, RulePack,
      RuleVersion, EvaluationRun, EvaluationDetail, ConflictReport,
      ValidationReport, RuleTemplate, RuleDependency, SLAThreshold,
      AuditEntry

Request models (8):
    - CreateRuleRequest, UpdateRuleRequest, CreateRuleSetRequest,
      UpdateRuleSetRequest, EvaluateRequest, BatchEvaluateRequest,
      DetectConflictsRequest, GenerateReportRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Layer 1 Re-exports (best-effort with stubs on ImportError)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.quality_rule_engine import (  # type: ignore[import]
        QualityRuleEngine as L1QualityRuleEngine,
    )

    QualityRuleEngine = L1QualityRuleEngine
    _QRE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QRE_AVAILABLE = False

    class QualityRuleEngine:  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler.quality_rule_engine is unavailable."""

        pass


try:
    from greenlang.data_quality_profiler.validity_checker import (  # type: ignore[import]
        ValidityChecker as L1ValidityChecker,
    )

    ValidityChecker = L1ValidityChecker
    _VC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VC_AVAILABLE = False

    class ValidityChecker:  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler.validity_checker is unavailable."""

        pass


try:
    from greenlang.data_quality_profiler.models import (  # type: ignore[import]
        QualityDimension as L1QualityDimension,
    )

    QualityDimension = L1QualityDimension
    _QD_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QD_AVAILABLE = False

    class QualityDimension(str, Enum):  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler.models is unavailable."""

        COMPLETENESS = "completeness"
        VALIDITY = "validity"
        CONSISTENCY = "consistency"
        TIMELINESS = "timeliness"
        UNIQUENESS = "uniqueness"
        ACCURACY = "accuracy"


try:
    from greenlang.data_quality_profiler.models import (  # type: ignore[import]
        RuleType as L1RuleType,
    )

    RuleType = L1RuleType
    _RT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RT_AVAILABLE = False

    class RuleType(str, Enum):  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler.models is unavailable."""

        COMPLETENESS = "completeness"
        RANGE = "range"
        FORMAT = "format"
        UNIQUENESS = "uniqueness"
        CUSTOM = "custom"
        FRESHNESS = "freshness"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of validation rules per tenant / namespace.
MAX_RULES_PER_NAMESPACE: int = 100_000

#: Maximum number of rules that can be included in a single rule set.
MAX_RULES_PER_SET: int = 5_000

#: Maximum number of rule sets per tenant / namespace.
MAX_RULE_SETS_PER_NAMESPACE: int = 10_000

#: Maximum nesting depth for compound (AND/OR/NOT) rule expressions.
MAX_COMPOUND_NESTING_DEPTH: int = 10

#: Maximum number of versions retained per validation rule.
MAX_VERSIONS_PER_RULE: int = 500

#: Maximum number of records in a single batch evaluation request.
MAX_BATCH_RECORDS: int = 100_000

#: Default batch size for evaluation pipeline operations.
DEFAULT_EVALUATION_BATCH_SIZE: int = 1_000

#: Default confidence threshold for rule evaluation scoring.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8

#: Maximum number of conflicts that can be reported in a single detection run.
MAX_CONFLICTS_PER_REPORT: int = 10_000

#: Maximum number of evaluation detail entries stored per evaluation run.
MAX_EVALUATION_DETAILS_PER_RUN: int = 50_000

#: Severity ordering from least to most severe (for comparisons).
SEVERITY_ORDER: tuple = ("low", "medium", "high", "critical")

#: Default SLA threshold for critical rule execution time in milliseconds.
DEFAULT_CRITICAL_SLA_MS: float = 500.0

#: Default SLA threshold for all rules execution time in milliseconds.
DEFAULT_ALL_RULES_SLA_MS: float = 5_000.0

#: Supported report output formats.
SUPPORTED_REPORT_FORMATS: tuple = ("text", "json", "html", "markdown", "csv")

#: Rule pack framework identifiers for built-in compliance packs.
BUILT_IN_RULE_PACKS: tuple = ("ghg_protocol", "csrd_esrs", "eudr", "soc2")


# =============================================================================
# Enumerations (12)
# =============================================================================


class ValidationRuleType(str, Enum):
    """Type classification for a validation rule.

    Determines the evaluation strategy the rule engine uses to assess
    data against this rule. Each type implies different parameter
    schemas, evaluation logic, and applicable operators.

    COMPLETENESS: Checks that required fields are present and non-null.
    RANGE: Validates numeric or date values fall within allowed bounds.
    FORMAT: Validates string values match a pattern (regex, format mask).
    UNIQUENESS: Ensures values within a field or field combination are unique.
    CUSTOM: User-defined validation logic expressed as a Python expression.
    FRESHNESS: Verifies data recency against an age threshold.
    CROSS_FIELD: Validates relationships or constraints across multiple fields.
    CONDITIONAL: Applies validation only when a precondition is satisfied.
    STATISTICAL: Validates values against statistical distribution properties.
    REFERENTIAL: Validates foreign-key-style references to a lookup dataset.
    """

    COMPLETENESS = "completeness"
    RANGE = "range"
    FORMAT = "format"
    UNIQUENESS = "uniqueness"
    CUSTOM = "custom"
    FRESHNESS = "freshness"
    CROSS_FIELD = "cross_field"
    CONDITIONAL = "conditional"
    STATISTICAL = "statistical"
    REFERENTIAL = "referential"


class RuleOperator(str, Enum):
    """Comparison operators for validation rule evaluation.

    Defines how rule thresholds or expected values are compared
    against actual data values during validation. Operators are
    type-aware: numeric operators (GREATER_THAN, LESS_THAN, etc.)
    apply to numbers and dates, while pattern operators (MATCHES,
    CONTAINS) apply to strings.

    EQUALS: Value must exactly equal the expected value.
    NOT_EQUALS: Value must not equal the expected value.
    GREATER_THAN: Value must be strictly greater than the threshold.
    LESS_THAN: Value must be strictly less than the threshold.
    GREATER_EQUAL: Value must be greater than or equal to the threshold.
    LESS_EQUAL: Value must be less than or equal to the threshold.
    BETWEEN: Value must fall within an inclusive [min, max] range.
    MATCHES: String value must match the given regular expression.
    CONTAINS: String value must contain the given substring.
    IN_SET: Value must be a member of a specified set of allowed values.
    NOT_IN_SET: Value must not be a member of a specified set of values.
    IS_NULL: Value must be null or absent (for optional field checks).
    """

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    BETWEEN = "between"
    MATCHES = "matches"
    CONTAINS = "contains"
    IN_SET = "in_set"
    NOT_IN_SET = "not_in_set"
    IS_NULL = "is_null"


class RuleSeverity(str, Enum):
    """Severity classification for a validation rule violation.

    Determines the urgency and impact of a rule failure. Severity
    drives alert routing, SLA enforcement, and gate-pass decisions
    in compliance pipelines.

    CRITICAL: Immediate compliance risk; blocks pipeline progression.
    HIGH: Significant data quality issue; requires prompt remediation.
    MEDIUM: Moderate issue; should be addressed in the current cycle.
    LOW: Minor or informational issue; fix at convenience.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleStatus(str, Enum):
    """Lifecycle status of a validation rule definition.

    Controls whether the rule is evaluated during rule set execution
    and whether it can be modified or deleted.

    DRAFT: Rule is under development; not evaluated in production runs.
    ACTIVE: Rule is in production use and evaluated in all applicable runs.
    DEPRECATED: Rule is superseded; consumers should migrate to successor.
    ARCHIVED: Rule is retired and excluded from all evaluation runs.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class CompoundOperator(str, Enum):
    """Logical operators for combining validation rules into compound expressions.

    Used to construct complex validation logic by composing multiple
    individual rules into boolean expressions with nesting support.

    AND: All child rules must pass for the compound rule to pass.
    OR: At least one child rule must pass for the compound rule to pass.
    NOT: The single child rule must fail for the compound rule to pass.
    """

    AND = "and"
    OR = "or"
    NOT = "not"


class EvaluationResult(str, Enum):
    """Outcome of a single validation rule evaluation against a data record.

    PASS_RESULT: The data satisfies the rule with no issues.
    WARN: The data marginally satisfies the rule but triggers a warning.
    FAIL: The data violates the rule and fails validation.

    Note:
        The value "pass" is used for serialization; the Python member
        name is PASS_RESULT to avoid conflict with the reserved keyword.
    """

    PASS_RESULT = "pass"
    WARN = "warn"
    FAIL = "fail"


class ConflictType(str, Enum):
    """Type of conflict detected between validation rules.

    Identifies logical inconsistencies within a rule set that
    could cause contradictory evaluation results or unreachable
    pass conditions for data records.

    RANGE_OVERLAP: Two range rules have overlapping but inconsistent bounds.
    RANGE_CONTRADICTION: Two range rules create an impossible pass condition.
    FORMAT_CONFLICT: Two format rules specify incompatible patterns.
    SEVERITY_INCONSISTENCY: Same violation produces different severities.
    REDUNDANCY: Two rules evaluate the same condition identically.
    """

    RANGE_OVERLAP = "range_overlap"
    RANGE_CONTRADICTION = "range_contradiction"
    FORMAT_CONFLICT = "format_conflict"
    SEVERITY_INCONSISTENCY = "severity_inconsistency"
    REDUNDANCY = "redundancy"


class RulePackType(str, Enum):
    """Classification of a pre-built validation rule pack.

    Determines which compliance framework or standard the rule pack
    implements and which fields, thresholds, and severity mappings
    are included in the pack.

    GHG_PROTOCOL: GHG Protocol Scope 1/2/3 data quality validation rules.
    CSRD_ESRS: EU Corporate Sustainability Reporting Directive ESRS rules.
    EUDR: EU Deforestation Regulation validation rules.
    SOC2: SOC 2 Type II data quality and audit trail rules.
    CUSTOM: User-defined rule pack with custom validation rules.
    """

    GHG_PROTOCOL = "ghg_protocol"
    CSRD_ESRS = "csrd_esrs"
    EUDR = "eudr"
    SOC2 = "soc2"
    CUSTOM = "custom"


class ReportType(str, Enum):
    """Type of validation report to generate.

    Determines the structure, content granularity, and audience
    for the generated validation report.

    SUMMARY: High-level pass/fail counts and severity breakdown.
    DETAILED: Record-level evaluation results with full diagnostics.
    COMPLIANCE: Framework-aligned compliance evidence report.
    TREND: Historical trend analysis of validation results over time.
    EXECUTIVE: Executive summary with key metrics and recommendations.
    """

    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPLIANCE = "compliance"
    TREND = "trend"
    EXECUTIVE = "executive"


class ReportFormat(str, Enum):
    """Output format for a validation report.

    TEXT: Plain-text summary for terminal or log output.
    JSON: Structured JSON for programmatic consumption.
    HTML: Self-contained HTML page with formatting and styling.
    MARKDOWN: Markdown-formatted report for documentation systems.
    CSV: Comma-separated values for spreadsheet import.
    """

    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


class VersionBumpType(str, Enum):
    """Semantic version bump classification for a rule change.

    Follows SemVer conventions to classify the nature of a change
    to a validation rule definition.

    MAJOR: Breaking change that alters evaluation behavior significantly.
    MINOR: Additive change that extends rule coverage without breaking.
    PATCH: Cosmetic or documentation change with no evaluation impact.
    """

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


class SLALevel(str, Enum):
    """SLA enforcement level for validation rule evaluation performance.

    Determines which rules are subject to performance SLA tracking
    and whether custom thresholds override defaults.

    CRITICAL_RULES: SLA applies only to rules with CRITICAL severity.
    ALL_RULES: SLA applies to all rules in the evaluation run.
    CUSTOM: SLA applies to a user-defined subset of rules.
    """

    CRITICAL_RULES = "critical_rules"
    ALL_RULES = "all_rules"
    CUSTOM = "custom"


# =============================================================================
# SDK Data Models (14)
# =============================================================================


class ValidationRule(BaseModel):
    """A single validation rule definition in the Validation Rule Engine.

    Represents an atomic data quality check that can be applied to one or
    more fields of a dataset. Each rule specifies the validation type,
    operator, expected value or threshold, severity on violation, and
    metadata for governance and audit. Rules are versioned and carry
    SHA-256 provenance hashes for tamper-evident audit trails.

    Attributes:
        id: Unique rule identifier (UUID v4).
        name: Human-readable rule name, unique within a namespace.
        description: Detailed description of what this rule validates.
        rule_type: Classification of the validation check type.
        operator: Comparison operator for threshold evaluation.
        target_field: Name of the dataset field this rule validates.
        target_fields: List of fields for cross-field or multi-field rules.
        expected_value: Expected value for equality or membership checks.
        threshold_min: Minimum threshold for range-based validation.
        threshold_max: Maximum threshold for range-based validation.
        pattern: Regular expression pattern for FORMAT rule types.
        allowed_values: Set of allowed values for IN_SET operator rules.
        severity: Severity classification for rule violations.
        status: Current lifecycle status of this rule definition.
        namespace: Tenant or organizational namespace for isolation.
        tags: Arbitrary key-value labels for filtering and discovery.
        framework: Optional compliance framework association.
        expression: Custom Python expression for CUSTOM rule types.
        condition: Precondition expression for CONDITIONAL rule types.
        parameters: Additional rule-specific parameters and configuration.
        version: Current version number of this rule definition.
        provenance_hash: SHA-256 hash of the rule definition for audit.
        created_by: Actor (user or service) that created the rule.
        created_at: UTC timestamp when the rule was first created.
        updated_at: UTC timestamp when the rule was last modified.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique rule identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable rule name, unique within a namespace",
    )
    description: str = Field(
        default="",
        description="Detailed description of what this rule validates",
    )
    rule_type: ValidationRuleType = Field(
        ...,
        description="Classification of the validation check type",
    )
    operator: RuleOperator = Field(
        default=RuleOperator.EQUALS,
        description="Comparison operator for threshold evaluation",
    )
    target_field: str = Field(
        default="",
        description="Name of the dataset field this rule validates",
    )
    target_fields: List[str] = Field(
        default_factory=list,
        description="List of fields for cross-field or multi-field rules",
    )
    expected_value: Optional[Any] = Field(
        None,
        description="Expected value for equality or membership checks",
    )
    threshold_min: Optional[float] = Field(
        None,
        description="Minimum threshold for range-based validation",
    )
    threshold_max: Optional[float] = Field(
        None,
        description="Maximum threshold for range-based validation",
    )
    pattern: Optional[str] = Field(
        None,
        description="Regular expression pattern for FORMAT rule types",
    )
    allowed_values: List[Any] = Field(
        default_factory=list,
        description="Set of allowed values for IN_SET operator rules",
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.MEDIUM,
        description="Severity classification for rule violations",
    )
    status: RuleStatus = Field(
        default=RuleStatus.DRAFT,
        description="Current lifecycle status of this rule definition",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and discovery",
    )
    framework: Optional[str] = Field(
        None,
        description="Optional compliance framework association (e.g., 'ghg_protocol')",
    )
    expression: Optional[str] = Field(
        None,
        description="Custom Python expression for CUSTOM rule types",
    )
    condition: Optional[str] = Field(
        None,
        description="Precondition expression for CONDITIONAL rule types",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rule-specific parameters and configuration",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Current version number of this rule definition",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the rule definition for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor (user or service) that created the rule",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule was first created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("rule_type")
    @classmethod
    def validate_rule_type(cls, v: ValidationRuleType) -> ValidationRuleType:
        """Validate rule_type is a valid enum member."""
        if not isinstance(v, ValidationRuleType):
            raise ValueError(f"Invalid rule_type: {v}")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        """Validate version is a positive integer."""
        if v < 1:
            raise ValueError(f"version must be >= 1, got {v}")
        return v


class RuleSet(BaseModel):
    """A named collection of validation rules evaluated as a unit.

    Rule sets group related validation rules for batch evaluation
    against a dataset. A rule set defines the evaluation scope,
    gate-pass criteria, and ordering of rule execution.

    Attributes:
        id: Unique rule set identifier (UUID v4).
        name: Human-readable rule set name, unique within a namespace.
        description: Detailed description of the rule set's purpose.
        namespace: Tenant or organizational namespace for isolation.
        status: Current lifecycle status of this rule set.
        framework: Optional compliance framework association.
        gate_pass_threshold: Minimum pass rate (0.0-1.0) to pass the gate.
        fail_on_critical: Whether any CRITICAL severity failure fails the gate.
        evaluation_order: Ordered list of rule IDs for sequential evaluation.
        tags: Arbitrary key-value labels for filtering and discovery.
        rule_count: Number of rules currently in this rule set.
        version: Current version number of this rule set definition.
        provenance_hash: SHA-256 hash of the rule set definition for audit.
        created_by: Actor (user or service) that created the rule set.
        created_at: UTC timestamp when the rule set was first created.
        updated_at: UTC timestamp when the rule set was last modified.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique rule set identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable rule set name, unique within a namespace",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule set's purpose",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    status: RuleStatus = Field(
        default=RuleStatus.DRAFT,
        description="Current lifecycle status of this rule set",
    )
    framework: Optional[str] = Field(
        None,
        description="Optional compliance framework association (e.g., 'csrd_esrs')",
    )
    gate_pass_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum pass rate (0.0 to 1.0) to pass the quality gate",
    )
    fail_on_critical: bool = Field(
        default=True,
        description="Whether any CRITICAL severity failure automatically fails the gate",
    )
    evaluation_order: List[str] = Field(
        default_factory=list,
        description="Ordered list of rule IDs for sequential evaluation",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and discovery",
    )
    rule_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules currently in this rule set",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Current version number of this rule set definition",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the rule set definition for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor (user or service) that created the rule set",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule set was first created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule set was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("gate_pass_threshold")
    @classmethod
    def validate_gate_pass_threshold(cls, v: float) -> float:
        """Validate gate_pass_threshold is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"gate_pass_threshold must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        """Validate version is a positive integer."""
        if v < 1:
            raise ValueError(f"version must be >= 1, got {v}")
        return v


class RuleSetMember(BaseModel):
    """Membership record linking a validation rule to a rule set.

    Tracks which rules belong to which rule sets with ordering,
    override parameters, and activation status. Enables the same
    rule to belong to multiple rule sets with different configurations.

    Attributes:
        id: Unique membership record identifier (UUID v4).
        rule_set_id: ID of the parent rule set.
        rule_id: ID of the validation rule in this membership.
        ordinal: Execution order position within the rule set (1-based).
        enabled: Whether this rule is active within the rule set.
        severity_override: Optional severity override for this rule in this set.
        parameters_override: Optional parameter overrides for this membership.
        added_by: Actor that added this rule to the rule set.
        added_at: UTC timestamp when the rule was added to the set.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique membership record identifier (UUID v4)",
    )
    rule_set_id: str = Field(
        ...,
        description="ID of the parent rule set",
    )
    rule_id: str = Field(
        ...,
        description="ID of the validation rule in this membership",
    )
    ordinal: int = Field(
        default=0,
        ge=0,
        description="Execution order position within the rule set (1-based)",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this rule is active within the rule set",
    )
    severity_override: Optional[RuleSeverity] = Field(
        None,
        description="Optional severity override for this rule in this set",
    )
    parameters_override: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameter overrides for this membership context",
    )
    added_by: str = Field(
        default="system",
        description="Actor that added this rule to the rule set",
    )
    added_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule was added to the set",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_set_id")
    @classmethod
    def validate_rule_set_id(cls, v: str) -> str:
        """Validate rule_set_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_set_id must be non-empty")
        return v

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Validate rule_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_id must be non-empty")
        return v


class CompoundRule(BaseModel):
    """A compound validation rule composed of multiple child rules.

    Enables construction of complex validation logic by combining
    individual rules with boolean operators (AND, OR, NOT). Compound
    rules support nesting up to MAX_COMPOUND_NESTING_DEPTH levels
    for arbitrarily complex validation expressions.

    Attributes:
        id: Unique compound rule identifier (UUID v4).
        name: Human-readable compound rule name.
        description: Description of the compound validation logic.
        operator: Boolean operator combining the child rules.
        child_rule_ids: List of child validation rule IDs (for AND/OR).
        child_compound_ids: List of nested compound rule IDs (for nesting).
        negated_rule_id: Single child rule ID (for NOT operator).
        severity: Severity classification for compound rule violations.
        status: Current lifecycle status of this compound rule.
        nesting_depth: Current nesting depth of this compound rule.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the compound rule for audit trail.
        created_by: Actor that created the compound rule.
        created_at: UTC timestamp when the compound rule was created.
        updated_at: UTC timestamp when the compound rule was last modified.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique compound rule identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable compound rule name",
    )
    description: str = Field(
        default="",
        description="Description of the compound validation logic",
    )
    operator: CompoundOperator = Field(
        ...,
        description="Boolean operator combining the child rules",
    )
    child_rule_ids: List[str] = Field(
        default_factory=list,
        description="List of child validation rule IDs (for AND/OR operators)",
    )
    child_compound_ids: List[str] = Field(
        default_factory=list,
        description="List of nested compound rule IDs (for nesting support)",
    )
    negated_rule_id: Optional[str] = Field(
        None,
        description="Single child rule ID (used only with NOT operator)",
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.MEDIUM,
        description="Severity classification for compound rule violations",
    )
    status: RuleStatus = Field(
        default=RuleStatus.DRAFT,
        description="Current lifecycle status of this compound rule",
    )
    nesting_depth: int = Field(
        default=1,
        ge=1,
        le=MAX_COMPOUND_NESTING_DEPTH,
        description="Current nesting depth of this compound rule",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the compound rule for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor that created the compound rule",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the compound rule was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the compound rule was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("nesting_depth")
    @classmethod
    def validate_nesting_depth(cls, v: int) -> int:
        """Validate nesting_depth does not exceed the maximum."""
        if v > MAX_COMPOUND_NESTING_DEPTH:
            raise ValueError(
                f"nesting_depth must be <= {MAX_COMPOUND_NESTING_DEPTH}, got {v}"
            )
        return v


class RulePack(BaseModel):
    """A pre-built collection of validation rules for a compliance framework.

    Rule packs bundle validation rules that implement the data quality
    requirements of a specific regulatory framework. Packs can be
    installed into a namespace to immediately apply framework-specific
    validation checks to datasets.

    Attributes:
        id: Unique rule pack identifier (UUID v4).
        name: Human-readable rule pack name.
        description: Detailed description of the rule pack and its coverage.
        pack_type: Compliance framework or classification of this pack.
        version: Current version string of the rule pack (SemVer).
        rule_ids: List of validation rule IDs included in this pack.
        rule_set_ids: List of rule set IDs that organize pack rules.
        framework_version: Version of the compliance framework implemented.
        coverage_areas: List of compliance areas or topics covered by the pack.
        total_rules: Total number of rules in this pack.
        tags: Arbitrary key-value labels for filtering and discovery.
        is_built_in: Whether this is a platform-provided built-in pack.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the rule pack for audit trail.
        published_by: Actor that published this rule pack.
        published_at: UTC timestamp when the rule pack was published.
        updated_at: UTC timestamp when the rule pack was last modified.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique rule pack identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable rule pack name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule pack and its coverage",
    )
    pack_type: RulePackType = Field(
        ...,
        description="Compliance framework or classification of this pack",
    )
    version: str = Field(
        default="1.0.0",
        description="Current version string of the rule pack (SemVer)",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="List of validation rule IDs included in this pack",
    )
    rule_set_ids: List[str] = Field(
        default_factory=list,
        description="List of rule set IDs that organize pack rules",
    )
    framework_version: str = Field(
        default="",
        description="Version of the compliance framework implemented",
    )
    coverage_areas: List[str] = Field(
        default_factory=list,
        description="List of compliance areas or topics covered by the pack",
    )
    total_rules: int = Field(
        default=0,
        ge=0,
        description="Total number of rules in this pack",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and discovery",
    )
    is_built_in: bool = Field(
        default=False,
        description="Whether this is a platform-provided built-in pack",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the rule pack for audit trail",
    )
    published_by: str = Field(
        default="system",
        description="Actor that published this rule pack",
    )
    published_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule pack was published",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rule pack was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version is non-empty."""
        if not v or not v.strip():
            raise ValueError("version must be non-empty")
        return v


class RuleVersion(BaseModel):
    """A versioned snapshot of a validation rule definition.

    Each time a rule is modified, a new RuleVersion is created to
    preserve the complete change history. Versions enable rollback,
    audit comparison, and impact analysis of rule changes over time.

    Attributes:
        id: Unique version record identifier (UUID v4).
        rule_id: ID of the validation rule this version belongs to.
        version_number: Sequential version number (1-based).
        bump_type: Classification of the version change (major/minor/patch).
        rule_snapshot: Complete serialized state of the rule at this version.
        change_summary: Human-readable summary of changes in this version.
        changed_fields: List of field names that changed in this version.
        previous_version_id: ID of the immediately preceding version record.
        provenance_hash: SHA-256 hash of this version for audit trail.
        created_by: Actor that created this version.
        created_at: UTC timestamp when this version was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique version record identifier (UUID v4)",
    )
    rule_id: str = Field(
        ...,
        description="ID of the validation rule this version belongs to",
    )
    version_number: int = Field(
        default=1,
        ge=1,
        description="Sequential version number (1-based)",
    )
    bump_type: VersionBumpType = Field(
        default=VersionBumpType.PATCH,
        description="Classification of the version change (major/minor/patch)",
    )
    rule_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete serialized state of the rule at this version",
    )
    change_summary: str = Field(
        default="",
        description="Human-readable summary of changes in this version",
    )
    changed_fields: List[str] = Field(
        default_factory=list,
        description="List of field names that changed in this version",
    )
    previous_version_id: Optional[str] = Field(
        None,
        description="ID of the immediately preceding version record",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of this version for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor that created this version",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when this version was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Validate rule_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_id must be non-empty")
        return v

    @field_validator("version_number")
    @classmethod
    def validate_version_number(cls, v: int) -> int:
        """Validate version_number is a positive integer."""
        if v < 1:
            raise ValueError(f"version_number must be >= 1, got {v}")
        return v


class EvaluationRun(BaseModel):
    """A completed evaluation run of a rule set against a dataset.

    Captures aggregate results, performance metrics, and gate-pass
    outcome for a single invocation of the validation rule engine
    against a batch of data records.

    Attributes:
        id: Unique evaluation run identifier (UUID v4).
        rule_set_id: ID of the rule set that was evaluated.
        dataset_id: Optional identifier for the evaluated dataset.
        total_records: Total number of data records evaluated.
        total_rules: Total number of rules evaluated in this run.
        passed_count: Number of rule-record evaluations that passed.
        warned_count: Number of rule-record evaluations with warnings.
        failed_count: Number of rule-record evaluations that failed.
        pass_rate: Fraction of evaluations that passed (0.0 to 1.0).
        gate_result: Overall gate-pass result for this evaluation run.
        critical_failures: Number of CRITICAL severity failures.
        high_failures: Number of HIGH severity failures.
        medium_failures: Number of MEDIUM severity failures.
        low_failures: Number of LOW severity failures.
        duration_ms: Total wall-clock duration in milliseconds.
        sla_met: Whether the evaluation met the SLA time threshold.
        provenance_hash: SHA-256 hash of the evaluation results for audit.
        evaluated_by: Actor or service that triggered the evaluation.
        started_at: UTC timestamp when the evaluation began.
        completed_at: UTC timestamp when the evaluation completed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evaluation run identifier (UUID v4)",
    )
    rule_set_id: str = Field(
        ...,
        description="ID of the rule set that was evaluated",
    )
    dataset_id: str = Field(
        default="",
        description="Optional identifier for the evaluated dataset",
    )
    total_records: int = Field(
        default=0,
        ge=0,
        description="Total number of data records evaluated",
    )
    total_rules: int = Field(
        default=0,
        ge=0,
        description="Total number of rules evaluated in this run",
    )
    passed_count: int = Field(
        default=0,
        ge=0,
        description="Number of rule-record evaluations that passed",
    )
    warned_count: int = Field(
        default=0,
        ge=0,
        description="Number of rule-record evaluations with warnings",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of rule-record evaluations that failed",
    )
    pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of evaluations that passed (0.0 to 1.0)",
    )
    gate_result: EvaluationResult = Field(
        default=EvaluationResult.FAIL,
        description="Overall gate-pass result for this evaluation run",
    )
    critical_failures: int = Field(
        default=0,
        ge=0,
        description="Number of CRITICAL severity failures",
    )
    high_failures: int = Field(
        default=0,
        ge=0,
        description="Number of HIGH severity failures",
    )
    medium_failures: int = Field(
        default=0,
        ge=0,
        description="Number of MEDIUM severity failures",
    )
    low_failures: int = Field(
        default=0,
        ge=0,
        description="Number of LOW severity failures",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock duration in milliseconds",
    )
    sla_met: bool = Field(
        default=True,
        description="Whether the evaluation met the SLA time threshold",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the evaluation results for audit trail",
    )
    evaluated_by: str = Field(
        default="system",
        description="Actor or service that triggered the evaluation",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the evaluation began",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the evaluation completed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_set_id")
    @classmethod
    def validate_rule_set_id(cls, v: str) -> str:
        """Validate rule_set_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_set_id must be non-empty")
        return v

    @field_validator("pass_rate")
    @classmethod
    def validate_pass_rate(cls, v: float) -> float:
        """Validate pass_rate is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"pass_rate must be between 0.0 and 1.0, got {v}")
        return v


class EvaluationDetail(BaseModel):
    """A single rule-record evaluation result within an evaluation run.

    Records the outcome of applying one validation rule to one data
    record. Provides the actual value, expected value, comparison
    result, and diagnostic message for audit and debugging.

    Attributes:
        id: Unique evaluation detail identifier (UUID v4).
        evaluation_run_id: ID of the parent evaluation run.
        rule_id: ID of the validation rule that was evaluated.
        record_index: Zero-based index of the evaluated data record.
        record_id: Optional unique identifier of the evaluated record.
        result: Outcome of this rule-record evaluation (pass/warn/fail).
        actual_value: The actual value found in the data record.
        expected_value: The expected value or threshold from the rule.
        field_name: Name of the field that was evaluated.
        message: Human-readable diagnostic message explaining the result.
        severity: Severity classification of this evaluation result.
        duration_ms: Time taken to evaluate this single rule-record pair.
        metadata: Additional diagnostic metadata for this evaluation.
        evaluated_at: UTC timestamp when the evaluation was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evaluation detail identifier (UUID v4)",
    )
    evaluation_run_id: str = Field(
        ...,
        description="ID of the parent evaluation run",
    )
    rule_id: str = Field(
        ...,
        description="ID of the validation rule that was evaluated",
    )
    record_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based index of the evaluated data record",
    )
    record_id: str = Field(
        default="",
        description="Optional unique identifier of the evaluated record",
    )
    result: EvaluationResult = Field(
        ...,
        description="Outcome of this rule-record evaluation (pass/warn/fail)",
    )
    actual_value: Optional[Any] = Field(
        None,
        description="The actual value found in the data record",
    )
    expected_value: Optional[Any] = Field(
        None,
        description="The expected value or threshold from the rule definition",
    )
    field_name: str = Field(
        default="",
        description="Name of the field that was evaluated",
    )
    message: str = Field(
        default="",
        description="Human-readable diagnostic message explaining the result",
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.MEDIUM,
        description="Severity classification of this evaluation result",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to evaluate this single rule-record pair (ms)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic metadata for this evaluation",
    )
    evaluated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the evaluation was performed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("evaluation_run_id")
    @classmethod
    def validate_evaluation_run_id(cls, v: str) -> str:
        """Validate evaluation_run_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("evaluation_run_id must be non-empty")
        return v

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Validate rule_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_id must be non-empty")
        return v


class ConflictReport(BaseModel):
    """Result of a conflict detection analysis across validation rules.

    Produced by the conflict detection engine to identify logical
    inconsistencies between rules within a rule set or across
    rule sets that could lead to contradictory evaluation results.

    Attributes:
        id: Unique conflict report identifier (UUID v4).
        scope: Scope of the conflict analysis (e.g., rule_set_id or "all").
        total_rules_analyzed: Number of rules included in the analysis.
        total_conflicts: Total number of conflicts detected.
        conflicts: List of individual conflict details.
        conflict_by_type: Count of conflicts broken down by ConflictType.
        recommendations: List of suggested remediation actions.
        resolution_required: Whether any conflict requires manual resolution.
        provenance_hash: SHA-256 hash of the conflict report for audit.
        analyzed_at: UTC timestamp when the analysis was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique conflict report identifier (UUID v4)",
    )
    scope: str = Field(
        default="all",
        description="Scope of the conflict analysis (e.g., rule_set_id or 'all')",
    )
    total_rules_analyzed: int = Field(
        default=0,
        ge=0,
        description="Number of rules included in the analysis",
    )
    total_conflicts: int = Field(
        default=0,
        ge=0,
        description="Total number of conflicts detected",
    )
    conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of individual conflict details",
    )
    conflict_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of conflicts broken down by ConflictType",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of suggested remediation actions",
    )
    resolution_required: bool = Field(
        default=False,
        description="Whether any conflict requires manual resolution",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the conflict report for audit trail",
    )
    analyzed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the analysis was performed",
    )

    model_config = {"extra": "forbid"}


class ValidationReport(BaseModel):
    """A generated validation report in a specified format.

    Produced by the reporting engine to render validation results
    for compliance evidence, data governance documentation, or
    operational dashboards. Reports are immutable once generated
    and include a SHA-256 hash for tamper detection.

    Attributes:
        id: Unique report identifier (UUID v4).
        report_type: Type of validation report (summary, detailed, etc.).
        report_format: Output format (text, json, html, markdown, csv).
        evaluation_run_id: Optional reference to the evaluation run.
        rule_set_id: Optional reference to the evaluated rule set.
        scope: Scope of the report (e.g., "full", "rule_set:xyz").
        parameters: Report generation parameters and configuration.
        content: The rendered report content as a string.
        report_hash: SHA-256 hash of the report content for tamper detection.
        total_rules: Total number of rules covered in the report.
        total_records: Total number of records covered in the report.
        pass_rate: Overall pass rate reported (0.0 to 1.0).
        severity_summary: Failure counts by severity level.
        framework_compliance: Optional framework compliance percentage (0.0-1.0).
        recommendations: List of suggested remediation actions.
        generated_by: Actor (user or service) that requested the report.
        generated_at: UTC timestamp when the report was generated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier (UUID v4)",
    )
    report_type: ReportType = Field(
        default=ReportType.SUMMARY,
        description="Type of validation report (summary, detailed, etc.)",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format (text, json, html, markdown, csv)",
    )
    evaluation_run_id: Optional[str] = Field(
        None,
        description="Optional reference to the evaluation run",
    )
    rule_set_id: Optional[str] = Field(
        None,
        description="Optional reference to the evaluated rule set",
    )
    scope: str = Field(
        default="full",
        description="Scope of the report (e.g., 'full', 'rule_set:xyz')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report generation parameters and configuration",
    )
    content: str = Field(
        default="",
        description="The rendered report content as a string",
    )
    report_hash: str = Field(
        default="",
        description="SHA-256 hash of the report content for tamper detection",
    )
    total_rules: int = Field(
        default=0,
        ge=0,
        description="Total number of rules covered in the report",
    )
    total_records: int = Field(
        default=0,
        ge=0,
        description="Total number of records covered in the report",
    )
    pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall pass rate reported (0.0 to 1.0)",
    )
    severity_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Failure counts by severity level",
    )
    framework_compliance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional framework compliance percentage (0.0 to 1.0)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of suggested remediation actions",
    )
    generated_by: str = Field(
        default="system",
        description="Actor (user or service) that requested the report",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the report was generated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("pass_rate")
    @classmethod
    def validate_pass_rate(cls, v: float) -> float:
        """Validate pass_rate is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"pass_rate must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("framework_compliance")
    @classmethod
    def validate_framework_compliance(cls, v: Optional[float]) -> Optional[float]:
        """Validate framework_compliance is in range [0.0, 1.0] if set."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"framework_compliance must be between 0.0 and 1.0, got {v}"
            )
        return v


class RuleTemplate(BaseModel):
    """A reusable template for creating validation rules.

    Templates capture common validation patterns with parameterized
    placeholders that are resolved when a concrete rule is instantiated
    from the template. Reduces duplication and ensures consistency
    across similar rules within and across rule sets.

    Attributes:
        id: Unique template identifier (UUID v4).
        name: Human-readable template name.
        description: Detailed description of the template's purpose.
        rule_type: Default validation rule type for instantiated rules.
        operator: Default operator for instantiated rules.
        severity: Default severity for instantiated rules.
        parameter_schema: JSON-Schema describing template parameters.
        default_parameters: Default values for template parameters.
        target_field_pattern: Pattern for matching applicable field names.
        expression_template: Template expression with placeholders.
        tags: Arbitrary key-value labels for filtering and discovery.
        usage_count: Number of rules instantiated from this template.
        namespace: Tenant or organizational namespace for isolation.
        created_by: Actor that created the template.
        created_at: UTC timestamp when the template was created.
        updated_at: UTC timestamp when the template was last modified.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique template identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template's purpose",
    )
    rule_type: ValidationRuleType = Field(
        default=ValidationRuleType.CUSTOM,
        description="Default validation rule type for instantiated rules",
    )
    operator: RuleOperator = Field(
        default=RuleOperator.EQUALS,
        description="Default operator for instantiated rules",
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.MEDIUM,
        description="Default severity for instantiated rules",
    )
    parameter_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-Schema describing template parameters",
    )
    default_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default values for template parameters",
    )
    target_field_pattern: str = Field(
        default="*",
        description="Pattern for matching applicable field names",
    )
    expression_template: str = Field(
        default="",
        description="Template expression with placeholders (e.g., '{field} > {min}')",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and discovery",
    )
    usage_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules instantiated from this template",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    created_by: str = Field(
        default="system",
        description="Actor that created the template",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the template was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the template was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class RuleDependency(BaseModel):
    """A declared dependency between two validation rules.

    Tracks execution order constraints where one rule must be
    evaluated before another. Used by the evaluation engine to
    determine correct evaluation sequencing and to detect
    circular dependency chains.

    Attributes:
        id: Unique dependency record identifier (UUID v4).
        rule_id: ID of the dependent rule (the rule that requires another).
        depends_on_rule_id: ID of the prerequisite rule.
        dependency_type: Nature of the dependency (e.g., "data", "result").
        is_mandatory: Whether the prerequisite must pass for this rule to run.
        description: Human-readable description of the dependency.
        namespace: Tenant or organizational namespace for isolation.
        created_at: UTC timestamp when the dependency was declared.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique dependency record identifier (UUID v4)",
    )
    rule_id: str = Field(
        ...,
        description="ID of the dependent rule (the rule that requires another)",
    )
    depends_on_rule_id: str = Field(
        ...,
        description="ID of the prerequisite rule",
    )
    dependency_type: str = Field(
        default="result",
        description="Nature of the dependency (e.g., 'data', 'result', 'order')",
    )
    is_mandatory: bool = Field(
        default=True,
        description="Whether the prerequisite must pass for this rule to evaluate",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the dependency relationship",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the dependency was declared",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Validate rule_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_id must be non-empty")
        return v

    @field_validator("depends_on_rule_id")
    @classmethod
    def validate_depends_on_rule_id(cls, v: str) -> str:
        """Validate depends_on_rule_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("depends_on_rule_id must be non-empty")
        return v


class SLAThreshold(BaseModel):
    """Performance SLA threshold for validation rule evaluation.

    Defines acceptable execution time thresholds for the rule
    evaluation engine. SLA thresholds are tracked per evaluation
    run and reported for capacity planning and performance monitoring.

    Attributes:
        id: Unique SLA threshold identifier (UUID v4).
        name: Human-readable name for this SLA threshold.
        level: SLA enforcement level (critical_rules, all_rules, custom).
        warning_threshold_ms: Threshold (ms) that triggers a performance warning.
        critical_threshold_ms: Threshold (ms) that triggers a performance failure.
        rule_ids: List of rule IDs subject to this SLA (for CUSTOM level).
        namespace: Tenant or organizational namespace for isolation.
        enabled: Whether this SLA threshold is actively enforced.
        created_at: UTC timestamp when the SLA threshold was created.
        updated_at: UTC timestamp when the SLA threshold was last modified.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique SLA threshold identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable name for this SLA threshold",
    )
    level: SLALevel = Field(
        default=SLALevel.ALL_RULES,
        description="SLA enforcement level (critical_rules, all_rules, custom)",
    )
    warning_threshold_ms: float = Field(
        default=DEFAULT_ALL_RULES_SLA_MS * 0.8,
        ge=0.0,
        description="Threshold (ms) that triggers a performance warning",
    )
    critical_threshold_ms: float = Field(
        default=DEFAULT_ALL_RULES_SLA_MS,
        ge=0.0,
        description="Threshold (ms) that triggers a performance failure",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="List of rule IDs subject to this SLA (for CUSTOM level)",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this SLA threshold is actively enforced",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the SLA threshold was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the SLA threshold was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("critical_threshold_ms")
    @classmethod
    def validate_critical_threshold_ms(cls, v: float) -> float:
        """Validate critical_threshold_ms is non-negative."""
        if v < 0.0:
            raise ValueError(
                f"critical_threshold_ms must be >= 0.0, got {v}"
            )
        return v

    @field_validator("warning_threshold_ms")
    @classmethod
    def validate_warning_threshold_ms(cls, v: float) -> float:
        """Validate warning_threshold_ms is non-negative."""
        if v < 0.0:
            raise ValueError(
                f"warning_threshold_ms must be >= 0.0, got {v}"
            )
        return v


class AuditEntry(BaseModel):
    """An immutable audit log entry for any validation rule engine action.

    All create, update, delete, and evaluation actions in the Validation
    Rule Engine produce an AuditEntry. Entries form a provenance chain
    using SHA-256 hashes linking each entry to its parent for
    tamper-evident audit trails.

    Attributes:
        id: Unique audit entry identifier (UUID v4).
        action: Action verb (e.g., "create_rule", "evaluate_rule_set").
        entity_type: Type of entity acted upon (e.g., "ValidationRule").
        entity_id: ID of the entity that was acted upon.
        actor: User, service, or system that performed the action.
        details: Structured details about the action and its parameters.
        previous_state: Snapshot of the entity state before the action.
        new_state: Snapshot of the entity state after the action.
        provenance_hash: SHA-256 hash of this entry's content.
        parent_hash: SHA-256 hash of the immediately preceding audit entry.
        ip_address: IP address of the actor (for security audit).
        user_agent: User agent string of the actor (for security audit).
        created_at: UTC timestamp when the audit entry was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique audit entry identifier (UUID v4)",
    )
    action: str = Field(
        ...,
        description="Action verb (e.g., 'create_rule', 'evaluate_rule_set')",
    )
    entity_type: str = Field(
        ...,
        description="Type of entity acted upon (e.g., 'ValidationRule')",
    )
    entity_id: str = Field(
        ...,
        description="ID of the entity that was acted upon",
    )
    actor: str = Field(
        default="system",
        description="User, service, or system that performed the action",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the action and its parameters",
    )
    previous_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot of the entity state before the action",
    )
    new_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot of the entity state after the action",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of this entry's content for tamper detection",
    )
    parent_hash: str = Field(
        default="",
        description="SHA-256 hash of the immediately preceding audit entry",
    )
    ip_address: str = Field(
        default="",
        description="IP address of the actor (for security audit)",
    )
    user_agent: str = Field(
        default="",
        description="User agent string of the actor (for security audit)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the audit entry was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is non-empty."""
        if not v or not v.strip():
            raise ValueError("action must be non-empty")
        return v

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate entity_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_type must be non-empty")
        return v

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v


# =============================================================================
# Request Models (8)
# =============================================================================


class CreateRuleRequest(BaseModel):
    """Request body for creating a new validation rule.

    Attributes:
        name: Human-readable rule name, unique within a namespace.
        description: Detailed description of what this rule validates.
        rule_type: Classification of the validation check type.
        operator: Comparison operator for threshold evaluation.
        target_field: Name of the dataset field this rule validates.
        target_fields: List of fields for cross-field or multi-field rules.
        expected_value: Expected value for equality or membership checks.
        threshold_min: Minimum threshold for range-based validation.
        threshold_max: Maximum threshold for range-based validation.
        pattern: Regular expression pattern for FORMAT rule types.
        allowed_values: Set of allowed values for IN_SET operator rules.
        severity: Severity classification for rule violations.
        namespace: Tenant or organizational namespace for isolation.
        tags: Arbitrary key-value labels for filtering and discovery.
        framework: Optional compliance framework association.
        expression: Custom Python expression for CUSTOM rule types.
        condition: Precondition expression for CONDITIONAL rule types.
        parameters: Additional rule-specific parameters and configuration.
    """

    name: str = Field(
        ...,
        description="Human-readable rule name, unique within a namespace",
    )
    description: str = Field(
        default="",
        description="Detailed description of what this rule validates",
    )
    rule_type: ValidationRuleType = Field(
        ...,
        description="Classification of the validation check type",
    )
    operator: RuleOperator = Field(
        default=RuleOperator.EQUALS,
        description="Comparison operator for threshold evaluation",
    )
    target_field: str = Field(
        default="",
        description="Name of the dataset field this rule validates",
    )
    target_fields: List[str] = Field(
        default_factory=list,
        description="List of fields for cross-field or multi-field rules",
    )
    expected_value: Optional[Any] = Field(
        None,
        description="Expected value for equality or membership checks",
    )
    threshold_min: Optional[float] = Field(
        None,
        description="Minimum threshold for range-based validation",
    )
    threshold_max: Optional[float] = Field(
        None,
        description="Maximum threshold for range-based validation",
    )
    pattern: Optional[str] = Field(
        None,
        description="Regular expression pattern for FORMAT rule types",
    )
    allowed_values: List[Any] = Field(
        default_factory=list,
        description="Set of allowed values for IN_SET operator rules",
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.MEDIUM,
        description="Severity classification for rule violations",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and discovery",
    )
    framework: Optional[str] = Field(
        None,
        description="Optional compliance framework association (e.g., 'ghg_protocol')",
    )
    expression: Optional[str] = Field(
        None,
        description="Custom Python expression for CUSTOM rule types",
    )
    condition: Optional[str] = Field(
        None,
        description="Precondition expression for CONDITIONAL rule types",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rule-specific parameters and configuration",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class UpdateRuleRequest(BaseModel):
    """Request body for updating mutable fields of an existing validation rule.

    Only fields explicitly included in this model can be updated.
    The name and rule_type are immutable once created. All fields
    are optional; only provided fields will be updated.

    Attributes:
        description: Updated description of the rule.
        operator: Updated comparison operator.
        target_field: Updated target field name.
        target_fields: Updated list of target fields.
        expected_value: Updated expected value.
        threshold_min: Updated minimum threshold.
        threshold_max: Updated maximum threshold.
        pattern: Updated regular expression pattern.
        allowed_values: Updated set of allowed values.
        severity: Updated severity classification.
        status: Updated lifecycle status.
        tags: Updated key-value labels.
        framework: Updated compliance framework association.
        expression: Updated custom expression.
        condition: Updated precondition expression.
        parameters: Updated rule-specific parameters.
    """

    description: Optional[str] = Field(
        None,
        description="Updated description of the rule",
    )
    operator: Optional[RuleOperator] = Field(
        None,
        description="Updated comparison operator",
    )
    target_field: Optional[str] = Field(
        None,
        description="Updated target field name",
    )
    target_fields: Optional[List[str]] = Field(
        None,
        description="Updated list of target fields for multi-field rules",
    )
    expected_value: Optional[Any] = Field(
        None,
        description="Updated expected value for equality or membership checks",
    )
    threshold_min: Optional[float] = Field(
        None,
        description="Updated minimum threshold for range-based validation",
    )
    threshold_max: Optional[float] = Field(
        None,
        description="Updated maximum threshold for range-based validation",
    )
    pattern: Optional[str] = Field(
        None,
        description="Updated regular expression pattern for FORMAT rule types",
    )
    allowed_values: Optional[List[Any]] = Field(
        None,
        description="Updated set of allowed values for IN_SET operator rules",
    )
    severity: Optional[RuleSeverity] = Field(
        None,
        description="Updated severity classification for rule violations",
    )
    status: Optional[RuleStatus] = Field(
        None,
        description="Updated lifecycle status (draft, active, deprecated, archived)",
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Updated key-value labels for filtering and discovery",
    )
    framework: Optional[str] = Field(
        None,
        description="Updated compliance framework association",
    )
    expression: Optional[str] = Field(
        None,
        description="Updated custom Python expression for CUSTOM rule types",
    )
    condition: Optional[str] = Field(
        None,
        description="Updated precondition expression for CONDITIONAL rule types",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated rule-specific parameters and configuration",
    )

    model_config = {"extra": "forbid"}


class CreateRuleSetRequest(BaseModel):
    """Request body for creating a new rule set.

    Attributes:
        name: Human-readable rule set name, unique within a namespace.
        description: Detailed description of the rule set's purpose.
        namespace: Tenant or organizational namespace for isolation.
        framework: Optional compliance framework association.
        gate_pass_threshold: Minimum pass rate to pass the quality gate.
        fail_on_critical: Whether CRITICAL failures auto-fail the gate.
        rule_ids: Initial list of rule IDs to include in the set.
        tags: Arbitrary key-value labels for filtering and discovery.
    """

    name: str = Field(
        ...,
        description="Human-readable rule set name, unique within a namespace",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule set's purpose",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    framework: Optional[str] = Field(
        None,
        description="Optional compliance framework association (e.g., 'csrd_esrs')",
    )
    gate_pass_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum pass rate (0.0 to 1.0) to pass the quality gate",
    )
    fail_on_critical: bool = Field(
        default=True,
        description="Whether CRITICAL severity failures auto-fail the gate",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="Initial list of rule IDs to include in the set",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and discovery",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("gate_pass_threshold")
    @classmethod
    def validate_gate_pass_threshold(cls, v: float) -> float:
        """Validate gate_pass_threshold is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"gate_pass_threshold must be between 0.0 and 1.0, got {v}"
            )
        return v


class UpdateRuleSetRequest(BaseModel):
    """Request body for updating mutable fields of an existing rule set.

    All fields are optional; only provided fields will be updated.
    The name is immutable once created.

    Attributes:
        description: Updated description of the rule set.
        status: Updated lifecycle status.
        framework: Updated compliance framework association.
        gate_pass_threshold: Updated minimum pass rate for gate-pass.
        fail_on_critical: Updated critical failure gate behavior.
        evaluation_order: Updated ordered list of rule IDs for evaluation.
        tags: Updated key-value labels for filtering and discovery.
    """

    description: Optional[str] = Field(
        None,
        description="Updated description of the rule set",
    )
    status: Optional[RuleStatus] = Field(
        None,
        description="Updated lifecycle status (draft, active, deprecated, archived)",
    )
    framework: Optional[str] = Field(
        None,
        description="Updated compliance framework association",
    )
    gate_pass_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Updated minimum pass rate for gate-pass (0.0 to 1.0)",
    )
    fail_on_critical: Optional[bool] = Field(
        None,
        description="Updated critical failure gate behavior",
    )
    evaluation_order: Optional[List[str]] = Field(
        None,
        description="Updated ordered list of rule IDs for evaluation",
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Updated key-value labels for filtering and discovery",
    )

    model_config = {"extra": "forbid"}

    @field_validator("gate_pass_threshold")
    @classmethod
    def validate_gate_pass_threshold(cls, v: Optional[float]) -> Optional[float]:
        """Validate gate_pass_threshold is in range [0.0, 1.0] if set."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"gate_pass_threshold must be between 0.0 and 1.0, got {v}"
            )
        return v


class EvaluateRequest(BaseModel):
    """Request body for evaluating a rule set against a dataset.

    Triggers the evaluation engine to apply all active rules in the
    specified rule set against the provided data records.

    Attributes:
        rule_set_id: ID of the rule set to evaluate.
        dataset_id: Optional identifier for the dataset being evaluated.
        records: List of data records to evaluate (each record is a dict).
        options: Evaluation-specific options and configuration.
        include_details: Whether to include per-record evaluation details.
        fail_fast: Whether to stop evaluation on first CRITICAL failure.
        dry_run: Whether to run in dry-run mode (no side effects).
    """

    rule_set_id: str = Field(
        ...,
        description="ID of the rule set to evaluate",
    )
    dataset_id: str = Field(
        default="",
        description="Optional identifier for the dataset being evaluated",
    )
    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of data records to evaluate (each record is a dict)",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation-specific options and configuration",
    )
    include_details: bool = Field(
        default=True,
        description="Whether to include per-record evaluation detail entries",
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop evaluation on first CRITICAL failure",
    )
    dry_run: bool = Field(
        default=False,
        description="Whether to run in dry-run mode (no persistent side effects)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_set_id")
    @classmethod
    def validate_rule_set_id(cls, v: str) -> str:
        """Validate rule_set_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_set_id must be non-empty")
        return v


class BatchEvaluateRequest(BaseModel):
    """Request body for batch evaluation of multiple rule sets or datasets.

    Enables evaluating multiple rule sets against multiple datasets
    in a single request for pipeline and orchestration efficiency.

    Attributes:
        evaluations: List of individual evaluation requests in the batch.
        batch_size: Number of records to process per evaluation chunk.
        parallel: Whether to evaluate rule sets in parallel.
        fail_fast: Whether to stop the entire batch on first CRITICAL failure.
        max_workers: Maximum number of parallel workers (if parallel is True).
        dry_run: Whether to run in dry-run mode (no side effects).
    """

    evaluations: List[EvaluateRequest] = Field(
        ...,
        description="List of individual evaluation requests in the batch",
    )
    batch_size: int = Field(
        default=DEFAULT_EVALUATION_BATCH_SIZE,
        ge=1,
        le=MAX_BATCH_RECORDS,
        description="Number of records to process per evaluation chunk",
    )
    parallel: bool = Field(
        default=False,
        description="Whether to evaluate rule sets in parallel",
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop the entire batch on first CRITICAL failure",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum number of parallel workers (if parallel is True)",
    )
    dry_run: bool = Field(
        default=False,
        description="Whether to run in dry-run mode (no persistent side effects)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("evaluations")
    @classmethod
    def validate_evaluations(
        cls, v: List[EvaluateRequest]
    ) -> List[EvaluateRequest]:
        """Validate evaluations list is non-empty."""
        if not v:
            raise ValueError("evaluations must contain at least one request")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch_size is within allowed range."""
        if v < 1 or v > MAX_BATCH_RECORDS:
            raise ValueError(
                f"batch_size must be between 1 and {MAX_BATCH_RECORDS}, got {v}"
            )
        return v


class DetectConflictsRequest(BaseModel):
    """Request body for detecting conflicts between validation rules.

    Triggers the conflict detection engine to analyze rules within
    a rule set or across all rules in a namespace for logical
    inconsistencies.

    Attributes:
        scope: Scope of conflict detection (e.g., "rule_set:xyz" or "all").
        rule_set_id: Optional rule set ID to limit analysis scope.
        rule_ids: Optional list of specific rule IDs to analyze.
        conflict_types: Optional list of conflict types to check for.
        include_recommendations: Whether to include remediation suggestions.
        namespace: Tenant or organizational namespace for isolation.
    """

    scope: str = Field(
        default="all",
        description="Scope of conflict detection (e.g., 'rule_set:xyz' or 'all')",
    )
    rule_set_id: Optional[str] = Field(
        None,
        description="Optional rule set ID to limit analysis scope",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="Optional list of specific rule IDs to analyze",
    )
    conflict_types: List[ConflictType] = Field(
        default_factory=list,
        description="Optional list of conflict types to check for (empty = all)",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to include remediation suggestions",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}


class GenerateReportRequest(BaseModel):
    """Request body for generating a validation report.

    Triggers the reporting engine to produce a validation report in
    the specified format and type, scoped to the requested evaluation
    run, rule set, or the full namespace.

    Attributes:
        report_type: Type of validation report to generate.
        report_format: Output format for the report.
        evaluation_run_id: Optional reference to a specific evaluation run.
        rule_set_id: Optional reference to a specific rule set.
        scope: Scope of the report (e.g., "full", "rule_set:xyz").
        parameters: Report generation parameters and configuration.
        include_recommendations: Whether to include remediation suggestions.
        time_range_start: Optional start of the time range for trend reports.
        time_range_end: Optional end of the time range for trend reports.
        namespace: Tenant or organizational namespace for isolation.
    """

    report_type: ReportType = Field(
        default=ReportType.SUMMARY,
        description="Type of validation report to generate",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format for the report",
    )
    evaluation_run_id: Optional[str] = Field(
        None,
        description="Optional reference to a specific evaluation run",
    )
    rule_set_id: Optional[str] = Field(
        None,
        description="Optional reference to a specific rule set",
    )
    scope: str = Field(
        default="full",
        description="Scope of the report (e.g., 'full', 'rule_set:xyz')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report generation parameters and configuration",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to include remediation suggestions in the report",
    )
    time_range_start: Optional[datetime] = Field(
        None,
        description="Optional start of the time range for TREND reports",
    )
    time_range_end: Optional[datetime] = Field(
        None,
        description="Optional end of the time range for TREND reports",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (data_quality_profiler)
    # -------------------------------------------------------------------------
    "QualityRuleEngine",
    "ValidityChecker",
    "QualityDimension",
    "RuleType",
    # -------------------------------------------------------------------------
    # Availability flags (for downstream feature detection)
    # -------------------------------------------------------------------------
    "_QRE_AVAILABLE",
    "_VC_AVAILABLE",
    "_QD_AVAILABLE",
    "_RT_AVAILABLE",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "VERSION",
    "MAX_RULES_PER_NAMESPACE",
    "MAX_RULES_PER_SET",
    "MAX_RULE_SETS_PER_NAMESPACE",
    "MAX_COMPOUND_NESTING_DEPTH",
    "MAX_VERSIONS_PER_RULE",
    "MAX_BATCH_RECORDS",
    "DEFAULT_EVALUATION_BATCH_SIZE",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MAX_CONFLICTS_PER_REPORT",
    "MAX_EVALUATION_DETAILS_PER_RUN",
    "SEVERITY_ORDER",
    "DEFAULT_CRITICAL_SLA_MS",
    "DEFAULT_ALL_RULES_SLA_MS",
    "SUPPORTED_REPORT_FORMATS",
    "BUILT_IN_RULE_PACKS",
    # -------------------------------------------------------------------------
    # Enumerations (12)
    # -------------------------------------------------------------------------
    "ValidationRuleType",
    "RuleOperator",
    "RuleSeverity",
    "RuleStatus",
    "CompoundOperator",
    "EvaluationResult",
    "ConflictType",
    "RulePackType",
    "ReportType",
    "ReportFormat",
    "VersionBumpType",
    "SLALevel",
    # -------------------------------------------------------------------------
    # SDK data models (14)
    # -------------------------------------------------------------------------
    "ValidationRule",
    "RuleSet",
    "RuleSetMember",
    "CompoundRule",
    "RulePack",
    "RuleVersion",
    "EvaluationRun",
    "EvaluationDetail",
    "ConflictReport",
    "ValidationReport",
    "RuleTemplate",
    "RuleDependency",
    "SLAThreshold",
    "AuditEntry",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "CreateRuleRequest",
    "UpdateRuleRequest",
    "CreateRuleSetRequest",
    "UpdateRuleSetRequest",
    "EvaluateRequest",
    "BatchEvaluateRequest",
    "DetectConflictsRequest",
    "GenerateReportRequest",
]
