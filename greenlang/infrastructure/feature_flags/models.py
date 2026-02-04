"""
Feature Flag Data Models - INFRA-008

Pydantic v2 models for the GreenLang feature flag system. Provides strongly-typed
data structures for flag definitions, evaluation contexts, targeting rules,
flag variants, overrides, evaluation results, and audit log entries.

All datetime fields use UTC. All models enforce strict validation via Pydantic v2
field validators and model configuration.

Models:
    - FlagType: Enumeration of supported flag evaluation strategies
    - FlagStatus: Lifecycle status of a feature flag
    - FeatureFlag: Core flag definition with rollout and scheduling support
    - FlagRule: Conditional targeting rule attached to a flag
    - FlagVariant: Multivariate flag variant with weighted distribution
    - FlagOverride: Scoped override for a flag (user, tenant, segment, environment)
    - EvaluationContext: Runtime context provided during flag evaluation
    - FlagEvaluationResult: Outcome of evaluating a flag for a given context
    - AuditLogEntry: Immutable record of a flag change for compliance audit trails

Example:
    >>> from greenlang.infrastructure.feature_flags.models import (
    ...     FeatureFlag, FlagType, FlagStatus, EvaluationContext
    ... )
    >>> flag = FeatureFlag(
    ...     key="enable-scope3-calculation",
    ...     name="Enable Scope 3 Calculation Engine",
    ...     description="Rollout of the new Scope 3 calculation pipeline",
    ...     flag_type=FlagType.PERCENTAGE,
    ...     status=FlagStatus.ACTIVE,
    ...     default_value=False,
    ...     rollout_percentage=25.0,
    ...     owner="platform-team",
    ... )
    >>> ctx = EvaluationContext(
    ...     user_id="user-42",
    ...     tenant_id="tenant-acme",
    ...     environment="staging",
    ... )
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLAG_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{1,127}$")
"""Valid flag key: lowercase alphanumeric, dots, hyphens, underscores. 2-128 chars."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FlagType(str, Enum):
    """Supported feature flag evaluation strategies.

    Each type determines how the flag engine resolves the enabled/disabled
    state and optional variant for a given evaluation context.
    """

    BOOLEAN = "boolean"
    """Simple on/off toggle."""

    PERCENTAGE = "percentage"
    """Gradual rollout based on a deterministic hash of the context identifier."""

    USER_LIST = "user_list"
    """Enabled for an explicit allow-list of user IDs."""

    ENVIRONMENT = "environment"
    """Enabled only in specified deployment environments."""

    SEGMENT = "segment"
    """Enabled for users matching one or more audience segments."""

    SCHEDULED = "scheduled"
    """Enabled within a defined time window (start_time to end_time)."""

    MULTIVARIATE = "multivariate"
    """Returns one of several weighted variants instead of a simple boolean."""


class FlagStatus(str, Enum):
    """Lifecycle status of a feature flag.

    Transitions follow the lifecycle: DRAFT -> ACTIVE -> ROLLED_OUT -> PERMANENT
    Flags can be ARCHIVED or KILLED from any active state.
    """

    DRAFT = "draft"
    """Flag is defined but not yet evaluating. Always returns default_value."""

    ACTIVE = "active"
    """Flag is live and being evaluated against rules and rollout percentage."""

    ROLLED_OUT = "rolled_out"
    """Flag has been fully rolled out (100%). Candidate for promotion to PERMANENT."""

    PERMANENT = "permanent"
    """Flag is permanent infrastructure. Will not be cleaned up."""

    ARCHIVED = "archived"
    """Flag is retired. Evaluates to default_value. Kept for audit history."""

    KILLED = "killed"
    """Emergency kill-switch activated. Always returns False/default regardless of rules."""


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class FeatureFlag(BaseModel):
    """Core feature flag definition.

    Represents a single feature flag with its configuration, targeting,
    scheduling, and metadata. Supports all FlagType evaluation strategies.

    Attributes:
        key: Unique, immutable identifier for the flag (lowercase, 2-128 chars).
        name: Human-readable display name.
        description: Detailed description of what this flag controls and why.
        flag_type: Evaluation strategy used to resolve the flag.
        status: Current lifecycle status.
        default_value: Value returned when the flag is off, killed, or archived.
        rollout_percentage: Percentage of traffic that receives the enabled value
            (used with PERCENTAGE type). 0.0 = off, 100.0 = fully rolled out.
        environments: Environments where the flag is active (e.g. ["staging", "prod"]).
        tags: Free-form labels for organizing and filtering flags.
        owner: Team or individual responsible for this flag.
        metadata: Arbitrary key-value pairs for integrations and tooling.
        start_time: Scheduled activation time (UTC). Used with SCHEDULED type.
        end_time: Scheduled deactivation time (UTC). Used with SCHEDULED type.
        created_at: Timestamp when the flag was first created (UTC).
        updated_at: Timestamp of the most recent modification (UTC).
        version: Optimistic concurrency version. Incremented on every update.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "key": "enable-scope3-calc",
                    "name": "Enable Scope 3 Calculation",
                    "flag_type": "percentage",
                    "status": "active",
                    "default_value": False,
                    "rollout_percentage": 25.0,
                    "owner": "platform-team",
                }
            ]
        },
    )

    key: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Unique flag identifier. Lowercase alphanumeric with dots, hyphens, underscores.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable display name for the flag.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Detailed description of the flag's purpose and behavior.",
    )
    flag_type: FlagType = Field(
        default=FlagType.BOOLEAN,
        description="Evaluation strategy used to resolve this flag.",
    )
    status: FlagStatus = Field(
        default=FlagStatus.DRAFT,
        description="Current lifecycle status.",
    )
    default_value: Any = Field(
        default=False,
        description="Value returned when the flag is off, killed, or archived.",
    )
    rollout_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of traffic receiving the enabled value (0.0-100.0).",
    )
    environments: List[str] = Field(
        default_factory=list,
        description="Environments where this flag is active.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Free-form labels for filtering and grouping.",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team or individual responsible for this flag.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for integrations.",
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Scheduled activation time (UTC). Used with SCHEDULED type.",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Scheduled deactivation time (UTC). Used with SCHEDULED type.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Optimistic concurrency version number.",
    )

    # -- Field Validators --------------------------------------------------

    @field_validator("key")
    @classmethod
    def validate_key_format(cls, v: str) -> str:
        """Ensure flag key matches the required pattern.

        Keys must start with a lowercase letter and contain only lowercase
        alphanumeric characters, dots, hyphens, and underscores.
        """
        if not _FLAG_KEY_PATTERN.match(v):
            raise ValueError(
                f"Flag key '{v}' is invalid. Must match pattern: "
                f"lowercase alpha start, 2-128 chars, "
                f"allowed characters: [a-z0-9._-]"
            )
        return v

    @field_validator("environments")
    @classmethod
    def validate_environments(cls, v: List[str]) -> List[str]:
        """Validate and normalize environment names to lowercase."""
        allowed = {"dev", "development", "staging", "prod", "production", "test"}
        normalized: List[str] = []
        for env in v:
            env_lower = env.strip().lower()
            if env_lower not in allowed:
                raise ValueError(
                    f"Invalid environment '{env}'. Allowed: {sorted(allowed)}"
                )
            normalized.append(env_lower)
        return normalized

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Strip whitespace from tags and remove empty entries."""
        cleaned = [tag.strip().lower() for tag in v if tag.strip()]
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: List[str] = []
        for tag in cleaned:
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        return deduped

    @field_validator("start_time", "end_time")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    # -- Model Validators ---------------------------------------------------

    @model_validator(mode="after")
    def validate_scheduled_times(self) -> "FeatureFlag":
        """Validate scheduled flag time window consistency.

        For SCHEDULED flags, start_time must be before end_time.
        For non-scheduled flags, time fields should not both be set.
        """
        if self.flag_type == FlagType.SCHEDULED:
            if self.start_time is None or self.end_time is None:
                raise ValueError(
                    "SCHEDULED flags require both start_time and end_time."
                )
            if self.start_time >= self.end_time:
                raise ValueError(
                    f"start_time ({self.start_time.isoformat()}) must be "
                    f"before end_time ({self.end_time.isoformat()})."
                )
        return self

    @model_validator(mode="after")
    def validate_rollout_percentage_type(self) -> "FeatureFlag":
        """Warn if rollout_percentage is set for non-percentage flags."""
        if (
            self.flag_type != FlagType.PERCENTAGE
            and self.rollout_percentage > 0.0
            and self.flag_type != FlagType.MULTIVARIATE
        ):
            # Allow it but it will be ignored during evaluation
            pass
        return self


class FlagRule(BaseModel):
    """Conditional targeting rule attached to a feature flag.

    Rules are evaluated in priority order (lower number = higher priority).
    The first matching rule determines the evaluation result. If no rules
    match, the flag falls back to its default rollout logic.

    Attributes:
        rule_id: Unique identifier for this rule.
        flag_key: The feature flag this rule belongs to.
        rule_type: Type of targeting rule (e.g. "user_list", "segment", "attribute").
        priority: Evaluation order. Lower values are evaluated first.
        conditions: Rule-specific targeting conditions as a dict.
        enabled: Whether this rule is currently active.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this rule.",
    )
    flag_key: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="The feature flag this rule targets.",
    )
    rule_type: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Type of targeting rule (e.g. 'user_list', 'segment', 'attribute').",
    )
    priority: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Evaluation priority. Lower = higher priority.",
    )
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule conditions as structured key-value pairs.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this rule is currently active.",
    )

    @field_validator("flag_key")
    @classmethod
    def validate_flag_key_format(cls, v: str) -> str:
        """Ensure flag_key matches the standard key pattern."""
        if not _FLAG_KEY_PATTERN.match(v):
            raise ValueError(f"Flag key '{v}' does not match required pattern.")
        return v


class FlagVariant(BaseModel):
    """Multivariate flag variant with weighted distribution.

    Used with MULTIVARIATE flags to return different values to different
    user segments. Weights across all variants for a flag should sum to 100.

    Attributes:
        variant_key: Unique identifier for this variant within its flag.
        flag_key: The feature flag this variant belongs to.
        variant_value: The value returned when this variant is selected.
        weight: Traffic allocation weight (0-100). All variants should sum to 100.
        description: Human-readable explanation of this variant.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    variant_key: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique variant identifier within its flag.",
    )
    flag_key: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="The feature flag this variant belongs to.",
    )
    variant_value: Dict[str, Any] = Field(
        default_factory=dict,
        description="The structured value returned when this variant is selected.",
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Traffic allocation weight (0.0-100.0). All variants should sum to 100.",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Human-readable description of this variant.",
    )

    @field_validator("flag_key")
    @classmethod
    def validate_flag_key_format(cls, v: str) -> str:
        """Ensure flag_key matches the standard key pattern."""
        if not _FLAG_KEY_PATTERN.match(v):
            raise ValueError(f"Flag key '{v}' does not match required pattern.")
        return v

    @field_validator("variant_key")
    @classmethod
    def validate_variant_key(cls, v: str) -> str:
        """Variant keys should be lowercase alphanumeric with hyphens/underscores."""
        v_stripped = v.strip().lower()
        if not re.match(r"^[a-z][a-z0-9_-]{0,127}$", v_stripped):
            raise ValueError(
                f"Variant key '{v}' must start with a lowercase letter "
                f"and contain only [a-z0-9_-]."
            )
        return v_stripped


class FlagOverride(BaseModel):
    """Scoped override for a feature flag.

    Overrides take precedence over all rules and rollout logic.
    They allow specific users, tenants, segments, or environments to receive
    a forced-on/off state or a specific variant. Overrides may have an
    expiration to support time-limited experiments and testing.

    Attributes:
        flag_key: The feature flag being overridden.
        scope_type: The scope of the override (user, tenant, segment, environment).
        scope_value: The specific identifier within the scope (e.g. a user_id).
        enabled: Whether the flag is force-enabled or force-disabled.
        variant_key: Optional variant to force for this override scope.
        expires_at: Optional expiration time (UTC). Override is ignored after this.
        created_by: Identifier of the person/system that created this override.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    flag_key: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="The feature flag being overridden.",
    )
    scope_type: str = Field(
        ...,
        description="Override scope: 'user', 'tenant', 'segment', or 'environment'.",
    )
    scope_value: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Specific identifier within the scope (e.g. user ID, tenant ID).",
    )
    enabled: bool = Field(
        default=True,
        description="Force the flag to this enabled state for the scope.",
    )
    variant_key: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional variant key to force for this scope.",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration time (UTC). Override is ignored after this time.",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="Identifier of the person or system that created this override.",
    )

    @field_validator("scope_type")
    @classmethod
    def validate_scope_type(cls, v: str) -> str:
        """Ensure scope_type is one of the allowed values."""
        allowed = {"user", "tenant", "segment", "environment"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid scope_type '{v}'. Allowed values: {sorted(allowed)}"
            )
        return v_lower

    @field_validator("flag_key")
    @classmethod
    def validate_flag_key_format(cls, v: str) -> str:
        """Ensure flag_key matches the standard key pattern."""
        if not _FLAG_KEY_PATTERN.match(v):
            raise ValueError(f"Flag key '{v}' does not match required pattern.")
        return v

    @field_validator("expires_at")
    @classmethod
    def ensure_utc_expires(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure expiration datetime is timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Evaluation Models
# ---------------------------------------------------------------------------


class EvaluationContext(BaseModel):
    """Runtime context provided during flag evaluation.

    Contains all the information the flag engine needs to resolve a flag
    for a specific request: user identity, tenant, environment, segments,
    and arbitrary user attributes for attribute-based targeting.

    Attributes:
        user_id: Unique identifier for the current user.
        tenant_id: Unique identifier for the current tenant/organization.
        environment: Deployment environment (e.g. "prod", "staging").
        user_segments: Audience segments the user belongs to.
        user_attributes: Arbitrary key-value attributes for attribute-based targeting.
        request_id: Unique identifier for the current request (for tracing).
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Unique identifier for the current user.",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Unique identifier for the current tenant/organization.",
    )
    environment: str = Field(
        default="dev",
        max_length=64,
        description="Deployment environment (e.g. 'prod', 'staging', 'dev').",
    )
    user_segments: List[str] = Field(
        default_factory=list,
        description="Audience segments the user belongs to.",
    )
    user_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary attributes for attribute-based targeting.",
    )
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier for distributed tracing.",
    )

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment name to lowercase."""
        return v.strip().lower()

    @field_validator("user_segments")
    @classmethod
    def normalize_segments(cls, v: List[str]) -> List[str]:
        """Strip and lowercase segment names."""
        return [s.strip().lower() for s in v if s.strip()]

    @property
    def identity_key(self) -> str:
        """Return the primary identity key for deterministic hashing.

        Prefers user_id, falls back to tenant_id, then request_id.
        """
        return self.user_id or self.tenant_id or self.request_id or ""


class FlagEvaluationResult(BaseModel):
    """Outcome of evaluating a feature flag for a given context.

    Contains the resolved enabled state, the reason for the decision,
    optional variant information, and performance metadata.

    Attributes:
        flag_key: The flag that was evaluated.
        enabled: Whether the flag resolved to enabled.
        reason: Human-readable explanation of how the result was determined.
        rule_id: ID of the matching rule, if any.
        variant_key: Selected variant key for multivariate flags.
        metadata: Additional metadata about the evaluation.
        cache_layer: Which cache layer served the result (l1, l2, db, default).
        duration_us: Evaluation duration in microseconds.
    """

    model_config = ConfigDict(extra="forbid")

    flag_key: str = Field(
        ...,
        description="The flag key that was evaluated.",
    )
    enabled: bool = Field(
        ...,
        description="Whether the flag resolved to enabled for the given context.",
    )
    reason: str = Field(
        default="default",
        max_length=512,
        description="Explanation of how the result was determined.",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="ID of the matching targeting rule, if any.",
    )
    variant_key: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Selected variant key for multivariate flags.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation.",
    )
    cache_layer: str = Field(
        default="default",
        description="Cache layer that served the flag definition (l1, l2, db, default).",
    )
    duration_us: int = Field(
        default=0,
        ge=0,
        description="Evaluation duration in microseconds.",
    )


# ---------------------------------------------------------------------------
# Audit Models
# ---------------------------------------------------------------------------


class AuditLogEntry(BaseModel):
    """Immutable audit record of a feature flag change.

    Every modification to a flag definition, rule, variant, or override
    produces an AuditLogEntry for regulatory compliance and operational
    forensics. Entries are append-only and never modified.

    Attributes:
        flag_key: The flag that was changed.
        action: What happened (e.g. "created", "updated", "killed", "archived").
        old_value: Previous state before the change (empty dict for creation).
        new_value: New state after the change (empty dict for deletion).
        changed_by: Identifier of the person or system that made the change.
        change_reason: Free-text explanation of why the change was made.
        ip_address: IP address of the requester (for security auditing).
        created_at: Timestamp of the change (UTC).
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    flag_key: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="The feature flag that was changed.",
    )
    action: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Action performed: created, updated, killed, archived, etc.",
    )
    old_value: Dict[str, Any] = Field(
        default_factory=dict,
        description="Previous state before the change.",
    )
    new_value: Dict[str, Any] = Field(
        default_factory=dict,
        description="New state after the change.",
    )
    changed_by: str = Field(
        default="",
        max_length=256,
        description="Who made the change (user ID, service account, etc.).",
    )
    change_reason: str = Field(
        default="",
        max_length=2048,
        description="Why the change was made.",
    )
    ip_address: Optional[str] = Field(
        default=None,
        max_length=45,
        description="IP address of the requester (IPv4 or IPv6).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the change (UTC).",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Normalize and validate the action field."""
        allowed_actions = {
            "created",
            "updated",
            "deleted",
            "enabled",
            "disabled",
            "killed",
            "archived",
            "rolled_out",
            "promoted",
            "rule_added",
            "rule_removed",
            "rule_updated",
            "variant_added",
            "variant_removed",
            "variant_updated",
            "override_added",
            "override_removed",
            "override_updated",
        }
        v_lower = v.strip().lower()
        if v_lower not in allowed_actions:
            raise ValueError(
                f"Invalid action '{v}'. Allowed actions: {sorted(allowed_actions)}"
            )
        return v_lower

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, v: Optional[str]) -> Optional[str]:
        """Basic validation of IP address format."""
        if v is None:
            return v
        v_stripped = v.strip()
        if not v_stripped:
            return None
        # Accept IPv4 and IPv6 - basic length/character check
        # Full validation is done at the network layer
        if len(v_stripped) < 3 or len(v_stripped) > 45:
            raise ValueError(
                f"IP address '{v_stripped}' has invalid length. "
                f"Expected 3-45 characters."
            )
        return v_stripped

    @field_validator("created_at")
    @classmethod
    def ensure_utc_created(cls, v: datetime) -> datetime:
        """Ensure created_at is timezone-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
