# -*- coding: utf-8 -*-
"""
Feature Flags API Schemas - INFRA-008

Pydantic v2 request/response models for the feature flag REST API.
All schemas enforce strict validation and provide comprehensive
OpenAPI documentation via Field descriptions and JSON examples.

Schemas:
    - CreateFlagRequest / UpdateFlagRequest: Flag mutation inputs
    - FlagResponse / FlagListResponse: Flag query outputs
    - EvaluateRequest / EvaluateResponse: Single flag evaluation
    - BatchEvaluateRequest / BatchEvaluateResponse: Batch evaluation
    - RolloutRequest / KillSwitchRequest: Operational controls
    - OverrideRequest / VariantRequest: Targeting controls
    - AuditLogResponse / StatisticsResponse: Observability outputs
    - PaginationParams: Reusable pagination query parameters

Example:
    >>> from greenlang.infrastructure.feature_flags.api.schemas import CreateFlagRequest
    >>> req = CreateFlagRequest(
    ...     key="enable-scope3-calc",
    ...     name="Enable Scope 3 Calculation",
    ...     flag_type="percentage",
    ...     rollout_percentage=25.0,
    ... )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PaginationParams(BaseModel):
    """Reusable pagination query parameters.

    Attributes:
        page: 1-indexed page number.
        page_size: Number of items per page (1-100).
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed).",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page.",
    )


# ---------------------------------------------------------------------------
# Flag Mutation Schemas
# ---------------------------------------------------------------------------


class CreateFlagRequest(BaseModel):
    """Request schema for creating a new feature flag.

    Attributes:
        key: Unique flag identifier (lowercase, 2-128 chars).
        name: Human-readable display name.
        description: Detailed description.
        flag_type: Evaluation strategy (boolean, percentage, user_list, etc.).
        default_value: Default value when flag is off.
        rollout_percentage: Initial rollout percentage (0-100).
        environments: Environments where the flag is active.
        tags: Labels for organization and filtering.
        owner: Team or individual responsible.
        metadata: Arbitrary key-value metadata.
        start_time: Scheduled activation time (for scheduled flags).
        end_time: Scheduled deactivation time (for scheduled flags).
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "key": "enable-scope3-calc",
                    "name": "Enable Scope 3 Calculation",
                    "flag_type": "percentage",
                    "rollout_percentage": 25.0,
                    "owner": "platform-team",
                    "tags": ["emissions", "scope3"],
                }
            ]
        },
    )

    key: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Unique flag key. Lowercase alphanumeric with dots, hyphens, underscores.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable display name.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Detailed description of the flag.",
    )
    flag_type: str = Field(
        default="boolean",
        description="Evaluation strategy: boolean, percentage, user_list, environment, segment, scheduled, multivariate.",
    )
    default_value: Any = Field(
        default=False,
        description="Default value when the flag is off.",
    )
    rollout_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Initial rollout percentage (0.0-100.0).",
    )
    environments: List[str] = Field(
        default_factory=list,
        description="Environments where this flag is active.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Labels for filtering and grouping.",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Responsible team or individual.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Scheduled activation time (UTC).",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Scheduled deactivation time (UTC).",
    )

    @field_validator("flag_type")
    @classmethod
    def validate_flag_type(cls, v: str) -> str:
        """Validate flag_type is one of the allowed values."""
        allowed = {
            "boolean", "percentage", "user_list",
            "environment", "segment", "scheduled", "multivariate",
        }
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid flag_type '{v}'. Allowed: {sorted(allowed)}"
            )
        return v_lower


class UpdateFlagRequest(BaseModel):
    """Request schema for updating an existing feature flag.

    All fields are optional. Only provided fields are updated.

    Attributes:
        name: New display name.
        description: New description.
        flag_type: New evaluation strategy.
        default_value: New default value.
        rollout_percentage: New rollout percentage.
        environments: New environment list.
        tags: New tag list.
        owner: New owner.
        metadata: New metadata.
        start_time: New schedule start time.
        end_time: New schedule end time.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Updated display name.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Updated description.",
    )
    flag_type: Optional[str] = Field(
        default=None,
        description="Updated evaluation strategy.",
    )
    default_value: Optional[Any] = Field(
        default=None,
        description="Updated default value.",
    )
    rollout_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Updated rollout percentage.",
    )
    environments: Optional[List[str]] = Field(
        default=None,
        description="Updated environments list.",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Updated tags list.",
    )
    owner: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Updated owner.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated metadata.",
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Updated schedule start time.",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Updated schedule end time.",
    )


# ---------------------------------------------------------------------------
# Flag Response Schemas
# ---------------------------------------------------------------------------


class FlagResponse(BaseModel):
    """Response schema for a single feature flag.

    Attributes:
        key: Unique flag identifier.
        name: Display name.
        description: Flag description.
        flag_type: Evaluation strategy.
        status: Current lifecycle status.
        default_value: Default value when off.
        rollout_percentage: Current rollout percentage.
        environments: Active environments.
        tags: Labels.
        owner: Responsible team/individual.
        metadata: Arbitrary metadata.
        start_time: Scheduled start time.
        end_time: Scheduled end time.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
        version: Optimistic concurrency version.
    """

    model_config = ConfigDict(from_attributes=True)

    key: str = Field(..., description="Unique flag identifier.")
    name: str = Field(..., description="Display name.")
    description: str = Field(default="", description="Flag description.")
    flag_type: str = Field(..., description="Evaluation strategy.")
    status: str = Field(..., description="Current lifecycle status.")
    default_value: Any = Field(default=False, description="Default value when off.")
    rollout_percentage: float = Field(default=0.0, description="Current rollout %.")
    environments: List[str] = Field(default_factory=list, description="Active environments.")
    tags: List[str] = Field(default_factory=list, description="Labels.")
    owner: str = Field(default="", description="Responsible team/individual.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata.")
    start_time: Optional[datetime] = Field(default=None, description="Scheduled start.")
    end_time: Optional[datetime] = Field(default=None, description="Scheduled end.")
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Last modification timestamp.")
    version: int = Field(default=1, description="Optimistic concurrency version.")


class FlagListResponse(BaseModel):
    """Paginated list of feature flags.

    Attributes:
        items: List of flags for the current page.
        total: Total number of matching flags.
        page: Current page number.
        page_size: Items per page.
        total_pages: Total number of pages.
        has_next: Whether there is a next page.
        has_prev: Whether there is a previous page.
    """

    items: List[FlagResponse] = Field(..., description="Flags for this page.")
    total: int = Field(..., ge=0, description="Total matching flags.")
    page: int = Field(..., ge=1, description="Current page number.")
    page_size: int = Field(..., ge=1, description="Items per page.")
    total_pages: int = Field(..., ge=0, description="Total pages.")
    has_next: bool = Field(..., description="Has next page.")
    has_prev: bool = Field(..., description="Has previous page.")


# ---------------------------------------------------------------------------
# Evaluation Schemas
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    """Request schema for evaluating a flag.

    Attributes:
        user_id: Current user identifier.
        tenant_id: Current tenant identifier.
        environment: Deployment environment.
        user_segments: User's audience segments.
        user_attributes: Additional attributes for targeting.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Current user identifier.",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Current tenant identifier.",
    )
    environment: str = Field(
        default="dev",
        max_length=64,
        description="Deployment environment.",
    )
    user_segments: List[str] = Field(
        default_factory=list,
        description="User's audience segments.",
    )
    user_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes for targeting.",
    )


class EvaluateResponse(BaseModel):
    """Response schema for a flag evaluation.

    Attributes:
        flag_key: The flag that was evaluated.
        enabled: Whether the flag is enabled.
        reason: How the result was determined.
        variant_key: Selected variant for multivariate flags.
        metadata: Additional evaluation metadata.
        duration_us: Evaluation duration in microseconds.
    """

    flag_key: str = Field(..., description="The evaluated flag key.")
    enabled: bool = Field(..., description="Whether the flag is enabled.")
    reason: str = Field(default="default", description="Resolution reason.")
    variant_key: Optional[str] = Field(default=None, description="Selected variant.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Eval metadata.")
    duration_us: int = Field(default=0, ge=0, description="Duration in microseconds.")


class BatchEvaluateRequest(BaseModel):
    """Request schema for batch flag evaluation.

    Attributes:
        flag_keys: List of flag keys to evaluate.
        context: Evaluation context shared across all flags.
    """

    model_config = ConfigDict(extra="forbid")

    flag_keys: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Flag keys to evaluate (max 100).",
    )
    context: EvaluateRequest = Field(
        default_factory=EvaluateRequest,
        description="Evaluation context.",
    )


class BatchEvaluateResponse(BaseModel):
    """Response schema for batch flag evaluation.

    Attributes:
        results: Mapping of flag_key to evaluation result.
        total_duration_us: Total evaluation duration in microseconds.
    """

    results: Dict[str, EvaluateResponse] = Field(
        ..., description="Flag key -> evaluation result."
    )
    total_duration_us: int = Field(
        default=0, ge=0, description="Total evaluation duration."
    )


# ---------------------------------------------------------------------------
# Operational Schemas
# ---------------------------------------------------------------------------


class RolloutRequest(BaseModel):
    """Request schema for setting rollout percentage.

    Attributes:
        percentage: New rollout percentage (0.0 to 100.0).
        updated_by: Identity of who is changing the rollout.
    """

    model_config = ConfigDict(extra="forbid")

    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Rollout percentage (0.0-100.0).",
    )
    updated_by: str = Field(
        default="",
        max_length=256,
        description="Identity of who is changing the rollout.",
    )


class KillSwitchRequest(BaseModel):
    """Request schema for kill switch operations.

    Attributes:
        actor: Identity of who is operating the kill switch.
        reason: Explanation for the action.
    """

    model_config = ConfigDict(extra="forbid")

    actor: str = Field(
        default="",
        max_length=256,
        description="Identity of the actor.",
    )
    reason: str = Field(
        default="",
        max_length=2048,
        description="Reason for the action.",
    )


class OverrideRequest(BaseModel):
    """Request schema for setting a flag override.

    Attributes:
        scope_type: Override scope (user, tenant, segment, environment).
        scope_value: Specific identifier within the scope.
        enabled: Force the flag to this state.
        variant_key: Optional variant to force.
        expires_at: Optional expiration datetime (UTC).
        created_by: Identity of who created the override.
    """

    model_config = ConfigDict(extra="forbid")

    scope_type: str = Field(
        ...,
        description="Override scope: user, tenant, segment, environment.",
    )
    scope_value: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Specific scope identifier.",
    )
    enabled: bool = Field(
        default=True,
        description="Force-enable or force-disable.",
    )
    variant_key: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional variant to force.",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Optional expiration (UTC).",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="Creator identity.",
    )

    @field_validator("scope_type")
    @classmethod
    def validate_scope_type(cls, v: str) -> str:
        """Validate scope_type is one of the allowed values."""
        allowed = {"user", "tenant", "segment", "environment"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid scope_type '{v}'. Allowed: {sorted(allowed)}"
            )
        return v_lower


class VariantRequest(BaseModel):
    """Request schema for adding/updating a flag variant.

    Attributes:
        variant_key: Unique variant identifier within the flag.
        variant_value: Structured value for this variant.
        weight: Traffic allocation weight (0-100).
        description: Human-readable description.
    """

    model_config = ConfigDict(extra="forbid")

    variant_key: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Variant identifier.",
    )
    variant_value: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured variant value.",
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Traffic allocation weight (0-100).",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Variant description.",
    )


# ---------------------------------------------------------------------------
# Observability Schemas
# ---------------------------------------------------------------------------


class AuditLogEntryResponse(BaseModel):
    """Single audit log entry in API response format.

    Attributes:
        flag_key: The flag that was changed.
        action: Action performed.
        old_value: Previous state.
        new_value: New state.
        changed_by: Who made the change.
        change_reason: Why the change was made.
        ip_address: Requester IP address.
        created_at: Timestamp of the change.
    """

    model_config = ConfigDict(from_attributes=True)

    flag_key: str = Field(..., description="Changed flag key.")
    action: str = Field(..., description="Action performed.")
    old_value: Dict[str, Any] = Field(default_factory=dict, description="Previous state.")
    new_value: Dict[str, Any] = Field(default_factory=dict, description="New state.")
    changed_by: str = Field(default="", description="Who made the change.")
    change_reason: str = Field(default="", description="Why the change was made.")
    ip_address: Optional[str] = Field(default=None, description="Requester IP.")
    created_at: datetime = Field(..., description="Change timestamp.")


class AuditLogResponse(BaseModel):
    """Paginated audit log response.

    Attributes:
        items: Audit log entries for this page.
        total: Total number of entries.
        page: Current page number.
        page_size: Items per page.
    """

    items: List[AuditLogEntryResponse] = Field(
        ..., description="Audit log entries."
    )
    total: int = Field(..., ge=0, description="Total entries.")
    page: int = Field(..., ge=1, description="Current page.")
    page_size: int = Field(..., ge=1, description="Items per page.")


class StatisticsResponse(BaseModel):
    """Feature flag system statistics.

    Attributes:
        total_flags: Total number of flags.
        by_status: Count of flags by lifecycle status.
        by_type: Count of flags by flag type.
        killed_flags: List of currently killed flag keys.
        killed_count: Number of currently killed flags.
        storage_healthy: Whether the storage backend is healthy.
        environment: Current deployment environment.
    """

    total_flags: int = Field(..., ge=0, description="Total flags.")
    by_status: Dict[str, int] = Field(
        default_factory=dict, description="Counts by status."
    )
    by_type: Dict[str, int] = Field(
        default_factory=dict, description="Counts by type."
    )
    killed_flags: List[str] = Field(
        default_factory=list, description="Currently killed flag keys."
    )
    killed_count: int = Field(default=0, ge=0, description="Killed flag count.")
    storage_healthy: bool = Field(default=True, description="Storage health.")
    environment: str = Field(default="dev", description="Environment.")


__all__ = [
    "AuditLogEntryResponse",
    "AuditLogResponse",
    "BatchEvaluateRequest",
    "BatchEvaluateResponse",
    "CreateFlagRequest",
    "EvaluateRequest",
    "EvaluateResponse",
    "FlagListResponse",
    "FlagResponse",
    "KillSwitchRequest",
    "OverrideRequest",
    "PaginationParams",
    "RolloutRequest",
    "StatisticsResponse",
    "UpdateFlagRequest",
    "VariantRequest",
]
