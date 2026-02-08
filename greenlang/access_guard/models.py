# -*- coding: utf-8 -*-
"""
Access Guard Data Models - AGENT-FOUND-006: Access & Policy Guard

Pydantic v2 data models for the Access & Policy Guard SDK. These models
are clean SDK versions that mirror the foundation agent enumerations
and models while providing a stable public API.

Models:
    - Enums: AccessDecision, PolicyType, DataClassification, RoleType,
             AuditEventType, PolicyChangeType, SimulationMode
    - Core: Principal, Resource, AccessRequest, PolicyRule, Policy,
            AccessDecisionResult, AuditEvent, RateLimitConfig,
            ComplianceReport, PolicySimulationResult
    - Versioning: PolicyVersion, PolicyChangeLogEntry
    - Constants: CLASSIFICATION_HIERARCHY, DEFAULT_ROLE_PERMISSIONS

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enumerations (mirrored from foundation agent)
# =============================================================================


class AccessDecision(str, Enum):
    """Result of an access control decision."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class PolicyType(str, Enum):
    """Types of policies supported by the guard."""
    DATA_ACCESS = "data_access"
    AGENT_EXECUTION = "agent_execution"
    EXPORT = "export"
    RETENTION = "retention"
    GEOGRAPHIC = "geographic"
    RATE_LIMIT = "rate_limit"


class DataClassification(str, Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RoleType(str, Enum):
    """Standard role types for RBAC."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    EDITOR = "editor"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    SERVICE_ACCOUNT = "service_account"


class AuditEventType(str, Enum):
    """Types of audit events."""
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    POLICY_EVALUATED = "policy_evaluated"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TENANT_BOUNDARY_VIOLATION = "tenant_boundary_violation"
    CLASSIFICATION_CHECK = "classification_check"
    EXPORT_APPROVED = "export_approved"
    EXPORT_DENIED = "export_denied"
    POLICY_UPDATED = "policy_updated"
    SIMULATION_RUN = "simulation_run"


# -- New enums added by the SDK layer ----------------------------------------


class PolicyChangeType(str, Enum):
    """Types of policy mutations for change tracking."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ENABLE = "enable"
    DISABLE = "disable"


class SimulationMode(str, Enum):
    """Simulation execution modes."""
    DRY_RUN = "dry_run"
    SHADOW = "shadow"
    CANARY = "canary"


# =============================================================================
# Constants
# =============================================================================


CLASSIFICATION_HIERARCHY: Dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
    DataClassification.TOP_SECRET: 4,
}

DEFAULT_ROLE_PERMISSIONS: Dict[RoleType, Set[str]] = {
    RoleType.VIEWER: {"read"},
    RoleType.ANALYST: {"read", "analyze", "export_internal"},
    RoleType.EDITOR: {"read", "write", "analyze", "export_internal"},
    RoleType.ADMIN: {
        "read", "write", "delete", "analyze",
        "export_internal", "export_external", "manage_users",
    },
    RoleType.SUPER_ADMIN: {"*"},
    RoleType.SERVICE_ACCOUNT: {"read", "write", "execute"},
}


# =============================================================================
# Utility
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# =============================================================================
# Core Data Models
# =============================================================================


class Principal(BaseModel):
    """Represents a user, service, or entity requesting access."""
    principal_id: str = Field(..., description="Unique identifier for the principal")
    principal_type: str = Field(default="user", description="Type: user, service, agent")
    tenant_id: str = Field(..., description="Tenant the principal belongs to")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="ABAC attributes",
    )
    clearance_level: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Maximum data classification the principal can access",
    )
    groups: List[str] = Field(default_factory=list, description="Group memberships")
    authenticated: bool = Field(
        default=True, description="Whether the principal is authenticated",
    )
    session_id: Optional[str] = Field(None, description="Current session ID")

    @field_validator("clearance_level", mode="before")
    @classmethod
    def _coerce_clearance(cls, v: Any) -> Any:
        """Convert string to DataClassification if needed."""
        if isinstance(v, str):
            return DataClassification(v.lower())
        return v


class Resource(BaseModel):
    """Represents a resource being accessed."""
    resource_id: str = Field(..., description="Unique identifier for the resource")
    resource_type: str = Field(
        ..., description="Type of resource: data, agent, report, etc.",
    )
    tenant_id: str = Field(..., description="Tenant the resource belongs to")
    classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Data classification level",
    )
    owner_id: Optional[str] = Field(None, description="Principal ID of the resource owner")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Resource attributes",
    )
    geographic_location: Optional[str] = Field(
        None, description="Geographic location of the data",
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    retention_policy: Optional[str] = Field(None, description="Retention policy ID")

    @field_validator("classification", mode="before")
    @classmethod
    def _coerce_classification(cls, v: Any) -> Any:
        """Convert string to DataClassification if needed."""
        if isinstance(v, str):
            return DataClassification(v.lower())
        return v


class AccessRequest(BaseModel):
    """Represents an access control request."""
    request_id: str = Field(default_factory=_new_uuid, description="Request ID")
    principal: Principal = Field(..., description="The requesting principal")
    resource: Resource = Field(..., description="The resource being accessed")
    action: str = Field(
        ..., description="Action requested: read, write, delete, execute, export",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for ABAC",
    )
    timestamp: datetime = Field(default_factory=_utcnow, description="Request timestamp")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")


class PolicyRule(BaseModel):
    """A single policy rule definition."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(default="", description="Rule description")
    policy_type: PolicyType = Field(..., description="Type of policy")
    priority: int = Field(
        default=100, ge=0, le=1000,
        description="Rule priority (lower = higher priority)",
    )
    enabled: bool = Field(default=True, description="Whether the rule is active")

    # Conditions
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions that must be met for the rule to apply",
    )

    # Actions
    effect: AccessDecision = Field(
        default=AccessDecision.DENY, description="Allow or deny",
    )
    actions: List[str] = Field(
        default_factory=list, description="Actions this rule applies to",
    )
    resources: List[str] = Field(
        default_factory=list,
        description="Resource patterns this rule applies to",
    )
    principals: List[str] = Field(
        default_factory=list,
        description="Principal patterns this rule applies to",
    )

    # Additional constraints
    time_constraints: Optional[Dict[str, Any]] = Field(
        None, description="Time-based constraints",
    )
    geographic_constraints: Optional[List[str]] = Field(
        None, description="Allowed geographic regions",
    )
    classification_max: Optional[DataClassification] = Field(
        None, description="Maximum classification allowed",
    )

    # Metadata
    version: str = Field(default="1.0.0", description="Rule version")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    created_by: Optional[str] = Field(
        None, description="Principal who created the rule",
    )
    tags: List[str] = Field(default_factory=list, description="Tags for organization")

    @field_validator("classification_max", mode="before")
    @classmethod
    def _coerce_class_max(cls, v: Any) -> Any:
        """Convert string to DataClassification if needed."""
        if isinstance(v, str):
            return DataClassification(v.lower())
        return v


class Policy(BaseModel):
    """A collection of policy rules."""
    policy_id: str = Field(..., description="Unique policy identifier")
    name: str = Field(..., description="Policy name")
    description: str = Field(default="", description="Policy description")
    version: str = Field(default="1.0.0", description="Policy version")
    enabled: bool = Field(default=True, description="Whether the policy is active")

    rules: List[PolicyRule] = Field(
        default_factory=list, description="Policy rules",
    )

    # Inheritance
    parent_policy_id: Optional[str] = Field(
        None, description="Parent policy for inheritance",
    )
    allow_override: bool = Field(
        default=True, description="Whether child policies can override",
    )

    # Scope
    tenant_id: Optional[str] = Field(
        None, description="Tenant scope (None = global)",
    )
    applies_to: List[str] = Field(
        default_factory=list, description="Resource types this applies to",
    )

    # Metadata
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last update time")
    created_by: Optional[str] = Field(
        None, description="Principal who created the policy",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail",
    )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the policy for provenance tracking.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        policy_str = json.dumps(
            {
                "policy_id": self.policy_id,
                "rules": [r.model_dump(mode="json") for r in self.rules],
                "version": self.version,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(policy_str.encode()).hexdigest()


class AccessDecisionResult(BaseModel):
    """Result of an access control decision."""
    request_id: str = Field(..., description="Original request ID")
    decision: AccessDecision = Field(..., description="The access decision")
    allowed: bool = Field(..., description="Whether access is allowed")

    # Decision details
    matching_rules: List[str] = Field(
        default_factory=list, description="Rules that matched",
    )
    deny_reasons: List[str] = Field(
        default_factory=list, description="Reasons for denial if denied",
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions if conditional access",
    )

    # Audit info
    evaluated_at: datetime = Field(
        default_factory=_utcnow, description="Evaluation timestamp",
    )
    evaluation_time_ms: float = Field(
        default=0.0, description="Time to evaluate in milliseconds",
    )
    policy_versions: Dict[str, str] = Field(
        default_factory=dict, description="Policy versions evaluated",
    )

    # Provenance
    decision_hash: str = Field(
        default="", description="SHA-256 hash of the decision",
    )


class AuditEvent(BaseModel):
    """An audit log entry."""
    event_id: str = Field(default_factory=_new_uuid, description="Event ID")
    event_type: AuditEventType = Field(..., description="Type of audit event")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Event timestamp",
    )

    # Context
    tenant_id: str = Field(..., description="Tenant context")
    principal_id: Optional[str] = Field(None, description="Acting principal")
    resource_id: Optional[str] = Field(None, description="Affected resource")
    action: Optional[str] = Field(None, description="Action attempted")

    # Decision
    decision: Optional[AccessDecision] = Field(
        None, description="Access decision",
    )
    decision_hash: Optional[str] = Field(
        None, description="Decision provenance hash",
    )

    # Details
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details",
    )
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent")

    # Retention
    retention_days: int = Field(
        default=365, description="Days to retain this event",
    )


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(
        default=100, ge=1, description="Requests per minute",
    )
    requests_per_hour: int = Field(
        default=1000, ge=1, description="Requests per hour",
    )
    requests_per_day: int = Field(
        default=10000, ge=1, description="Requests per day",
    )
    burst_limit: int = Field(default=20, ge=1, description="Burst limit")

    # Per-role overrides
    role_overrides: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "admin": {
                "requests_per_minute": 500,
                "requests_per_hour": 5000,
            },
            "super_admin": {
                "requests_per_minute": 1000,
                "requests_per_hour": 10000,
            },
            "service_account": {
                "requests_per_minute": 1000,
                "requests_per_hour": 50000,
            },
        },
        description="Per-role rate limit overrides",
    )


class ComplianceReport(BaseModel):
    """Compliance report for policy enforcement."""
    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Generation timestamp",
    )
    report_period_start: datetime = Field(
        ..., description="Report period start",
    )
    report_period_end: datetime = Field(
        ..., description="Report period end",
    )
    tenant_id: str = Field(..., description="Tenant for this report")

    # Statistics
    total_requests: int = Field(default=0, description="Total requests evaluated")
    allowed_requests: int = Field(default=0, description="Requests allowed")
    denied_requests: int = Field(default=0, description="Requests denied")
    rate_limited_requests: int = Field(
        default=0, description="Requests rate-limited",
    )

    # Breakdown by policy type
    decisions_by_type: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Decisions by action type",
    )

    # Top denial reasons
    top_denial_reasons: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top denial reasons with counts",
    )

    # Policy coverage
    policies_evaluated: List[str] = Field(
        default_factory=list, description="Policies evaluated in period",
    )
    rules_triggered: Dict[str, int] = Field(
        default_factory=dict, description="Rules triggered with counts",
    )

    # Geographic access patterns
    access_by_region: Dict[str, int] = Field(
        default_factory=dict, description="Access by geographic region",
    )

    # Data classification access
    access_by_classification: Dict[str, int] = Field(
        default_factory=dict, description="Access by classification level",
    )

    # Provenance
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of the report",
    )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the report for provenance tracking.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        report_str = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, default=str,
        )
        return hashlib.sha256(report_str.encode()).hexdigest()


class PolicySimulationResult(BaseModel):
    """Result of a policy simulation run."""
    simulation_id: str = Field(
        default_factory=_new_uuid, description="Simulation ID",
    )
    run_at: datetime = Field(
        default_factory=_utcnow, description="Simulation timestamp",
    )

    # Input
    test_requests: int = Field(
        default=0, description="Number of test requests",
    )
    policies_tested: List[str] = Field(
        default_factory=list, description="Policies tested",
    )

    # Results
    results: List[AccessDecisionResult] = Field(
        default_factory=list, description="Per-request results",
    )
    summary: Dict[str, int] = Field(
        default_factory=dict, description="Aggregated summary",
    )

    # Potential issues
    conflicts_detected: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected conflicts",
    )
    coverage_gaps: List[str] = Field(
        default_factory=list, description="Uncovered resource patterns",
    )


# =============================================================================
# Versioning Models (new in SDK layer)
# =============================================================================


class PolicyVersion(BaseModel):
    """A snapshot of a policy at a point in time."""
    version_id: str = Field(
        default_factory=_new_uuid, description="Version record ID",
    )
    policy_id: str = Field(..., description="Policy this version belongs to")
    version: str = Field(..., description="Semantic version string")
    snapshot: Dict[str, Any] = Field(
        ..., description="Serialized policy snapshot",
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the snapshot",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Version creation time",
    )
    created_by: Optional[str] = Field(
        None, description="User who created this version",
    )


class PolicyChangeLogEntry(BaseModel):
    """An entry in the policy change log for auditing mutations."""
    log_id: str = Field(
        default_factory=_new_uuid, description="Change log entry ID",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Change timestamp",
    )
    change_type: PolicyChangeType = Field(
        ..., description="Type of policy change",
    )
    policy_id: str = Field(..., description="Affected policy ID")
    user_id: str = Field(default="system", description="User who made the change")
    change_reason: str = Field(default="", description="Reason for the change")
    old_hash: Optional[str] = Field(
        None, description="SHA-256 hash before change",
    )
    new_hash: Optional[str] = Field(
        None, description="SHA-256 hash after change",
    )
    provenance_hash: str = Field(
        default="", description="Chain hash for tamper evidence",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional change details",
    )


__all__ = [
    # Enumerations
    "AccessDecision",
    "PolicyType",
    "DataClassification",
    "RoleType",
    "AuditEventType",
    "PolicyChangeType",
    "SimulationMode",
    # Constants
    "CLASSIFICATION_HIERARCHY",
    "DEFAULT_ROLE_PERMISSIONS",
    # Core models
    "Principal",
    "Resource",
    "AccessRequest",
    "PolicyRule",
    "Policy",
    "AccessDecisionResult",
    "AuditEvent",
    "RateLimitConfig",
    "ComplianceReport",
    "PolicySimulationResult",
    # Versioning models
    "PolicyVersion",
    "PolicyChangeLogEntry",
]
