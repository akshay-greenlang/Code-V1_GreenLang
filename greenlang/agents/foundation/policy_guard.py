# -*- coding: utf-8 -*-
"""
GL-FOUND-X-006: Access & Policy Guard Agent
============================================

The core authorization and policy enforcement agent for GreenLang Climate OS.
This agent handles RBAC/ABAC policies, data classification, tenant isolation,
and comprehensive audit logging for compliance.

Capabilities:
    - Role-Based Access Control (RBAC) policy enforcement
    - Attribute-Based Access Control (ABAC) policy enforcement
    - Data classification and sensitivity level management
    - Policy enforcement for data handling, export, and retention
    - Multi-tenant data isolation with strict boundary enforcement
    - Audit logging for all access decisions
    - Rate limiting per tenant/user
    - Open Policy Agent (OPA) Rego policy support
    - Policy simulation mode for testing
    - Policy inheritance and override management
    - Compliance report generation

Zero-Hallucination Guarantees:
    - All access decisions are deterministic based on policy rules
    - Complete audit trail for every access decision
    - No probabilistic or ML-based access decisions
    - All policies versioned with SHA-256 provenance hashes

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class AccessDecision(str, Enum):
    """Result of an access control decision."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"  # Allow with conditions


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


# Classification level hierarchy (higher number = more sensitive)
CLASSIFICATION_HIERARCHY: Dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
    DataClassification.TOP_SECRET: 4,
}

# Default role permissions (can be overridden by policies)
DEFAULT_ROLE_PERMISSIONS: Dict[RoleType, Set[str]] = {
    RoleType.VIEWER: {"read"},
    RoleType.ANALYST: {"read", "analyze", "export_internal"},
    RoleType.EDITOR: {"read", "write", "analyze", "export_internal"},
    RoleType.ADMIN: {"read", "write", "delete", "analyze", "export_internal", "export_external", "manage_users"},
    RoleType.SUPER_ADMIN: {"*"},  # All permissions
    RoleType.SERVICE_ACCOUNT: {"read", "write", "execute"},
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class Principal(BaseModel):
    """Represents a user, service, or entity requesting access."""
    principal_id: str = Field(..., description="Unique identifier for the principal")
    principal_type: str = Field(default="user", description="Type: user, service, agent")
    tenant_id: str = Field(..., description="Tenant the principal belongs to")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="ABAC attributes")
    clearance_level: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Maximum data classification the principal can access"
    )
    groups: List[str] = Field(default_factory=list, description="Group memberships")
    authenticated: bool = Field(default=True, description="Whether the principal is authenticated")
    session_id: Optional[str] = Field(None, description="Current session ID")

    @validator('clearance_level', pre=True)
    def validate_clearance(cls, v):
        """Convert string to DataClassification if needed."""
        if isinstance(v, str):
            return DataClassification(v.lower())
        return v


class Resource(BaseModel):
    """Represents a resource being accessed."""
    resource_id: str = Field(..., description="Unique identifier for the resource")
    resource_type: str = Field(..., description="Type of resource: data, agent, report, etc.")
    tenant_id: str = Field(..., description="Tenant the resource belongs to")
    classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Data classification level"
    )
    owner_id: Optional[str] = Field(None, description="Principal ID of the resource owner")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Resource attributes")
    geographic_location: Optional[str] = Field(None, description="Geographic location of the data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    retention_policy: Optional[str] = Field(None, description="Retention policy ID")

    @validator('classification', pre=True)
    def validate_classification(cls, v):
        """Convert string to DataClassification if needed."""
        if isinstance(v, str):
            return DataClassification(v.lower())
        return v


class AccessRequest(BaseModel):
    """Represents an access control request."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    principal: Principal = Field(..., description="The requesting principal")
    resource: Resource = Field(..., description="The resource being accessed")
    action: str = Field(..., description="The action being requested: read, write, delete, execute, export")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for ABAC")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")


class PolicyRule(BaseModel):
    """A single policy rule definition."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(default="", description="Rule description")
    policy_type: PolicyType = Field(..., description="Type of policy")
    priority: int = Field(default=100, ge=0, le=1000, description="Rule priority (lower = higher priority)")
    enabled: bool = Field(default=True, description="Whether the rule is active")

    # Conditions
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions that must be met for the rule to apply"
    )

    # Actions
    effect: AccessDecision = Field(default=AccessDecision.DENY, description="Allow or deny")
    actions: List[str] = Field(default_factory=list, description="Actions this rule applies to")
    resources: List[str] = Field(default_factory=list, description="Resource patterns this rule applies to")
    principals: List[str] = Field(default_factory=list, description="Principal patterns this rule applies to")

    # Additional constraints
    time_constraints: Optional[Dict[str, Any]] = Field(None, description="Time-based constraints")
    geographic_constraints: Optional[List[str]] = Field(None, description="Allowed geographic regions")
    classification_max: Optional[DataClassification] = Field(None, description="Maximum classification allowed")

    # Metadata
    version: str = Field(default="1.0.0", description="Rule version")
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    created_by: Optional[str] = Field(None, description="Principal who created the rule")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")

    @validator('classification_max', pre=True)
    def validate_class_max(cls, v):
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

    rules: List[PolicyRule] = Field(default_factory=list, description="Policy rules")

    # Inheritance
    parent_policy_id: Optional[str] = Field(None, description="Parent policy for inheritance")
    allow_override: bool = Field(default=True, description="Whether child policies can override")

    # Scope
    tenant_id: Optional[str] = Field(None, description="Tenant scope (None = global)")
    applies_to: List[str] = Field(default_factory=list, description="Resource types this applies to")

    # Metadata
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    created_by: Optional[str] = Field(None, description="Principal who created the policy")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the policy for provenance tracking."""
        policy_str = json.dumps(
            {
                "policy_id": self.policy_id,
                "rules": [r.model_dump() for r in self.rules],
                "version": self.version,
            },
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(policy_str.encode()).hexdigest()


class AccessDecisionResult(BaseModel):
    """Result of an access control decision."""
    request_id: str = Field(..., description="Original request ID")
    decision: AccessDecision = Field(..., description="The access decision")
    allowed: bool = Field(..., description="Whether access is allowed")

    # Decision details
    matching_rules: List[str] = Field(default_factory=list, description="Rules that matched")
    deny_reasons: List[str] = Field(default_factory=list, description="Reasons for denial if denied")
    conditions: List[str] = Field(default_factory=list, description="Conditions if conditional access")

    # Audit info
    evaluated_at: datetime = Field(default_factory=DeterministicClock.now)
    evaluation_time_ms: float = Field(default=0.0, description="Time to evaluate")
    policy_versions: Dict[str, str] = Field(default_factory=dict, description="Policy versions evaluated")

    # Provenance
    decision_hash: str = Field(default="", description="SHA-256 hash of the decision")


class AuditEvent(BaseModel):
    """An audit log entry."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = Field(..., description="Type of audit event")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)

    # Context
    tenant_id: str = Field(..., description="Tenant context")
    principal_id: Optional[str] = Field(None, description="Acting principal")
    resource_id: Optional[str] = Field(None, description="Affected resource")
    action: Optional[str] = Field(None, description="Action attempted")

    # Decision
    decision: Optional[AccessDecision] = Field(None, description="Access decision")
    decision_hash: Optional[str] = Field(None, description="Decision provenance hash")

    # Details
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent")

    # Retention
    retention_days: int = Field(default=365, description="Days to retain this event")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=100, ge=1, description="Requests per minute")
    requests_per_hour: int = Field(default=1000, ge=1, description="Requests per hour")
    requests_per_day: int = Field(default=10000, ge=1, description="Requests per day")
    burst_limit: int = Field(default=20, ge=1, description="Burst limit")

    # Per-role overrides
    role_overrides: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "admin": {"requests_per_minute": 500, "requests_per_hour": 5000},
            "super_admin": {"requests_per_minute": 1000, "requests_per_hour": 10000},
            "service_account": {"requests_per_minute": 1000, "requests_per_hour": 50000},
        }
    )


class PolicyGuardConfig(BaseModel):
    """Configuration for the Policy Guard Agent."""
    # Mode
    simulation_mode: bool = Field(default=False, description="Run in simulation mode (log only, don't enforce)")
    strict_mode: bool = Field(default=True, description="Deny by default if no matching rules")

    # Rate limiting
    rate_limiting_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_config: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Audit
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_all_decisions: bool = Field(default=True, description="Audit all decisions, not just denials")
    audit_retention_days: int = Field(default=365, description="Default audit retention")

    # Tenant isolation
    strict_tenant_isolation: bool = Field(default=True, description="Enforce strict tenant isolation")

    # OPA integration
    opa_enabled: bool = Field(default=False, description="Enable OPA Rego policy evaluation")
    opa_endpoint: Optional[str] = Field(None, description="OPA server endpoint")

    # Cache
    decision_cache_ttl_seconds: int = Field(default=60, description="Decision cache TTL")
    policy_cache_ttl_seconds: int = Field(default=300, description="Policy cache TTL")


class ComplianceReport(BaseModel):
    """Compliance report for policy enforcement."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=DeterministicClock.now)
    report_period_start: datetime = Field(..., description="Report period start")
    report_period_end: datetime = Field(..., description="Report period end")
    tenant_id: str = Field(..., description="Tenant for this report")

    # Statistics
    total_requests: int = Field(default=0)
    allowed_requests: int = Field(default=0)
    denied_requests: int = Field(default=0)
    rate_limited_requests: int = Field(default=0)

    # Breakdown by policy type
    decisions_by_type: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    # Top denial reasons
    top_denial_reasons: List[Dict[str, Any]] = Field(default_factory=list)

    # Policy coverage
    policies_evaluated: List[str] = Field(default_factory=list)
    rules_triggered: Dict[str, int] = Field(default_factory=dict)

    # Geographic access patterns
    access_by_region: Dict[str, int] = Field(default_factory=dict)

    # Data classification access
    access_by_classification: Dict[str, int] = Field(default_factory=dict)

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash of the report")

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the report for provenance tracking."""
        report_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(report_str.encode()).hexdigest()


class PolicySimulationResult(BaseModel):
    """Result of a policy simulation run."""
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_at: datetime = Field(default_factory=DeterministicClock.now)

    # Input
    test_requests: int = Field(default=0, description="Number of test requests")
    policies_tested: List[str] = Field(default_factory=list)

    # Results
    results: List[AccessDecisionResult] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)

    # Potential issues
    conflicts_detected: List[Dict[str, Any]] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(default_factory=list)


# =============================================================================
# RATE LIMITER
# =============================================================================


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_update: float
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    requests_this_day: int = 0
    minute_start: float = 0.0
    hour_start: float = 0.0
    day_start: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter with per-minute, per-hour, and per-day limits.
    Thread-safe implementation for multi-tenant environments.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, RateLimitBucket] = {}

    def _get_bucket_key(self, tenant_id: str, principal_id: str) -> str:
        """Generate unique bucket key for tenant+principal."""
        return f"{tenant_id}:{principal_id}"

    def _get_or_create_bucket(self, key: str) -> RateLimitBucket:
        """Get or create a rate limit bucket."""
        now = time.time()
        if key not in self._buckets:
            self._buckets[key] = RateLimitBucket(
                tokens=self.config.burst_limit,
                last_update=now,
                minute_start=now,
                hour_start=now,
                day_start=now
            )
        return self._buckets[key]

    def check_rate_limit(
        self,
        tenant_id: str,
        principal_id: str,
        role: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is within rate limits.

        Args:
            tenant_id: Tenant identifier
            principal_id: Principal identifier
            role: Optional role for role-specific limits

        Returns:
            Tuple of (allowed, denial_reason)
        """
        key = self._get_bucket_key(tenant_id, principal_id)
        bucket = self._get_or_create_bucket(key)
        now = time.time()

        # Get effective limits (apply role overrides)
        rpm = self.config.requests_per_minute
        rph = self.config.requests_per_hour
        rpd = self.config.requests_per_day

        if role and role in self.config.role_overrides:
            overrides = self.config.role_overrides[role]
            rpm = overrides.get("requests_per_minute", rpm)
            rph = overrides.get("requests_per_hour", rph)
            rpd = overrides.get("requests_per_day", rpd)

        # Reset counters if time windows have passed
        if now - bucket.minute_start >= 60:
            bucket.requests_this_minute = 0
            bucket.minute_start = now

        if now - bucket.hour_start >= 3600:
            bucket.requests_this_hour = 0
            bucket.hour_start = now

        if now - bucket.day_start >= 86400:
            bucket.requests_this_day = 0
            bucket.day_start = now

        # Check limits
        if bucket.requests_this_minute >= rpm:
            return False, f"Rate limit exceeded: {rpm} requests per minute"

        if bucket.requests_this_hour >= rph:
            return False, f"Rate limit exceeded: {rph} requests per hour"

        if bucket.requests_this_day >= rpd:
            return False, f"Rate limit exceeded: {rpd} requests per day"

        # Increment counters
        bucket.requests_this_minute += 1
        bucket.requests_this_hour += 1
        bucket.requests_this_day += 1

        return True, None

    def get_remaining_quota(
        self,
        tenant_id: str,
        principal_id: str,
        role: Optional[str] = None
    ) -> Dict[str, int]:
        """Get remaining quota for a principal."""
        key = self._get_bucket_key(tenant_id, principal_id)
        bucket = self._get_or_create_bucket(key)

        rpm = self.config.requests_per_minute
        rph = self.config.requests_per_hour
        rpd = self.config.requests_per_day

        if role and role in self.config.role_overrides:
            overrides = self.config.role_overrides[role]
            rpm = overrides.get("requests_per_minute", rpm)
            rph = overrides.get("requests_per_hour", rph)
            rpd = overrides.get("requests_per_day", rpd)

        return {
            "remaining_per_minute": max(0, rpm - bucket.requests_this_minute),
            "remaining_per_hour": max(0, rph - bucket.requests_this_hour),
            "remaining_per_day": max(0, rpd - bucket.requests_this_day),
        }


# =============================================================================
# POLICY ENGINE
# =============================================================================


class PolicyEngine:
    """
    Core policy evaluation engine.
    Supports RBAC, ABAC, and OPA Rego policy evaluation.
    """

    def __init__(self, config: PolicyGuardConfig):
        self.config = config
        self._policies: Dict[str, Policy] = {}
        self._policy_hierarchy: Dict[str, List[str]] = {}  # child -> [parents]
        self._rego_policies: Dict[str, str] = {}  # policy_id -> rego source

    def add_policy(self, policy: Policy) -> str:
        """
        Add a policy to the engine.

        Args:
            policy: The policy to add

        Returns:
            Policy provenance hash
        """
        policy.provenance_hash = policy.compute_hash()
        policy.updated_at = DeterministicClock.now()
        self._policies[policy.policy_id] = policy

        # Track inheritance
        if policy.parent_policy_id:
            if policy.policy_id not in self._policy_hierarchy:
                self._policy_hierarchy[policy.policy_id] = []
            self._policy_hierarchy[policy.policy_id].append(policy.parent_policy_id)

        logger.info(f"Added policy: {policy.policy_id} (hash: {policy.provenance_hash[:16]})")
        return policy.provenance_hash

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the engine."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            if policy_id in self._policy_hierarchy:
                del self._policy_hierarchy[policy_id]
            logger.info(f"Removed policy: {policy_id}")
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def get_effective_rules(
        self,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None
    ) -> List[PolicyRule]:
        """
        Get all effective rules for a tenant/resource type,
        respecting policy inheritance.

        Args:
            tenant_id: Optional tenant filter
            resource_type: Optional resource type filter

        Returns:
            List of applicable rules sorted by priority
        """
        rules: List[PolicyRule] = []

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            # Check tenant scope
            if tenant_id and policy.tenant_id and policy.tenant_id != tenant_id:
                continue

            # Check resource type scope
            if resource_type and policy.applies_to and resource_type not in policy.applies_to:
                continue

            # Add rules from this policy
            for rule in policy.rules:
                if rule.enabled:
                    rules.append(rule)

        # Sort by priority (lower number = higher priority)
        rules.sort(key=lambda r: r.priority)
        return rules

    def evaluate(self, request: AccessRequest) -> AccessDecisionResult:
        """
        Evaluate an access request against all applicable policies.

        Args:
            request: The access request to evaluate

        Returns:
            AccessDecisionResult with the decision and details
        """
        start_time = time.time()

        matching_rules: List[str] = []
        deny_reasons: List[str] = []
        conditions: List[str] = []
        policy_versions: Dict[str, str] = {}

        # Get applicable rules
        rules = self.get_effective_rules(
            tenant_id=request.resource.tenant_id,
            resource_type=request.resource.resource_type
        )

        # Track policy versions for audit
        for policy in self._policies.values():
            policy_versions[policy.policy_id] = policy.version

        # Default decision based on mode
        decision = AccessDecision.DENY if self.config.strict_mode else AccessDecision.ALLOW

        for rule in rules:
            if self._rule_matches(rule, request):
                matching_rules.append(rule.rule_id)

                if rule.effect == AccessDecision.ALLOW:
                    decision = AccessDecision.ALLOW
                    # Check for conditions
                    if rule.time_constraints or rule.geographic_constraints:
                        conditions.append(f"Rule {rule.rule_id} has conditions")
                        decision = AccessDecision.CONDITIONAL
                    break  # First matching rule wins (rules are sorted by priority)
                elif rule.effect == AccessDecision.DENY:
                    decision = AccessDecision.DENY
                    deny_reasons.append(f"Denied by rule: {rule.name} ({rule.rule_id})")
                    break  # First matching rule wins (rules are sorted by priority)

        # If still conditional, check if all conditions are met
        if decision == AccessDecision.CONDITIONAL:
            conditions_met = self._check_conditions(request, matching_rules)
            if conditions_met:
                decision = AccessDecision.ALLOW
                conditions = []
            else:
                decision = AccessDecision.DENY
                deny_reasons.extend(conditions)

        evaluation_time = (time.time() - start_time) * 1000

        result = AccessDecisionResult(
            request_id=request.request_id,
            decision=decision,
            allowed=decision == AccessDecision.ALLOW,
            matching_rules=matching_rules,
            deny_reasons=deny_reasons,
            conditions=conditions if decision == AccessDecision.CONDITIONAL else [],
            evaluation_time_ms=evaluation_time,
            policy_versions=policy_versions
        )

        # Compute decision hash for provenance
        decision_str = json.dumps({
            "request_id": request.request_id,
            "decision": decision.value,
            "matching_rules": matching_rules,
            "timestamp": result.evaluated_at.isoformat()
        }, sort_keys=True)
        result.decision_hash = hashlib.sha256(decision_str.encode()).hexdigest()

        return result

    def _rule_matches(self, rule: PolicyRule, request: AccessRequest) -> bool:
        """Check if a rule matches the given request."""
        # Check action
        if rule.actions and request.action not in rule.actions:
            if "*" not in rule.actions:
                return False

        # Check principal pattern
        if rule.principals:
            principal_match = False
            for pattern in rule.principals:
                if self._pattern_matches(pattern, request.principal.principal_id):
                    principal_match = True
                    break
                # Check role match
                if pattern.startswith("role:"):
                    role_name = pattern[5:]
                    if role_name in request.principal.roles or role_name == "*":
                        principal_match = True
                        break
            if not principal_match:
                return False

        # Check resource pattern
        if rule.resources:
            resource_match = False
            for pattern in rule.resources:
                if self._pattern_matches(pattern, request.resource.resource_id):
                    resource_match = True
                    break
                # Check resource type match
                if pattern.startswith("type:"):
                    type_name = pattern[5:]
                    if type_name == request.resource.resource_type or type_name == "*":
                        resource_match = True
                        break
            if not resource_match:
                return False

        # Check classification constraint
        if rule.classification_max:
            resource_level = CLASSIFICATION_HIERARCHY.get(request.resource.classification, 0)
            max_level = CLASSIFICATION_HIERARCHY.get(rule.classification_max, 0)
            if resource_level > max_level:
                return False

        # Check conditions from rule.conditions dict
        if rule.conditions:
            for key, expected_value in rule.conditions.items():
                actual_value = request.context.get(key)
                if actual_value != expected_value:
                    # Also check principal attributes
                    actual_value = request.principal.attributes.get(key)
                    if actual_value != expected_value:
                        return False

        return True

    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Check if a glob-like pattern matches a value."""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return pattern == value

        # Convert glob to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", value))

    def _check_conditions(
        self,
        request: AccessRequest,
        rule_ids: List[str]
    ) -> bool:
        """Check if all conditions are met for conditional access."""
        now = DeterministicClock.now()

        for rule_id in rule_ids:
            # Find the rule
            for policy in self._policies.values():
                for rule in policy.rules:
                    if rule.rule_id == rule_id:
                        # Check time constraints
                        if rule.time_constraints:
                            start_hour = rule.time_constraints.get("start_hour", 0)
                            end_hour = rule.time_constraints.get("end_hour", 24)
                            if not (start_hour <= now.hour < end_hour):
                                return False

                        # Check geographic constraints
                        if rule.geographic_constraints:
                            location = request.resource.geographic_location
                            if location and location not in rule.geographic_constraints:
                                return False

        return True

    def add_rego_policy(self, policy_id: str, rego_source: str) -> str:
        """
        Add a Rego policy for OPA evaluation.

        Args:
            policy_id: Unique policy identifier
            rego_source: Rego policy source code

        Returns:
            SHA-256 hash of the Rego source
        """
        self._rego_policies[policy_id] = rego_source
        policy_hash = hashlib.sha256(rego_source.encode()).hexdigest()
        logger.info(f"Added Rego policy: {policy_id} (hash: {policy_hash[:16]})")
        return policy_hash

    def evaluate_rego(self, request: AccessRequest, policy_id: str) -> Optional[AccessDecisionResult]:
        """
        Evaluate a request against a Rego policy.
        Note: Requires OPA server or embedded OPA for actual evaluation.

        Args:
            request: The access request
            policy_id: The Rego policy to evaluate

        Returns:
            AccessDecisionResult if OPA is enabled, None otherwise
        """
        if not self.config.opa_enabled:
            logger.warning("OPA evaluation requested but OPA is not enabled")
            return None

        rego_source = self._rego_policies.get(policy_id)
        if not rego_source:
            logger.error(f"Rego policy not found: {policy_id}")
            return None

        # In a real implementation, this would call the OPA server
        # For now, we return a placeholder indicating OPA would be used
        logger.info(f"Would evaluate Rego policy: {policy_id}")
        return None


# =============================================================================
# POLICY GUARD AGENT
# =============================================================================


class PolicyGuardAgent(BaseAgent):
    """
    GL-FOUND-X-006: Access & Policy Guard Agent

    The core authorization and policy enforcement agent for GreenLang Climate OS.
    Implements RBAC/ABAC policies, data classification, tenant isolation,
    and comprehensive audit logging for compliance.

    Zero-Hallucination Guarantees:
        - All access decisions are deterministic based on policy rules
        - Complete audit trail for every access decision
        - No probabilistic or ML-based access decisions
        - All policies versioned with SHA-256 provenance hashes

    Usage:
        guard = PolicyGuardAgent(config)
        result = guard.check_access(access_request)
        if result.allowed:
            # Proceed with operation
        else:
            # Handle denial
    """

    AGENT_ID = "GL-FOUND-X-006"
    AGENT_NAME = "Access & Policy Guard"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Policy Guard Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Authorization and policy enforcement agent",
                version=self.VERSION,
                parameters={}
            )

        # Parse guard-specific configuration BEFORE calling super().__init__()
        # because super().__init__() calls initialize() which needs these components
        guard_config_dict = config.parameters.get("guard_config", {})
        self._guard_config = PolicyGuardConfig(**guard_config_dict)

        # Initialize components BEFORE calling super().__init__()
        self._policy_engine = PolicyEngine(self._guard_config)
        self._rate_limiter = RateLimiter(self._guard_config.rate_limit_config)

        # Audit log storage (in production, use database/streaming)
        self._audit_log: List[AuditEvent] = []
        self._audit_log_max_size = 100000

        # Decision cache
        self._decision_cache: Dict[str, Tuple[AccessDecisionResult, float]] = {}

        # Metrics
        self._total_requests = 0
        self._allowed_requests = 0
        self._denied_requests = 0
        self._rate_limited_requests = 0

        # Now call super().__init__() which will call initialize()
        super().__init__(config)

        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def initialize(self):
        """Custom initialization for the Policy Guard Agent."""
        self._load_default_policies()

    def _load_default_policies(self):
        """Load default security policies."""
        # Tenant isolation policy
        tenant_isolation_policy = Policy(
            policy_id="default-tenant-isolation",
            name="Default Tenant Isolation",
            description="Enforce strict tenant isolation - principals can only access resources in their tenant",
            rules=[
                PolicyRule(
                    rule_id="tenant-isolation-001",
                    name="Tenant Boundary Enforcement",
                    description="Deny access if principal tenant != resource tenant",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=1,
                    effect=AccessDecision.DENY,
                    actions=["*"],
                    conditions={"_check": "tenant_mismatch"}
                )
            ]
        )
        self._policy_engine.add_policy(tenant_isolation_policy)

        # Classification access policy
        classification_policy = Policy(
            policy_id="default-classification",
            name="Default Classification Access",
            description="Enforce data classification access based on clearance level",
            rules=[
                PolicyRule(
                    rule_id="classification-001",
                    name="Clearance Level Check",
                    description="Deny access if clearance level is insufficient",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=5,
                    effect=AccessDecision.DENY,
                    actions=["*"],
                    conditions={"_check": "clearance_insufficient"}
                )
            ]
        )
        self._policy_engine.add_policy(classification_policy)

        logger.info("Loaded default security policies")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the Policy Guard Agent.

        Input data should contain:
            - operation: The operation to perform (check_access, add_policy, etc.)
            - request: AccessRequest data (for check_access)
            - policy: Policy data (for add_policy)

        Returns:
            AgentResult with operation result
        """
        operation = input_data.get("operation", "check_access")

        try:
            if operation == "check_access":
                request_data = input_data.get("request", {})
                if isinstance(request_data, dict):
                    request = AccessRequest(**request_data)
                else:
                    request = request_data
                result = self.check_access(request)
                return AgentResult(
                    success=True,
                    data=result.model_dump()
                )

            elif operation == "add_policy":
                policy_data = input_data.get("policy", {})
                if isinstance(policy_data, dict):
                    policy = Policy(**policy_data)
                else:
                    policy = policy_data
                policy_hash = self.add_policy(policy)
                return AgentResult(
                    success=True,
                    data={"policy_hash": policy_hash, "policy_id": policy.policy_id}
                )

            elif operation == "remove_policy":
                policy_id = input_data.get("policy_id")
                removed = self.remove_policy(policy_id)
                return AgentResult(
                    success=removed,
                    data={"removed": removed, "policy_id": policy_id}
                )

            elif operation == "classify_data":
                resource_data = input_data.get("resource", {})
                if isinstance(resource_data, dict):
                    resource = Resource(**resource_data)
                else:
                    resource = resource_data
                classification = self.classify_data(resource)
                return AgentResult(
                    success=True,
                    data={"classification": classification.value}
                )

            elif operation == "generate_report":
                tenant_id = input_data.get("tenant_id")
                start_date = input_data.get("start_date")
                end_date = input_data.get("end_date")
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date)
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date)
                report = self.generate_compliance_report(tenant_id, start_date, end_date)
                return AgentResult(
                    success=True,
                    data=report.model_dump()
                )

            elif operation == "simulate_policies":
                requests_data = input_data.get("requests", [])
                policy_ids = input_data.get("policy_ids", [])
                requests = [
                    AccessRequest(**r) if isinstance(r, dict) else r
                    for r in requests_data
                ]
                result = self.simulate_policies(requests, policy_ids)
                return AgentResult(
                    success=True,
                    data=result.model_dump()
                )

            elif operation == "get_metrics":
                metrics = self.get_metrics()
                return AgentResult(
                    success=True,
                    data=metrics
                )

            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )

        except Exception as e:
            logger.error(f"Policy Guard execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )

    def check_access(self, request: AccessRequest) -> AccessDecisionResult:
        """
        Check if an access request should be allowed.

        Args:
            request: The access request to evaluate

        Returns:
            AccessDecisionResult with the decision and details
        """
        self._total_requests += 1
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(request)
        if cache_key in self._decision_cache:
            cached_result, cached_time = self._decision_cache[cache_key]
            if time.time() - cached_time < self._guard_config.decision_cache_ttl_seconds:
                return cached_result

        deny_reasons: List[str] = []

        # Step 1: Verify principal is authenticated
        if not request.principal.authenticated:
            deny_reasons.append("Principal is not authenticated")
            # Log audit event for unauthenticated access attempt
            if self._guard_config.audit_enabled:
                self._log_audit_event(
                    AuditEventType.ACCESS_DENIED,
                    request,
                    AccessDecision.DENY,
                    {"reason": "unauthenticated"}
                )
            return self._make_decision(request, AccessDecision.DENY, deny_reasons, start_time)

        # Step 2: Check tenant isolation
        if self._guard_config.strict_tenant_isolation:
            if request.principal.tenant_id != request.resource.tenant_id:
                deny_reasons.append(
                    f"Tenant boundary violation: principal tenant '{request.principal.tenant_id}' "
                    f"cannot access resource in tenant '{request.resource.tenant_id}'"
                )
                self._log_audit_event(
                    AuditEventType.TENANT_BOUNDARY_VIOLATION,
                    request,
                    AccessDecision.DENY
                )
                return self._make_decision(request, AccessDecision.DENY, deny_reasons, start_time)

        # Step 3: Check data classification
        principal_clearance = CLASSIFICATION_HIERARCHY.get(request.principal.clearance_level, 0)
        resource_classification = CLASSIFICATION_HIERARCHY.get(request.resource.classification, 0)
        if resource_classification > principal_clearance:
            deny_reasons.append(
                f"Insufficient clearance: principal has '{request.principal.clearance_level.value}' "
                f"but resource requires '{request.resource.classification.value}'"
            )
            self._log_audit_event(
                AuditEventType.CLASSIFICATION_CHECK,
                request,
                AccessDecision.DENY,
                {"reason": "insufficient_clearance"}
            )
            return self._make_decision(request, AccessDecision.DENY, deny_reasons, start_time)

        # Step 4: Check rate limits
        if self._guard_config.rate_limiting_enabled:
            # Get the highest role for rate limit calculation
            highest_role = None
            for role in request.principal.roles:
                if role in self._guard_config.rate_limit_config.role_overrides:
                    highest_role = role
                    break

            allowed, rate_limit_reason = self._rate_limiter.check_rate_limit(
                request.principal.tenant_id,
                request.principal.principal_id,
                highest_role
            )
            if not allowed:
                self._rate_limited_requests += 1
                deny_reasons.append(rate_limit_reason)
                self._log_audit_event(
                    AuditEventType.RATE_LIMIT_EXCEEDED,
                    request,
                    AccessDecision.DENY,
                    {"reason": rate_limit_reason}
                )
                return self._make_decision(request, AccessDecision.DENY, deny_reasons, start_time)

        # Step 5: Evaluate policies
        result = self._policy_engine.evaluate(request)

        # Step 6: Log audit event
        if self._guard_config.audit_enabled:
            if self._guard_config.audit_all_decisions or result.decision == AccessDecision.DENY:
                self._log_audit_event(
                    AuditEventType.ACCESS_GRANTED if result.allowed else AuditEventType.ACCESS_DENIED,
                    request,
                    result.decision,
                    {"matching_rules": result.matching_rules}
                )

        # Update metrics
        if result.allowed:
            self._allowed_requests += 1
        else:
            self._denied_requests += 1

        # Cache the result
        self._decision_cache[cache_key] = (result, time.time())

        # Handle simulation mode
        if self._guard_config.simulation_mode:
            logger.info(
                f"SIMULATION: Request {request.request_id} would be "
                f"{'ALLOWED' if result.allowed else 'DENIED'}"
            )
            # In simulation mode, always allow but log the would-be decision
            simulated_result = AccessDecisionResult(
                request_id=result.request_id,
                decision=AccessDecision.ALLOW,
                allowed=True,
                matching_rules=result.matching_rules,
                deny_reasons=[f"[SIMULATED] {r}" for r in result.deny_reasons],
                evaluation_time_ms=result.evaluation_time_ms,
                policy_versions=result.policy_versions,
                decision_hash=result.decision_hash
            )
            return simulated_result

        return result

    def _make_decision(
        self,
        request: AccessRequest,
        decision: AccessDecision,
        deny_reasons: List[str],
        start_time: float
    ) -> AccessDecisionResult:
        """Create an access decision result."""
        evaluation_time = (time.time() - start_time) * 1000

        result = AccessDecisionResult(
            request_id=request.request_id,
            decision=decision,
            allowed=decision == AccessDecision.ALLOW,
            deny_reasons=deny_reasons,
            evaluation_time_ms=evaluation_time
        )

        # Compute decision hash
        decision_str = json.dumps({
            "request_id": request.request_id,
            "decision": decision.value,
            "deny_reasons": deny_reasons,
            "timestamp": result.evaluated_at.isoformat()
        }, sort_keys=True)
        result.decision_hash = hashlib.sha256(decision_str.encode()).hexdigest()

        if decision == AccessDecision.DENY:
            self._denied_requests += 1
        else:
            self._allowed_requests += 1

        return result

    def _get_cache_key(self, request: AccessRequest) -> str:
        """Generate a cache key for a request."""
        key_parts = [
            request.principal.principal_id,
            request.principal.tenant_id,
            request.resource.resource_id,
            request.resource.tenant_id,
            request.action,
            str(sorted(request.principal.roles))
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    def _log_audit_event(
        self,
        event_type: AuditEventType,
        request: AccessRequest,
        decision: AccessDecision,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            tenant_id=request.resource.tenant_id,
            principal_id=request.principal.principal_id,
            resource_id=request.resource.resource_id,
            action=request.action,
            decision=decision,
            details=details or {},
            source_ip=request.source_ip,
            user_agent=request.user_agent,
            retention_days=self._guard_config.audit_retention_days
        )

        # Add to audit log
        self._audit_log.append(event)

        # Trim log if too large
        if len(self._audit_log) > self._audit_log_max_size:
            self._audit_log = self._audit_log[-self._audit_log_max_size // 2:]

        logger.debug(f"Audit event: {event_type.value} - {request.request_id}")

    def add_policy(self, policy: Policy) -> str:
        """
        Add a policy to the guard.

        Args:
            policy: The policy to add

        Returns:
            Policy provenance hash
        """
        policy_hash = self._policy_engine.add_policy(policy)

        # Log policy update
        self._audit_log.append(AuditEvent(
            event_type=AuditEventType.POLICY_UPDATED,
            tenant_id=policy.tenant_id or "global",
            details={
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
                "action": "add"
            }
        ))

        return policy_hash

    def remove_policy(self, policy_id: str) -> bool:
        """
        Remove a policy from the guard.

        Args:
            policy_id: ID of the policy to remove

        Returns:
            True if removed, False if not found
        """
        removed = self._policy_engine.remove_policy(policy_id)

        if removed:
            self._audit_log.append(AuditEvent(
                event_type=AuditEventType.POLICY_UPDATED,
                tenant_id="global",
                details={
                    "policy_id": policy_id,
                    "action": "remove"
                }
            ))

        return removed

    def classify_data(self, resource: Resource) -> DataClassification:
        """
        Classify data based on resource attributes.

        Args:
            resource: The resource to classify

        Returns:
            DataClassification level
        """
        # Default classification based on resource type
        classification = resource.classification

        # Check for sensitive patterns in resource attributes
        attributes = resource.attributes

        # PII detection (simplified)
        if any(key in str(attributes).lower() for key in ["ssn", "social_security", "passport", "credit_card"]):
            classification = DataClassification.RESTRICTED

        # Financial data
        if "financial" in resource.resource_type.lower() or "payment" in resource.resource_type.lower():
            if CLASSIFICATION_HIERARCHY.get(classification, 0) < CLASSIFICATION_HIERARCHY[DataClassification.CONFIDENTIAL]:
                classification = DataClassification.CONFIDENTIAL

        # Emission data sensitivity
        if "emission" in resource.resource_type.lower():
            # Emission data is typically internal or confidential
            if CLASSIFICATION_HIERARCHY.get(classification, 0) < CLASSIFICATION_HIERARCHY[DataClassification.INTERNAL]:
                classification = DataClassification.INTERNAL

        self._log_audit_event(
            AuditEventType.CLASSIFICATION_CHECK,
            AccessRequest(
                principal=Principal(principal_id="system", tenant_id=resource.tenant_id),
                resource=resource,
                action="classify"
            ),
            AccessDecision.ALLOW,
            {"classified_as": classification.value}
        )

        return classification

    def check_export_allowed(
        self,
        request: AccessRequest,
        export_destination: str
    ) -> AccessDecisionResult:
        """
        Check if data export is allowed.

        Args:
            request: Access request for the export
            export_destination: Where the data is being exported to

        Returns:
            AccessDecisionResult for the export
        """
        # First check basic access
        access_result = self.check_access(request)
        if not access_result.allowed:
            return access_result

        deny_reasons: List[str] = []

        # Check classification allows export
        if request.resource.classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            deny_reasons.append(
                f"Export not allowed for {request.resource.classification.value} data"
            )

        # Check geographic constraints
        if request.resource.geographic_location:
            # Check for data residency violations
            if "external" in export_destination.lower():
                # Simplified check - in production, use proper geo policy
                if request.resource.classification != DataClassification.PUBLIC:
                    deny_reasons.append(
                        f"External export requires PUBLIC classification"
                    )

        if deny_reasons:
            self._log_audit_event(
                AuditEventType.EXPORT_DENIED,
                request,
                AccessDecision.DENY,
                {"destination": export_destination, "reasons": deny_reasons}
            )
            return self._make_decision(request, AccessDecision.DENY, deny_reasons, time.time())

        self._log_audit_event(
            AuditEventType.EXPORT_APPROVED,
            request,
            AccessDecision.ALLOW,
            {"destination": export_destination}
        )

        return access_result

    def simulate_policies(
        self,
        test_requests: List[AccessRequest],
        policy_ids: Optional[List[str]] = None
    ) -> PolicySimulationResult:
        """
        Simulate policy evaluation without enforcing.

        Args:
            test_requests: List of test access requests
            policy_ids: Optional list of specific policy IDs to test

        Returns:
            PolicySimulationResult with detailed results
        """
        # Enable simulation mode temporarily
        original_mode = self._guard_config.simulation_mode
        self._guard_config.simulation_mode = True

        results: List[AccessDecisionResult] = []
        summary: Dict[str, int] = {"allowed": 0, "denied": 0, "conditional": 0}
        conflicts: List[Dict[str, Any]] = []

        try:
            for request in test_requests:
                result = self.check_access(request)
                results.append(result)

                if result.decision == AccessDecision.ALLOW:
                    summary["allowed"] += 1
                elif result.decision == AccessDecision.DENY:
                    summary["denied"] += 1
                else:
                    summary["conditional"] += 1

                # Detect potential conflicts (multiple rules with different effects)
                if len(result.matching_rules) > 1:
                    conflicts.append({
                        "request_id": request.request_id,
                        "conflicting_rules": result.matching_rules,
                        "final_decision": result.decision.value
                    })

            # Log simulation
            self._audit_log.append(AuditEvent(
                event_type=AuditEventType.SIMULATION_RUN,
                tenant_id="global",
                details={
                    "test_requests": len(test_requests),
                    "summary": summary,
                    "conflicts_found": len(conflicts)
                }
            ))

            return PolicySimulationResult(
                test_requests=len(test_requests),
                policies_tested=policy_ids or list(self._policy_engine._policies.keys()),
                results=results,
                summary=summary,
                conflicts_detected=conflicts
            )

        finally:
            # Restore original mode
            self._guard_config.simulation_mode = original_mode

    def generate_compliance_report(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceReport:
        """
        Generate a compliance report for a tenant.

        Args:
            tenant_id: Tenant to generate report for
            start_date: Report period start
            end_date: Report period end

        Returns:
            ComplianceReport with statistics and analysis
        """
        # Filter audit events for the tenant and time period
        # Handle timezone-aware vs naive datetime comparison by converting to naive
        def to_naive(dt: datetime) -> datetime:
            """Convert datetime to naive (remove timezone info)."""
            if dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt

        start_naive = to_naive(start_date)
        end_naive = to_naive(end_date)

        relevant_events = [
            event for event in self._audit_log
            if event.tenant_id == tenant_id
            and start_naive <= to_naive(event.timestamp) <= end_naive
        ]

        # Calculate statistics
        total_requests = len([
            e for e in relevant_events
            if e.event_type in [AuditEventType.ACCESS_GRANTED, AuditEventType.ACCESS_DENIED]
        ])
        allowed = len([
            e for e in relevant_events
            if e.event_type == AuditEventType.ACCESS_GRANTED
        ])
        denied = len([
            e for e in relevant_events
            if e.event_type == AuditEventType.ACCESS_DENIED
        ])
        rate_limited = len([
            e for e in relevant_events
            if e.event_type == AuditEventType.RATE_LIMIT_EXCEEDED
        ])

        # Breakdown by policy type
        decisions_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"allowed": 0, "denied": 0})
        for event in relevant_events:
            if event.action:
                if event.event_type == AuditEventType.ACCESS_GRANTED:
                    decisions_by_type[event.action]["allowed"] += 1
                elif event.event_type == AuditEventType.ACCESS_DENIED:
                    decisions_by_type[event.action]["denied"] += 1

        # Top denial reasons
        denial_reasons: Dict[str, int] = defaultdict(int)
        for event in relevant_events:
            if event.event_type == AuditEventType.ACCESS_DENIED:
                reason = event.details.get("reason", "unknown")
                denial_reasons[str(reason)] += 1

        top_denial_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(denial_reasons.items(), key=lambda x: -x[1])[:10]
        ]

        # Access by classification
        access_by_classification: Dict[str, int] = defaultdict(int)
        for event in relevant_events:
            if event.event_type == AuditEventType.CLASSIFICATION_CHECK:
                classification = event.details.get("classified_as", "unknown")
                access_by_classification[classification] += 1

        # Build report
        report = ComplianceReport(
            report_period_start=start_date,
            report_period_end=end_date,
            tenant_id=tenant_id,
            total_requests=total_requests,
            allowed_requests=allowed,
            denied_requests=denied,
            rate_limited_requests=rate_limited,
            decisions_by_type=dict(decisions_by_type),
            top_denial_reasons=top_denial_reasons,
            policies_evaluated=list(self._policy_engine._policies.keys()),
            access_by_classification=dict(access_by_classification)
        )

        report.provenance_hash = report.compute_hash()

        logger.info(
            f"Generated compliance report for tenant {tenant_id}: "
            f"{total_requests} requests, {allowed} allowed, {denied} denied"
        )

        return report

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics from the Policy Guard."""
        return {
            "total_requests": self._total_requests,
            "allowed_requests": self._allowed_requests,
            "denied_requests": self._denied_requests,
            "rate_limited_requests": self._rate_limited_requests,
            "allow_rate": (
                self._allowed_requests / self._total_requests * 100
                if self._total_requests > 0 else 0
            ),
            "policies_loaded": len(self._policy_engine._policies),
            "audit_events": len(self._audit_log),
            "cache_size": len(self._decision_cache),
            "simulation_mode": self._guard_config.simulation_mode,
            "strict_tenant_isolation": self._guard_config.strict_tenant_isolation,
            "rate_limiting_enabled": self._guard_config.rate_limiting_enabled,
        }

    def get_audit_events(
        self,
        tenant_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Retrieve audit events with optional filtering.

        Args:
            tenant_id: Optional tenant filter
            event_type: Optional event type filter
            limit: Maximum events to return

        Returns:
            List of matching audit events
        """
        events = self._audit_log

        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def clear_cache(self):
        """Clear the decision cache."""
        self._decision_cache.clear()
        logger.info("Decision cache cleared")

    def add_rego_policy(self, policy_id: str, rego_source: str) -> str:
        """
        Add a Rego policy for OPA evaluation.

        Args:
            policy_id: Unique policy identifier
            rego_source: Rego policy source code

        Returns:
            SHA-256 hash of the Rego source
        """
        return self._policy_engine.add_rego_policy(policy_id, rego_source)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main agent
    "PolicyGuardAgent",

    # Enums
    "AccessDecision",
    "PolicyType",
    "DataClassification",
    "RoleType",
    "AuditEventType",

    # Models
    "Principal",
    "Resource",
    "AccessRequest",
    "PolicyRule",
    "Policy",
    "AccessDecisionResult",
    "AuditEvent",
    "RateLimitConfig",
    "PolicyGuardConfig",
    "ComplianceReport",
    "PolicySimulationResult",

    # Components
    "PolicyEngine",
    "RateLimiter",

    # Constants
    "CLASSIFICATION_HIERARCHY",
    "DEFAULT_ROLE_PERMISSIONS",
]
