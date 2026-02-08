# -*- coding: utf-8 -*-
"""
GL-FOUND-X-006: GreenLang Access & Policy Guard SDK
====================================================

This package provides the access control, policy enforcement, and
audit logging SDK for the GreenLang framework. It supports:

- RBAC/ABAC policy engine with first-match-wins evaluation
- Data classification with built-in PII detection patterns
- Token bucket rate limiting with role-based overrides
- In-memory audit logger with compliance report generation
- OPA Rego policy management with syntax validation
- SHA-256 provenance tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_ACCESS_GUARD_ env prefix

Key Components:
    - policy_engine: PolicyEngine for rule evaluation
    - rate_limiter: RateLimiter for request throttling
    - classifier: DataClassifier for sensitivity detection
    - audit_logger: AuditLogger for event recording
    - opa_integration: OPAClient for Rego policies
    - provenance: ProvenanceTracker for SHA-256 audit trails
    - config: AccessGuardConfig with GL_ACCESS_GUARD_ env prefix
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: AccessGuardService facade

Example:
    >>> from greenlang.access_guard import AccessGuardService
    >>> service = AccessGuardService()
    >>> service.startup()
    >>> result = service.check_access(access_request)
    >>> print(result.allowed)

    >>> from greenlang.access_guard import PolicyEngine
    >>> engine = PolicyEngine()
    >>> engine.add_policy(policy)
    >>> result = engine.evaluate(request)

Agent ID: GL-FOUND-X-006
Agent Name: Access & Policy Guard
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-006"
__agent_name__ = "Access & Policy Guard"

# SDK availability flag
ACCESS_GUARD_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.access_guard.config import (
    AccessGuardConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, core, versioning, constants)
# ---------------------------------------------------------------------------
from greenlang.access_guard.models import (
    # Enumerations
    AccessDecision,
    PolicyType,
    DataClassification,
    RoleType,
    AuditEventType,
    PolicyChangeType,
    SimulationMode,
    # Constants
    CLASSIFICATION_HIERARCHY,
    DEFAULT_ROLE_PERMISSIONS,
    # Core models
    Principal,
    Resource,
    AccessRequest,
    PolicyRule,
    Policy,
    AccessDecisionResult,
    AuditEvent,
    RateLimitConfig,
    ComplianceReport,
    PolicySimulationResult,
    # Versioning models
    PolicyVersion,
    PolicyChangeLogEntry,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.access_guard.policy_engine import PolicyEngine
from greenlang.access_guard.rate_limiter import RateLimiter
from greenlang.access_guard.classifier import DataClassifier
from greenlang.access_guard.audit_logger import AuditLogger
from greenlang.access_guard.opa_integration import OPAClient
from greenlang.access_guard.provenance import ProvenanceTracker, ProvenanceEntry

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.access_guard.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    access_guard_decisions_total,
    access_guard_decision_duration_seconds,
    access_guard_denials_total,
    access_guard_rate_limits_total,
    access_guard_policy_evaluations_total,
    access_guard_tenant_violations_total,
    access_guard_classification_checks_total,
    access_guard_policies_total,
    access_guard_rules_total,
    access_guard_cache_hits_total,
    access_guard_cache_misses_total,
    access_guard_audit_events_total,
    # Helper functions
    record_decision,
    record_denial,
    record_rate_limit,
    record_policy_evaluation,
    record_tenant_violation,
    record_classification_check,
    update_policies_count,
    update_rules_count,
    record_cache_hit,
    record_cache_miss,
    update_audit_events_count,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.access_guard.setup import (
    AccessGuardService,
    configure_access_guard,
    get_access_guard,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "ACCESS_GUARD_SDK_AVAILABLE",
    # Configuration
    "AccessGuardConfig",
    "get_config",
    "set_config",
    "reset_config",
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
    # Core engines
    "PolicyEngine",
    "RateLimiter",
    "DataClassifier",
    "AuditLogger",
    "OPAClient",
    "ProvenanceTracker",
    "ProvenanceEntry",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "access_guard_decisions_total",
    "access_guard_decision_duration_seconds",
    "access_guard_denials_total",
    "access_guard_rate_limits_total",
    "access_guard_policy_evaluations_total",
    "access_guard_tenant_violations_total",
    "access_guard_classification_checks_total",
    "access_guard_policies_total",
    "access_guard_rules_total",
    "access_guard_cache_hits_total",
    "access_guard_cache_misses_total",
    "access_guard_audit_events_total",
    # Metric helper functions
    "record_decision",
    "record_denial",
    "record_rate_limit",
    "record_policy_evaluation",
    "record_tenant_violation",
    "record_classification_check",
    "update_policies_count",
    "update_rules_count",
    "record_cache_hit",
    "record_cache_miss",
    "update_audit_events_count",
    # Service setup facade
    "AccessGuardService",
    "configure_access_guard",
    "get_access_guard",
    "get_router",
]
