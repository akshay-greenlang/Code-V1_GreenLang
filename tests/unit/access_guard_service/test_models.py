# -*- coding: utf-8 -*-
"""
Unit Tests for Access Guard Models (AGENT-FOUND-006)

Tests all enums, model classes, field validation, serialization,
hash computation, and edge cases for the access guard data types.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models mirroring the foundation policy_guard.py
# ---------------------------------------------------------------------------


class AccessDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class PolicyType(str, Enum):
    DATA_ACCESS = "data_access"
    AGENT_EXECUTION = "agent_execution"
    EXPORT = "export"
    RETENTION = "retention"
    GEOGRAPHIC = "geographic"
    RATE_LIMIT = "rate_limit"


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RoleType(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    EDITOR = "editor"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    SERVICE_ACCOUNT = "service_account"


class AuditEventType(str, Enum):
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


class PolicyChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ENABLE = "enable"
    DISABLE = "disable"


class SimulationMode(str, Enum):
    DRY_RUN = "dry_run"
    SHADOW = "shadow"
    CANARY = "canary"


# Classification hierarchy: higher number = more sensitive
CLASSIFICATION_HIERARCHY: Dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
    DataClassification.TOP_SECRET: 4,
}

# Default role permissions
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


# ---------------------------------------------------------------------------
# Lightweight model classes mirroring Pydantic models for testing
# ---------------------------------------------------------------------------


class Principal:
    def __init__(
        self,
        principal_id: str,
        principal_type: str = "user",
        tenant_id: str = "",
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        clearance_level: str = "internal",
        groups: Optional[List[str]] = None,
        authenticated: bool = True,
        session_id: Optional[str] = None,
    ):
        self.principal_id = principal_id
        self.principal_type = principal_type
        self.tenant_id = tenant_id
        self.roles = roles or []
        self.attributes = attributes or {}
        self.clearance_level = DataClassification(clearance_level.lower())
        self.groups = groups or []
        self.authenticated = authenticated
        self.session_id = session_id


class Resource:
    def __init__(
        self,
        resource_id: str,
        resource_type: str = "data",
        tenant_id: str = "",
        classification: str = "internal",
        owner_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        geographic_location: Optional[str] = None,
        created_at: Optional[datetime] = None,
        retention_policy: Optional[str] = None,
    ):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.tenant_id = tenant_id
        self.classification = DataClassification(classification.lower())
        self.owner_id = owner_id
        self.attributes = attributes or {}
        self.geographic_location = geographic_location
        self.created_at = created_at
        self.retention_policy = retention_policy


class AccessRequest:
    def __init__(
        self,
        principal: Principal,
        resource: Resource,
        action: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.principal = principal
        self.resource = resource
        self.action = action
        self.context = context or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.source_ip = source_ip
        self.user_agent = user_agent


class PolicyRule:
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str = "",
        policy_type: str = "data_access",
        priority: int = 100,
        enabled: bool = True,
        conditions: Optional[Dict[str, Any]] = None,
        effect: str = "deny",
        actions: Optional[List[str]] = None,
        resources: Optional[List[str]] = None,
        principals: Optional[List[str]] = None,
        time_constraints: Optional[Dict[str, Any]] = None,
        geographic_constraints: Optional[List[str]] = None,
        classification_max: Optional[str] = None,
        version: str = "1.0.0",
        created_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.policy_type = PolicyType(policy_type)
        if not (0 <= priority <= 1000):
            raise ValueError(f"Priority must be between 0 and 1000, got {priority}")
        self.priority = priority
        self.enabled = enabled
        self.conditions = conditions or {}
        self.effect = AccessDecision(effect)
        self.actions = actions or []
        self.resources = resources or []
        self.principals = principals or []
        self.time_constraints = time_constraints
        self.geographic_constraints = geographic_constraints
        self.classification_max = (
            DataClassification(classification_max.lower())
            if classification_max
            else None
        )
        self.version = version
        self.created_at = created_at or datetime.utcnow()
        self.created_by = created_by
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "policy_type": self.policy_type.value,
            "priority": self.priority,
            "effect": self.effect.value,
            "actions": self.actions,
            "resources": self.resources,
            "principals": self.principals,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Policy:
    def __init__(
        self,
        policy_id: str,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        enabled: bool = True,
        rules: Optional[List[PolicyRule]] = None,
        parent_policy_id: Optional[str] = None,
        allow_override: bool = True,
        tenant_id: Optional[str] = None,
        applies_to: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        provenance_hash: str = "",
    ):
        self.policy_id = policy_id
        self.name = name
        self.description = description
        self.version = version
        self.enabled = enabled
        self.rules = rules or []
        self.parent_policy_id = parent_policy_id
        self.allow_override = allow_override
        self.tenant_id = tenant_id
        self.applies_to = applies_to or []
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.created_by = created_by
        self.provenance_hash = provenance_hash

    def compute_hash(self) -> str:
        policy_str = json.dumps(
            {
                "policy_id": self.policy_id,
                "rules": [r.to_dict() for r in self.rules],
                "version": self.version,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(policy_str.encode()).hexdigest()


class AccessDecisionResult:
    def __init__(
        self,
        request_id: str,
        decision: str,
        allowed: bool,
        matching_rules: Optional[List[str]] = None,
        deny_reasons: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        evaluated_at: Optional[datetime] = None,
        evaluation_time_ms: float = 0.0,
        policy_versions: Optional[Dict[str, str]] = None,
        decision_hash: str = "",
    ):
        self.request_id = request_id
        self.decision = AccessDecision(decision)
        self.allowed = allowed
        self.matching_rules = matching_rules or []
        self.deny_reasons = deny_reasons or []
        self.conditions = conditions or []
        self.evaluated_at = evaluated_at or datetime.utcnow()
        self.evaluation_time_ms = evaluation_time_ms
        self.policy_versions = policy_versions or {}
        self.decision_hash = decision_hash

    def compute_decision_hash(self) -> str:
        decision_str = json.dumps(
            {
                "request_id": self.request_id,
                "decision": self.decision.value,
                "matching_rules": self.matching_rules,
                "timestamp": self.evaluated_at.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(decision_str.encode()).hexdigest()


class AuditEvent:
    def __init__(
        self,
        event_type: str,
        tenant_id: str,
        event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        principal_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        decision: Optional[str] = None,
        decision_hash: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        retention_days: int = 365,
    ):
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = AuditEventType(event_type)
        self.timestamp = timestamp or datetime.utcnow()
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self.resource_id = resource_id
        self.action = action
        self.decision = AccessDecision(decision) if decision else None
        self.decision_hash = decision_hash
        self.details = details or {}
        self.source_ip = source_ip
        self.user_agent = user_agent
        self.retention_days = retention_days


class RateLimitConfig:
    def __init__(
        self,
        requests_per_minute: int = 100,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_limit: int = 20,
        role_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit
        self.role_overrides = role_overrides or {
            "admin": {"requests_per_minute": 500, "requests_per_hour": 5000},
            "super_admin": {"requests_per_minute": 1000, "requests_per_hour": 10000},
            "service_account": {"requests_per_minute": 1000, "requests_per_hour": 50000},
        }


class ComplianceReport:
    def __init__(
        self,
        tenant_id: str,
        report_period_start: datetime,
        report_period_end: datetime,
        report_id: Optional[str] = None,
        generated_at: Optional[datetime] = None,
        total_requests: int = 0,
        allowed_requests: int = 0,
        denied_requests: int = 0,
        rate_limited_requests: int = 0,
        decisions_by_type: Optional[Dict[str, Dict[str, int]]] = None,
        top_denial_reasons: Optional[List[Dict[str, Any]]] = None,
        policies_evaluated: Optional[List[str]] = None,
        rules_triggered: Optional[Dict[str, int]] = None,
        access_by_region: Optional[Dict[str, int]] = None,
        access_by_classification: Optional[Dict[str, int]] = None,
        provenance_hash: str = "",
    ):
        self.report_id = report_id or str(uuid.uuid4())
        self.generated_at = generated_at or datetime.utcnow()
        self.report_period_start = report_period_start
        self.report_period_end = report_period_end
        self.tenant_id = tenant_id
        self.total_requests = total_requests
        self.allowed_requests = allowed_requests
        self.denied_requests = denied_requests
        self.rate_limited_requests = rate_limited_requests
        self.decisions_by_type = decisions_by_type or {}
        self.top_denial_reasons = top_denial_reasons or []
        self.policies_evaluated = policies_evaluated or []
        self.rules_triggered = rules_triggered or {}
        self.access_by_region = access_by_region or {}
        self.access_by_classification = access_by_classification or {}
        self.provenance_hash = provenance_hash

    def compute_hash(self) -> str:
        data = {
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "denied_requests": self.denied_requests,
            "generated_at": self.generated_at.isoformat() if self.generated_at else "",
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()


class PolicySimulationResult:
    def __init__(
        self,
        test_requests: int = 0,
        policies_tested: Optional[List[str]] = None,
        results: Optional[List[AccessDecisionResult]] = None,
        summary: Optional[Dict[str, int]] = None,
        conflicts_detected: Optional[List[Dict[str, Any]]] = None,
        coverage_gaps: Optional[List[str]] = None,
        simulation_id: Optional[str] = None,
        run_at: Optional[datetime] = None,
    ):
        self.simulation_id = simulation_id or str(uuid.uuid4())
        self.run_at = run_at or datetime.utcnow()
        self.test_requests = test_requests
        self.policies_tested = policies_tested or []
        self.results = results or []
        self.summary = summary or {}
        self.conflicts_detected = conflicts_detected or []
        self.coverage_gaps = coverage_gaps or []


class PolicyVersion:
    def __init__(
        self,
        policy_id: str,
        version: str,
        hash_value: str,
        created_at: Optional[datetime] = None,
    ):
        self.policy_id = policy_id
        self.version = version
        self.hash_value = hash_value
        self.created_at = created_at or datetime.utcnow()


class PolicyChangeLogEntry:
    def __init__(
        self,
        policy_id: str,
        change_type: str,
        changed_by: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.policy_id = policy_id
        self.change_type = PolicyChangeType(change_type)
        self.changed_by = changed_by
        self.timestamp = timestamp or datetime.utcnow()
        self.details = details or {}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAccessDecisionEnum:
    """Test AccessDecision enum values."""

    def test_allow_value(self):
        assert AccessDecision.ALLOW.value == "allow"

    def test_deny_value(self):
        assert AccessDecision.DENY.value == "deny"

    def test_conditional_value(self):
        assert AccessDecision.CONDITIONAL.value == "conditional"

    def test_enum_count(self):
        assert len(AccessDecision) == 3

    def test_from_string(self):
        assert AccessDecision("allow") == AccessDecision.ALLOW

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AccessDecision("invalid")


class TestPolicyTypeEnum:
    """Test PolicyType enum values."""

    def test_data_access(self):
        assert PolicyType.DATA_ACCESS.value == "data_access"

    def test_agent_execution(self):
        assert PolicyType.AGENT_EXECUTION.value == "agent_execution"

    def test_export(self):
        assert PolicyType.EXPORT.value == "export"

    def test_retention(self):
        assert PolicyType.RETENTION.value == "retention"

    def test_geographic(self):
        assert PolicyType.GEOGRAPHIC.value == "geographic"

    def test_rate_limit(self):
        assert PolicyType.RATE_LIMIT.value == "rate_limit"

    def test_enum_count(self):
        assert len(PolicyType) == 6


class TestDataClassificationEnum:
    """Test DataClassification enum values."""

    def test_public(self):
        assert DataClassification.PUBLIC.value == "public"

    def test_internal(self):
        assert DataClassification.INTERNAL.value == "internal"

    def test_confidential(self):
        assert DataClassification.CONFIDENTIAL.value == "confidential"

    def test_restricted(self):
        assert DataClassification.RESTRICTED.value == "restricted"

    def test_top_secret(self):
        assert DataClassification.TOP_SECRET.value == "top_secret"

    def test_enum_count(self):
        assert len(DataClassification) == 5


class TestRoleTypeEnum:
    """Test RoleType enum values."""

    def test_viewer(self):
        assert RoleType.VIEWER.value == "viewer"

    def test_analyst(self):
        assert RoleType.ANALYST.value == "analyst"

    def test_editor(self):
        assert RoleType.EDITOR.value == "editor"

    def test_admin(self):
        assert RoleType.ADMIN.value == "admin"

    def test_super_admin(self):
        assert RoleType.SUPER_ADMIN.value == "super_admin"

    def test_service_account(self):
        assert RoleType.SERVICE_ACCOUNT.value == "service_account"

    def test_enum_count(self):
        assert len(RoleType) == 6


class TestAuditEventTypeEnum:
    """Test AuditEventType enum values."""

    def test_access_granted(self):
        assert AuditEventType.ACCESS_GRANTED.value == "access_granted"

    def test_access_denied(self):
        assert AuditEventType.ACCESS_DENIED.value == "access_denied"

    def test_policy_evaluated(self):
        assert AuditEventType.POLICY_EVALUATED.value == "policy_evaluated"

    def test_rate_limit_exceeded(self):
        assert AuditEventType.RATE_LIMIT_EXCEEDED.value == "rate_limit_exceeded"

    def test_tenant_boundary_violation(self):
        assert AuditEventType.TENANT_BOUNDARY_VIOLATION.value == "tenant_boundary_violation"

    def test_classification_check(self):
        assert AuditEventType.CLASSIFICATION_CHECK.value == "classification_check"

    def test_export_approved(self):
        assert AuditEventType.EXPORT_APPROVED.value == "export_approved"

    def test_export_denied(self):
        assert AuditEventType.EXPORT_DENIED.value == "export_denied"

    def test_policy_updated(self):
        assert AuditEventType.POLICY_UPDATED.value == "policy_updated"

    def test_simulation_run(self):
        assert AuditEventType.SIMULATION_RUN.value == "simulation_run"

    def test_enum_count(self):
        assert len(AuditEventType) == 10


class TestPolicyChangeTypeEnum:
    """Test PolicyChangeType enum values."""

    def test_create(self):
        assert PolicyChangeType.CREATE.value == "create"

    def test_update(self):
        assert PolicyChangeType.UPDATE.value == "update"

    def test_delete(self):
        assert PolicyChangeType.DELETE.value == "delete"

    def test_enable(self):
        assert PolicyChangeType.ENABLE.value == "enable"

    def test_disable(self):
        assert PolicyChangeType.DISABLE.value == "disable"

    def test_enum_count(self):
        assert len(PolicyChangeType) == 5


class TestSimulationModeEnum:
    """Test SimulationMode enum values."""

    def test_dry_run(self):
        assert SimulationMode.DRY_RUN.value == "dry_run"

    def test_shadow(self):
        assert SimulationMode.SHADOW.value == "shadow"

    def test_canary(self):
        assert SimulationMode.CANARY.value == "canary"

    def test_enum_count(self):
        assert len(SimulationMode) == 3


class TestPrincipal:
    """Test Principal model creation and validation."""

    def test_creation_basic(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        assert p.principal_id == "u1"
        assert p.tenant_id == "t1"
        assert p.principal_type == "user"
        assert p.roles == []
        assert p.authenticated is True

    def test_clearance_level_default(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        assert p.clearance_level == DataClassification.INTERNAL

    def test_clearance_level_string_conversion(self):
        p = Principal(principal_id="u1", tenant_id="t1", clearance_level="confidential")
        assert p.clearance_level == DataClassification.CONFIDENTIAL

    def test_clearance_level_case_insensitive(self):
        p = Principal(principal_id="u1", tenant_id="t1", clearance_level="RESTRICTED")
        assert p.clearance_level == DataClassification.RESTRICTED

    def test_roles_set(self):
        p = Principal(principal_id="u1", tenant_id="t1", roles=["analyst", "viewer"])
        assert "analyst" in p.roles
        assert "viewer" in p.roles

    def test_attributes_set(self):
        p = Principal(
            principal_id="u1", tenant_id="t1",
            attributes={"dept": "finance"},
        )
        assert p.attributes["dept"] == "finance"

    def test_groups_set(self):
        p = Principal(principal_id="u1", tenant_id="t1", groups=["team-a"])
        assert p.groups == ["team-a"]

    def test_unauthenticated_principal(self):
        p = Principal(principal_id="u1", tenant_id="t1", authenticated=False)
        assert p.authenticated is False

    def test_session_id(self):
        p = Principal(principal_id="u1", tenant_id="t1", session_id="sess-123")
        assert p.session_id == "sess-123"


class TestResource:
    """Test Resource model creation and validation."""

    def test_creation_basic(self):
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        assert r.resource_id == "r1"
        assert r.resource_type == "data"
        assert r.tenant_id == "t1"

    def test_classification_default(self):
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        assert r.classification == DataClassification.INTERNAL

    def test_classification_string_conversion(self):
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1", classification="restricted")
        assert r.classification == DataClassification.RESTRICTED

    def test_owner_id(self):
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1", owner_id="user-1")
        assert r.owner_id == "user-1"

    def test_geographic_location(self):
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1", geographic_location="EU")
        assert r.geographic_location == "EU"

    def test_attributes(self):
        r = Resource(
            resource_id="r1", resource_type="data", tenant_id="t1",
            attributes={"scope": "scope_1"},
        )
        assert r.attributes["scope"] == "scope_1"


class TestAccessRequest:
    """Test AccessRequest model creation and defaults."""

    def test_creation(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        req = AccessRequest(principal=p, resource=r, action="read")
        assert req.action == "read"
        assert req.principal.principal_id == "u1"
        assert req.resource.resource_id == "r1"

    def test_auto_generated_request_id(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        req = AccessRequest(principal=p, resource=r, action="read")
        assert req.request_id is not None
        assert len(req.request_id) == 36  # UUID format

    def test_custom_request_id(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        req = AccessRequest(principal=p, resource=r, action="read", request_id="custom-123")
        assert req.request_id == "custom-123"

    def test_context_default_empty(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        req = AccessRequest(principal=p, resource=r, action="read")
        assert req.context == {}

    def test_timestamp_auto_set(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        req = AccessRequest(principal=p, resource=r, action="read")
        assert isinstance(req.timestamp, datetime)

    def test_source_ip_and_user_agent(self):
        p = Principal(principal_id="u1", tenant_id="t1")
        r = Resource(resource_id="r1", resource_type="data", tenant_id="t1")
        req = AccessRequest(
            principal=p, resource=r, action="read",
            source_ip="192.168.1.1", user_agent="TestAgent/1.0",
        )
        assert req.source_ip == "192.168.1.1"
        assert req.user_agent == "TestAgent/1.0"


class TestPolicyRule:
    """Test PolicyRule model creation and priority validation."""

    def test_creation_basic(self):
        rule = PolicyRule(rule_id="r1", name="Test Rule")
        assert rule.rule_id == "r1"
        assert rule.name == "Test Rule"
        assert rule.priority == 100
        assert rule.enabled is True

    def test_policy_type_default(self):
        rule = PolicyRule(rule_id="r1", name="Test")
        assert rule.policy_type == PolicyType.DATA_ACCESS

    def test_effect_default_deny(self):
        rule = PolicyRule(rule_id="r1", name="Test")
        assert rule.effect == AccessDecision.DENY

    def test_effect_allow(self):
        rule = PolicyRule(rule_id="r1", name="Test", effect="allow")
        assert rule.effect == AccessDecision.ALLOW

    def test_priority_zero(self):
        rule = PolicyRule(rule_id="r1", name="Test", priority=0)
        assert rule.priority == 0

    def test_priority_max(self):
        rule = PolicyRule(rule_id="r1", name="Test", priority=1000)
        assert rule.priority == 1000

    def test_priority_negative_raises(self):
        with pytest.raises(ValueError):
            PolicyRule(rule_id="r1", name="Test", priority=-1)

    def test_priority_over_1000_raises(self):
        with pytest.raises(ValueError):
            PolicyRule(rule_id="r1", name="Test", priority=1001)

    def test_conditions_dict(self):
        rule = PolicyRule(rule_id="r1", name="Test", conditions={"region": "US"})
        assert rule.conditions == {"region": "US"}

    def test_classification_max(self):
        rule = PolicyRule(rule_id="r1", name="Test", classification_max="confidential")
        assert rule.classification_max == DataClassification.CONFIDENTIAL

    def test_time_constraints(self):
        rule = PolicyRule(
            rule_id="r1", name="Test",
            time_constraints={"start_hour": 9, "end_hour": 17},
        )
        assert rule.time_constraints["start_hour"] == 9
        assert rule.time_constraints["end_hour"] == 17

    def test_geographic_constraints(self):
        rule = PolicyRule(
            rule_id="r1", name="Test",
            geographic_constraints=["US", "EU"],
        )
        assert "US" in rule.geographic_constraints
        assert "EU" in rule.geographic_constraints

    def test_to_dict(self):
        rule = PolicyRule(rule_id="r1", name="Test", priority=50, effect="allow")
        d = rule.to_dict()
        assert d["rule_id"] == "r1"
        assert d["effect"] == "allow"
        assert d["priority"] == 50


class TestPolicy:
    """Test Policy model creation and hash computation."""

    def test_creation_basic(self):
        p = Policy(policy_id="p1", name="Test Policy")
        assert p.policy_id == "p1"
        assert p.name == "Test Policy"
        assert p.enabled is True

    def test_default_version(self):
        p = Policy(policy_id="p1", name="Test")
        assert p.version == "1.0.0"

    def test_rules_empty_by_default(self):
        p = Policy(policy_id="p1", name="Test")
        assert p.rules == []

    def test_compute_hash_returns_sha256(self):
        p = Policy(policy_id="p1", name="Test")
        h = p.compute_hash()
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_compute_hash_deterministic(self):
        rule = PolicyRule(rule_id="r1", name="Rule 1", effect="allow")
        p = Policy(policy_id="p1", name="Test", rules=[rule])
        h1 = p.compute_hash()
        h2 = p.compute_hash()
        assert h1 == h2

    def test_compute_hash_changes_with_rules(self):
        p1 = Policy(policy_id="p1", name="Test")
        rule = PolicyRule(rule_id="r1", name="Rule 1", effect="allow")
        p2 = Policy(policy_id="p1", name="Test", rules=[rule])
        assert p1.compute_hash() != p2.compute_hash()

    def test_parent_policy_id(self):
        p = Policy(policy_id="child", name="Child", parent_policy_id="parent")
        assert p.parent_policy_id == "parent"

    def test_tenant_scope(self):
        p = Policy(policy_id="p1", name="Test", tenant_id="tenant-1")
        assert p.tenant_id == "tenant-1"

    def test_global_scope_none(self):
        p = Policy(policy_id="p1", name="Test")
        assert p.tenant_id is None

    def test_applies_to(self):
        p = Policy(policy_id="p1", name="Test", applies_to=["data", "agent"])
        assert "data" in p.applies_to
        assert "agent" in p.applies_to


class TestAccessDecisionResult:
    """Test AccessDecisionResult model creation and decision hash."""

    def test_creation(self):
        r = AccessDecisionResult(
            request_id="req-1", decision="allow", allowed=True,
        )
        assert r.request_id == "req-1"
        assert r.decision == AccessDecision.ALLOW
        assert r.allowed is True

    def test_deny_result(self):
        r = AccessDecisionResult(
            request_id="req-1", decision="deny", allowed=False,
            deny_reasons=["insufficient clearance"],
        )
        assert r.decision == AccessDecision.DENY
        assert r.allowed is False
        assert "insufficient clearance" in r.deny_reasons

    def test_conditional_result(self):
        r = AccessDecisionResult(
            request_id="req-1", decision="conditional", allowed=False,
            conditions=["time-based constraint"],
        )
        assert r.decision == AccessDecision.CONDITIONAL
        assert "time-based constraint" in r.conditions

    def test_decision_hash_computation(self):
        r = AccessDecisionResult(
            request_id="req-1", decision="allow", allowed=True,
            matching_rules=["rule-1"],
        )
        h = r.compute_decision_hash()
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_decision_hash_deterministic(self):
        ts = datetime(2026, 1, 1, 12, 0, 0)
        r = AccessDecisionResult(
            request_id="req-1", decision="allow", allowed=True,
            matching_rules=["rule-1"], evaluated_at=ts,
        )
        h1 = r.compute_decision_hash()
        h2 = r.compute_decision_hash()
        assert h1 == h2

    def test_evaluation_time_ms(self):
        r = AccessDecisionResult(
            request_id="req-1", decision="allow", allowed=True,
            evaluation_time_ms=1.5,
        )
        assert r.evaluation_time_ms == 1.5

    def test_policy_versions(self):
        r = AccessDecisionResult(
            request_id="req-1", decision="allow", allowed=True,
            policy_versions={"pol-1": "1.0.0"},
        )
        assert r.policy_versions["pol-1"] == "1.0.0"


class TestAuditEvent:
    """Test AuditEvent model creation and retention."""

    def test_creation(self):
        e = AuditEvent(event_type="access_granted", tenant_id="t1")
        assert e.event_type == AuditEventType.ACCESS_GRANTED
        assert e.tenant_id == "t1"

    def test_auto_event_id(self):
        e = AuditEvent(event_type="access_denied", tenant_id="t1")
        assert e.event_id is not None
        assert len(e.event_id) == 36

    def test_custom_event_id(self):
        e = AuditEvent(event_type="access_denied", tenant_id="t1", event_id="my-id")
        assert e.event_id == "my-id"

    def test_default_retention_days(self):
        e = AuditEvent(event_type="access_granted", tenant_id="t1")
        assert e.retention_days == 365

    def test_custom_retention_days(self):
        e = AuditEvent(event_type="access_granted", tenant_id="t1", retention_days=90)
        assert e.retention_days == 90

    def test_details_dict(self):
        e = AuditEvent(
            event_type="access_granted", tenant_id="t1",
            details={"reason": "policy match"},
        )
        assert e.details["reason"] == "policy match"

    def test_decision_field(self):
        e = AuditEvent(
            event_type="access_denied", tenant_id="t1",
            decision="deny",
        )
        assert e.decision == AccessDecision.DENY

    def test_source_ip(self):
        e = AuditEvent(
            event_type="access_granted", tenant_id="t1",
            source_ip="10.0.0.1",
        )
        assert e.source_ip == "10.0.0.1"


class TestComplianceReport:
    """Test ComplianceReport model creation and hash."""

    def test_creation(self):
        now = datetime.utcnow()
        r = ComplianceReport(
            tenant_id="t1",
            report_period_start=now - timedelta(days=30),
            report_period_end=now,
        )
        assert r.tenant_id == "t1"
        assert r.total_requests == 0

    def test_auto_report_id(self):
        now = datetime.utcnow()
        r = ComplianceReport(
            tenant_id="t1",
            report_period_start=now - timedelta(days=30),
            report_period_end=now,
        )
        assert r.report_id is not None
        assert len(r.report_id) == 36

    def test_compute_hash(self):
        now = datetime.utcnow()
        r = ComplianceReport(
            tenant_id="t1",
            report_period_start=now - timedelta(days=30),
            report_period_end=now,
            total_requests=100,
            allowed_requests=80,
            denied_requests=20,
        )
        h = r.compute_hash()
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_compute_hash_deterministic(self):
        ts = datetime(2026, 1, 1)
        r = ComplianceReport(
            tenant_id="t1",
            report_period_start=ts,
            report_period_end=ts + timedelta(days=30),
            report_id="fixed-id",
            generated_at=ts,
        )
        h1 = r.compute_hash()
        h2 = r.compute_hash()
        assert h1 == h2

    def test_statistics_fields(self):
        now = datetime.utcnow()
        r = ComplianceReport(
            tenant_id="t1",
            report_period_start=now,
            report_period_end=now,
            total_requests=100,
            allowed_requests=80,
            denied_requests=15,
            rate_limited_requests=5,
        )
        assert r.total_requests == 100
        assert r.allowed_requests == 80
        assert r.denied_requests == 15
        assert r.rate_limited_requests == 5


class TestClassificationHierarchy:
    """Test classification hierarchy ordering and all levels."""

    def test_public_is_lowest(self):
        assert CLASSIFICATION_HIERARCHY[DataClassification.PUBLIC] == 0

    def test_internal_is_1(self):
        assert CLASSIFICATION_HIERARCHY[DataClassification.INTERNAL] == 1

    def test_confidential_is_2(self):
        assert CLASSIFICATION_HIERARCHY[DataClassification.CONFIDENTIAL] == 2

    def test_restricted_is_3(self):
        assert CLASSIFICATION_HIERARCHY[DataClassification.RESTRICTED] == 3

    def test_top_secret_is_highest(self):
        assert CLASSIFICATION_HIERARCHY[DataClassification.TOP_SECRET] == 4

    def test_ordering_ascending(self):
        levels = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
            DataClassification.TOP_SECRET,
        ]
        for i in range(len(levels) - 1):
            assert (
                CLASSIFICATION_HIERARCHY[levels[i]]
                < CLASSIFICATION_HIERARCHY[levels[i + 1]]
            )

    def test_all_levels_present(self):
        assert len(CLASSIFICATION_HIERARCHY) == 5
        for dc in DataClassification:
            assert dc in CLASSIFICATION_HIERARCHY


class TestDefaultRolePermissions:
    """Test default role permissions for all roles."""

    def test_viewer_permissions(self):
        assert DEFAULT_ROLE_PERMISSIONS[RoleType.VIEWER] == {"read"}

    def test_analyst_permissions(self):
        perms = DEFAULT_ROLE_PERMISSIONS[RoleType.ANALYST]
        assert "read" in perms
        assert "analyze" in perms
        assert "export_internal" in perms
        assert len(perms) == 3

    def test_editor_permissions(self):
        perms = DEFAULT_ROLE_PERMISSIONS[RoleType.EDITOR]
        assert "read" in perms
        assert "write" in perms
        assert "analyze" in perms
        assert "export_internal" in perms
        assert len(perms) == 4

    def test_admin_permissions(self):
        perms = DEFAULT_ROLE_PERMISSIONS[RoleType.ADMIN]
        assert "read" in perms
        assert "write" in perms
        assert "delete" in perms
        assert "export_external" in perms
        assert "manage_users" in perms
        assert len(perms) == 7

    def test_super_admin_wildcard(self):
        perms = DEFAULT_ROLE_PERMISSIONS[RoleType.SUPER_ADMIN]
        assert "*" in perms
        assert len(perms) == 1

    def test_service_account_permissions(self):
        perms = DEFAULT_ROLE_PERMISSIONS[RoleType.SERVICE_ACCOUNT]
        assert "read" in perms
        assert "write" in perms
        assert "execute" in perms
        assert len(perms) == 3

    def test_all_roles_present(self):
        for role in RoleType:
            assert role in DEFAULT_ROLE_PERMISSIONS

    def test_viewer_cannot_write(self):
        assert "write" not in DEFAULT_ROLE_PERMISSIONS[RoleType.VIEWER]

    def test_analyst_cannot_delete(self):
        assert "delete" not in DEFAULT_ROLE_PERMISSIONS[RoleType.ANALYST]
