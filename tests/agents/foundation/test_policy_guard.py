# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-006: Access & Policy Guard Agent

Tests cover:
    - Access control decisions (RBAC/ABAC)
    - Data classification
    - Policy enforcement
    - Tenant isolation
    - Rate limiting
    - Audit logging
    - Policy inheritance and overrides
    - OPA Rego policy support
    - Policy simulation mode
    - Compliance report generation
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time
import sys
import os

# Add the project root to path to ensure direct imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from greenlang.agents.base import AgentConfig, AgentResult

# Import directly from policy_guard module to avoid broken imports in __init__.py
# This bypasses the unit_normalizer import issue
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "policy_guard_direct",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                 "greenlang", "agents", "foundation", "policy_guard.py")
)
_policy_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy_guard)

# Now we can import from the loaded module
from greenlang.agents.foundation.policy_guard import (
    # Main agent
    PolicyGuardAgent,

    # Enums
    AccessDecision,
    PolicyType,
    DataClassification,
    RoleType,
    AuditEventType,

    # Models
    Principal,
    Resource,
    AccessRequest,
    PolicyRule,
    Policy,
    AccessDecisionResult,
    AuditEvent,
    RateLimitConfig,
    PolicyGuardConfig,
    ComplianceReport,
    PolicySimulationResult,

    # Components
    PolicyEngine,
    RateLimiter,

    # Constants
    CLASSIFICATION_HIERARCHY,
    DEFAULT_ROLE_PERMISSIONS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def policy_guard() -> PolicyGuardAgent:
    """Create a PolicyGuardAgent instance for testing."""
    config = AgentConfig(
        name="TestPolicyGuard",
        description="Test Policy Guard Agent",
        version="1.0.0",
        parameters={
            "guard_config": {
                "simulation_mode": False,
                "strict_mode": True,
                "rate_limiting_enabled": True,
                "audit_enabled": True,
                "audit_all_decisions": True,
                "strict_tenant_isolation": True,
            }
        }
    )
    return PolicyGuardAgent(config)


@pytest.fixture
def sample_principal() -> Principal:
    """Create a sample principal for testing."""
    return Principal(
        principal_id="user-001",
        principal_type="user",
        tenant_id="tenant-001",
        roles=["analyst"],
        attributes={"department": "sustainability", "region": "EU"},
        clearance_level=DataClassification.CONFIDENTIAL,
        groups=["emissions-team"],
        authenticated=True,
        session_id="session-123"
    )


@pytest.fixture
def sample_resource() -> Resource:
    """Create a sample resource for testing."""
    return Resource(
        resource_id="emission-report-2024",
        resource_type="report",
        tenant_id="tenant-001",
        classification=DataClassification.INTERNAL,
        owner_id="user-admin",
        attributes={"year": 2024, "scope": "scope1"},
        geographic_location="EU",
        created_at=datetime.now()
    )


@pytest.fixture
def sample_request(sample_principal: Principal, sample_resource: Resource) -> AccessRequest:
    """Create a sample access request for testing."""
    return AccessRequest(
        principal=sample_principal,
        resource=sample_resource,
        action="read",
        context={"source": "web_ui"},
        source_ip="192.168.1.100"
    )


@pytest.fixture
def admin_principal() -> Principal:
    """Create an admin principal for testing."""
    return Principal(
        principal_id="admin-001",
        principal_type="user",
        tenant_id="tenant-001",
        roles=["admin"],
        clearance_level=DataClassification.TOP_SECRET,
        authenticated=True
    )


@pytest.fixture
def restricted_resource() -> Resource:
    """Create a restricted resource for testing."""
    return Resource(
        resource_id="confidential-data-001",
        resource_type="data",
        tenant_id="tenant-001",
        classification=DataClassification.RESTRICTED,
        owner_id="admin-001"
    )


@pytest.fixture
def sample_policy() -> Policy:
    """Create a sample policy for testing."""
    return Policy(
        policy_id="test-policy-001",
        name="Test Access Policy",
        description="Allow analysts to read reports",
        rules=[
            PolicyRule(
                rule_id="rule-001",
                name="Analysts Read Reports",
                description="Allow analysts to read reports",
                policy_type=PolicyType.DATA_ACCESS,
                priority=10,
                effect=AccessDecision.ALLOW,
                actions=["read"],
                principals=["role:analyst"],
                resources=["type:report"],
                classification_max=DataClassification.CONFIDENTIAL
            )
        ],
        tenant_id="tenant-001"
    )


# =============================================================================
# PRINCIPAL AND RESOURCE MODEL TESTS
# =============================================================================


class TestPrincipalModel:
    """Tests for the Principal model."""

    def test_principal_creation(self):
        """Test creating a principal with all fields."""
        principal = Principal(
            principal_id="user-123",
            tenant_id="tenant-001",
            roles=["analyst", "viewer"],
            clearance_level=DataClassification.CONFIDENTIAL
        )

        assert principal.principal_id == "user-123"
        assert principal.tenant_id == "tenant-001"
        assert "analyst" in principal.roles
        assert principal.clearance_level == DataClassification.CONFIDENTIAL

    def test_principal_default_values(self):
        """Test principal default values."""
        principal = Principal(
            principal_id="user-minimal",
            tenant_id="tenant-001"
        )

        assert principal.principal_type == "user"
        assert principal.roles == []
        assert principal.authenticated is True
        assert principal.clearance_level == DataClassification.INTERNAL

    def test_principal_clearance_string_conversion(self):
        """Test that string clearance levels are converted to enum."""
        principal = Principal(
            principal_id="user-str",
            tenant_id="tenant-001",
            clearance_level="confidential"
        )

        assert principal.clearance_level == DataClassification.CONFIDENTIAL


class TestResourceModel:
    """Tests for the Resource model."""

    def test_resource_creation(self):
        """Test creating a resource with all fields."""
        resource = Resource(
            resource_id="report-001",
            resource_type="report",
            tenant_id="tenant-001",
            classification=DataClassification.CONFIDENTIAL,
            geographic_location="EU"
        )

        assert resource.resource_id == "report-001"
        assert resource.resource_type == "report"
        assert resource.classification == DataClassification.CONFIDENTIAL
        assert resource.geographic_location == "EU"

    def test_resource_default_values(self):
        """Test resource default values."""
        resource = Resource(
            resource_id="minimal-resource",
            resource_type="data",
            tenant_id="tenant-001"
        )

        assert resource.classification == DataClassification.INTERNAL
        assert resource.owner_id is None
        assert resource.attributes == {}


# =============================================================================
# ACCESS REQUEST TESTS
# =============================================================================


class TestAccessRequest:
    """Tests for the AccessRequest model."""

    def test_access_request_creation(self, sample_principal, sample_resource):
        """Test creating an access request."""
        request = AccessRequest(
            principal=sample_principal,
            resource=sample_resource,
            action="read"
        )

        assert request.principal == sample_principal
        assert request.resource == sample_resource
        assert request.action == "read"
        assert request.request_id is not None

    def test_access_request_with_context(self, sample_principal, sample_resource):
        """Test access request with additional context."""
        request = AccessRequest(
            principal=sample_principal,
            resource=sample_resource,
            action="write",
            context={"reason": "update_report", "approved_by": "manager-001"},
            source_ip="10.0.0.1"
        )

        assert request.context["reason"] == "update_report"
        assert request.source_ip == "10.0.0.1"


# =============================================================================
# POLICY GUARD AGENT TESTS
# =============================================================================


class TestPolicyGuardAgent:
    """Tests for the PolicyGuardAgent."""

    def test_agent_initialization(self, policy_guard):
        """Test that the agent initializes correctly."""
        assert policy_guard.AGENT_ID == "GL-FOUND-X-006"
        assert policy_guard.AGENT_NAME == "Access & Policy Guard"
        assert policy_guard._guard_config.strict_mode is True

    def test_check_access_allows_authenticated_same_tenant(
        self,
        policy_guard,
        sample_request
    ):
        """Test that authenticated users can access resources in their tenant."""
        # Add a permissive policy
        policy = Policy(
            policy_id="allow-all-tenant",
            name="Allow Tenant Access",
            rules=[
                PolicyRule(
                    rule_id="allow-read",
                    name="Allow Read",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=10,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["*"],
                    resources=["*"]
                )
            ],
            tenant_id="tenant-001"
        )
        policy_guard.add_policy(policy)

        result = policy_guard.check_access(sample_request)

        assert result.allowed is True
        assert result.decision == AccessDecision.ALLOW

    def test_check_access_denies_unauthenticated(self, policy_guard, sample_request):
        """Test that unauthenticated users are denied."""
        sample_request.principal.authenticated = False

        result = policy_guard.check_access(sample_request)

        assert result.allowed is False
        assert "not authenticated" in result.deny_reasons[0].lower()

    def test_check_access_denies_cross_tenant(self, policy_guard, sample_request):
        """Test that cross-tenant access is denied."""
        sample_request.resource.tenant_id = "different-tenant"

        result = policy_guard.check_access(sample_request)

        assert result.allowed is False
        assert "tenant boundary" in result.deny_reasons[0].lower()

    def test_check_access_denies_insufficient_clearance(
        self,
        policy_guard,
        sample_principal,
        restricted_resource
    ):
        """Test that access is denied when clearance is insufficient."""
        # Set principal clearance to INTERNAL (below RESTRICTED)
        sample_principal.clearance_level = DataClassification.INTERNAL

        request = AccessRequest(
            principal=sample_principal,
            resource=restricted_resource,
            action="read"
        )

        result = policy_guard.check_access(request)

        assert result.allowed is False
        assert "clearance" in result.deny_reasons[0].lower()

    def test_add_and_remove_policy(self, policy_guard, sample_policy):
        """Test adding and removing policies."""
        # Add policy
        policy_hash = policy_guard.add_policy(sample_policy)

        assert policy_hash is not None
        assert len(policy_hash) == 64  # SHA-256 hex

        # Remove policy
        removed = policy_guard.remove_policy(sample_policy.policy_id)
        assert removed is True

        # Try removing again
        removed_again = policy_guard.remove_policy(sample_policy.policy_id)
        assert removed_again is False

    def test_get_metrics(self, policy_guard, sample_request):
        """Test getting metrics from the guard."""
        # Make some requests
        policy = Policy(
            policy_id="metrics-test",
            name="Metrics Test Policy",
            rules=[
                PolicyRule(
                    rule_id="allow-all",
                    name="Allow All",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        policy_guard.check_access(sample_request)

        metrics = policy_guard.get_metrics()

        assert "total_requests" in metrics
        assert "allowed_requests" in metrics
        assert "denied_requests" in metrics
        assert "policies_loaded" in metrics
        assert metrics["total_requests"] >= 1


class TestTenantIsolation:
    """Tests for tenant isolation enforcement."""

    def test_strict_tenant_isolation(self, policy_guard, sample_principal, sample_resource):
        """Test strict tenant isolation is enforced."""
        # Create request with mismatched tenants
        sample_resource.tenant_id = "tenant-002"  # Different from principal

        request = AccessRequest(
            principal=sample_principal,
            resource=sample_resource,
            action="read"
        )

        result = policy_guard.check_access(request)

        assert result.allowed is False
        assert "tenant boundary violation" in result.deny_reasons[0].lower()

    def test_same_tenant_allowed(self, policy_guard, sample_principal, sample_resource):
        """Test that same-tenant access is allowed with proper policy."""
        # Ensure same tenant
        sample_resource.tenant_id = sample_principal.tenant_id

        # Add permissive policy
        policy = Policy(
            policy_id="allow-same-tenant",
            name="Allow Same Tenant",
            rules=[
                PolicyRule(
                    rule_id="allow-read",
                    name="Allow Read",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        request = AccessRequest(
            principal=sample_principal,
            resource=sample_resource,
            action="read"
        )

        result = policy_guard.check_access(request)

        assert result.allowed is True


class TestDataClassification:
    """Tests for data classification and clearance checks."""

    def test_classification_hierarchy(self):
        """Test that classification hierarchy is correct."""
        assert CLASSIFICATION_HIERARCHY[DataClassification.PUBLIC] == 0
        assert CLASSIFICATION_HIERARCHY[DataClassification.INTERNAL] == 1
        assert CLASSIFICATION_HIERARCHY[DataClassification.CONFIDENTIAL] == 2
        assert CLASSIFICATION_HIERARCHY[DataClassification.RESTRICTED] == 3
        assert CLASSIFICATION_HIERARCHY[DataClassification.TOP_SECRET] == 4

    def test_classify_data_pii_detection(self, policy_guard):
        """Test that PII data is classified as restricted."""
        resource = Resource(
            resource_id="pii-data",
            resource_type="data",
            tenant_id="tenant-001",
            attributes={"ssn": "123-45-6789"}
        )

        classification = policy_guard.classify_data(resource)

        assert classification == DataClassification.RESTRICTED

    def test_classify_data_financial(self, policy_guard):
        """Test that financial data is classified appropriately."""
        resource = Resource(
            resource_id="financial-report",
            resource_type="financial_report",
            tenant_id="tenant-001",
            classification=DataClassification.INTERNAL
        )

        classification = policy_guard.classify_data(resource)

        assert classification in [DataClassification.CONFIDENTIAL, DataClassification.INTERNAL]

    def test_clearance_sufficient(self, policy_guard, admin_principal, restricted_resource):
        """Test access with sufficient clearance."""
        # Admin has TOP_SECRET clearance, resource is RESTRICTED
        request = AccessRequest(
            principal=admin_principal,
            resource=restricted_resource,
            action="read"
        )

        # Add permissive policy
        policy = Policy(
            policy_id="admin-access",
            name="Admin Access",
            rules=[
                PolicyRule(
                    rule_id="admin-all",
                    name="Admin All Access",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["role:admin"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        result = policy_guard.check_access(request)

        assert result.allowed is True


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limiter_creation(self):
        """Test rate limiter creation."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        )
        limiter = RateLimiter(config)

        assert limiter.config.requests_per_minute == 10

    def test_rate_limit_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        for i in range(10):
            allowed, reason = limiter.check_rate_limit("tenant-001", "user-001")
            assert allowed is True
            assert reason is None

    def test_rate_limit_blocks_excess(self):
        """Test that requests exceeding limit are blocked."""
        config = RateLimitConfig(requests_per_minute=5)
        limiter = RateLimiter(config)

        # Use up the limit
        for i in range(5):
            limiter.check_rate_limit("tenant-001", "user-001")

        # Next request should be blocked
        allowed, reason = limiter.check_rate_limit("tenant-001", "user-001")

        assert allowed is False
        assert "rate limit exceeded" in reason.lower()

    def test_rate_limit_role_override(self):
        """Test that role overrides apply higher limits."""
        config = RateLimitConfig(
            requests_per_minute=5,
            role_overrides={"admin": {"requests_per_minute": 100}}
        )
        limiter = RateLimiter(config)

        # Admin should have higher limit
        for i in range(50):
            allowed, _ = limiter.check_rate_limit("tenant-001", "admin-user", "admin")
            assert allowed is True

    def test_get_remaining_quota(self):
        """Test getting remaining quota."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100
        )
        limiter = RateLimiter(config)

        # Make some requests
        for i in range(3):
            limiter.check_rate_limit("tenant-001", "user-001")

        quota = limiter.get_remaining_quota("tenant-001", "user-001")

        assert quota["remaining_per_minute"] == 7
        assert quota["remaining_per_hour"] == 97


class TestPolicyEngine:
    """Tests for the policy engine."""

    def test_policy_engine_creation(self):
        """Test policy engine creation."""
        config = PolicyGuardConfig()
        engine = PolicyEngine(config)

        assert engine.config == config

    def test_add_and_get_policy(self):
        """Test adding and retrieving policies."""
        config = PolicyGuardConfig()
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="test-policy",
            name="Test Policy",
            rules=[]
        )

        policy_hash = engine.add_policy(policy)
        retrieved = engine.get_policy("test-policy")

        assert retrieved is not None
        assert retrieved.policy_id == "test-policy"
        assert len(policy_hash) == 64

    def test_policy_evaluation_allow(self, sample_request):
        """Test policy evaluation with allow rule."""
        config = PolicyGuardConfig(strict_mode=False)
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="allow-policy",
            name="Allow Policy",
            rules=[
                PolicyRule(
                    rule_id="allow-read",
                    name="Allow Read",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        engine.add_policy(policy)

        result = engine.evaluate(sample_request)

        assert result.decision == AccessDecision.ALLOW
        assert "allow-read" in result.matching_rules

    def test_policy_evaluation_deny(self, sample_request):
        """Test policy evaluation with deny rule."""
        config = PolicyGuardConfig(strict_mode=True)
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="deny-policy",
            name="Deny Policy",
            rules=[
                PolicyRule(
                    rule_id="deny-all",
                    name="Deny All",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.DENY,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        engine.add_policy(policy)

        result = engine.evaluate(sample_request)

        assert result.decision == AccessDecision.DENY
        assert len(result.deny_reasons) > 0

    def test_policy_priority_ordering(self, sample_request):
        """Test that policies are evaluated in priority order."""
        config = PolicyGuardConfig(strict_mode=True)
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="priority-policy",
            name="Priority Policy",
            rules=[
                PolicyRule(
                    rule_id="low-priority-deny",
                    name="Low Priority Deny",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=100,  # Lower priority
                    effect=AccessDecision.DENY,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                ),
                PolicyRule(
                    rule_id="high-priority-allow",
                    name="High Priority Allow",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=1,  # Higher priority (lower number)
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        engine.add_policy(policy)

        result = engine.evaluate(sample_request)

        # The allow rule should be evaluated first and match
        assert result.decision == AccessDecision.ALLOW

    def test_get_effective_rules_tenant_filter(self):
        """Test getting effective rules with tenant filter."""
        config = PolicyGuardConfig()
        engine = PolicyEngine(config)

        policy1 = Policy(
            policy_id="tenant-001-policy",
            name="Tenant 001 Policy",
            tenant_id="tenant-001",
            rules=[
                PolicyRule(
                    rule_id="t1-rule",
                    name="Tenant 1 Rule",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"]
                )
            ]
        )
        policy2 = Policy(
            policy_id="tenant-002-policy",
            name="Tenant 002 Policy",
            tenant_id="tenant-002",
            rules=[
                PolicyRule(
                    rule_id="t2-rule",
                    name="Tenant 2 Rule",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"]
                )
            ]
        )

        engine.add_policy(policy1)
        engine.add_policy(policy2)

        rules = engine.get_effective_rules(tenant_id="tenant-001")

        rule_ids = [r.rule_id for r in rules]
        assert "t1-rule" in rule_ids
        assert "t2-rule" not in rule_ids


class TestPolicyRules:
    """Tests for policy rule matching."""

    def test_rule_action_matching(self, sample_request):
        """Test that rules match on action."""
        config = PolicyGuardConfig()
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="action-policy",
            name="Action Policy",
            rules=[
                PolicyRule(
                    rule_id="write-only",
                    name="Write Only",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["write"],  # Only write, not read
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        engine.add_policy(policy)

        # Request is for "read", rule is for "write"
        result = engine.evaluate(sample_request)

        # Should not match (strict mode defaults to deny)
        assert result.decision == AccessDecision.DENY

    def test_rule_role_matching(self, sample_request):
        """Test that rules match on role."""
        config = PolicyGuardConfig()
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="role-policy",
            name="Role Policy",
            rules=[
                PolicyRule(
                    rule_id="analyst-only",
                    name="Analyst Only",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["role:analyst"],
                    resources=["*"]
                )
            ]
        )
        engine.add_policy(policy)

        # Principal has "analyst" role
        result = engine.evaluate(sample_request)

        assert result.decision == AccessDecision.ALLOW

    def test_rule_resource_type_matching(self, sample_request):
        """Test that rules match on resource type."""
        config = PolicyGuardConfig()
        engine = PolicyEngine(config)

        policy = Policy(
            policy_id="resource-policy",
            name="Resource Policy",
            rules=[
                PolicyRule(
                    rule_id="reports-only",
                    name="Reports Only",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["*"],
                    resources=["type:report"]
                )
            ]
        )
        engine.add_policy(policy)

        # Resource type is "report"
        result = engine.evaluate(sample_request)

        assert result.decision == AccessDecision.ALLOW


class TestAuditLogging:
    """Tests for audit logging functionality."""

    def test_audit_events_created(self, policy_guard, sample_request):
        """Test that audit events are created for access decisions."""
        # Add a policy to allow access
        policy = Policy(
            policy_id="audit-test-policy",
            name="Audit Test Policy",
            rules=[
                PolicyRule(
                    rule_id="allow-all",
                    name="Allow All",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        # Make a request
        policy_guard.check_access(sample_request)

        # Get audit events
        events = policy_guard.get_audit_events(limit=10)

        assert len(events) > 0
        assert any(e.event_type == AuditEventType.ACCESS_GRANTED for e in events)

    def test_audit_events_for_denial(self, policy_guard, sample_request):
        """Test that audit events are created for denials."""
        # Make request that will be denied (unauthenticated)
        sample_request.principal.authenticated = False
        policy_guard.check_access(sample_request)

        events = policy_guard.get_audit_events(
            event_type=AuditEventType.ACCESS_DENIED,
            limit=10
        )

        assert len(events) > 0

    def test_audit_events_tenant_filter(self, policy_guard, sample_request):
        """Test filtering audit events by tenant."""
        # Add policy and make request
        policy = Policy(
            policy_id="tenant-audit",
            name="Tenant Audit",
            rules=[
                PolicyRule(
                    rule_id="allow",
                    name="Allow",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)
        policy_guard.check_access(sample_request)

        # Filter by tenant
        events = policy_guard.get_audit_events(
            tenant_id="tenant-001",
            limit=10
        )

        assert all(e.tenant_id == "tenant-001" for e in events)


class TestPolicySimulation:
    """Tests for policy simulation mode."""

    def test_simulation_mode(self, sample_request):
        """Test simulation mode allows all but logs decisions."""
        config = AgentConfig(
            name="SimulationGuard",
            description="Simulation Mode Guard",
            version="1.0.0",
            parameters={
                "guard_config": {
                    "simulation_mode": True,
                    "strict_mode": True
                }
            }
        )
        guard = PolicyGuardAgent(config)

        # In simulation mode, should allow but mark as simulated
        result = guard.check_access(sample_request)

        assert result.allowed is True
        assert any("[SIMULATED]" in r for r in result.deny_reasons) or len(result.deny_reasons) == 0

    def test_simulate_policies(self, policy_guard, sample_request):
        """Test policy simulation with test requests."""
        # Create test requests
        test_requests = [sample_request]

        result = policy_guard.simulate_policies(test_requests)

        assert isinstance(result, PolicySimulationResult)
        assert result.test_requests == 1
        assert "allowed" in result.summary
        assert "denied" in result.summary


class TestComplianceReporting:
    """Tests for compliance report generation."""

    def test_generate_compliance_report(self, policy_guard, sample_request):
        """Test generating a compliance report."""
        # Add policy and make some requests
        policy = Policy(
            policy_id="compliance-test",
            name="Compliance Test",
            rules=[
                PolicyRule(
                    rule_id="allow",
                    name="Allow",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        # Make some requests
        for _ in range(5):
            policy_guard.check_access(sample_request)

        # Generate report
        now = datetime.now()
        report = policy_guard.generate_compliance_report(
            tenant_id="tenant-001",
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1)
        )

        assert isinstance(report, ComplianceReport)
        assert report.tenant_id == "tenant-001"
        assert report.total_requests >= 0
        assert report.provenance_hash is not None

    def test_report_provenance_hash(self, policy_guard):
        """Test that report has valid provenance hash."""
        now = datetime.now()
        report = policy_guard.generate_compliance_report(
            tenant_id="tenant-001",
            start_date=now - timedelta(days=1),
            end_date=now
        )

        assert len(report.provenance_hash) == 64  # SHA-256 hex


class TestExportPolicies:
    """Tests for export policy enforcement."""

    def test_export_allowed_public_data(self, policy_guard, sample_request):
        """Test that public data can be exported."""
        sample_request.resource.classification = DataClassification.PUBLIC
        sample_request.action = "export"

        # Add export policy
        policy = Policy(
            policy_id="export-policy",
            name="Export Policy",
            rules=[
                PolicyRule(
                    rule_id="allow-export",
                    name="Allow Export",
                    policy_type=PolicyType.EXPORT,
                    effect=AccessDecision.ALLOW,
                    actions=["export"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        result = policy_guard.check_export_allowed(sample_request, "external-system")

        assert result.allowed is True

    def test_export_denied_restricted_data(self, policy_guard, sample_request):
        """Test that restricted data cannot be exported."""
        sample_request.resource.classification = DataClassification.RESTRICTED
        sample_request.action = "export"

        result = policy_guard.check_export_allowed(sample_request, "external-system")

        assert result.allowed is False


class TestRegoPolicy:
    """Tests for OPA Rego policy support."""

    def test_add_rego_policy(self, policy_guard):
        """Test adding a Rego policy."""
        rego_source = """
        package greenlang.test

        default allow = false

        allow {
            input.user.authenticated == true
        }
        """

        policy_hash = policy_guard.add_rego_policy("test-rego", rego_source)

        assert len(policy_hash) == 64  # SHA-256 hex


class TestPolicyInheritance:
    """Tests for policy inheritance."""

    def test_policy_with_parent(self, policy_guard):
        """Test creating a policy with a parent."""
        parent_policy = Policy(
            policy_id="parent-policy",
            name="Parent Policy",
            rules=[
                PolicyRule(
                    rule_id="parent-rule",
                    name="Parent Rule",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )

        child_policy = Policy(
            policy_id="child-policy",
            name="Child Policy",
            parent_policy_id="parent-policy",
            rules=[
                PolicyRule(
                    rule_id="child-rule",
                    name="Child Rule",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=5,  # Higher priority than parent
                    effect=AccessDecision.DENY,
                    actions=["delete"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )

        policy_guard.add_policy(parent_policy)
        policy_guard.add_policy(child_policy)

        # Both policies should be loaded
        metrics = policy_guard.get_metrics()
        assert metrics["policies_loaded"] >= 2


class TestCacheManagement:
    """Tests for decision cache management."""

    def test_cache_hit(self, policy_guard, sample_request):
        """Test that cached decisions are returned."""
        # Add policy
        policy = Policy(
            policy_id="cache-test",
            name="Cache Test",
            rules=[
                PolicyRule(
                    rule_id="allow",
                    name="Allow",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        # First request
        result1 = policy_guard.check_access(sample_request)

        # Second identical request (should hit cache)
        result2 = policy_guard.check_access(sample_request)

        assert result1.decision == result2.decision
        assert result1.request_id == result2.request_id

    def test_clear_cache(self, policy_guard, sample_request):
        """Test clearing the decision cache."""
        # Add policy and make request
        policy = Policy(
            policy_id="cache-clear-test",
            name="Cache Clear Test",
            rules=[
                PolicyRule(
                    rule_id="allow",
                    name="Allow",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)
        policy_guard.check_access(sample_request)

        # Verify cache has entries
        assert policy_guard.get_metrics()["cache_size"] > 0

        # Clear cache
        policy_guard.clear_cache()

        assert policy_guard.get_metrics()["cache_size"] == 0


class TestAgentExecute:
    """Tests for the agent execute method."""

    def test_execute_check_access(self, policy_guard, sample_principal, sample_resource):
        """Test execute with check_access operation."""
        result = policy_guard.run({
            "operation": "check_access",
            "request": {
                "principal": sample_principal.model_dump(),
                "resource": sample_resource.model_dump(),
                "action": "read"
            }
        })

        assert result.success is True
        assert "decision" in result.data

    def test_execute_add_policy(self, policy_guard, sample_policy):
        """Test execute with add_policy operation."""
        result = policy_guard.run({
            "operation": "add_policy",
            "policy": sample_policy.model_dump()
        })

        assert result.success is True
        assert "policy_hash" in result.data
        assert "policy_id" in result.data

    def test_execute_get_metrics(self, policy_guard):
        """Test execute with get_metrics operation."""
        result = policy_guard.run({
            "operation": "get_metrics"
        })

        assert result.success is True
        assert "total_requests" in result.data

    def test_execute_unknown_operation(self, policy_guard):
        """Test execute with unknown operation."""
        result = policy_guard.run({
            "operation": "unknown_operation"
        })

        assert result.success is False
        assert "unknown operation" in result.error.lower()


class TestDefaultRolePermissions:
    """Tests for default role permissions."""

    def test_viewer_permissions(self):
        """Test viewer role has read permission."""
        assert "read" in DEFAULT_ROLE_PERMISSIONS[RoleType.VIEWER]
        assert "write" not in DEFAULT_ROLE_PERMISSIONS[RoleType.VIEWER]

    def test_admin_permissions(self):
        """Test admin role has full permissions."""
        admin_perms = DEFAULT_ROLE_PERMISSIONS[RoleType.ADMIN]
        assert "read" in admin_perms
        assert "write" in admin_perms
        assert "delete" in admin_perms
        assert "manage_users" in admin_perms

    def test_super_admin_wildcard(self):
        """Test super admin has wildcard permission."""
        assert "*" in DEFAULT_ROLE_PERMISSIONS[RoleType.SUPER_ADMIN]


class TestPolicyProvenance:
    """Tests for policy provenance tracking."""

    def test_policy_hash_computation(self, sample_policy):
        """Test that policy hash is computed correctly."""
        hash1 = sample_policy.compute_hash()

        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

        # Same policy should produce same hash
        hash2 = sample_policy.compute_hash()
        assert hash1 == hash2

    def test_different_policies_different_hashes(self):
        """Test that different policies have different hashes."""
        policy1 = Policy(
            policy_id="policy-1",
            name="Policy 1",
            rules=[]
        )
        policy2 = Policy(
            policy_id="policy-2",
            name="Policy 2",
            rules=[]
        )

        assert policy1.compute_hash() != policy2.compute_hash()

    def test_decision_provenance_hash(self, policy_guard, sample_request):
        """Test that access decisions have provenance hashes."""
        policy = Policy(
            policy_id="provenance-test",
            name="Provenance Test",
            rules=[
                PolicyRule(
                    rule_id="allow",
                    name="Allow",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["*"],
                    resources=["*"]
                )
            ]
        )
        policy_guard.add_policy(policy)

        result = policy_guard.check_access(sample_request)

        assert result.decision_hash is not None
        assert len(result.decision_hash) == 64


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPolicyGuardIntegration:
    """Integration tests for the Policy Guard Agent."""

    def test_full_workflow(self):
        """Test a complete access control workflow."""
        # 1. Create guard
        guard = PolicyGuardAgent()

        # 2. Add policies
        data_policy = Policy(
            policy_id="data-access-policy",
            name="Data Access Policy",
            description="Control access to emission data",
            rules=[
                PolicyRule(
                    rule_id="analysts-read-emissions",
                    name="Analysts Read Emissions",
                    policy_type=PolicyType.DATA_ACCESS,
                    effect=AccessDecision.ALLOW,
                    actions=["read"],
                    principals=["role:analyst"],
                    resources=["type:emission_data"],
                    classification_max=DataClassification.CONFIDENTIAL
                ),
                PolicyRule(
                    rule_id="admins-all-access",
                    name="Admins All Access",
                    policy_type=PolicyType.DATA_ACCESS,
                    priority=1,
                    effect=AccessDecision.ALLOW,
                    actions=["*"],
                    principals=["role:admin"],
                    resources=["*"]
                )
            ],
            tenant_id="acme-corp"
        )
        guard.add_policy(data_policy)

        # 3. Create principals
        analyst = Principal(
            principal_id="analyst-jane",
            tenant_id="acme-corp",
            roles=["analyst"],
            clearance_level=DataClassification.CONFIDENTIAL
        )

        admin = Principal(
            principal_id="admin-bob",
            tenant_id="acme-corp",
            roles=["admin"],
            clearance_level=DataClassification.TOP_SECRET
        )

        # 4. Create resources
        emission_report = Resource(
            resource_id="q1-2024-emissions",
            resource_type="emission_data",
            tenant_id="acme-corp",
            classification=DataClassification.INTERNAL
        )

        confidential_report = Resource(
            resource_id="audit-findings",
            resource_type="audit_report",
            tenant_id="acme-corp",
            classification=DataClassification.RESTRICTED
        )

        # 5. Test access scenarios

        # Analyst can read emission data
        result = guard.check_access(AccessRequest(
            principal=analyst,
            resource=emission_report,
            action="read"
        ))
        assert result.allowed is True

        # Analyst cannot write emission data (no write rule)
        result = guard.check_access(AccessRequest(
            principal=analyst,
            resource=emission_report,
            action="write"
        ))
        assert result.allowed is False

        # Admin can access everything
        result = guard.check_access(AccessRequest(
            principal=admin,
            resource=confidential_report,
            action="read"
        ))
        assert result.allowed is True

        # 6. Check metrics
        metrics = guard.get_metrics()
        assert metrics["total_requests"] >= 3

        # 7. Generate compliance report
        now = datetime.now()
        report = guard.generate_compliance_report(
            tenant_id="acme-corp",
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1)
        )
        assert report.total_requests >= 0

    def test_multi_tenant_isolation(self):
        """Test that tenant isolation is strictly enforced."""
        guard = PolicyGuardAgent()

        # Create users from different tenants
        tenant1_user = Principal(
            principal_id="user-t1",
            tenant_id="tenant-1",
            roles=["analyst"],
            clearance_level=DataClassification.TOP_SECRET
        )

        tenant2_resource = Resource(
            resource_id="data-t2",
            resource_type="data",
            tenant_id="tenant-2",
            classification=DataClassification.PUBLIC  # Even public data
        )

        # Should be denied due to tenant mismatch
        result = guard.check_access(AccessRequest(
            principal=tenant1_user,
            resource=tenant2_resource,
            action="read"
        ))

        assert result.allowed is False
        assert "tenant" in result.deny_reasons[0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
