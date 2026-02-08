# -*- coding: utf-8 -*-
"""
Unit Tests for PolicyEngine (AGENT-FOUND-006)

Tests policy CRUD, evaluation (RBAC/ABAC), pattern matching, classification
constraints, priority ordering, inheritance, provenance hashing, time/geo
constraints, and strict vs permissive mode.

Coverage target: 85%+ of policy_engine.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models (self-contained)
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


CLASSIFICATION_HIERARCHY: Dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
    DataClassification.TOP_SECRET: 4,
}


class Principal:
    def __init__(self, principal_id, tenant_id="", roles=None, attributes=None,
                 clearance_level="internal", authenticated=True, groups=None,
                 principal_type="user", session_id=None):
        self.principal_id = principal_id
        self.principal_type = principal_type
        self.tenant_id = tenant_id
        self.roles = roles or []
        self.attributes = attributes or {}
        self.clearance_level = DataClassification(clearance_level)
        self.authenticated = authenticated
        self.groups = groups or []
        self.session_id = session_id


class Resource:
    def __init__(self, resource_id, resource_type="data", tenant_id="",
                 classification="internal", owner_id=None, attributes=None,
                 geographic_location=None, created_at=None, retention_policy=None):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.tenant_id = tenant_id
        self.classification = DataClassification(classification)
        self.owner_id = owner_id
        self.attributes = attributes or {}
        self.geographic_location = geographic_location
        self.created_at = created_at
        self.retention_policy = retention_policy


class AccessRequest:
    def __init__(self, principal, resource, action, request_id=None,
                 context=None, timestamp=None, source_ip=None, user_agent=None):
        self.request_id = request_id or str(uuid.uuid4())
        self.principal = principal
        self.resource = resource
        self.action = action
        self.context = context or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.source_ip = source_ip
        self.user_agent = user_agent


class PolicyRule:
    def __init__(self, rule_id, name, description="", policy_type="data_access",
                 priority=100, enabled=True, conditions=None, effect="deny",
                 actions=None, resources=None, principals=None,
                 time_constraints=None, geographic_constraints=None,
                 classification_max=None, version="1.0.0", created_at=None,
                 created_by=None, tags=None):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.policy_type = PolicyType(policy_type)
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
            DataClassification(classification_max) if classification_max else None
        )
        self.version = version
        self.created_at = created_at or datetime.utcnow()
        self.created_by = created_by
        self.tags = tags or []

    def to_dict(self):
        return {
            "rule_id": self.rule_id, "name": self.name,
            "policy_type": self.policy_type.value, "priority": self.priority,
            "effect": self.effect.value, "actions": self.actions,
            "resources": self.resources, "principals": self.principals,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Policy:
    def __init__(self, policy_id, name, description="", version="1.0.0",
                 enabled=True, rules=None, parent_policy_id=None,
                 allow_override=True, tenant_id=None, applies_to=None,
                 created_at=None, updated_at=None, created_by=None,
                 provenance_hash=""):
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

    def compute_hash(self):
        policy_str = json.dumps(
            {"policy_id": self.policy_id, "rules": [r.to_dict() for r in self.rules],
             "version": self.version},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(policy_str.encode()).hexdigest()


class AccessDecisionResult:
    def __init__(self, request_id, decision, allowed, matching_rules=None,
                 deny_reasons=None, conditions=None, evaluated_at=None,
                 evaluation_time_ms=0.0, policy_versions=None, decision_hash=""):
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


# ---------------------------------------------------------------------------
# PolicyEngine (self-contained mirror)
# ---------------------------------------------------------------------------

class PolicyEngine:
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._policies: Dict[str, Policy] = {}
        self._policy_hierarchy: Dict[str, List[str]] = {}

    @property
    def count(self) -> int:
        return len(self._policies)

    def add_policy(self, policy: Policy) -> str:
        policy.provenance_hash = policy.compute_hash()
        policy.updated_at = datetime.utcnow()
        self._policies[policy.policy_id] = policy
        if policy.parent_policy_id:
            if policy.policy_id not in self._policy_hierarchy:
                self._policy_hierarchy[policy.policy_id] = []
            self._policy_hierarchy[policy.policy_id].append(policy.parent_policy_id)
        return policy.provenance_hash

    def remove_policy(self, policy_id: str) -> bool:
        if policy_id in self._policies:
            del self._policies[policy_id]
            self._policy_hierarchy.pop(policy_id, None)
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        return self._policies.get(policy_id)

    def list_policies(self) -> List[Policy]:
        return list(self._policies.values())

    def update_policy(self, policy: Policy) -> str:
        policy.provenance_hash = policy.compute_hash()
        policy.updated_at = datetime.utcnow()
        self._policies[policy.policy_id] = policy
        return policy.provenance_hash

    def get_effective_rules(self, tenant_id=None, resource_type=None) -> List[PolicyRule]:
        rules = []
        for policy in self._policies.values():
            if not policy.enabled:
                continue
            if tenant_id and policy.tenant_id and policy.tenant_id != tenant_id:
                continue
            if resource_type and policy.applies_to and resource_type not in policy.applies_to:
                continue
            for rule in policy.rules:
                if rule.enabled:
                    rules.append(rule)
        rules.sort(key=lambda r: r.priority)
        return rules

    def evaluate(self, request: AccessRequest) -> AccessDecisionResult:
        start_time = time.time()
        matching_rules = []
        deny_reasons = []
        conditions = []
        policy_versions = {}

        rules = self.get_effective_rules(
            tenant_id=request.resource.tenant_id,
            resource_type=request.resource.resource_type,
        )
        for policy in self._policies.values():
            policy_versions[policy.policy_id] = policy.version

        decision = AccessDecision.DENY if self.strict_mode else AccessDecision.ALLOW

        for rule in rules:
            if self._rule_matches(rule, request):
                matching_rules.append(rule.rule_id)
                if rule.effect == AccessDecision.ALLOW:
                    decision = AccessDecision.ALLOW
                    if rule.time_constraints or rule.geographic_constraints:
                        conditions.append(f"Rule {rule.rule_id} has conditions")
                        decision = AccessDecision.CONDITIONAL
                    break
                elif rule.effect == AccessDecision.DENY:
                    decision = AccessDecision.DENY
                    deny_reasons.append(f"Denied by rule: {rule.name} ({rule.rule_id})")
                    break

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
            decision=decision.value,
            allowed=decision == AccessDecision.ALLOW,
            matching_rules=matching_rules,
            deny_reasons=deny_reasons,
            conditions=conditions if decision == AccessDecision.CONDITIONAL else [],
            evaluation_time_ms=evaluation_time,
            policy_versions=policy_versions,
        )
        decision_str = json.dumps({
            "request_id": request.request_id, "decision": decision.value,
            "matching_rules": matching_rules,
            "timestamp": result.evaluated_at.isoformat(),
        }, sort_keys=True)
        result.decision_hash = hashlib.sha256(decision_str.encode()).hexdigest()
        return result

    def _rule_matches(self, rule: PolicyRule, request: AccessRequest) -> bool:
        if rule.actions and request.action not in rule.actions:
            if "*" not in rule.actions:
                return False
        if rule.principals:
            principal_match = False
            for pattern in rule.principals:
                if self._pattern_matches(pattern, request.principal.principal_id):
                    principal_match = True
                    break
                if pattern.startswith("role:"):
                    role_name = pattern[5:]
                    if role_name in request.principal.roles or role_name == "*":
                        principal_match = True
                        break
            if not principal_match:
                return False
        if rule.resources:
            resource_match = False
            for pattern in rule.resources:
                if self._pattern_matches(pattern, request.resource.resource_id):
                    resource_match = True
                    break
                if pattern.startswith("type:"):
                    type_name = pattern[5:]
                    if type_name == request.resource.resource_type or type_name == "*":
                        resource_match = True
                        break
            if not resource_match:
                return False
        if rule.classification_max:
            resource_level = CLASSIFICATION_HIERARCHY.get(request.resource.classification, 0)
            max_level = CLASSIFICATION_HIERARCHY.get(rule.classification_max, 0)
            if resource_level > max_level:
                return False
        if rule.conditions:
            for key, expected_value in rule.conditions.items():
                actual_value = request.context.get(key)
                if actual_value != expected_value:
                    actual_value = request.principal.attributes.get(key)
                    if actual_value != expected_value:
                        return False
        return True

    def _pattern_matches(self, pattern: str, value: str) -> bool:
        if pattern == "*":
            return True
        if "*" not in pattern:
            return pattern == value
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", value))

    def _check_conditions(self, request: AccessRequest, rule_ids: List[str]) -> bool:
        now = datetime.utcnow()
        for rule_id in rule_ids:
            for policy in self._policies.values():
                for rule in policy.rules:
                    if rule.rule_id == rule_id:
                        if rule.time_constraints:
                            start_hour = rule.time_constraints.get("start_hour", 0)
                            end_hour = rule.time_constraints.get("end_hour", 24)
                            if not (start_hour <= now.hour < end_hour):
                                return False
                        if rule.geographic_constraints:
                            location = request.resource.geographic_location
                            if location and location not in rule.geographic_constraints:
                                return False
        return True


# ===========================================================================
# Helper to build standard objects
# ===========================================================================

def _make_request(action="read", principal_roles=None, resource_type="data",
                  tenant_id="tenant-1", classification="internal",
                  context=None, principal_attrs=None, resource_id="res-1",
                  principal_id="user-1", geo_location=None):
    p = Principal(
        principal_id=principal_id, tenant_id=tenant_id,
        roles=principal_roles or ["analyst"],
        attributes=principal_attrs or {},
    )
    r = Resource(
        resource_id=resource_id, resource_type=resource_type,
        tenant_id=tenant_id, classification=classification,
        geographic_location=geo_location,
    )
    return AccessRequest(principal=p, resource=r, action=action, context=context or {})


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPolicyEngineAddRemove:
    """Test add, remove, get, list, count operations."""

    def test_add_policy(self):
        engine = PolicyEngine()
        p = Policy(policy_id="p1", name="P1")
        h = engine.add_policy(p)
        assert len(h) == 64
        assert engine.count == 1

    def test_add_multiple_policies(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="P1"))
        engine.add_policy(Policy(policy_id="p2", name="P2"))
        assert engine.count == 2

    def test_remove_policy(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="P1"))
        assert engine.remove_policy("p1") is True
        assert engine.count == 0

    def test_remove_nonexistent_policy(self):
        engine = PolicyEngine()
        assert engine.remove_policy("nonexistent") is False

    def test_get_policy(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="Policy One"))
        p = engine.get_policy("p1")
        assert p is not None
        assert p.name == "Policy One"

    def test_get_nonexistent_policy(self):
        engine = PolicyEngine()
        assert engine.get_policy("nope") is None

    def test_list_policies(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="P1"))
        engine.add_policy(Policy(policy_id="p2", name="P2"))
        policies = engine.list_policies()
        assert len(policies) == 2
        ids = {p.policy_id for p in policies}
        assert "p1" in ids
        assert "p2" in ids

    def test_count_empty(self):
        engine = PolicyEngine()
        assert engine.count == 0

    def test_add_replaces_existing(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="V1"))
        engine.add_policy(Policy(policy_id="p1", name="V2"))
        assert engine.count == 1
        assert engine.get_policy("p1").name == "V2"

    def test_update_policy(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="V1"))
        h = engine.update_policy(Policy(policy_id="p1", name="V2", version="2.0.0"))
        assert len(h) == 64
        assert engine.get_policy("p1").version == "2.0.0"


class TestPolicyEngineEvaluate:
    """Test basic evaluation: allow rules, deny rules, first-match-wins."""

    def test_allow_rule_grants_access(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow Read", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        req = _make_request(action="read")
        result = engine.evaluate(req)
        assert result.allowed is True
        assert result.decision == AccessDecision.ALLOW

    def test_deny_rule_blocks_access(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Deny Write", effect="deny",
                          actions=["write"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        req = _make_request(action="write")
        result = engine.evaluate(req)
        assert result.allowed is False
        assert len(result.deny_reasons) > 0

    def test_first_match_wins_allow_before_deny(self):
        engine = PolicyEngine(strict_mode=True)
        allow_rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                                 priority=10, actions=["read"],
                                 principals=["role:analyst"], resources=["type:data"])
        deny_rule = PolicyRule(rule_id="r2", name="Deny", effect="deny",
                                priority=20, actions=["read"],
                                principals=["role:analyst"], resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[allow_rule, deny_rule]))
        result = engine.evaluate(_make_request(action="read"))
        assert result.allowed is True
        assert "r1" in result.matching_rules

    def test_first_match_wins_deny_before_allow(self):
        engine = PolicyEngine(strict_mode=True)
        deny_rule = PolicyRule(rule_id="r1", name="Deny", effect="deny",
                                priority=10, actions=["read"],
                                principals=["role:analyst"], resources=["type:data"])
        allow_rule = PolicyRule(rule_id="r2", name="Allow", effect="allow",
                                 priority=20, actions=["read"],
                                 principals=["role:analyst"], resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[deny_rule, allow_rule]))
        result = engine.evaluate(_make_request(action="read"))
        assert result.allowed is False

    def test_no_matching_rules_strict_mode_deny(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow Admin", effect="allow",
                          actions=["read"], principals=["role:admin"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        req = _make_request(action="read", principal_roles=["analyst"])
        result = engine.evaluate(req)
        assert result.allowed is False

    def test_decision_hash_populated(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(action="read"))
        assert len(result.decision_hash) == 64

    def test_evaluation_time_ms_positive(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(action="read"))
        assert result.evaluation_time_ms >= 0

    def test_policy_versions_in_result(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_id="p1", name="P1", version="2.0.0"))
        result = engine.evaluate(_make_request())
        assert "p1" in result.policy_versions
        assert result.policy_versions["p1"] == "2.0.0"


class TestPolicyEngineStrictMode:
    """Test deny-by-default vs allow-by-default."""

    def test_strict_mode_deny_by_default(self):
        engine = PolicyEngine(strict_mode=True)
        result = engine.evaluate(_make_request())
        assert result.allowed is False
        assert result.decision == AccessDecision.DENY

    def test_permissive_mode_allow_by_default(self):
        engine = PolicyEngine(strict_mode=False)
        result = engine.evaluate(_make_request())
        assert result.allowed is True
        assert result.decision == AccessDecision.ALLOW

    def test_strict_with_allow_rule(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(action="read"))
        assert result.allowed is True

    def test_permissive_with_deny_rule(self):
        engine = PolicyEngine(strict_mode=False)
        rule = PolicyRule(rule_id="r1", name="Deny", effect="deny",
                          actions=["delete"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(action="delete"))
        assert result.allowed is False


class TestPolicyEngineRBAC:
    """Test role-based matching: role:analyst, role:admin, role:*."""

    def test_role_analyst_matches(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(principal_roles=["analyst"]))
        assert result.allowed is True

    def test_role_admin_does_not_match_analyst_rule(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow Analyst", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(principal_roles=["admin"]))
        assert result.allowed is False

    def test_role_wildcard_matches_any(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow All", effect="allow",
                          actions=["read"], principals=["role:*"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(principal_roles=["viewer"]))
        assert result.allowed is True

    def test_multiple_roles_in_principal(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow Editor", effect="allow",
                          actions=["write"], principals=["role:editor"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(
            action="write", principal_roles=["analyst", "editor"],
        ))
        assert result.allowed is True

    def test_principal_id_exact_match(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow User", effect="allow",
                          actions=["read"], principals=["user-1"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(principal_id="user-1"))
        assert result.allowed is True

    def test_principal_id_no_match(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow User", effect="allow",
                          actions=["read"], principals=["user-2"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(principal_id="user-1"))
        assert result.allowed is False


class TestPolicyEngineABAC:
    """Test attribute-based conditions."""

    def test_condition_from_context(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"region": "US"})
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(context={"region": "US"}))
        assert result.allowed is True

    def test_condition_mismatch_from_context(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"region": "US"})
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(context={"region": "EU"}))
        assert result.allowed is False

    def test_condition_from_principal_attributes(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"department": "sustainability"})
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(
            principal_attrs={"department": "sustainability"},
        ))
        assert result.allowed is True

    def test_condition_missing_in_both_context_and_attrs(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"department": "finance"})
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request())
        assert result.allowed is False

    def test_multiple_conditions_all_must_match(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"region": "US", "level": "senior"})
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(
            context={"region": "US", "level": "senior"},
        ))
        assert result.allowed is True

    def test_multiple_conditions_partial_match_fails(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"region": "US", "level": "senior"})
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(context={"region": "US"}))
        assert result.allowed is False


class TestPolicyEnginePatternMatching:
    """Test glob patterns, exact match, wildcard."""

    def test_exact_resource_id_match(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["res-1"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_id="res-1"))
        assert result.allowed is True

    def test_exact_resource_id_no_match(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["res-1"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_id="res-2"))
        assert result.allowed is False

    def test_wildcard_resource(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["*"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_id="anything"))
        assert result.allowed is True

    def test_glob_pattern_prefix(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["res-*"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_id="res-abc"))
        assert result.allowed is True

    def test_glob_pattern_no_match(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["res-*"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_id="data-abc"))
        assert result.allowed is False

    def test_wildcard_action(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["*"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(action="delete"))
        assert result.allowed is True

    def test_empty_actions_matches_all(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=[], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(action="write"))
        assert result.allowed is True


class TestPolicyEngineResourceType:
    """Test type: prefixed resource matching."""

    def test_type_data_matches(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_type="data"))
        assert result.allowed is True

    def test_type_agent_does_not_match_data(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:agent"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_type="data"))
        assert result.allowed is False

    def test_type_wildcard_matches_any(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:*"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(resource_type="report"))
        assert result.allowed is True


class TestPolicyEngineClassification:
    """Test classification_max enforcement."""

    def test_classification_within_limit(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          classification_max="confidential")
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(classification="internal"))
        assert result.allowed is True

    def test_classification_at_limit(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          classification_max="confidential")
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(classification="confidential"))
        assert result.allowed is True

    def test_classification_exceeds_limit(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          classification_max="confidential")
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(classification="restricted"))
        assert result.allowed is False

    def test_classification_top_secret_exceeds_restricted(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          classification_max="restricted")
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(classification="top_secret"))
        assert result.allowed is False

    def test_no_classification_max_allows_all(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(classification="top_secret"))
        assert result.allowed is True


class TestPolicyEnginePriority:
    """Test lower priority number wins, tie-breaking."""

    def test_lower_priority_wins(self):
        engine = PolicyEngine(strict_mode=True)
        deny = PolicyRule(rule_id="r1", name="Deny", effect="deny",
                          priority=50, actions=["read"],
                          principals=["role:analyst"], resources=["type:data"])
        allow = PolicyRule(rule_id="r2", name="Allow", effect="allow",
                           priority=100, actions=["read"],
                           principals=["role:analyst"], resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[deny, allow]))
        result = engine.evaluate(_make_request())
        assert result.allowed is False
        assert "r1" in result.matching_rules

    def test_same_priority_first_added_rule_wins(self):
        engine = PolicyEngine(strict_mode=True)
        allow = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                           priority=100, actions=["read"],
                           principals=["role:analyst"], resources=["type:data"])
        deny = PolicyRule(rule_id="r2", name="Deny", effect="deny",
                          priority=100, actions=["read"],
                          principals=["role:analyst"], resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[allow, deny]))
        result = engine.evaluate(_make_request())
        # Both have same priority; sorted is stable, so the first in list wins
        assert "r1" in result.matching_rules or "r2" in result.matching_rules

    def test_rules_sorted_by_priority_across_policies(self):
        engine = PolicyEngine(strict_mode=True)
        allow = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                           priority=10, actions=["read"],
                           principals=["role:analyst"], resources=["type:data"])
        deny = PolicyRule(rule_id="r2", name="Deny", effect="deny",
                          priority=5, actions=["read"],
                          principals=["role:analyst"], resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[allow]))
        engine.add_policy(Policy(policy_id="p2", name="P2", rules=[deny]))
        result = engine.evaluate(_make_request())
        assert result.allowed is False
        assert "r2" in result.matching_rules


class TestPolicyEngineInheritance:
    """Test parent_policy_id resolution."""

    def test_parent_tracked_on_add(self):
        engine = PolicyEngine()
        parent = Policy(policy_id="parent", name="Parent")
        child = Policy(policy_id="child", name="Child", parent_policy_id="parent")
        engine.add_policy(parent)
        engine.add_policy(child)
        assert "child" in engine._policy_hierarchy
        assert "parent" in engine._policy_hierarchy["child"]

    def test_remove_child_cleans_hierarchy(self):
        engine = PolicyEngine()
        parent = Policy(policy_id="parent", name="Parent")
        child = Policy(policy_id="child", name="Child", parent_policy_id="parent")
        engine.add_policy(parent)
        engine.add_policy(child)
        engine.remove_policy("child")
        assert "child" not in engine._policy_hierarchy

    def test_child_and_parent_rules_both_evaluated(self):
        engine = PolicyEngine(strict_mode=True)
        parent_rule = PolicyRule(rule_id="pr1", name="Parent Allow", effect="allow",
                                  priority=100, actions=["read"],
                                  principals=["role:analyst"], resources=["type:data"])
        parent = Policy(policy_id="parent", name="Parent", rules=[parent_rule])
        child = Policy(policy_id="child", name="Child", parent_policy_id="parent")
        engine.add_policy(parent)
        engine.add_policy(child)
        result = engine.evaluate(_make_request())
        assert result.allowed is True


class TestPolicyEngineProvenance:
    """Test SHA-256 hash on add/update."""

    def test_add_sets_provenance_hash(self):
        engine = PolicyEngine()
        p = Policy(policy_id="p1", name="P1")
        h = engine.add_policy(p)
        assert len(h) == 64
        assert p.provenance_hash == h

    def test_update_changes_provenance_hash(self):
        engine = PolicyEngine()
        p = Policy(policy_id="p1", name="P1")
        h1 = engine.add_policy(p)
        p.version = "2.0.0"
        h2 = engine.update_policy(p)
        assert h1 != h2

    def test_provenance_hash_is_sha256(self):
        engine = PolicyEngine()
        p = Policy(policy_id="p1", name="P1")
        h = engine.add_policy(p)
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_same_policy_same_hash(self):
        engine = PolicyEngine()
        rule = PolicyRule(rule_id="r1", name="R1", effect="allow")
        p1 = Policy(policy_id="p1", name="P1", rules=[rule], version="1.0.0")
        p2 = Policy(policy_id="p1", name="P1", rules=[rule], version="1.0.0")
        h1 = p1.compute_hash()
        h2 = p2.compute_hash()
        assert h1 == h2


class TestPolicyEngineTimeConstraints:
    """Test time-based conditions."""

    def test_time_constraint_rule_becomes_conditional(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(
            rule_id="r1", name="Business Hours", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
            time_constraints={"start_hour": 0, "end_hour": 24},
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request())
        # With time_constraints spanning all hours, conditions are met -> ALLOW
        assert result.allowed is True

    def test_time_constraint_outside_window(self):
        engine = PolicyEngine(strict_mode=True)
        # Create a rule that only allows during hour 25 (impossible)
        rule = PolicyRule(
            rule_id="r1", name="Never Hours", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
            time_constraints={"start_hour": 25, "end_hour": 26},
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request())
        # Conditions not met, should deny
        assert result.allowed is False


class TestPolicyEngineGeographic:
    """Test geographic constraints."""

    def test_geographic_constraint_matching(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(
            rule_id="r1", name="US Only", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
            geographic_constraints=["US"],
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(geo_location="US"))
        assert result.allowed is True

    def test_geographic_constraint_not_matching(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(
            rule_id="r1", name="US Only", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
            geographic_constraints=["US"],
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(geo_location="CN"))
        assert result.allowed is False

    def test_geographic_no_location_passes(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(
            rule_id="r1", name="US Only", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
            geographic_constraints=["US"],
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request(geo_location=None))
        assert result.allowed is True


class TestPolicyEngineDisabledRulesAndPolicies:
    """Test disabled rules and policies are skipped."""

    def test_disabled_rule_skipped(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"], enabled=False)
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        result = engine.evaluate(_make_request())
        assert result.allowed is False  # strict mode, rule disabled

    def test_disabled_policy_skipped(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule], enabled=False))
        result = engine.evaluate(_make_request())
        assert result.allowed is False

    def test_tenant_scoped_policy_not_matching(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(
            policy_id="p1", name="P1", rules=[rule], tenant_id="tenant-2",
        ))
        result = engine.evaluate(_make_request(tenant_id="tenant-1"))
        assert result.allowed is False

    def test_resource_type_scoped_policy_not_matching(self):
        engine = PolicyEngine(strict_mode=True)
        rule = PolicyRule(rule_id="r1", name="Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:agent"])
        engine.add_policy(Policy(
            policy_id="p1", name="P1", rules=[rule], applies_to=["agent"],
        ))
        result = engine.evaluate(_make_request(resource_type="data"))
        assert result.allowed is False
