# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Access & Policy Guard Service (AGENT-FOUND-006)

Tests full workflows: multi-tenant isolation, classification enforcement,
rate limiting, compliance report generation, policy simulation, and
provenance chain integrity.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for end-to-end testing
# ---------------------------------------------------------------------------


class AccessDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


CLASSIFICATION_HIERARCHY = {
    DataClassification.PUBLIC: 0, DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2, DataClassification.RESTRICTED: 3,
    DataClassification.TOP_SECRET: 4,
}


class Principal:
    def __init__(self, pid, tid, roles=None, clearance="internal",
                 authenticated=True, attrs=None):
        self.principal_id = pid
        self.tenant_id = tid
        self.roles = roles or []
        self.clearance_level = DataClassification(clearance)
        self.authenticated = authenticated
        self.attributes = attrs or {}


class Resource:
    def __init__(self, rid, rtype="data", tid="", classification="internal",
                 attrs=None, geo=None):
        self.resource_id = rid
        self.resource_type = rtype
        self.tenant_id = tid
        self.classification = DataClassification(classification)
        self.attributes = attrs or {}
        self.geographic_location = geo


class AccessRequest:
    def __init__(self, principal, resource, action, context=None):
        self.request_id = str(uuid.uuid4())
        self.principal = principal
        self.resource = resource
        self.action = action
        self.context = context or {}
        self.timestamp = datetime.utcnow()


class AccessDecisionResult:
    def __init__(self, request_id, decision, allowed, deny_reasons=None,
                 matching_rules=None, decision_hash=""):
        self.request_id = request_id
        self.decision = AccessDecision(decision)
        self.allowed = allowed
        self.deny_reasons = deny_reasons or []
        self.matching_rules = matching_rules or []
        self.decision_hash = decision_hash


class PolicyRule:
    def __init__(self, rule_id, name, effect="deny", priority=100,
                 actions=None, principals=None, resources=None,
                 conditions=None, classification_max=None,
                 enabled=True):
        self.rule_id = rule_id
        self.name = name
        self.effect = AccessDecision(effect)
        self.priority = priority
        self.actions = actions or []
        self.principals = principals or []
        self.resources = resources or []
        self.conditions = conditions or {}
        self.classification_max = (
            DataClassification(classification_max) if classification_max else None
        )
        self.enabled = enabled

    def to_dict(self):
        return {"rule_id": self.rule_id, "name": self.name, "effect": self.effect.value,
                "priority": self.priority}


class Policy:
    def __init__(self, pid, name, rules=None, tenant_id=None,
                 enabled=True, applies_to=None, version="1.0.0"):
        self.policy_id = pid
        self.name = name
        self.rules = rules or []
        self.tenant_id = tenant_id
        self.enabled = enabled
        self.applies_to = applies_to or []
        self.version = version
        self.provenance_hash = ""

    def compute_hash(self):
        s = json.dumps({"policy_id": self.policy_id, "rules": [r.to_dict() for r in self.rules],
                         "version": self.version}, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()


class EndToEndGuardService:
    """Full-stack access guard for E2E testing."""

    def __init__(self, strict_mode=True, strict_tenant=True,
                 rate_limit_rpm=100, audit_enabled=True):
        self.strict_mode = strict_mode
        self.strict_tenant = strict_tenant
        self.rate_limit_rpm = rate_limit_rpm
        self.audit_enabled = audit_enabled
        self._policies: Dict[str, Policy] = {}
        self._audit: List[Dict] = []
        self._rate: Dict[str, int] = defaultdict(int)
        self._provenance: Dict[str, List[Dict]] = {}

    def add_policy(self, policy):
        policy.provenance_hash = policy.compute_hash()
        self._policies[policy.policy_id] = policy
        self._record_provenance(policy.policy_id, "policy", "add", policy.provenance_hash)
        return policy.provenance_hash

    def check_access(self, request):
        deny_reasons = []

        if not request.principal.authenticated:
            deny_reasons.append("Not authenticated")
            self._log_audit(request, "deny", deny_reasons)
            return AccessDecisionResult(request.request_id, "deny", False, deny_reasons)

        if self.strict_tenant and request.principal.tenant_id != request.resource.tenant_id:
            deny_reasons.append("Tenant boundary violation")
            self._log_audit(request, "deny", deny_reasons)
            return AccessDecisionResult(request.request_id, "deny", False, deny_reasons)

        p_level = CLASSIFICATION_HIERARCHY.get(request.principal.clearance_level, 0)
        r_level = CLASSIFICATION_HIERARCHY.get(request.resource.classification, 0)
        if r_level > p_level:
            deny_reasons.append("Insufficient clearance")
            self._log_audit(request, "deny", deny_reasons)
            return AccessDecisionResult(request.request_id, "deny", False, deny_reasons)

        key = f"{request.principal.tenant_id}:{request.principal.principal_id}"
        self._rate[key] += 1
        if self._rate[key] > self.rate_limit_rpm:
            deny_reasons.append("Rate limit exceeded")
            self._log_audit(request, "deny", deny_reasons)
            return AccessDecisionResult(request.request_id, "deny", False, deny_reasons)

        # Evaluate policies
        rules = self._get_effective_rules(request.resource.tenant_id, request.resource.resource_type)
        decision = "deny" if self.strict_mode else "allow"
        matching = []

        for rule in rules:
            if self._rule_matches(rule, request):
                matching.append(rule.rule_id)
                decision = rule.effect.value
                if decision == "deny":
                    deny_reasons.append(f"Denied by {rule.name}")
                break

        allowed = decision == "allow"
        self._log_audit(request, decision, deny_reasons)
        h = hashlib.sha256(json.dumps({
            "req": request.request_id, "decision": decision,
        }, sort_keys=True).encode()).hexdigest()
        return AccessDecisionResult(request.request_id, decision, allowed, deny_reasons, matching, h)

    def _get_effective_rules(self, tenant_id, resource_type):
        rules = []
        for p in self._policies.values():
            if not p.enabled:
                continue
            if p.tenant_id and p.tenant_id != tenant_id:
                continue
            if p.applies_to and resource_type not in p.applies_to:
                continue
            for r in p.rules:
                if r.enabled:
                    rules.append(r)
        rules.sort(key=lambda r: r.priority)
        return rules

    def _rule_matches(self, rule, request):
        if rule.actions and request.action not in rule.actions and "*" not in rule.actions:
            return False
        if rule.principals:
            matched = False
            for p in rule.principals:
                if p.startswith("role:"):
                    if p[5:] in request.principal.roles or p[5:] == "*":
                        matched = True
                elif p == request.principal.principal_id:
                    matched = True
            if not matched:
                return False
        if rule.resources:
            matched = False
            for r in rule.resources:
                if r.startswith("type:"):
                    if r[5:] == request.resource.resource_type or r[5:] == "*":
                        matched = True
                elif r == request.resource.resource_id or r == "*":
                    matched = True
            if not matched:
                return False
        if rule.classification_max:
            if CLASSIFICATION_HIERARCHY.get(request.resource.classification, 0) > \
               CLASSIFICATION_HIERARCHY.get(rule.classification_max, 0):
                return False
        if rule.conditions:
            for k, v in rule.conditions.items():
                if request.context.get(k) != v and request.principal.attributes.get(k) != v:
                    return False
        return True

    def _log_audit(self, request, decision, reasons):
        if self.audit_enabled:
            self._audit.append({
                "event_id": str(uuid.uuid4()),
                "request_id": request.request_id,
                "principal_id": request.principal.principal_id,
                "resource_id": request.resource.resource_id,
                "tenant_id": request.resource.tenant_id,
                "action": request.action,
                "decision": decision,
                "deny_reasons": reasons,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def _record_provenance(self, entity_id, entity_type, action, data_hash):
        chain = self._provenance.get(entity_id, [])
        prev = chain[-1]["chain_hash"] if chain else ""
        entry = {
            "entity_id": entity_id, "action": action, "data_hash": data_hash,
            "previous_hash": prev,
            "chain_hash": hashlib.sha256(
                f"{entity_id}:{action}:{data_hash}:{prev}".encode()
            ).hexdigest(),
        }
        chain.append(entry)
        self._provenance[entity_id] = chain

    def get_audit_events(self, tenant_id=None, limit=100):
        events = self._audit
        if tenant_id:
            events = [e for e in events if e["tenant_id"] == tenant_id]
        return events[-limit:]

    def generate_report(self, tenant_id):
        events = [e for e in self._audit if e["tenant_id"] == tenant_id]
        return {
            "tenant_id": tenant_id,
            "total": len(events),
            "allowed": len([e for e in events if e["decision"] == "allow"]),
            "denied": len([e for e in events if e["decision"] == "deny"]),
        }

    def verify_provenance(self, entity_id):
        chain = self._provenance.get(entity_id, [])
        if not chain:
            return {"valid": False, "error": "Not found"}
        for i in range(1, len(chain)):
            if chain[i]["previous_hash"] != chain[i - 1]["chain_hash"]:
                return {"valid": False, "error": f"Break at entry {i}"}
        return {"valid": True, "entries": len(chain)}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestMultiTenantIsolation:
    """Test strict tenant boundary enforcement."""

    def test_same_tenant_allowed(self):
        svc = EndToEndGuardService(strict_mode=False)
        p = Principal("u1", "t1", roles=["analyst"])
        r = Resource("r1", "data", "t1")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is True

    def test_cross_tenant_denied(self):
        svc = EndToEndGuardService()
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t2")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is False
        assert "Tenant" in result.deny_reasons[0]

    def test_cross_tenant_allowed_when_disabled(self):
        svc = EndToEndGuardService(strict_tenant=False, strict_mode=False)
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t2")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is True

    def test_multiple_tenants_isolated(self):
        svc = EndToEndGuardService(strict_mode=False)
        for tid in ["t1", "t2", "t3"]:
            p = Principal(f"u-{tid}", tid)
            r = Resource(f"r-{tid}", "data", tid)
            result = svc.check_access(AccessRequest(p, r, "read"))
            assert result.allowed is True

        # Cross-tenant should fail
        p = Principal("u-t1", "t1")
        r = Resource("r-t2", "data", "t2")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is False

    def test_audit_records_tenant_violation(self):
        svc = EndToEndGuardService()
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t2")
        svc.check_access(AccessRequest(p, r, "read"))
        events = svc.get_audit_events()
        assert len(events) == 1
        assert events[0]["decision"] == "deny"


class TestClassificationEnforcement:
    """Test classification-based access control."""

    def test_internal_clearance_accesses_internal(self):
        svc = EndToEndGuardService(strict_mode=False)
        p = Principal("u1", "t1", clearance="internal")
        r = Resource("r1", "data", "t1", "internal")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is True

    def test_internal_clearance_blocked_from_confidential(self):
        svc = EndToEndGuardService()
        p = Principal("u1", "t1", clearance="internal")
        r = Resource("r1", "data", "t1", "confidential")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is False

    def test_top_secret_accesses_everything(self):
        svc = EndToEndGuardService(strict_mode=False)
        p = Principal("u1", "t1", clearance="top_secret")
        for level in ["public", "internal", "confidential", "restricted", "top_secret"]:
            r = Resource(f"r-{level}", "data", "t1", level)
            result = svc.check_access(AccessRequest(p, r, "read"))
            assert result.allowed is True, f"Failed for {level}"

    def test_public_blocked_from_restricted(self):
        svc = EndToEndGuardService()
        p = Principal("u1", "t1", clearance="public")
        r = Resource("r1", "data", "t1", "restricted")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is False

    def test_classification_denial_reason(self):
        svc = EndToEndGuardService()
        p = Principal("u1", "t1", clearance="internal")
        r = Resource("r1", "data", "t1", "top_secret")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert "clearance" in result.deny_reasons[0].lower()


class TestRateLimitingEndToEnd:
    """Test rate limiting enforcement."""

    def test_within_limit_allowed(self):
        svc = EndToEndGuardService(rate_limit_rpm=10, strict_mode=False)
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t1")
        for _ in range(10):
            result = svc.check_access(AccessRequest(p, r, "read"))
            assert result.allowed is True

    def test_exceed_limit_denied(self):
        svc = EndToEndGuardService(rate_limit_rpm=5, strict_mode=False)
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t1")
        for _ in range(5):
            svc.check_access(AccessRequest(p, r, "read"))
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is False
        assert "rate limit" in result.deny_reasons[0].lower()

    def test_different_principals_independent_limits(self):
        svc = EndToEndGuardService(rate_limit_rpm=3, strict_mode=False)
        for _ in range(3):
            svc.check_access(AccessRequest(
                Principal("u1", "t1"), Resource("r1", "data", "t1"), "read",
            ))
        # u1 is exhausted
        r = svc.check_access(AccessRequest(
            Principal("u1", "t1"), Resource("r1", "data", "t1"), "read",
        ))
        assert r.allowed is False

        # u2 still fresh
        r = svc.check_access(AccessRequest(
            Principal("u2", "t1"), Resource("r1", "data", "t1"), "read",
        ))
        assert r.allowed is True


class TestComplianceReportEndToEnd:
    """Test compliance report generation."""

    def test_report_counts(self):
        svc = EndToEndGuardService(strict_mode=False)
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t1")
        for _ in range(5):
            svc.check_access(AccessRequest(p, r, "read"))
        # Denied request
        svc.check_access(AccessRequest(
            Principal("u1", "t1", authenticated=False),
            Resource("r1", "data", "t1"), "read",
        ))
        report = svc.generate_report("t1")
        assert report["total"] == 6
        assert report["allowed"] == 5
        assert report["denied"] == 1

    def test_report_tenant_filter(self):
        svc = EndToEndGuardService(strict_mode=False)
        svc.check_access(AccessRequest(
            Principal("u1", "t1"), Resource("r1", "data", "t1"), "read",
        ))
        svc.check_access(AccessRequest(
            Principal("u2", "t2"), Resource("r2", "data", "t2"), "read",
        ))
        report = svc.generate_report("t1")
        assert report["total"] == 1

    def test_empty_report(self):
        svc = EndToEndGuardService()
        report = svc.generate_report("empty")
        assert report["total"] == 0


class TestPolicySimulationEndToEnd:
    """Test policy evaluation with policies loaded."""

    def test_policy_allows_analyst_read(self):
        svc = EndToEndGuardService(strict_mode=True)
        rule = PolicyRule("r1", "Allow Analyst Read", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        svc.add_policy(Policy("p1", "Analyst Policy", rules=[rule], tenant_id="t1",
                               applies_to=["data"]))

        p = Principal("u1", "t1", roles=["analyst"], clearance="internal")
        r = Resource("r1", "data", "t1", "internal")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert result.allowed is True

    def test_policy_denies_write(self):
        svc = EndToEndGuardService(strict_mode=True)
        deny_rule = PolicyRule("r1", "Deny Write", effect="deny", priority=10,
                                actions=["write"], principals=["role:analyst"],
                                resources=["type:data"])
        allow_rule = PolicyRule("r2", "Allow Read", effect="allow", priority=20,
                                 actions=["read"], principals=["role:analyst"],
                                 resources=["type:data"])
        svc.add_policy(Policy("p1", "Policy", rules=[deny_rule, allow_rule],
                               tenant_id="t1", applies_to=["data"]))

        p = Principal("u1", "t1", roles=["analyst"], clearance="internal")
        r = Resource("r1", "data", "t1", "internal")
        result = svc.check_access(AccessRequest(p, r, "write"))
        assert result.allowed is False

    def test_classification_max_enforcement(self):
        svc = EndToEndGuardService(strict_mode=True)
        rule = PolicyRule("r1", "Allow Internal Only", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          classification_max="internal")
        svc.add_policy(Policy("p1", "Policy", rules=[rule], tenant_id="t1",
                               applies_to=["data"]))

        # Internal allowed
        p = Principal("u1", "t1", roles=["analyst"], clearance="confidential")
        r = Resource("r1", "data", "t1", "internal")
        assert svc.check_access(AccessRequest(p, r, "read")).allowed is True

        # Confidential not matched by rule
        r2 = Resource("r2", "data", "t1", "confidential")
        assert svc.check_access(AccessRequest(p, r2, "read")).allowed is False

    def test_abac_condition_matching(self):
        svc = EndToEndGuardService(strict_mode=True)
        rule = PolicyRule("r1", "US Only", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"],
                          conditions={"region": "US"})
        svc.add_policy(Policy("p1", "Geo Policy", rules=[rule], tenant_id="t1",
                               applies_to=["data"]))

        p = Principal("u1", "t1", roles=["analyst"], clearance="internal")
        r = Resource("r1", "data", "t1", "internal")

        # With matching context
        result = svc.check_access(AccessRequest(p, r, "read", context={"region": "US"}))
        assert result.allowed is True

        # Without matching context
        result = svc.check_access(AccessRequest(p, r, "read", context={"region": "EU"}))
        assert result.allowed is False


class TestProvenanceEndToEnd:
    """Test provenance chain integrity."""

    def test_policy_provenance_chain(self):
        svc = EndToEndGuardService()
        svc.add_policy(Policy("p1", "V1"))
        result = svc.verify_provenance("p1")
        assert result["valid"] is True
        assert result["entries"] == 1

    def test_provenance_hash_is_sha256(self):
        svc = EndToEndGuardService()
        h = svc.add_policy(Policy("p1", "V1"))
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_nonexistent_provenance(self):
        svc = EndToEndGuardService()
        result = svc.verify_provenance("nope")
        assert result["valid"] is False

    def test_decision_has_hash(self):
        svc = EndToEndGuardService(strict_mode=False)
        p = Principal("u1", "t1")
        r = Resource("r1", "data", "t1")
        result = svc.check_access(AccessRequest(p, r, "read"))
        assert len(result.decision_hash) == 64
