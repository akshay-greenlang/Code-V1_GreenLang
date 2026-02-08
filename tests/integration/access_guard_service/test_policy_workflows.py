# -*- coding: utf-8 -*-
"""
Policy Workflow Integration Tests for Access & Policy Guard (AGENT-FOUND-006)

Tests policy CRUD lifecycle, inheritance, simulation, and OPA Rego workflows.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline models for policy workflow testing
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


class PolicyRule:
    def __init__(self, rule_id, name, effect="deny", priority=100,
                 actions=None, principals=None, resources=None,
                 conditions=None, classification_max=None, enabled=True):
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
                "priority": self.priority, "actions": self.actions,
                "principals": self.principals, "resources": self.resources}


class Policy:
    def __init__(self, pid, name, rules=None, parent_policy_id=None,
                 tenant_id=None, enabled=True, applies_to=None,
                 version="1.0.0"):
        self.policy_id = pid
        self.name = name
        self.rules = rules or []
        self.parent_policy_id = parent_policy_id
        self.tenant_id = tenant_id
        self.enabled = enabled
        self.applies_to = applies_to or []
        self.version = version
        self.provenance_hash = ""
        self.updated_at = datetime.utcnow()

    def compute_hash(self):
        s = json.dumps({
            "policy_id": self.policy_id,
            "rules": [r.to_dict() for r in self.rules],
            "version": self.version,
        }, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()


class PolicyWorkflowEngine:
    """Engine for testing policy lifecycle workflows."""

    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._hierarchy: Dict[str, List[str]] = {}
        self._rego_policies: Dict[str, str] = {}
        self._rego_versions: Dict[str, int] = {}
        self._change_log: List[Dict] = []

    @property
    def count(self):
        return len(self._policies)

    def create_policy(self, policy):
        if policy.policy_id in self._policies:
            raise ValueError(f"Policy {policy.policy_id} already exists")
        policy.provenance_hash = policy.compute_hash()
        self._policies[policy.policy_id] = policy
        if policy.parent_policy_id:
            self._hierarchy.setdefault(policy.policy_id, []).append(policy.parent_policy_id)
        self._change_log.append({
            "policy_id": policy.policy_id, "action": "create",
            "hash": policy.provenance_hash, "timestamp": datetime.utcnow().isoformat(),
        })
        return policy.provenance_hash

    def update_policy(self, pid, updates):
        if pid not in self._policies:
            raise ValueError(f"Policy {pid} not found")
        policy = self._policies[pid]
        for k, v in updates.items():
            setattr(policy, k, v)
        policy.version = str(float(policy.version.split(".")[0]) + 0.1)
        old_hash = policy.provenance_hash
        policy.provenance_hash = policy.compute_hash()
        policy.updated_at = datetime.utcnow()
        self._change_log.append({
            "policy_id": pid, "action": "update",
            "old_hash": old_hash, "new_hash": policy.provenance_hash,
        })
        return policy.provenance_hash

    def delete_policy(self, pid):
        if pid not in self._policies:
            return False
        del self._policies[pid]
        self._hierarchy.pop(pid, None)
        self._change_log.append({"policy_id": pid, "action": "delete"})
        return True

    def get_policy(self, pid):
        return self._policies.get(pid)

    def list_policies(self):
        return list(self._policies.values())

    def enable_policy(self, pid):
        if pid not in self._policies:
            return False
        self._policies[pid].enabled = True
        self._change_log.append({"policy_id": pid, "action": "enable"})
        return True

    def disable_policy(self, pid):
        if pid not in self._policies:
            return False
        self._policies[pid].enabled = False
        self._change_log.append({"policy_id": pid, "action": "disable"})
        return True

    def get_children(self, pid):
        children = []
        for child_id, parents in self._hierarchy.items():
            if pid in parents:
                children.append(child_id)
        return children

    def get_parents(self, pid):
        return self._hierarchy.get(pid, [])

    def add_rego_policy(self, pid, source):
        h = hashlib.sha256(source.encode()).hexdigest()
        self._rego_policies[pid] = source
        self._rego_versions[pid] = self._rego_versions.get(pid, 0) + 1
        return h

    def get_rego_policy(self, pid):
        return self._rego_policies.get(pid)

    def get_change_log(self, pid=None):
        if pid:
            return [e for e in self._change_log if e.get("policy_id") == pid]
        return self._change_log

    def simulate(self, policy, test_requests):
        """Simulate policy against test requests without persisting."""
        results = []
        for req in test_requests:
            matched = False
            for rule in policy.rules:
                if not rule.enabled:
                    continue
                if rule.actions and req.get("action") not in rule.actions:
                    continue
                if rule.principals:
                    role_match = any(
                        p[5:] in req.get("roles", []) or p[5:] == "*"
                        for p in rule.principals if p.startswith("role:")
                    )
                    if not role_match:
                        continue
                matched = True
                results.append({
                    "request": req, "decision": rule.effect.value,
                    "rule": rule.rule_id,
                })
                break
            if not matched:
                results.append({"request": req, "decision": "deny", "rule": None})
        return results


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPolicyCRUDLifecycle:
    """Test full policy create/read/update/delete lifecycle."""

    def test_create_policy(self):
        engine = PolicyWorkflowEngine()
        h = engine.create_policy(Policy("p1", "Policy 1"))
        assert len(h) == 64
        assert engine.count == 1

    def test_create_duplicate_raises(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "P1"))
        with pytest.raises(ValueError, match="already exists"):
            engine.create_policy(Policy("p1", "P1 dup"))

    def test_read_policy(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "Policy 1"))
        p = engine.get_policy("p1")
        assert p.name == "Policy 1"

    def test_read_nonexistent(self):
        engine = PolicyWorkflowEngine()
        assert engine.get_policy("nope") is None

    def test_update_policy(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "V1"))
        h = engine.update_policy("p1", {"name": "V2"})
        assert engine.get_policy("p1").name == "V2"
        assert len(h) == 64

    def test_update_nonexistent_raises(self):
        engine = PolicyWorkflowEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.update_policy("nope", {})

    def test_update_changes_hash(self):
        engine = PolicyWorkflowEngine()
        h1 = engine.create_policy(Policy("p1", "V1"))
        rule = PolicyRule("r1", "New Rule", effect="allow")
        h2 = engine.update_policy("p1", {"rules": [rule]})
        assert h1 != h2

    def test_delete_policy(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "P1"))
        assert engine.delete_policy("p1") is True
        assert engine.count == 0

    def test_delete_nonexistent(self):
        engine = PolicyWorkflowEngine()
        assert engine.delete_policy("nope") is False

    def test_list_policies(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "P1"))
        engine.create_policy(Policy("p2", "P2"))
        policies = engine.list_policies()
        assert len(policies) == 2

    def test_enable_disable_policy(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "P1"))
        engine.disable_policy("p1")
        assert engine.get_policy("p1").enabled is False
        engine.enable_policy("p1")
        assert engine.get_policy("p1").enabled is True

    def test_change_log_tracks_all_operations(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "P1"))
        engine.update_policy("p1", {"name": "P1v2"})
        engine.disable_policy("p1")
        engine.enable_policy("p1")
        engine.delete_policy("p1")
        log = engine.get_change_log("p1")
        actions = [e["action"] for e in log]
        assert actions == ["create", "update", "disable", "enable", "delete"]


class TestPolicyInheritance:
    """Test parent-child policy relationships."""

    def test_parent_child_tracking(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("parent", "Parent"))
        engine.create_policy(Policy("child", "Child", parent_policy_id="parent"))
        assert "parent" in engine.get_parents("child")

    def test_get_children(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("parent", "Parent"))
        engine.create_policy(Policy("c1", "Child 1", parent_policy_id="parent"))
        engine.create_policy(Policy("c2", "Child 2", parent_policy_id="parent"))
        children = engine.get_children("parent")
        assert "c1" in children
        assert "c2" in children

    def test_delete_child_removes_hierarchy(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("parent", "Parent"))
        engine.create_policy(Policy("child", "Child", parent_policy_id="parent"))
        engine.delete_policy("child")
        assert "child" not in engine._hierarchy

    def test_no_parent(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("standalone", "Standalone"))
        assert engine.get_parents("standalone") == []

    def test_multi_level_hierarchy(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("root", "Root"))
        engine.create_policy(Policy("mid", "Mid", parent_policy_id="root"))
        engine.create_policy(Policy("leaf", "Leaf", parent_policy_id="mid"))
        assert "root" in engine.get_parents("mid")
        assert "mid" in engine.get_parents("leaf")


class TestPolicySimulation:
    """Test policy simulation without enforcement."""

    def test_simulation_allow(self):
        engine = PolicyWorkflowEngine()
        rule = PolicyRule("r1", "Allow Read", effect="allow",
                          actions=["read"], principals=["role:analyst"])
        policy = Policy("p1", "Test", rules=[rule])
        requests = [{"action": "read", "roles": ["analyst"]}]
        results = engine.simulate(policy, requests)
        assert results[0]["decision"] == "allow"

    def test_simulation_deny(self):
        engine = PolicyWorkflowEngine()
        rule = PolicyRule("r1", "Allow Read", effect="allow",
                          actions=["read"], principals=["role:admin"])
        policy = Policy("p1", "Test", rules=[rule])
        requests = [{"action": "read", "roles": ["analyst"]}]
        results = engine.simulate(policy, requests)
        assert results[0]["decision"] == "deny"

    def test_simulation_multiple_requests(self):
        engine = PolicyWorkflowEngine()
        rule = PolicyRule("r1", "Allow Read", effect="allow",
                          actions=["read"], principals=["role:analyst"])
        policy = Policy("p1", "Test", rules=[rule])
        requests = [
            {"action": "read", "roles": ["analyst"]},
            {"action": "write", "roles": ["analyst"]},
            {"action": "read", "roles": ["viewer"]},
        ]
        results = engine.simulate(policy, requests)
        assert results[0]["decision"] == "allow"
        assert results[1]["decision"] == "deny"  # write not in actions
        assert results[2]["decision"] == "deny"  # viewer not analyst

    def test_simulation_does_not_persist(self):
        engine = PolicyWorkflowEngine()
        policy = Policy("p1", "Test")
        engine.simulate(policy, [{"action": "read", "roles": []}])
        assert engine.count == 0

    def test_simulation_disabled_rule_skipped(self):
        engine = PolicyWorkflowEngine()
        rule = PolicyRule("r1", "Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          enabled=False)
        policy = Policy("p1", "Test", rules=[rule])
        results = engine.simulate(policy, [{"action": "read", "roles": ["analyst"]}])
        assert results[0]["decision"] == "deny"


class TestOPARegoWorkflows:
    """Test OPA Rego policy management workflows."""

    def test_add_rego_policy(self):
        engine = PolicyWorkflowEngine()
        h = engine.add_rego_policy("rego-1", "package test\nallow { true }")
        assert len(h) == 64

    def test_get_rego_policy(self):
        engine = PolicyWorkflowEngine()
        engine.add_rego_policy("rego-1", "package test\nallow { true }")
        source = engine.get_rego_policy("rego-1")
        assert "package test" in source

    def test_get_nonexistent_rego(self):
        engine = PolicyWorkflowEngine()
        assert engine.get_rego_policy("nope") is None

    def test_rego_versioning(self):
        engine = PolicyWorkflowEngine()
        engine.add_rego_policy("rego-1", "package v1\nallow { true }")
        engine.add_rego_policy("rego-1", "package v2\nallow { true }")
        assert engine._rego_versions["rego-1"] == 2

    def test_rego_hash_changes_on_update(self):
        engine = PolicyWorkflowEngine()
        h1 = engine.add_rego_policy("rego-1", "package v1\nallow { true }")
        h2 = engine.add_rego_policy("rego-1", "package v2\nallow { false }")
        assert h1 != h2


class TestPolicyProvenance:
    """Test provenance hash tracking across policy lifecycle."""

    def test_create_sets_provenance_hash(self):
        engine = PolicyWorkflowEngine()
        h = engine.create_policy(Policy("p1", "P1"))
        assert len(h) == 64
        assert engine.get_policy("p1").provenance_hash == h

    def test_update_changes_provenance_hash(self):
        engine = PolicyWorkflowEngine()
        h1 = engine.create_policy(Policy("p1", "P1"))
        h2 = engine.update_policy("p1", {"name": "P1 Updated"})
        assert h1 != h2

    def test_hash_is_deterministic(self):
        engine = PolicyWorkflowEngine()
        rule = PolicyRule("r1", "Rule", effect="allow")
        p1 = Policy("p1", "P1", rules=[rule], version="1.0.0")
        p2 = Policy("p1", "P1", rules=[rule], version="1.0.0")
        assert p1.compute_hash() == p2.compute_hash()

    def test_change_log_records_hashes(self):
        engine = PolicyWorkflowEngine()
        engine.create_policy(Policy("p1", "P1"))
        engine.update_policy("p1", {"name": "P1v2"})
        log = engine.get_change_log("p1")
        assert "hash" in log[0]
        assert "old_hash" in log[1]
        assert "new_hash" in log[1]
        assert log[0]["hash"] == log[1]["old_hash"]
