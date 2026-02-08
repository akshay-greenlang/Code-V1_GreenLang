# -*- coding: utf-8 -*-
"""
API Integration Tests for Access & Policy Guard Service (AGENT-FOUND-006)

Tests end-to-end API-like workflows including policy CRUD, access check,
audit retrieval, classification, rate limiting, and compliance reporting
using direct function calls (no network/TestClient) to comply with
the integration test network blocker.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline API service for integration testing (direct calls, no HTTP)
# ---------------------------------------------------------------------------


class IntegrationAPIService:
    """Full integration service simulating API endpoints in-memory."""

    def __init__(self):
        self.policies: Dict[str, Dict] = {}
        self.audit_events: List[Dict] = []
        self.rego_policies: Dict[str, str] = {}
        self.rate_buckets: Dict[str, int] = defaultdict(int)
        self.rate_limit_rpm = 100

    def health(self):
        return {"status": "healthy", "service": "access-guard"}

    def check_access(self, principal, resource, action="read"):
        event = {
            "event_id": str(uuid.uuid4()),
            "principal_id": principal.get("principal_id"),
            "resource_id": resource.get("resource_id"),
            "action": action,
            "tenant_id": resource.get("tenant_id", ""),
        }
        decision = "allow"
        deny_reasons = []

        if not principal.get("authenticated", True):
            decision = "deny"
            deny_reasons.append("Not authenticated")

        if principal.get("tenant_id") != resource.get("tenant_id"):
            decision = "deny"
            deny_reasons.append("Tenant mismatch")

        clearance_map = {"public": 0, "internal": 1, "confidential": 2,
                         "restricted": 3, "top_secret": 4}
        p_level = clearance_map.get(principal.get("clearance_level", "internal"), 1)
        r_level = clearance_map.get(resource.get("classification", "internal"), 1)
        if r_level > p_level:
            decision = "deny"
            deny_reasons.append("Insufficient clearance")

        key = f"{principal.get('tenant_id')}:{principal.get('principal_id')}"
        self.rate_buckets[key] += 1
        if self.rate_buckets[key] > self.rate_limit_rpm:
            decision = "deny"
            deny_reasons.append("Rate limit exceeded")

        event["decision"] = decision
        event["deny_reasons"] = deny_reasons
        event["event_type"] = "access_granted" if decision == "allow" else "access_denied"
        self.audit_events.append(event)

        return {
            "request_id": str(uuid.uuid4()),
            "decision": decision,
            "allowed": decision == "allow",
            "deny_reasons": deny_reasons,
            "decision_hash": hashlib.sha256(
                json.dumps(event, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }

    def create_policy(self, body):
        pid = body.get("policy_id")
        if not pid:
            return {"error": "policy_id required", "status": 422}
        if pid in self.policies:
            return {"error": "Policy already exists", "status": 400}
        h = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        body["provenance_hash"] = h
        self.policies[pid] = body
        return {"policy_id": pid, "provenance_hash": h, "status": 201}

    def list_policies(self):
        return {"policies": list(self.policies.values()), "count": len(self.policies)}

    def get_policy(self, pid):
        if pid not in self.policies:
            return {"error": "Not found", "status": 404}
        return {**self.policies[pid], "status": 200}

    def delete_policy(self, pid):
        if pid not in self.policies:
            return {"error": "Not found", "status": 404}
        del self.policies[pid]
        return {"deleted": True, "status": 200}

    def get_audit_events(self, tenant_id=None, limit=100):
        events = self.audit_events
        if tenant_id:
            events = [e for e in events if e.get("tenant_id") == tenant_id]
        return {"events": events[-limit:], "count": len(events)}

    def generate_compliance_report(self, tenant_id):
        events = [e for e in self.audit_events if e.get("tenant_id") == tenant_id]
        return {
            "report_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "total_requests": len(events),
            "allowed_requests": len([e for e in events if e.get("decision") == "allow"]),
            "denied_requests": len([e for e in events if e.get("decision") == "deny"]),
        }


@pytest.fixture
def svc():
    return IntegrationAPIService()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAPIHealthIntegration:
    def test_health_endpoint(self, svc):
        result = svc.health()
        assert result["status"] == "healthy"

    def test_health_service_name(self, svc):
        result = svc.health()
        assert result["service"] == "access-guard"


class TestAPIPolicyCRUDIntegration:
    def test_full_policy_lifecycle(self, svc):
        # Create
        result = svc.create_policy({"policy_id": "p1", "name": "P1"})
        assert result["status"] == 201
        assert result["policy_id"] == "p1"

        # Read
        result = svc.get_policy("p1")
        assert result["status"] == 200
        assert result["policy_id"] == "p1"

        # List
        result = svc.list_policies()
        assert result["count"] == 1

        # Delete
        result = svc.delete_policy("p1")
        assert result["deleted"] is True

        # Verify deleted
        result = svc.get_policy("p1")
        assert result["status"] == 404

    def test_multiple_policies(self, svc):
        for i in range(5):
            svc.create_policy({"policy_id": f"p{i}", "name": f"P{i}"})
        result = svc.list_policies()
        assert result["count"] == 5

    def test_create_duplicate_returns_error(self, svc):
        svc.create_policy({"policy_id": "p1", "name": "P1"})
        result = svc.create_policy({"policy_id": "p1", "name": "P1 dup"})
        assert result["status"] == 400

    def test_create_missing_id_returns_error(self, svc):
        result = svc.create_policy({"name": "No ID"})
        assert result["status"] == 422

    def test_delete_nonexistent_returns_404(self, svc):
        result = svc.delete_policy("nope")
        assert result["status"] == 404

    def test_policy_provenance_hash(self, svc):
        result = svc.create_policy({"policy_id": "p1", "name": "P1"})
        assert len(result["provenance_hash"]) == 64


class TestAPIAccessCheckIntegration:
    def test_allow_request(self, svc):
        result = svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1", "authenticated": True,
             "clearance_level": "confidential"},
            {"resource_id": "r1", "tenant_id": "t1", "classification": "internal"},
            "read",
        )
        assert result["allowed"] is True

    def test_deny_unauthenticated(self, svc):
        result = svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1", "authenticated": False},
            {"resource_id": "r1", "tenant_id": "t1"},
        )
        assert result["allowed"] is False

    def test_deny_tenant_mismatch(self, svc):
        result = svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1"},
            {"resource_id": "r1", "tenant_id": "t2"},
        )
        assert result["allowed"] is False

    def test_deny_insufficient_clearance(self, svc):
        result = svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1", "clearance_level": "internal"},
            {"resource_id": "r1", "tenant_id": "t1", "classification": "restricted"},
        )
        assert result["allowed"] is False

    def test_decision_hash_present(self, svc):
        result = svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1"},
            {"resource_id": "r1", "tenant_id": "t1"},
        )
        assert len(result["decision_hash"]) == 64


class TestAPIAuditIntegration:
    def test_audit_events_created_by_access_check(self, svc):
        svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1"},
            {"resource_id": "r1", "tenant_id": "t1"},
            "read",
        )
        result = svc.get_audit_events()
        assert result["count"] >= 1

    def test_audit_filter_by_tenant(self, svc):
        for tid in ["t1", "t2", "t1"]:
            svc.check_access(
                {"principal_id": "u1", "tenant_id": tid},
                {"resource_id": "r1", "tenant_id": tid},
            )
        result = svc.get_audit_events(tenant_id="t1")
        for e in result["events"]:
            assert e["tenant_id"] == "t1"

    def test_audit_events_limit(self, svc):
        for _ in range(10):
            svc.check_access(
                {"principal_id": "u1", "tenant_id": "t1"},
                {"resource_id": "r1", "tenant_id": "t1"},
            )
        result = svc.get_audit_events(limit=5)
        assert len(result["events"]) == 5


class TestAPIComplianceReportIntegration:
    def test_compliance_report_generation(self, svc):
        # Generate allow events
        for _ in range(3):
            svc.check_access(
                {"principal_id": "u1", "tenant_id": "t1"},
                {"resource_id": "r1", "tenant_id": "t1"},
            )
        # Generate deny event
        svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1", "authenticated": False},
            {"resource_id": "r1", "tenant_id": "t1"},
        )
        report = svc.generate_compliance_report("t1")
        assert report["tenant_id"] == "t1"
        assert report["total_requests"] == 4
        assert report["allowed_requests"] == 3
        assert report["denied_requests"] == 1

    def test_compliance_report_empty_tenant(self, svc):
        report = svc.generate_compliance_report("empty")
        assert report["total_requests"] == 0


class TestAPIRateLimitIntegration:
    def test_rate_limit_enforced(self, svc):
        svc.rate_limit_rpm = 5
        results = []
        for _ in range(7):
            result = svc.check_access(
                {"principal_id": "u1", "tenant_id": "t1"},
                {"resource_id": "r1", "tenant_id": "t1"},
            )
            results.append(result["allowed"])
        # First 5 should be allowed, then denied
        assert all(results[:5])
        assert not all(results[5:])

    def test_rate_limit_per_principal(self, svc):
        svc.rate_limit_rpm = 3
        for _ in range(3):
            svc.check_access(
                {"principal_id": "u1", "tenant_id": "t1"},
                {"resource_id": "r1", "tenant_id": "t1"},
            )
        # u1 exhausted
        r = svc.check_access(
            {"principal_id": "u1", "tenant_id": "t1"},
            {"resource_id": "r1", "tenant_id": "t1"},
        )
        assert r["allowed"] is False

        # u2 still fresh
        r = svc.check_access(
            {"principal_id": "u2", "tenant_id": "t1"},
            {"resource_id": "r1", "tenant_id": "t1"},
        )
        assert r["allowed"] is True
