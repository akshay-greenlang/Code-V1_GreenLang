# -*- coding: utf-8 -*-
"""
Load Tests for Access & Policy Guard Service (AGENT-FOUND-006)

Tests throughput and concurrency for access decisions, rate limiting,
policy evaluation, audit logging, single-decision latency, and
compliance report generation.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Import inline implementations from unit tests
# ---------------------------------------------------------------------------

from tests.unit.access_guard_service.test_policy_engine import (
    PolicyEngine,
    Policy,
    PolicyRule,
    AccessRequest,
    Principal,
    Resource,
    AccessDecision,
)
from tests.unit.access_guard_service.test_rate_limiter import (
    RateLimiter,
    RateLimitConfig,
)
from tests.unit.access_guard_service.test_audit_logger import (
    AuditLogger,
    AuditEvent,
)


# ===========================================================================
# Load Test Classes
# ===========================================================================


class TestDecisionThroughput:
    """Test 10000 access decisions in <5s."""

    @pytest.mark.slow
    def test_10000_sequential_decisions(self):
        engine = PolicyEngine(strict_mode=False)
        rule = PolicyRule(
            rule_id="r1", name="Allow All", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))

        start = time.time()
        for i in range(10000):
            p = Principal(f"user-{i % 100}", f"tenant-{i % 10}", roles=["analyst"])
            r = Resource(f"res-{i}", "data", f"tenant-{i % 10}")
            req = AccessRequest(p, r, "read")
            result = engine.evaluate(req)
            assert result.decision in (AccessDecision.ALLOW, AccessDecision.DENY)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"10000 decisions took {elapsed:.2f}s (target: <5s)"


class TestConcurrentDecisions:
    """Test 50 concurrent decision requests."""

    @pytest.mark.slow
    def test_50_concurrent_decisions(self):
        engine = PolicyEngine(strict_mode=False)
        rule = PolicyRule(
            rule_id="r1", name="Allow All", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))

        def do_decision(i):
            p = Principal(f"user-{i}", "tenant-1", roles=["analyst"])
            r = Resource(f"res-{i}", "data", "tenant-1")
            req = AccessRequest(p, r, "read")
            return engine.evaluate(req)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_decision, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 50
        for r in results:
            assert r.decision in (AccessDecision.ALLOW, AccessDecision.DENY)


class TestRateLimiterThroughput:
    """Test 5000 rate limit checks."""

    @pytest.mark.slow
    def test_5000_rate_limit_checks(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=100000,
            requests_per_hour=1000000,
            requests_per_day=10000000,
        ))

        start = time.time()
        for i in range(5000):
            allowed, _ = rl.check_rate_limit(
                f"tenant-{i % 10}",
                f"user-{i % 100}",
            )
            assert allowed is True
        elapsed = time.time() - start

        assert elapsed < 5.0, f"5000 rate checks took {elapsed:.2f}s (target: <5s)"


class TestPolicyEvaluationThroughput:
    """Test 1000 evaluations with 50 rules."""

    @pytest.mark.slow
    def test_1000_evaluations_50_rules(self):
        engine = PolicyEngine(strict_mode=True)

        # Create 50 rules across 5 policies
        for p_idx in range(5):
            rules = []
            for r_idx in range(10):
                rules.append(PolicyRule(
                    rule_id=f"r-{p_idx}-{r_idx}",
                    name=f"Rule {p_idx}-{r_idx}",
                    effect="allow" if r_idx == 9 else "deny",
                    priority=100 + r_idx,
                    actions=["read"],
                    principals=[f"role:role-{r_idx}"],
                    resources=["type:data"],
                ))
            engine.add_policy(Policy(
                policy_id=f"policy-{p_idx}",
                name=f"Policy {p_idx}",
                rules=rules,
            ))

        start = time.time()
        for i in range(1000):
            p = Principal(f"user-{i}", "tenant-1", roles=[f"role-{i % 10}"])
            r = Resource(f"res-{i}", "data", "tenant-1")
            engine.evaluate(AccessRequest(p, r, "read"))
        elapsed = time.time() - start

        assert elapsed < 5.0, f"1000 evaluations took {elapsed:.2f}s (target: <5s)"


class TestAuditLoggingThroughput:
    """Test 5000 audit events logged."""

    @pytest.mark.slow
    def test_5000_audit_events(self):
        logger = AuditLogger(max_size=100000)

        start = time.time()
        for i in range(5000):
            event = AuditEvent(
                event_type="access_granted",
                tenant_id=f"tenant-{i % 10}",
                principal_id=f"user-{i % 100}",
                resource_id=f"res-{i}",
                action="read",
                decision="allow",
            )
            logger.log_event(event)
        elapsed = time.time() - start

        assert logger.count == 5000
        assert elapsed < 5.0, f"5000 events took {elapsed:.2f}s (target: <5s)"


class TestSingleDecisionLatency:
    """Test <1ms per decision."""

    def test_single_decision_latency(self):
        engine = PolicyEngine(strict_mode=False)
        rule = PolicyRule(
            rule_id="r1", name="Allow", effect="allow",
            actions=["read"], principals=["role:analyst"],
            resources=["type:data"],
        )
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))

        p = Principal("user-1", "tenant-1", roles=["analyst"])
        r = Resource("res-1", "data", "tenant-1")
        req = AccessRequest(p, r, "read")

        # Warm up
        engine.evaluate(req)

        # Measure
        latencies = []
        for _ in range(100):
            start = time.time()
            engine.evaluate(req)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}ms (target: <1ms)"


class TestComplianceReportGeneration:
    """Test compliance report on 10000 events."""

    @pytest.mark.slow
    def test_report_on_10000_events(self):
        logger = AuditLogger(max_size=100000)
        now = datetime.utcnow()

        # Generate 10000 events
        for i in range(10000):
            event_type = "access_granted" if i % 3 != 0 else "access_denied"
            event = AuditEvent(
                event_type=event_type,
                tenant_id="tenant-1",
                principal_id=f"user-{i % 50}",
                resource_id=f"res-{i % 200}",
                action="read" if i % 2 == 0 else "write",
                decision="allow" if event_type == "access_granted" else "deny",
                timestamp=now,
                details={"reason": "policy"} if event_type == "access_denied" else {},
            )
            logger.log_event(event)

        assert logger.count == 10000

        start = time.time()
        report = logger.generate_compliance_report(
            "tenant-1",
            now - timedelta(hours=1),
            now + timedelta(hours=1),
        )
        elapsed = time.time() - start

        assert report.total_requests > 0
        assert report.allowed_requests + report.denied_requests == report.total_requests
        assert len(report.provenance_hash) == 64
        assert elapsed < 5.0, f"Report generation took {elapsed:.2f}s (target: <5s)"


class TestConcurrentAuditLogging:
    """Test concurrent audit event logging."""

    @pytest.mark.slow
    def test_100_concurrent_audit_events(self):
        logger = AuditLogger(max_size=100000)

        def log_event(i):
            event = AuditEvent(
                event_type="access_granted",
                tenant_id="tenant-1",
                principal_id=f"user-{i}",
            )
            return logger.log_event(event)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(log_event, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 100
        assert logger.count == 100


class TestMixedWorkloadThroughput:
    """Test mixed workload: decisions + audit + rate limit."""

    @pytest.mark.slow
    def test_mixed_workload_1000_ops(self):
        engine = PolicyEngine(strict_mode=False)
        rule = PolicyRule("r1", "Allow", effect="allow",
                          actions=["read"], principals=["role:analyst"],
                          resources=["type:data"])
        engine.add_policy(Policy(policy_id="p1", name="P1", rules=[rule]))
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=100000,
            requests_per_hour=1000000,
            requests_per_day=10000000,
        ))
        logger = AuditLogger()

        start = time.time()
        for i in range(1000):
            # Rate limit check
            rl.check_rate_limit(f"t-{i % 5}", f"u-{i % 50}")

            # Policy evaluation
            p = Principal(f"u-{i}", "t-1", roles=["analyst"])
            r = Resource(f"r-{i}", "data", "t-1")
            engine.evaluate(AccessRequest(p, r, "read"))

            # Audit log
            logger.log_event(AuditEvent(
                event_type="access_granted", tenant_id="t-1",
            ))

        elapsed = time.time() - start
        assert elapsed < 5.0, f"Mixed workload took {elapsed:.2f}s (target: <5s)"
        assert logger.count == 1000
