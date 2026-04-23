# -*- coding: utf-8 -*-
"""Tests for data-retention policy engine (SEC-5)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.middleware.retention import (
    InMemoryResourceAdapter,
    PurgeReason,
    RetentionEngine,
    RetentionResource,
    RetentionTier,
    TenantTierResolver,
    TIER_POLICY,
    factors_retention_purge,
    get_rule,
    set_audit_hook,
    set_metric_hook,
)


# ---------------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------------


def test_policy_coverage_every_tier_every_resource():
    for tier in RetentionTier:
        for resource in RetentionResource:
            assert resource in TIER_POLICY[tier], (tier, resource)


def test_enterprise_has_longer_log_retention_than_pro():
    assert TIER_POLICY[RetentionTier.ENTERPRISE][RetentionResource.LOGS] > TIER_POLICY[
        RetentionTier.PRO
    ][RetentionResource.LOGS]


def test_customer_private_is_indefinite_for_non_community_tiers():
    for tier in (RetentionTier.PRO, RetentionTier.PLATFORM, RetentionTier.ENTERPRISE):
        rule = get_rule(tier.value, RetentionResource.CUSTOMER_PRIVATE)
        assert rule.is_indefinite()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    adapters = {
        r: InMemoryResourceAdapter(r)
        for r in (
            RetentionResource.LOGS,
            RetentionResource.SIGNED_RECEIPTS,
            RetentionResource.EXPLAIN_HISTORY,
            RetentionResource.CUSTOMER_PRIVATE,
        )
    }
    resolver = TenantTierResolver(static={"pro-tenant": "pro", "ent-tenant": "enterprise"})
    return RetentionEngine(adapters=adapters, resolver=resolver), adapters


def test_scheduled_purge_removes_records_past_retention(engine):
    eng, adapters = engine
    now = datetime.now(timezone.utc)
    log_ad = adapters[RetentionResource.LOGS]
    # Pro tier: logs retained 90 days. Put one 100d old + one 10d old.
    log_ad.add("pro-tenant", now - timedelta(days=100))
    log_ad.add("pro-tenant", now - timedelta(days=10))

    reports = eng.run_scheduled_purge(["pro-tenant"], now=now)

    log_report = next(r for r in reports if r.resource == RetentionResource.LOGS)
    assert log_report.purged_rows == 1
    assert len(log_ad.records) == 1


def test_scheduled_purge_is_idempotent(engine):
    eng, adapters = engine
    now = datetime.now(timezone.utc)
    log_ad = adapters[RetentionResource.LOGS]
    log_ad.add("pro-tenant", now - timedelta(days=200))
    eng.run_scheduled_purge(["pro-tenant"], now=now)
    first = list(log_ad.records)
    eng.run_scheduled_purge(["pro-tenant"], now=now)
    assert log_ad.records == first


def test_indefinite_resource_is_never_purged_by_schedule(engine):
    eng, adapters = engine
    now = datetime.now(timezone.utc)
    cp_ad = adapters[RetentionResource.CUSTOMER_PRIVATE]
    cp_ad.add("pro-tenant", now - timedelta(days=3650))  # 10 years old
    eng.run_scheduled_purge(["pro-tenant"], now=now)
    assert len(cp_ad.records) == 1  # still there


def test_resurrect_within_retention(engine):
    eng, _ = engine
    # Pro logs = 90d; 45d-old record is still within retention.
    assert eng.resurrect_if_within_retention(
        RetentionResource.LOGS, "pro-tenant", record_age_days=45
    )
    assert not eng.resurrect_if_within_retention(
        RetentionResource.LOGS, "pro-tenant", record_age_days=120
    )


def test_tenant_deletion_cascade(engine):
    eng, adapters = engine
    now = datetime.now(timezone.utc)
    for ad in adapters.values():
        ad.add("ent-tenant", now)
        ad.add("other-tenant", now)
    reports = eng.purge_tenant_data("ent-tenant", reason=PurgeReason.GDPR_ART17)
    total = sum(r.purged_rows for r in reports)
    assert total == len(adapters)  # one per resource
    for ad in adapters.values():
        assert all(r["tenant_id"] != "ent-tenant" for r in ad.records)


def test_metric_hook_invoked(engine):
    eng, adapters = engine
    collected = []

    def _hook(resource, tier, reason, count):
        collected.append((resource, tier, reason, count))

    set_metric_hook(_hook)
    try:
        now = datetime.now(timezone.utc)
        adapters[RetentionResource.LOGS].add("pro-tenant", now - timedelta(days=200))
        eng.run_scheduled_purge(["pro-tenant"], now=now)
    finally:
        set_metric_hook(lambda *a, **k: None)
    resources = {c[0] for c in collected}
    assert RetentionResource.LOGS in resources


def test_audit_hook_captures_purges(engine):
    eng, adapters = engine
    audit_events = []
    set_audit_hook(audit_events.append)
    try:
        adapters[RetentionResource.LOGS].add(
            "pro-tenant", datetime.now(timezone.utc) - timedelta(days=500)
        )
        eng.run_scheduled_purge(["pro-tenant"])
    finally:
        set_audit_hook(lambda _e: None)
    assert any(e["event"] == "retention.cron.purge" for e in audit_events)


def test_factors_retention_purge_returns_summary(engine):
    eng, _ = engine
    summary = factors_retention_purge(eng, tenants=["pro-tenant"])
    assert summary["job"] == "factors_retention_purge"
    assert "reports" in summary
    assert "total_purged" in summary
