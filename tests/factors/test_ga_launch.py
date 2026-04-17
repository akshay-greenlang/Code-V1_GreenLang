# -*- coding: utf-8 -*-
"""Tests for Phase 12: GA readiness, billing, SLA tracking."""

from __future__ import annotations

import pytest

from greenlang.factors.ga.billing import (
    BillingEngine,
    BillingPlan,
    BillingTier,
    PLANS,
    UsageMeter,
)
from greenlang.factors.ga.readiness import (
    CheckStatus,
    ReadinessChecker,
    ReadinessReport,
)
from greenlang.factors.ga.sla_tracker import (
    DEFAULT_SLAS,
    SLADefinition,
    SLAMetric,
    SLAReport,
    SLAStatus,
    SLATracker,
)


# ── ReadinessChecker ─────────────────────────────────────────────────


class TestReadinessChecker:
    def test_all_pass(self):
        checker = ReadinessChecker()
        report = checker.run({
            "factor_count": 120000,
            "edition_id": "ed-2026-04",
            "source_count": 15,
        })
        assert report.overall_ready
        assert report.fail_count == 0

    def test_low_factor_count_fails(self):
        checker = ReadinessChecker()
        report = checker.run({"factor_count": 10000})
        assert not report.overall_ready
        assert "factor_count" in report.blockers

    def test_no_edition_fails(self):
        checker = ReadinessChecker()
        report = checker.run({"factor_count": 100000, "edition_id": None})
        assert "edition_available" in report.blockers

    def test_warn_for_low_sources(self):
        checker = ReadinessChecker()
        report = checker.run({"factor_count": 100000, "edition_id": "ed-1", "source_count": 3})
        assert report.warn_count > 0

    def test_custom_check(self):
        checker = ReadinessChecker()

        def custom():
            from greenlang.factors.ga.readiness import CheckResult, CheckStatus
            return CheckResult("custom", "test", CheckStatus.PASS, "ok")

        checker.add_check(custom)
        report = checker.run({"factor_count": 100000, "edition_id": "ed-1"})
        names = [c.name for c in report.checks]
        assert "custom" in names

    def test_report_to_dict(self):
        checker = ReadinessChecker()
        report = checker.run({"factor_count": 100000, "edition_id": "ed-1"})
        d = report.to_dict()
        assert "summary" in d
        assert "checks" in d
        assert "blockers" in d

    def test_check_error_handled(self):
        checker = ReadinessChecker()

        def bad_check():
            raise RuntimeError("boom")

        checker.add_check(bad_check)
        report = checker.run({"factor_count": 100000, "edition_id": "ed-1"})
        failed_names = [c.name for c in report.checks if c.status == CheckStatus.FAIL]
        assert "bad_check" in failed_names


# ── UsageMeter ───────────────────────────────────────────────────────


class TestUsageMeter:
    def test_record_call(self):
        meter = UsageMeter()
        rec = meter.record_call("t1", "search")
        assert rec.api_calls == 1
        assert rec.search_calls == 1

    def test_cumulative_recording(self):
        meter = UsageMeter()
        for _ in range(10):
            meter.record_call("t1", "api")
        rec = meter.get_current_usage("t1")
        assert rec.api_calls == 10

    def test_separate_tenants(self):
        meter = UsageMeter()
        meter.record_call("t1")
        meter.record_call("t2")
        meter.record_call("t2")
        r1 = meter.get_current_usage("t1")
        r2 = meter.get_current_usage("t2")
        assert r1.api_calls == 1
        assert r2.api_calls == 2

    def test_no_usage(self):
        meter = UsageMeter()
        assert meter.get_current_usage("nobody") is None


# ── BillingEngine ────────────────────────────────────────────────────


class TestBillingEngine:
    def _setup(self, tier: BillingTier = BillingTier.PRO, calls: int = 100) -> tuple:
        meter = UsageMeter()
        from datetime import datetime, timezone
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        for _ in range(calls):
            meter.record_call("t1")
        engine = BillingEngine(meter)
        engine.set_tenant_tier("t1", tier)
        return engine, month

    def test_community_invoice(self):
        engine, month = self._setup(BillingTier.COMMUNITY, 500)
        inv = engine.generate_invoice("t1", month)
        assert inv.base_price == 0.0
        assert inv.total_amount == 0.0  # No overage for community

    def test_pro_within_quota(self):
        engine, month = self._setup(BillingTier.PRO, 1000)
        inv = engine.generate_invoice("t1", month)
        assert inv.base_price == 299.0
        assert inv.overage_amount == 0.0
        assert inv.total_amount == 299.0

    def test_pro_with_overage(self):
        engine, month = self._setup(BillingTier.PRO, 60000)
        inv = engine.generate_invoice("t1", month)
        assert inv.overage_amount > 0
        # 10000 over × $5/1000 = $50
        assert inv.total_amount == 299.0 + inv.overage_amount

    def test_enterprise_pricing(self):
        engine, month = self._setup(BillingTier.ENTERPRISE, 100)
        inv = engine.generate_invoice("t1", month)
        assert inv.base_price == 999.0

    def test_invoice_to_dict(self):
        engine, month = self._setup()
        inv = engine.generate_invoice("t1", month)
        d = inv.to_dict()
        assert "invoice_id" in d
        assert "line_items" in d
        assert len(d["line_items"]) >= 1

    def test_quota_check_community(self):
        engine, _ = self._setup(BillingTier.COMMUNITY, 2000)
        assert not engine.is_within_quota("t1")

    def test_quota_check_pro(self):
        engine, _ = self._setup(BillingTier.PRO, 60000)
        assert engine.is_within_quota("t1")  # Pro allows overage

    def test_list_invoices(self):
        engine, month = self._setup()
        engine.generate_invoice("t1", month)
        assert len(engine.list_invoices("t1")) == 1
        assert len(engine.list_invoices("t2")) == 0

    def test_plans_have_all_tiers(self):
        assert BillingTier.COMMUNITY in PLANS
        assert BillingTier.PRO in PLANS
        assert BillingTier.ENTERPRISE in PLANS


# ── SLATracker ───────────────────────────────────────────────────────


class TestSLATracker:
    def test_record_compliant(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.UPTIME, 99.95)
        assert status == SLAStatus.COMPLIANT

    def test_record_at_risk(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.UPTIME, 99.7)
        assert status == SLAStatus.AT_RISK

    def test_record_violated(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.UPTIME, 98.0)
        assert status == SLAStatus.VIOLATED
        assert tracker.violation_count == 1

    def test_latency_compliant(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.LATENCY_P95, 200.0)
        assert status == SLAStatus.COMPLIANT

    def test_latency_violated(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.LATENCY_P95, 1500.0)
        assert status == SLAStatus.VIOLATED

    def test_error_rate_compliant(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.ERROR_RATE, 0.05)
        assert status == SLAStatus.COMPLIANT

    def test_error_rate_violated(self):
        tracker = SLATracker()
        status = tracker.record(SLAMetric.ERROR_RATE, 2.0)
        assert status == SLAStatus.VIOLATED

    def test_current_status(self):
        tracker = SLATracker()
        tracker.record(SLAMetric.UPTIME, 99.95)
        tracker.record(SLAMetric.LATENCY_P95, 300.0)
        status = tracker.current_status()
        assert "API Uptime" in status
        assert status["API Uptime"]["status"] == "compliant"

    def test_current_status_unknown(self):
        tracker = SLATracker()
        status = tracker.current_status()
        assert status["API Uptime"]["status"] == "unknown"

    def test_generate_report(self):
        tracker = SLATracker()
        for _ in range(10):
            tracker.record(SLAMetric.UPTIME, 99.95)
            tracker.record(SLAMetric.LATENCY_P95, 300.0)
            tracker.record(SLAMetric.ERROR_RATE, 0.05)
        report = tracker.generate_report()
        assert report.overall_status == SLAStatus.COMPLIANT
        assert report.uptime_pct > 99

    def test_report_with_violations(self):
        tracker = SLATracker()
        tracker.record(SLAMetric.UPTIME, 97.0)
        report = tracker.generate_report()
        assert report.overall_status == SLAStatus.VIOLATED
        assert len(report.violations) > 0

    def test_report_to_dict(self):
        tracker = SLATracker()
        tracker.record(SLAMetric.UPTIME, 99.95)
        d = tracker.generate_report().to_dict()
        assert "overall_status" in d
        assert "sla_results" in d
        assert "violations" in d

    def test_custom_sla(self):
        custom = [SLADefinition(
            name="Custom Metric",
            metric=SLAMetric.THROUGHPUT,
            target_value=200.0,
            warning_threshold=100.0,
            unit="req/s",
        )]
        tracker = SLATracker(slas=custom)
        status = tracker.record(SLAMetric.THROUGHPUT, 250.0)
        assert status == SLAStatus.COMPLIANT

    def test_default_slas_exist(self):
        assert len(DEFAULT_SLAS) == 5
