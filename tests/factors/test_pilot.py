# -*- coding: utf-8 -*-
"""Tests for Phase 11: Pilot provisioner, registry, telemetry, feedback."""

from __future__ import annotations

import pytest

from greenlang.factors.pilot.feedback import (
    FeedbackAnalyzer,
    FeedbackCategory,
    FeedbackCollector,
    FeedbackPriority,
    FeedbackStatus,
)
from greenlang.factors.pilot.provisioner import PilotConfig, PilotProvisioner
from greenlang.factors.pilot.registry import (
    PilotPartner,
    PilotRegistry,
    PilotStatus,
    PilotTier,
)
from greenlang.factors.pilot.telemetry import PilotTelemetry, UsageEvent


# ── PilotRegistry ────────────────────────────────────────────────────


class TestPilotRegistry:
    def test_enroll(self):
        reg = PilotRegistry()
        p = reg.enroll("Acme Corp", "alice@acme.com", "Acme Corp", PilotTier.PRO)
        assert p.status == PilotStatus.INVITED
        assert p.partner_id.startswith("pilot_")
        assert p.tenant_id.startswith("tenant_")

    def test_activate(self):
        reg = PilotRegistry()
        p = reg.enroll("Test", "t@t.com", "Test")
        result = reg.activate(p.partner_id, "key_123")
        assert result.status == PilotStatus.ACTIVE
        assert result.api_key == "key_123"

    def test_activate_nonexistent(self):
        reg = PilotRegistry()
        assert reg.activate("fake_id", "key") is None

    def test_pause_and_complete(self):
        reg = PilotRegistry()
        p = reg.enroll("Test", "t@t.com", "Test")
        reg.activate(p.partner_id, "key")
        reg.pause(p.partner_id)
        assert reg.get(p.partner_id).status == PilotStatus.PAUSED
        reg.complete(p.partner_id)
        assert reg.get(p.partner_id).status == PilotStatus.COMPLETED

    def test_list_active(self):
        reg = PilotRegistry()
        p1 = reg.enroll("A", "a@a.com", "A")
        p2 = reg.enroll("B", "b@b.com", "B")
        reg.activate(p1.partner_id, "k1")
        assert len(reg.list_active()) == 1

    def test_get_by_tenant(self):
        reg = PilotRegistry()
        p = reg.enroll("Test", "t@t.com", "Test")
        found = reg.get_by_tenant(p.tenant_id)
        assert found.partner_id == p.partner_id

    def test_summary(self):
        reg = PilotRegistry()
        reg.enroll("A", "a@a.com", "A")
        p = reg.enroll("B", "b@b.com", "B")
        reg.activate(p.partner_id, "k")
        s = reg.summary()
        assert s["total"] == 2

    def test_to_dict(self):
        reg = PilotRegistry()
        p = reg.enroll("Test", "t@t.com", "Test", use_cases=["ghg"])
        d = p.to_dict()
        assert d["name"] == "Test"
        assert d["target_use_cases"] == ["ghg"]


# ── PilotProvisioner ─────────────────────────────────────────────────


class TestPilotProvisioner:
    def test_provision(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        config = prov.provision("Acme", "a@acme.com", "Acme Corp", PilotTier.PRO)
        assert config.api_key.startswith("glf_")
        assert config.tier == "pro"
        assert config.rate_limit_per_day == 10000

    def test_provision_community(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        config = prov.provision("Small Co", "s@s.com", "Small", PilotTier.COMMUNITY)
        assert config.rate_limit_per_day == 1000
        assert config.connector_access == []

    def test_provision_enterprise(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        config = prov.provision("Big Corp", "b@b.com", "Big", PilotTier.ENTERPRISE)
        assert config.rate_limit_per_day == 100000
        assert "iea_statistics" in config.connector_access

    def test_provision_with_overrides(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        config = prov.provision("Test", "t@t.com", "Test", overrides={"rate_limit_per_day": 5000})
        assert config.rate_limit_per_day == 5000

    def test_deprovision(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        config = prov.provision("Test", "t@t.com", "Test")
        assert prov.deprovision(config.partner_id)
        assert prov.get_config(config.partner_id) is None
        assert reg.get(config.partner_id).status == PilotStatus.COMPLETED

    def test_list_provisions(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        prov.provision("A", "a@a.com", "A")
        prov.provision("B", "b@b.com", "B")
        assert len(prov.list_provisions()) == 2

    def test_config_to_dict(self):
        reg = PilotRegistry()
        prov = PilotProvisioner(reg)
        config = prov.provision("Test", "t@t.com", "Test")
        d = config.to_dict()
        assert "api_key_prefix" in d
        assert d["api_key_prefix"].endswith("...")


# ── PilotTelemetry ───────────────────────────────────────────────────


class TestPilotTelemetry:
    def _event(self, partner_id: str = "p1", endpoint: str = "/search", status: int = 200) -> UsageEvent:
        return UsageEvent(
            event_id="ev1",
            partner_id=partner_id,
            tenant_id="t1",
            event_type="api_call",
            endpoint=endpoint,
            status_code=status,
            latency_ms=50.0,
            query_params={"query": "electricity"},
        )

    def test_record(self):
        tel = PilotTelemetry()
        tel.record(self._event())
        assert tel.total_events == 1

    def test_partner_metrics(self):
        tel = PilotTelemetry()
        for _ in range(10):
            tel.record(self._event())
        tel.record(self._event(status=500))
        m = tel.get_partner_metrics("p1")
        assert m.total_requests == 11
        assert m.error_count == 1
        assert m.unique_endpoints == 1

    def test_partner_metrics_empty(self):
        tel = PilotTelemetry()
        m = tel.get_partner_metrics("nonexistent")
        assert m.total_requests == 0

    def test_engagement_score(self):
        tel = PilotTelemetry()
        for ep in ["/search", "/match", "/detail", "/batch", "/health"]:
            for _ in range(5):
                tel.record(self._event(endpoint=ep))
        score = tel.engagement_score("p1")
        assert 0 <= score <= 100

    def test_engagement_score_empty(self):
        tel = PilotTelemetry()
        assert tel.engagement_score("nobody") == 0.0

    def test_weekly_summary(self):
        tel = PilotTelemetry()
        for _ in range(5):
            tel.record(self._event())
        s = tel.weekly_summary()
        assert s["total_events"] == 5
        assert s["active_partners"] == 1

    def test_metrics_to_dict(self):
        tel = PilotTelemetry()
        tel.record(self._event())
        d = tel.get_partner_metrics("p1").to_dict()
        assert "total_requests" in d
        assert "error_rate" in d


# ── Feedback ─────────────────────────────────────────────────────────


class TestFeedbackCollector:
    def test_submit(self):
        fc = FeedbackCollector()
        entry = fc.submit("p1", FeedbackCategory.DATA_QUALITY, "Missing UK factors")
        assert entry.feedback_id.startswith("fb_")
        assert entry.status == FeedbackStatus.NEW

    def test_acknowledge_and_resolve(self):
        fc = FeedbackCollector()
        entry = fc.submit("p1", FeedbackCategory.BUG_REPORT, "500 error")
        fc.acknowledge(entry.feedback_id)
        assert fc.get(entry.feedback_id).status == FeedbackStatus.ACKNOWLEDGED
        fc.resolve(entry.feedback_id, "Fixed in v1.1")
        assert fc.get(entry.feedback_id).status == FeedbackStatus.RESOLVED

    def test_list_by_partner(self):
        fc = FeedbackCollector()
        fc.submit("p1", FeedbackCategory.API_USABILITY, "Slow")
        fc.submit("p2", FeedbackCategory.API_USABILITY, "Fast")
        assert len(fc.list_by_partner("p1")) == 1

    def test_list_by_category(self):
        fc = FeedbackCollector()
        fc.submit("p1", FeedbackCategory.DATA_QUALITY, "A")
        fc.submit("p1", FeedbackCategory.BUG_REPORT, "B")
        assert len(fc.list_by_category(FeedbackCategory.DATA_QUALITY)) == 1

    def test_list_open(self):
        fc = FeedbackCollector()
        e1 = fc.submit("p1", FeedbackCategory.OTHER, "A")
        e2 = fc.submit("p1", FeedbackCategory.OTHER, "B")
        fc.resolve(e1.feedback_id)
        assert len(fc.list_open()) == 1

    def test_to_dict(self):
        fc = FeedbackCollector()
        entry = fc.submit("p1", FeedbackCategory.FEATURE_REQUEST, "Webhook")
        d = entry.to_dict()
        assert d["category"] == "feature_request"


class TestFeedbackAnalyzer:
    def _setup(self) -> FeedbackCollector:
        fc = FeedbackCollector()
        fc.submit("p1", FeedbackCategory.DATA_QUALITY, "A", priority=FeedbackPriority.CRITICAL)
        fc.submit("p1", FeedbackCategory.BUG_REPORT, "B", priority=FeedbackPriority.HIGH)
        fc.submit("p2", FeedbackCategory.FEATURE_REQUEST, "C", priority=FeedbackPriority.LOW)
        e = fc.submit("p2", FeedbackCategory.API_USABILITY, "D", priority=FeedbackPriority.MEDIUM)
        fc.resolve(e.feedback_id)
        return fc

    def test_category_distribution(self):
        fc = self._setup()
        a = FeedbackAnalyzer(fc)
        dist = a.category_distribution()
        assert dist["data_quality"] == 1
        assert dist["bug_report"] == 1

    def test_priority_distribution(self):
        fc = self._setup()
        a = FeedbackAnalyzer(fc)
        dist = a.priority_distribution()
        assert dist["critical"] == 1

    def test_resolution_rate(self):
        fc = self._setup()
        a = FeedbackAnalyzer(fc)
        assert 0.0 < a.resolution_rate() < 1.0

    def test_top_issues(self):
        fc = self._setup()
        a = FeedbackAnalyzer(fc)
        top = a.top_issues()
        # Critical should come first
        assert top[0]["priority"] == "critical"

    def test_partner_satisfaction(self):
        fc = self._setup()
        a = FeedbackAnalyzer(fc)
        scores = a.partner_satisfaction_proxy()
        assert "p1" in scores
        assert "p2" in scores
        # p1 has critical issue, should score lower
        assert scores["p1"] < scores["p2"]

    def test_full_report(self):
        fc = self._setup()
        a = FeedbackAnalyzer(fc)
        report = a.full_report()
        assert report["total_feedback"] == 4
        assert "category_distribution" in report
        assert "partner_satisfaction" in report
