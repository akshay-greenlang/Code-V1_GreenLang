# -*- coding: utf-8 -*-
"""
Load tests - Alerting Service throughput (OBS-004)

Tests that key alerting operations meet throughput targets:
- Lifecycle: >1000 fire+ack+resolve cycles/sec
- Deduplication: >5000 checks/sec
- Routing: >5000 decisions/sec
- Template rendering: >2000 renders/sec
- Fingerprint generation: >10000/sec
- Analytics recording: >5000 records/sec

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from typing import List

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
)
from greenlang.infrastructure.alerting_service.config import AlertingConfig

from tests.unit.alerting_service.test_lifecycle import AlertLifecycle
from tests.unit.alerting_service.test_deduplication import AlertDeduplicator
from tests.unit.alerting_service.test_router import AlertRouter
from tests.unit.alerting_service.test_templates import TemplateEngine
from tests.unit.alerting_service.test_analytics import AlertAnalytics


# ============================================================================
# Constants
# ============================================================================

BENCHMARK_DURATION_SEC = 2.0


# ============================================================================
# Throughput tests
# ============================================================================


class TestAlertingThroughput:
    """Load tests for alerting service throughput."""

    def test_lifecycle_throughput(self):
        """Lifecycle: >1000 fire+ack+resolve cycles/sec."""
        lifecycle = AlertLifecycle()
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            alert = Alert(
                source="load-test",
                name=f"LoadAlert-{count}",
                severity=AlertSeverity.CRITICAL,
                title=f"Load alert {count}",
                labels={"instance": f"node-{count}"},
            )
            fired = lifecycle.fire(alert)
            lifecycle.acknowledge(fired.alert_id, user="bot")
            lifecycle.resolve(fired.alert_id, user="bot")
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > 1000, (
            f"Lifecycle rate {rate:.0f} cycles/sec below target 1000"
        )

    def test_dedup_throughput(self):
        """Deduplication: >5000 checks/sec."""
        deduplicator = AlertDeduplicator(window_minutes=60)

        # Pre-populate with some fingerprints
        for i in range(100):
            a = Alert(
                source="load", name=f"Dedup-{i}",
                severity=AlertSeverity.WARNING, title=f"Dedup {i}",
                labels={"k": f"v{i}"},
            )
            deduplicator.process(a)

        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            alert = Alert(
                source="load", name=f"Dedup-{count % 100}",
                severity=AlertSeverity.WARNING, title="Check",
                labels={"k": f"v{count % 100}"},
            )
            deduplicator.is_duplicate(alert)
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > 5000, (
            f"Dedup rate {rate:.0f} checks/sec below target 5000"
        )

    def test_routing_throughput(self):
        """Routing: >5000 decisions/sec."""
        config = AlertingConfig(
            pagerduty_enabled=True,
            opsgenie_enabled=True,
            slack_enabled=True,
            email_enabled=True,
        )
        router = AlertRouter(config)
        router.add_team_route("platform", ["pagerduty", "slack"])
        router.add_team_route("data", ["slack", "email"])

        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            alert = Alert(
                source="load", name=f"Route-{count}",
                severity=AlertSeverity.CRITICAL, title="Route test",
                team="platform" if count % 2 == 0 else "data",
            )
            router.route(alert)
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > 5000, (
            f"Routing rate {rate:.0f} decisions/sec below target 5000"
        )

    def test_template_rendering_throughput(self):
        """Template rendering: >2000 renders/sec."""
        engine = TemplateEngine()
        alert = Alert(
            source="load", name="TemplateTest",
            severity=AlertSeverity.CRITICAL, title="Template load test",
            team="platform", service="api",
        )
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            engine.render("firing", alert)
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > 2000, (
            f"Template rate {rate:.0f} renders/sec below target 2000"
        )

    def test_fingerprint_generation_throughput(self):
        """Fingerprint generation: >10000/sec."""
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            Alert.generate_fingerprint(
                "prometheus", f"Alert-{count}",
                {"instance": f"node-{count}", "job": "test"},
            )
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > 10000, (
            f"Fingerprint rate {rate:.0f}/sec below target 10000"
        )

    def test_analytics_recording_throughput(self):
        """Analytics recording: >5000 records/sec."""
        analytics = AlertAnalytics(enabled=True)
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            alert = Alert(
                source="load", name=f"Analytics-{count}",
                severity=AlertSeverity.WARNING, title=f"Analytics {count}",
                team="platform",
            )
            analytics.record_fired(alert)
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > 5000, (
            f"Analytics rate {rate:.0f} records/sec below target 5000"
        )
