# -*- coding: utf-8 -*-
"""Tests for the regulatory watch integration pipeline and event store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from greenlang.factors.watch.change_detector import detect_source_change
from greenlang.factors.watch.pipeline import (
    PipelineCycleResult,
    _dispatch_events_as_webhooks,
    _events_from_change_report,
)
from greenlang.factors.watch.regulatory_events import (
    RegulatoryChangeEvent,
    RegulatoryEventKind,
    RegulatoryEventStore,
    build_artifact_change_event,
    build_factor_event,
    build_source_unavailable_event,
)
from greenlang.factors.webhooks import (
    WebhookEvent,
    WebhookEventType,
    WebhookRegistry,
    deliver_webhook,
    dispatch_event,
    sign_webhook_payload,
)


# ---------------------------------------------------------------------------
# RegulatoryEventStore
# ---------------------------------------------------------------------------


class TestRegulatoryEventStore:
    def test_append_and_list_roundtrip(self, tmp_path: Path) -> None:
        store = RegulatoryEventStore(tmp_path / "rce.db")
        ev = build_artifact_change_event(
            source_id="egrid",
            artifact_hash_old="a" * 64,
            artifact_hash_new="b" * 64,
            url="https://example.test",
        )
        store.append(ev)
        rows = store.list_events()
        assert len(rows) == 1
        assert rows[0].source_id == "egrid"
        assert rows[0].event_kind == RegulatoryEventKind.SOURCE_ARTIFACT_CHANGED
        assert rows[0].artifact_hash_new == "b" * 64

    def test_filter_by_source(self, tmp_path: Path) -> None:
        store = RegulatoryEventStore(tmp_path / "rce.db")
        store.append(build_artifact_change_event(
            source_id="epa_hub", artifact_hash_old=None,
            artifact_hash_new="x", url=None,
        ))
        store.append(build_artifact_change_event(
            source_id="desnz", artifact_hash_old=None,
            artifact_hash_new="y", url=None,
        ))
        assert len(store.list_events(source_id="epa_hub")) == 1
        assert len(store.list_events(source_id="desnz")) == 1
        assert store.count() == 2
        assert store.count(source_id="desnz") == 1

    def test_filter_by_kind(self, tmp_path: Path) -> None:
        store = RegulatoryEventStore(tmp_path / "rce.db")
        store.append(build_source_unavailable_event(
            source_id="s1", url=None, error="timeout",
        ))
        store.append(build_factor_event(
            source_id="s1", kind=RegulatoryEventKind.FACTOR_UPDATED,
            factor_id="f1", old_value=1.0, new_value=1.1,
        ))
        ua = store.list_events(event_kind=RegulatoryEventKind.SOURCE_UNAVAILABLE)
        fu = store.list_events(event_kind=RegulatoryEventKind.FACTOR_UPDATED)
        assert len(ua) == 1
        assert len(fu) == 1
        assert fu[0].factor_id == "f1"

    def test_build_factor_event_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError):
            build_factor_event(source_id="s", kind="bogus", factor_id="f")

    def test_webhook_payload_shape(self) -> None:
        ev = build_factor_event(
            source_id="s1", kind=RegulatoryEventKind.FACTOR_UPDATED,
            factor_id="f1", old_value=0.5, new_value=0.51, unit="kgCO2e/kWh",
        )
        payload = ev.to_webhook_payload()
        assert payload["event_type"] == RegulatoryEventKind.FACTOR_UPDATED
        assert payload["factor_id"] == "f1"
        assert payload["old_value"] == pytest.approx(0.5)
        assert "payload" not in payload  # private detail excluded


# ---------------------------------------------------------------------------
# _events_from_change_report
# ---------------------------------------------------------------------------


def _factor(fid: str, co2e: float, unit: str = "kgCO2e/kWh") -> Dict[str, Any]:
    return {"factor_id": fid, "co2e_total": co2e, "unit": unit, "scope": 2}


class TestEventsFromChangeReport:
    def test_added_modified_removed_map_to_events(self) -> None:
        old = [_factor("a", 1.0), _factor("b", 2.0)]
        new = [_factor("a", 1.1), _factor("c", 3.0)]
        report = detect_source_change("egrid", old, new)
        events = _events_from_change_report("egrid", report)
        kinds = {e.event_kind for e in events}
        assert RegulatoryEventKind.FACTOR_UPDATED in kinds  # a changed
        assert RegulatoryEventKind.FACTOR_REMOVED in kinds  # b removed
        assert RegulatoryEventKind.FACTOR_ADDED in kinds    # c added
        # Removal triggers a breaking change event too.
        assert RegulatoryEventKind.BREAKING_CHANGE in kinds

    def test_unit_change_marks_event_breaking(self) -> None:
        old = [_factor("a", 1.0, unit="kgCO2e/kWh")]
        new = [_factor("a", 1.0, unit="tCO2e/MWh")]
        report = detect_source_change("egrid", old, new)
        events = _events_from_change_report("egrid", report)
        breaking = [e for e in events if e.severity == "breaking"]
        assert breaking, "unit change must surface a breaking event"


# ---------------------------------------------------------------------------
# Webhook delivery retry
# ---------------------------------------------------------------------------


class TestWebhookDeliveryRetry:
    def _subscription(self, tmp_path: Path):
        reg = WebhookRegistry(tmp_path / "wh.db")
        return reg.register(
            tenant_id="t1",
            target_url="https://example.test/hook",
            event_types=[WebhookEventType.FACTOR_UPDATED],
        )

    def test_success_on_first_attempt(self, tmp_path: Path) -> None:
        sub = self._subscription(tmp_path)
        calls: List[Any] = []

        def transport(url, body, headers):
            calls.append((url, headers["X-GL-Event"]))
            return 200, ""

        ev = WebhookEvent(event_type=WebhookEventType.FACTOR_UPDATED, payload={"k": 1})
        receipt = deliver_webhook(sub, ev, transport=transport, sleep=lambda s: None)
        assert receipt["status"] == "ok"
        assert receipt["attempts"] == 1
        assert len(calls) == 1
        # Signature header reconstructs against the canonical body.
        body = {"event": ev.event_type, "triggered_at": ev.triggered_at, "payload": ev.payload}
        assert sign_webhook_payload(body, sub.secret) == receipt["signature"]

    def test_retries_on_5xx_then_succeeds(self, tmp_path: Path) -> None:
        sub = self._subscription(tmp_path)
        statuses = iter([503, 500, 200])
        sleeps: List[float] = []

        def transport(url, body, headers):
            return next(statuses), ""

        ev = WebhookEvent(event_type=WebhookEventType.FACTOR_UPDATED, payload={})
        receipt = deliver_webhook(
            sub, ev,
            transport=transport,
            sleep=lambda s: sleeps.append(s),
            max_attempts=4,
            base_delay_s=1.0,
        )
        assert receipt["status"] == "ok"
        assert receipt["attempts"] == 3
        # Exponential backoff between attempts: 1s, 2s.
        assert sleeps == [1.0, 2.0]

    def test_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        sub = self._subscription(tmp_path)

        def transport(url, body, headers):
            return 500, ""

        ev = WebhookEvent(event_type=WebhookEventType.FACTOR_UPDATED, payload={})
        receipt = deliver_webhook(
            sub, ev,
            transport=transport,
            sleep=lambda s: None,
            max_attempts=3,
            base_delay_s=0.5,
        )
        assert receipt["status"] == "failed"
        assert receipt["attempts"] == 3
        assert receipt["last_http_status"] == 500

    def test_no_retry_on_4xx_except_rate_limit(self, tmp_path: Path) -> None:
        sub = self._subscription(tmp_path)
        attempts: List[int] = []

        def transport(url, body, headers):
            attempts.append(1)
            return 404, ""

        ev = WebhookEvent(event_type=WebhookEventType.FACTOR_UPDATED, payload={})
        receipt = deliver_webhook(
            sub, ev,
            transport=transport,
            sleep=lambda s: None,
            max_attempts=5,
        )
        assert receipt["status"] == "failed"
        # 404 is a permanent client error — only one attempt.
        assert receipt["attempts"] == 1

    def test_429_is_retryable(self, tmp_path: Path) -> None:
        sub = self._subscription(tmp_path)
        statuses = iter([429, 200])

        def transport(url, body, headers):
            return next(statuses), ""

        ev = WebhookEvent(event_type=WebhookEventType.FACTOR_UPDATED, payload={})
        receipt = deliver_webhook(
            sub, ev,
            transport=transport,
            sleep=lambda s: None,
            max_attempts=3,
        )
        assert receipt["status"] == "ok"
        assert receipt["attempts"] == 2


# ---------------------------------------------------------------------------
# Pipeline fan-out
# ---------------------------------------------------------------------------


class TestPipelineWebhookFanout:
    def test_dispatch_events_as_webhooks_skips_when_no_registry(self) -> None:
        ev = build_factor_event(
            source_id="s", kind=RegulatoryEventKind.FACTOR_UPDATED, factor_id="f",
        )
        assert _dispatch_events_as_webhooks([ev], None) == []

    def test_dispatch_events_delivers_to_matching_subscriber(self, tmp_path: Path) -> None:
        registry = WebhookRegistry(tmp_path / "wh.db")
        registry.register(
            tenant_id="tenant-a",
            target_url="https://example.test/a",
            event_types=[WebhookEventType.FACTOR_UPDATED],
        )
        delivered: List[Any] = []

        def transport(url, body, headers):
            delivered.append(url)
            return 200, ""

        ev = build_factor_event(
            source_id="s", kind=RegulatoryEventKind.FACTOR_UPDATED, factor_id="f",
        )
        receipts = _dispatch_events_as_webhooks(
            [ev], registry,
            delivery_kwargs={"transport": transport, "sleep": lambda s: None},
        )
        assert len(receipts) == 1
        assert receipts[0]["status"] == "ok"
        assert delivered == ["https://example.test/a"]

    def test_signed_payload_verifies(self, tmp_path: Path) -> None:
        registry = WebhookRegistry(tmp_path / "wh.db")
        sub = registry.register(
            tenant_id="t",
            target_url="https://example.test/hook",
            event_types=[WebhookEventType.SOURCE_ARTIFACT_CHANGED],
        )
        captured = {}

        def transport(url, body, headers):
            captured["body"] = body
            captured["headers"] = dict(headers)
            return 200, ""

        ev = WebhookEvent(
            event_type=WebhookEventType.SOURCE_ARTIFACT_CHANGED,
            payload={"source_id": "egrid"},
        )
        dispatch_event(
            registry, ev,
            transport=transport, sleep=lambda s: None,
        )
        body = json.loads(captured["body"].decode())
        sig = captured["headers"]["X-GL-Signature"].removeprefix("sha256=")
        assert sig == sign_webhook_payload(body, sub.secret)
