# -*- coding: utf-8 -*-
"""Hardened webhooks tests (W4-C / API12).

Covers event catalogue, HMAC signature, 5-attempt exponential retry,
DLQ landing, idempotency (recipient-side + sender-side),
customer_private scrubbing, and admin replay.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from greenlang.factors.webhooks import WebhookSubscription, sign_webhook_payload
from greenlang.factors.webhooks_hardened import (
    HardenedEvent,
    HardenedEventType,
    MAX_ATTEMPTS,
    RETRY_DELAYS_S,
    WebhookDeliveryTracker,
    deliver_hardened,
    replay_dead_letter,
    scrub_private_fields,
)


def _sub(url: str = "https://example.invalid/hook") -> WebhookSubscription:
    return WebhookSubscription(
        subscription_id="whs_test",
        tenant_id="t1",
        target_url=url,
        secret="sekrit",
        event_types=list(HardenedEventType.ALL),
        active=True,
    )


def _event(
    event_type: str = HardenedEventType.FACTOR_UPDATED,
    body: Dict[str, Any] = None,
) -> HardenedEvent:
    return HardenedEvent(
        event_type=event_type,
        resource_id="f_123",
        body=body or {"factor_id": "f_123", "redistribution_class": "open"},
        tenant_id="t1",
    )


# ---------------------------------------------------------------------------
# 1. All eight event types exist
# ---------------------------------------------------------------------------


def test_all_eight_event_types_present():
    for name in (
        "factor.created",
        "factor.updated",
        "factor.deprecated",
        "factor.superseded",
        "source.updated",
        "method_pack.released",
        "release.cut",
        "release.rolled_back",
    ):
        assert name in HardenedEventType.ALL


# ---------------------------------------------------------------------------
# 2. Successful delivery signs + acks
# ---------------------------------------------------------------------------


def test_delivery_on_first_try_is_acked():
    tracker = WebhookDeliveryTracker()
    received: List[Tuple[str, bytes, Dict[str, str]]] = []

    def transport(url: str, body: bytes, headers: Dict[str, str]):
        received.append((url, body, headers))
        return (200, "OK")

    sub = _sub()
    ev = _event()
    rec = deliver_hardened(
        subscription=sub,
        event=ev,
        tracker=tracker,
        transport=transport,
        sleep=lambda s: None,
    )
    assert rec["status"] == "delivered"
    assert rec["attempts"] == 1
    assert tracker.is_acked(event_id=ev.event_id, subscription_id=sub.subscription_id)

    # Signature sanity.
    url, body, headers = received[0]
    assert headers["X-GL-Signature"].startswith("sha256=")
    assert headers["X-GL-Event-Id"] == ev.event_id
    assert headers["X-GL-Event-Type"] == ev.event_type
    assert "X-GL-Delivery-Id" in headers


# ---------------------------------------------------------------------------
# 3. Exponential retry sequence on repeated failure → DLQ
# ---------------------------------------------------------------------------


def test_five_attempts_then_dead_letter():
    tracker = WebhookDeliveryTracker()
    attempts: List[int] = []

    def transport(url, body, headers):
        attempts.append(1)
        return (500, "server error")

    sub = _sub()
    ev = _event()
    rec = deliver_hardened(
        subscription=sub,
        event=ev,
        tracker=tracker,
        transport=transport,
        sleep=lambda s: None,
    )
    assert rec["status"] == "dead_letter"
    assert rec["attempts"] == MAX_ATTEMPTS
    assert len(attempts) == MAX_ATTEMPTS
    dlq = tracker.list_dead_letter()
    assert any(r["event_id"] == ev.event_id for r in dlq)


def test_retry_delays_exact_sequence():
    assert RETRY_DELAYS_S == (0, 30, 120, 600, 3600)


def test_sleep_called_with_configured_delays():
    tracker = WebhookDeliveryTracker()
    slept: List[float] = []

    def transport(url, body, headers):
        return (500, "fail")

    deliver_hardened(
        subscription=_sub(),
        event=_event(),
        tracker=tracker,
        transport=transport,
        sleep=lambda s: slept.append(s),
    )
    # Implementation skips sleep(0) for the first attempt, so we expect
    # MAX_ATTEMPTS - 1 sleeps corresponding to the nonzero delays.
    assert slept == list(RETRY_DELAYS_S[1:])


# ---------------------------------------------------------------------------
# 4. 4xx client error short-circuits retry
# ---------------------------------------------------------------------------


def test_permanent_4xx_goes_to_dlq_without_extra_retries():
    tracker = WebhookDeliveryTracker()
    count = {"n": 0}

    def transport(url, body, headers):
        count["n"] += 1
        return (403, "forbidden")

    rec = deliver_hardened(
        subscription=_sub(),
        event=_event(),
        tracker=tracker,
        transport=transport,
        sleep=lambda s: None,
    )
    assert rec["status"] == "dead_letter"
    # Single attempt then dead-letter.
    assert count["n"] == 1


# ---------------------------------------------------------------------------
# 5. Idempotency — the same event_id for the same subscription is not redelivered
# ---------------------------------------------------------------------------


def test_idempotent_replay_is_short_circuit():
    tracker = WebhookDeliveryTracker()
    sub = _sub()
    ev = _event()

    def ok(url, body, headers):
        return (200, "OK")

    first = deliver_hardened(subscription=sub, event=ev, tracker=tracker, transport=ok, sleep=lambda s: None)
    assert first["status"] == "delivered"
    # Second call with same event_id — no HTTP call made.
    calls = {"n": 0}
    def tracker_transport(url, body, headers):
        calls["n"] += 1
        return (200, "OK")
    second = deliver_hardened(
        subscription=sub, event=ev, tracker=tracker, transport=tracker_transport, sleep=lambda s: None,
    )
    assert second["status"] == "delivered"
    assert second.get("idempotent_replay") is True
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# 6. HMAC signature verifies on receiver side
# ---------------------------------------------------------------------------


def test_hmac_signature_roundtrip():
    body = {"event_id": "e1", "event_type": "factor.updated", "body": {"x": 1}}
    sig = sign_webhook_payload(body, "topsecret")
    # Same body + secret → same signature. Different secret → different.
    assert sign_webhook_payload(body, "topsecret") == sig
    assert sign_webhook_payload(body, "other") != sig


# ---------------------------------------------------------------------------
# 7. customer_private factors are SCRUBBED in the body
# ---------------------------------------------------------------------------


def test_scrub_private_fields_redacts_nested_payload():
    body = {
        "factor": {
            "factor_id": "f_private",
            "name": "Super Secret",
            "redistribution_class": "customer_private",
            "co2e": 42.0,
        }
    }
    out = scrub_private_fields(body)
    assert out["factor"]["redacted"] is True
    assert out["factor"]["factor_id"] == "f_private"
    assert "name" not in out["factor"]
    assert "co2e" not in out["factor"]


def test_misconfigured_endpoint_cannot_see_private_payload():
    """End-to-end: even if the subscription routes to a 3rd party, private
    factor bodies are scrubbed before the HMAC is computed."""
    tracker = WebhookDeliveryTracker()
    captured: List[bytes] = []

    def transport(url, body, headers):
        captured.append(body)
        return (200, "OK")

    ev = HardenedEvent(
        event_type=HardenedEventType.FACTOR_UPDATED,
        resource_id="f_private",
        body={
            "factor": {
                "factor_id": "f_private",
                "redistribution_class": "customer_private",
                "proprietary_value": 3.14,
            }
        },
        tenant_id="t-random",
    )
    deliver_hardened(
        subscription=_sub(),
        event=ev,
        tracker=tracker,
        transport=transport,
        sleep=lambda s: None,
    )
    assert captured, "transport not invoked"
    assert b"proprietary_value" not in captured[0]
    assert b"redacted" in captured[0]


# ---------------------------------------------------------------------------
# 8. DLQ listing & admin replay
# ---------------------------------------------------------------------------


def test_dlq_replay_succeeds():
    tracker = WebhookDeliveryTracker()
    sub = _sub()

    # Fail once → DLQ.
    def fail(url, body, headers):
        return (500, "err")

    dead = deliver_hardened(
        subscription=sub, event=_event(), tracker=tracker, transport=fail, sleep=lambda s: None
    )
    assert dead["status"] == "dead_letter"
    dlq = tracker.list_dead_letter()
    assert dlq
    delivery_id = dlq[0]["delivery_id"]

    # Now replay with an OK transport.
    def ok(url, body, headers):
        return (200, "OK")

    out = replay_dead_letter(
        tracker=tracker,
        delivery_id=delivery_id,
        subscription_lookup=lambda sid: sub if sid == sub.subscription_id else None,
        transport=ok,
        sleep=lambda s: None,
    )
    assert out["status"] == "delivered"


def test_dlq_replay_missing_delivery_raises():
    tracker = WebhookDeliveryTracker()
    with pytest.raises(KeyError):
        replay_dead_letter(
            tracker=tracker,
            delivery_id="nonexistent",
            subscription_lookup=lambda sid: None,
        )


# ---------------------------------------------------------------------------
# 9. Event envelope canonical shape includes all required fields
# ---------------------------------------------------------------------------


def test_event_envelope_shape():
    ev = _event()
    env = ev.canonical_body()
    for key in ("event_id", "event_type", "resource_id", "tenant_id", "timestamp", "body"):
        assert key in env
    # event_id is a UUID string.
    import uuid as _uuid
    _uuid.UUID(env["event_id"])


# ---------------------------------------------------------------------------
# 10-12. Tracker storage behaviour
# ---------------------------------------------------------------------------


def test_tracker_records_attempts():
    tracker = WebhookDeliveryTracker()

    def fail(url, body, headers):
        return (500, "err")

    rec = deliver_hardened(
        subscription=_sub(), event=_event(), tracker=tracker, transport=fail, sleep=lambda s: None,
    )
    dlq = tracker.list_dead_letter()
    assert dlq
    assert dlq[0]["attempts"] == MAX_ATTEMPTS


def test_is_acked_false_for_unknown_event():
    tracker = WebhookDeliveryTracker()
    assert not tracker.is_acked(event_id="nope", subscription_id="whs_test")


def test_requeue_dead_letter_updates_state():
    tracker = WebhookDeliveryTracker()

    def fail(url, body, headers):
        return (500, "err")

    deliver_hardened(
        subscription=_sub(), event=_event(), tracker=tracker, transport=fail, sleep=lambda s: None,
    )
    dlq = tracker.list_dead_letter()
    assert dlq
    delivery_id = dlq[0]["delivery_id"]
    assert tracker.requeue_dead_letter(delivery_id) is True
    # Row should no longer be in dead_letter status.
    assert all(r["delivery_id"] != delivery_id for r in tracker.list_dead_letter())
