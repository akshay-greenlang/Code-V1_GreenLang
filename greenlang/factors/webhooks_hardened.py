# -*- coding: utf-8 -*-
"""Hardened webhook delivery (W4-C / API12).

Builds on top of :mod:`greenlang.factors.webhooks` with four new
production concerns:

1. **Extended event catalogue** — adds ``factor.created``,
   ``factor.superseded``, ``source.updated``, ``method_pack.released``,
   ``release.cut``, and ``release.rolled_back``.
2. **Idempotency key.** Every event carries a UUIDv4 ``event_id``
   that recipients treat as the dedupe key. The sender tracks
   acknowledgements in a ``factors_webhook_deliveries`` row so redelivery
   never re-runs an ack'd delivery.
3. **Retry schedule.** 0s, 30s, 2min, 10min, 1hr — max 5 attempts.
   After the final attempt the event lands in a dead-letter queue.
4. **Dead-letter replay.** An admin endpoint pulls rows from the DLQ
   and re-enqueues them. See ``routes.py`` for the HTTP surface.

Security invariants:

* HMAC-SHA256 signature header ``X-GL-Signature: sha256=<hex>``.
* ``customer_private`` factors are NEVER serialized into an event body
  regardless of the recipient URL — see :func:`scrub_private_fields`.
* Recipients must speak HTTPS in staging/prod (http://localhost allowed
  only in dev — same policy as the legacy webhooks module).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .webhooks import WebhookSubscription, sign_webhook_payload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event catalogue (W4-C)
# ---------------------------------------------------------------------------


class HardenedEventType:
    """The full post-W4-C catalogue."""

    FACTOR_CREATED = "factor.created"
    FACTOR_UPDATED = "factor.updated"
    FACTOR_DEPRECATED = "factor.deprecated"
    FACTOR_SUPERSEDED = "factor.superseded"
    SOURCE_UPDATED = "source.updated"
    METHOD_PACK_RELEASED = "method_pack.released"
    RELEASE_CUT = "release.cut"
    RELEASE_ROLLED_BACK = "release.rolled_back"

    ALL = [
        FACTOR_CREATED,
        FACTOR_UPDATED,
        FACTOR_DEPRECATED,
        FACTOR_SUPERSEDED,
        SOURCE_UPDATED,
        METHOD_PACK_RELEASED,
        RELEASE_CUT,
        RELEASE_ROLLED_BACK,
    ]


#: Retry delays in seconds for attempts 1..5.
RETRY_DELAYS_S: Tuple[int, int, int, int, int] = (0, 30, 120, 600, 3600)
MAX_ATTEMPTS = len(RETRY_DELAYS_S)


# ---------------------------------------------------------------------------
# Payload scrubber — prevents customer_private leakage.
# ---------------------------------------------------------------------------

_PRIVATE_CLASSES = frozenset(
    {"customer_private", "customer-private", "private"}
)


def scrub_private_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields tagged with ``redistribution_class = customer_private``.

    Walks nested dicts; replaces factor bodies that are customer_private
    with a small stub (``{"factor_id", "redistribution_class", "redacted": True}``)
    so recipients never see the payload of a licensed-to-a-different-customer
    factor even if a misconfigured subscription routed them the event.
    """
    if not isinstance(body, dict):
        return body
    out: Dict[str, Any] = {}
    for k, v in body.items():
        if isinstance(v, dict):
            klass = (v.get("redistribution_class") or "").lower()
            if klass in _PRIVATE_CLASSES:
                out[k] = {
                    "factor_id": v.get("factor_id") or v.get("id"),
                    "redistribution_class": klass,
                    "redacted": True,
                }
                continue
            out[k] = scrub_private_fields(v)
        elif isinstance(v, list):
            out[k] = [
                scrub_private_fields(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Event envelope
# ---------------------------------------------------------------------------


@dataclass
class HardenedEvent:
    event_type: str
    resource_id: str
    body: Dict[str, Any]
    tenant_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def canonical_body(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "body": scrub_private_fields(self.body),
        }


# ---------------------------------------------------------------------------
# Delivery tracker + DLQ — SQLite
# ---------------------------------------------------------------------------


_DELIVERY_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors_webhook_deliveries (
    delivery_id      TEXT PRIMARY KEY,
    event_id         TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    subscription_id  TEXT NOT NULL,
    target_url       TEXT NOT NULL,
    tenant_id        TEXT,
    body_json        TEXT NOT NULL,
    status           TEXT NOT NULL,  -- queued | delivered | dead_letter
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_status_code INTEGER,
    last_error       TEXT,
    first_attempt_at TEXT,
    last_attempt_at  TEXT,
    acked            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fwd_status
    ON factors_webhook_deliveries (status, last_attempt_at);
CREATE INDEX IF NOT EXISTS idx_fwd_event
    ON factors_webhook_deliveries (event_id);

CREATE TABLE IF NOT EXISTS factors_webhook_idempotency (
    -- sender-side dedupe: an event_id is ack'd at most once per subscription.
    event_id         TEXT NOT NULL,
    subscription_id  TEXT NOT NULL,
    acked_at         TEXT NOT NULL,
    PRIMARY KEY (event_id, subscription_id)
);
"""


class WebhookDeliveryTracker:
    """SQLite-backed tracker with idempotency ledger + DLQ."""

    def __init__(self, sqlite_path: Union[str, Path, None] = None) -> None:
        self._lock = threading.Lock()
        if sqlite_path is None:
            self._conn = sqlite3.connect(
                ":memory:", check_same_thread=False, isolation_level=None
            )
        else:
            p = Path(sqlite_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(p), isolation_level=None, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DELIVERY_SCHEMA)

    def record_attempt(
        self,
        *,
        delivery_id: str,
        event: HardenedEvent,
        subscription: WebhookSubscription,
        status: str,
        attempt: int,
        last_status_code: Optional[int],
        last_error: Optional[str],
    ) -> None:
        with self._lock:
            existing = self._conn.execute(
                "SELECT first_attempt_at FROM factors_webhook_deliveries "
                "WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
            now = datetime.now(timezone.utc).isoformat()
            first = existing[0] if existing else now
            self._conn.execute(
                """
                INSERT INTO factors_webhook_deliveries (
                    delivery_id, event_id, event_type, subscription_id,
                    target_url, tenant_id, body_json, status, attempts,
                    last_status_code, last_error, first_attempt_at, last_attempt_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(delivery_id) DO UPDATE SET
                    status = excluded.status,
                    attempts = excluded.attempts,
                    last_status_code = excluded.last_status_code,
                    last_error = excluded.last_error,
                    last_attempt_at = excluded.last_attempt_at
                """,
                (
                    delivery_id,
                    event.event_id,
                    event.event_type,
                    subscription.subscription_id,
                    subscription.target_url,
                    event.tenant_id,
                    json.dumps(event.canonical_body(), default=str, sort_keys=True),
                    status,
                    attempt,
                    last_status_code,
                    last_error,
                    first,
                    now,
                ),
            )

    def mark_acked(self, *, event_id: str, subscription_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO factors_webhook_idempotency "
                "(event_id, subscription_id, acked_at) VALUES (?, ?, ?)",
                (event_id, subscription_id, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.execute(
                "UPDATE factors_webhook_deliveries SET acked = 1, status = 'delivered' "
                "WHERE event_id = ? AND subscription_id = ?",
                (event_id, subscription_id),
            )

    def is_acked(self, *, event_id: str, subscription_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM factors_webhook_idempotency "
                "WHERE event_id = ? AND subscription_id = ?",
                (event_id, subscription_id),
            ).fetchone()
        return bool(row)

    def list_dead_letter(
        self,
        *,
        limit: int = 100,
        subscription_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(1000, int(limit)))
        with self._lock:
            sql = (
                "SELECT delivery_id, event_id, event_type, subscription_id, "
                "target_url, tenant_id, body_json, status, attempts, "
                "last_status_code, last_error, first_attempt_at, last_attempt_at "
                "FROM factors_webhook_deliveries WHERE status = 'dead_letter'"
            )
            params: List[Any] = []
            if subscription_id:
                sql += " AND subscription_id = ?"
                params.append(subscription_id)
            sql += " ORDER BY last_attempt_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                body = json.loads(r[6])
            except (TypeError, json.JSONDecodeError):
                body = {}
            out.append({
                "delivery_id": r[0],
                "event_id": r[1],
                "event_type": r[2],
                "subscription_id": r[3],
                "target_url": r[4],
                "tenant_id": r[5],
                "body": body,
                "status": r[7],
                "attempts": int(r[8] or 0),
                "last_status_code": r[9],
                "last_error": r[10],
                "first_attempt_at": r[11],
                "last_attempt_at": r[12],
            })
        return out

    def requeue_dead_letter(self, delivery_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE factors_webhook_deliveries "
                "SET status = 'queued', attempts = 0, last_error = NULL "
                "WHERE delivery_id = ? AND status = 'dead_letter'",
                (delivery_id,),
            )
        return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Delivery orchestrator
# ---------------------------------------------------------------------------


def _build_headers(
    *,
    event: HardenedEvent,
    signature: str,
    delivery_id: str,
) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "User-Agent": "GreenLang-Factors-Webhooks/1.0",
        "X-GL-Event-Type": event.event_type,
        "X-GL-Event-Id": event.event_id,
        "X-GL-Delivery-Id": delivery_id,
        "X-GL-Signature": f"sha256={signature}",
        "X-GL-Timestamp": event.timestamp,
    }


def deliver_hardened(
    *,
    subscription: WebhookSubscription,
    event: HardenedEvent,
    tracker: Optional[WebhookDeliveryTracker] = None,
    transport: Optional[Callable[[str, bytes, Dict[str, str]], Tuple[int, str]]] = None,
    sleep: Optional[Callable[[float], None]] = None,
    retry_delays: Tuple[int, ...] = RETRY_DELAYS_S,
) -> Dict[str, Any]:
    """Deliver a single hardened webhook with retry + DLQ + idempotency.

    Returns a dict::

        {
          delivery_id, event_id, subscription_id, status ("delivered"|"dead_letter"),
          attempts, last_status_code, last_error, signature
        }
    """
    import time
    import urllib.error
    import urllib.request

    tracker = tracker or WebhookDeliveryTracker()
    _sleep = sleep or time.sleep

    # Short-circuit if already ack'd for this subscription.
    if tracker.is_acked(event_id=event.event_id, subscription_id=subscription.subscription_id):
        return {
            "status": "delivered",
            "event_id": event.event_id,
            "subscription_id": subscription.subscription_id,
            "idempotent_replay": True,
            "attempts": 0,
        }

    # Scrubbed + signed envelope.
    envelope = event.canonical_body()
    signature = sign_webhook_payload(envelope, subscription.secret)
    raw = json.dumps(envelope, sort_keys=True, default=str).encode("utf-8")

    delivery_id = f"dlv_{secrets.token_urlsafe(10)}"
    headers = _build_headers(event=event, signature=signature, delivery_id=delivery_id)

    last_status: Optional[int] = None
    last_error: Optional[str] = None

    for attempt_idx, delay_s in enumerate(retry_delays, start=1):
        if delay_s > 0:
            _sleep(delay_s)
        try:
            if transport is not None:
                last_status, _ = transport(subscription.target_url, raw, headers)
                last_error = None
            else:
                req = urllib.request.Request(
                    subscription.target_url,
                    data=raw,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:  # nosec B310
                    last_status = int(getattr(resp, "status", resp.getcode()))
                    last_error = None
            if last_status is not None and 200 <= last_status < 300:
                tracker.mark_acked(
                    event_id=event.event_id,
                    subscription_id=subscription.subscription_id,
                )
                tracker.record_attempt(
                    delivery_id=delivery_id, event=event, subscription=subscription,
                    status="delivered", attempt=attempt_idx,
                    last_status_code=last_status, last_error=None,
                )
                return {
                    "delivery_id": delivery_id,
                    "event_id": event.event_id,
                    "subscription_id": subscription.subscription_id,
                    "status": "delivered",
                    "attempts": attempt_idx,
                    "last_status_code": last_status,
                    "last_error": None,
                    "signature": signature,
                }
            last_error = f"http_status:{last_status}"
            # 4xx (except 408/425/429) are permanent; go straight to DLQ.
            if (
                last_status is not None
                and 400 <= last_status < 500
                and last_status not in (408, 425, 429)
            ):
                break
        except urllib.error.HTTPError as exc:
            last_status = int(exc.code)
            last_error = f"http_error:{exc.reason}"
            if last_status not in (408, 425, 429) and 400 <= last_status < 500:
                break
        except Exception as exc:
            last_status = None
            last_error = f"transport_error:{type(exc).__name__}:{exc}"

        tracker.record_attempt(
            delivery_id=delivery_id, event=event, subscription=subscription,
            status="queued", attempt=attempt_idx,
            last_status_code=last_status, last_error=last_error,
        )

    # Final: dead-letter.
    tracker.record_attempt(
        delivery_id=delivery_id, event=event, subscription=subscription,
        status="dead_letter", attempt=MAX_ATTEMPTS,
        last_status_code=last_status, last_error=last_error,
    )
    return {
        "delivery_id": delivery_id,
        "event_id": event.event_id,
        "subscription_id": subscription.subscription_id,
        "status": "dead_letter",
        "attempts": MAX_ATTEMPTS,
        "last_status_code": last_status,
        "last_error": last_error,
        "signature": signature,
    }


def replay_dead_letter(
    *,
    tracker: WebhookDeliveryTracker,
    delivery_id: str,
    subscription_lookup: Callable[[str], Optional[WebhookSubscription]],
    transport: Optional[Callable[[str, bytes, Dict[str, str]], Tuple[int, str]]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Replay a single DLQ row. Returns the new delivery receipt."""
    with tracker._lock:  # noqa: SLF001 - intentional: lookup needs the same conn
        row = tracker._conn.execute(
            "SELECT event_id, event_type, subscription_id, target_url, tenant_id, body_json "
            "FROM factors_webhook_deliveries WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
    if row is None:
        raise KeyError(delivery_id)
    sub = subscription_lookup(row[2])
    if sub is None:
        raise KeyError(f"subscription {row[2]} not found")
    try:
        envelope = json.loads(row[5])
    except (TypeError, json.JSONDecodeError):
        envelope = {}
    event = HardenedEvent(
        event_type=row[1],
        resource_id=envelope.get("resource_id", ""),
        body=envelope.get("body", {}),
        tenant_id=row[4],
        event_id=row[0],
        timestamp=envelope.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    )
    # Replay uses 0-delay retries to keep admin action snappy.
    return deliver_hardened(
        subscription=sub,
        event=event,
        tracker=tracker,
        transport=transport,
        sleep=sleep,
        retry_delays=(0, 0, 0, 0, 0),
    )


__all__ = [
    "HardenedEvent",
    "HardenedEventType",
    "MAX_ATTEMPTS",
    "RETRY_DELAYS_S",
    "WebhookDeliveryTracker",
    "deliver_hardened",
    "replay_dead_letter",
    "scrub_private_fields",
]
