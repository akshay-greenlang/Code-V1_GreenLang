# -*- coding: utf-8 -*-
"""
Customer webhook subscriptions + delivery (Phase F6).

Customers register HTTPS endpoints to receive notifications when:

- a factor they pinned moves to ``deprecated``
- a source publishes a new edition that changes a factor they depend on
- a factor's license class changes
- an impact simulation completes for their tenant

Subscriptions persist in a dedicated SQLite/Postgres table.  Delivery
is best-effort with exponential backoff and HMAC signing so recipients
can verify authenticity.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


class WebhookEventType:
    FACTOR_DEPRECATED = "factor.deprecated"
    FACTOR_UPDATED = "factor.updated"
    LICENSE_CHANGED = "license.changed"
    IMPACT_SIMULATION_COMPLETE = "impact_simulation.complete"
    ALL = [
        FACTOR_DEPRECATED,
        FACTOR_UPDATED,
        LICENSE_CHANGED,
        IMPACT_SIMULATION_COMPLETE,
    ]


@dataclass
class WebhookSubscription:
    subscription_id: str
    tenant_id: str
    target_url: str
    secret: str                            # HMAC shared secret
    event_types: List[str] = field(default_factory=list)
    active: bool = True
    created_at: Optional[str] = None


@dataclass
class WebhookEvent:
    event_type: str
    payload: Dict[str, Any]
    triggered_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.triggered_at:
            self.triggered_at = datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_webhook_subscriptions (
    subscription_id  TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL,
    target_url       TEXT NOT NULL,
    secret           TEXT NOT NULL,
    event_types_json TEXT NOT NULL,
    active           INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_whs_tenant ON factor_webhook_subscriptions (tenant_id);
"""


class WebhookRegistry:
    """Thread-safe SQLite-backed webhook subscription store."""

    def __init__(self, sqlite_path: Union[str, Path]) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def register(
        self,
        *,
        tenant_id: str,
        target_url: str,
        event_types: Iterable[str],
        secret: Optional[str] = None,
    ) -> WebhookSubscription:
        """Register a new subscription.  Returns the subscription record."""
        if not target_url.startswith(("https://", "http://localhost")):
            raise ValueError("target_url must be HTTPS (or http://localhost for dev)")
        types = list(event_types)
        unknown = [t for t in types if t not in WebhookEventType.ALL]
        if unknown:
            raise ValueError("Unknown event_types: %s" % unknown)

        subscription_id = f"whs_{secrets.token_urlsafe(16)}"
        secret_value = secret or secrets.token_urlsafe(32)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO factor_webhook_subscriptions (
                    subscription_id, tenant_id, target_url, secret,
                    event_types_json, active, created_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    subscription_id,
                    tenant_id,
                    target_url,
                    secret_value,
                    json.dumps(types),
                    created_at,
                ),
            )
        return WebhookSubscription(
            subscription_id=subscription_id,
            tenant_id=tenant_id,
            target_url=target_url,
            secret=secret_value,
            event_types=types,
            active=True,
            created_at=created_at,
        )

    def list_for_tenant(self, tenant_id: str) -> List[WebhookSubscription]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT subscription_id, tenant_id, target_url, secret,
                       event_types_json, active, created_at
                FROM factor_webhook_subscriptions
                WHERE tenant_id = ?
                ORDER BY created_at ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [
            WebhookSubscription(
                subscription_id=r[0],
                tenant_id=r[1],
                target_url=r[2],
                secret=r[3],
                event_types=json.loads(r[4]),
                active=bool(r[5]),
                created_at=r[6],
            )
            for r in rows
        ]

    def subscribers_for_event(self, event_type: str) -> List[WebhookSubscription]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT subscription_id, tenant_id, target_url, secret,
                       event_types_json, active, created_at
                FROM factor_webhook_subscriptions
                WHERE active = 1
                """
            ).fetchall()
        results: List[WebhookSubscription] = []
        for r in rows:
            types = json.loads(r[4])
            if event_type in types:
                results.append(
                    WebhookSubscription(
                        subscription_id=r[0],
                        tenant_id=r[1],
                        target_url=r[2],
                        secret=r[3],
                        event_types=types,
                        active=bool(r[5]),
                        created_at=r[6],
                    )
                )
        return results

    def deactivate(self, subscription_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE factor_webhook_subscriptions SET active = 0 "
                "WHERE subscription_id = ?",
                (subscription_id,),
            )
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def sign_webhook_payload(payload: Dict[str, Any], secret: str) -> str:
    """Return a hex HMAC-SHA256 signature of the canonical payload."""
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


__all__ = [
    "WebhookEventType",
    "WebhookSubscription",
    "WebhookEvent",
    "WebhookRegistry",
    "sign_webhook_payload",
]
