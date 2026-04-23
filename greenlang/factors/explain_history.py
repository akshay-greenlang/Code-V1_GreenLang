# -*- coding: utf-8 -*-
"""Hosted Explain Log retention (W4-C / API13).

Pro+ tenants can opt-in to retention of their ``/v1/resolve`` explain
payloads. Two concerns are baked in:

* **Tier-scoped retention.** Pro = 90 days, Platform = 1 year,
  Enterprise = indefinite (tenant can purge-on-request). The
  ``retention_days_for_tier`` helper is the single source of truth.
* **Tenant isolation.** Every row is tagged with ``tenant_id`` and the
  query API always filters by the caller's tenant. A missing tenant
  context returns empty results — never a cross-tenant read.

The backing store is the same SQLite-or-Postgres pattern used by
:mod:`greenlang.factors.webhooks`: SQLite in dev, Postgres in prod via
duck-typed connection factory.
"""
from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


RETENTION_BY_TIER: Dict[str, Optional[int]] = {
    "community": None,     # not allowed to subscribe
    "pro": 90,
    "platform": 365,
    "consulting": 365,     # parity with platform
    "enterprise": None,    # indefinite (purgeable)
    "internal": None,
}


def retention_days_for_tier(tier: Optional[str]) -> Optional[int]:
    """Return retention days for ``tier``; ``None`` means indefinite."""
    t = (tier or "community").strip().lower()
    if t not in RETENTION_BY_TIER:
        return None
    return RETENTION_BY_TIER[t]


def can_subscribe(tier: Optional[str]) -> bool:
    """Community tier cannot opt-in."""
    t = (tier or "community").strip().lower()
    return t != "community"


# ---------------------------------------------------------------------------
# Storage — SQLite implementation (dev + tests)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors_explain_subscriptions (
    tenant_id    TEXT PRIMARY KEY,
    tier         TEXT NOT NULL,
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factors_explain_history (
    receipt_id   TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    factor_id    TEXT,
    edition_id   TEXT,
    stored_at    TEXT NOT NULL,
    expires_at   TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eh_tenant_ts
    ON factors_explain_history (tenant_id, stored_at DESC);
CREATE INDEX IF NOT EXISTS idx_eh_factor
    ON factors_explain_history (factor_id);
"""


@dataclass
class ExplainRecord:
    receipt_id: str
    tenant_id: str
    factor_id: Optional[str]
    edition_id: Optional[str]
    stored_at: str
    expires_at: Optional[str]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "tenant_id": self.tenant_id,
            "factor_id": self.factor_id,
            "edition_id": self.edition_id,
            "stored_at": self.stored_at,
            "expires_at": self.expires_at,
            "payload": self.payload,
        }


class ExplainHistoryStore:
    """Thread-safe SQLite-backed explain history store."""

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
        self._conn.executescript(_SCHEMA)

    # ---- Subscriptions ----

    def subscribe(self, *, tenant_id: str, tier: str) -> Dict[str, Any]:
        if not can_subscribe(tier):
            raise ValueError(
                "Community tier cannot subscribe to hosted explain logs. "
                "Upgrade to Pro+."
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO factors_explain_subscriptions
                    (tenant_id, tier, enabled, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    tier = excluded.tier,
                    enabled = 1,
                    updated_at = excluded.updated_at
                """,
                (tenant_id, tier, now, now),
            )
        return {
            "tenant_id": tenant_id,
            "tier": tier,
            "enabled": True,
            "retention_days": retention_days_for_tier(tier),
        }

    def unsubscribe(self, tenant_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE factors_explain_subscriptions SET enabled = 0, "
                "updated_at = ? WHERE tenant_id = ?",
                (datetime.now(timezone.utc).isoformat(), tenant_id),
            )
        return cur.rowcount > 0

    def is_subscribed(self, tenant_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT enabled FROM factors_explain_subscriptions WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return bool(row and int(row[0] or 0))

    def get_subscription(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT tenant_id, tier, enabled, created_at, updated_at "
                "FROM factors_explain_subscriptions WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "tenant_id": row[0],
            "tier": row[1],
            "enabled": bool(row[2]),
            "created_at": row[3],
            "updated_at": row[4],
            "retention_days": retention_days_for_tier(row[1]),
        }

    # ---- History rows ----

    def record(
        self,
        *,
        tenant_id: str,
        tier: str,
        payload: Dict[str, Any],
        factor_id: Optional[str] = None,
        edition_id: Optional[str] = None,
        receipt_id: Optional[str] = None,
    ) -> Optional[str]:
        """Persist an explain payload.

        Returns the receipt_id if stored, or ``None`` if the tenant
        isn't subscribed (drop-on-floor is the required behaviour).
        """
        if not self.is_subscribed(tenant_id):
            return None
        now = datetime.now(timezone.utc)
        days = retention_days_for_tier(tier)
        expires_at = (now + timedelta(days=days)).isoformat() if days else None
        rid = receipt_id or f"exr_{secrets.token_urlsafe(16)}"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO factors_explain_history
                    (receipt_id, tenant_id, factor_id, edition_id, stored_at, expires_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    tenant_id,
                    factor_id,
                    edition_id,
                    now.isoformat(),
                    expires_at,
                    json.dumps(payload, default=str, sort_keys=True),
                ),
            )
        return rid

    def list_history(
        self,
        *,
        tenant_id: str,
        factor_id: Optional[str] = None,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExplainRecord]:
        """List retained explain rows for a tenant.

        Tenant isolation: the tenant_id clause is mandatory and the API
        layer MUST pass the authenticated caller's tenant_id — never a
        caller-supplied one.
        """
        if not tenant_id:
            return []
        limit = max(1, min(10_000, int(limit)))
        where = ["tenant_id = ?"]
        params: List[Any] = [tenant_id]
        if factor_id:
            where.append("factor_id = ?")
            params.append(factor_id)
        if from_ts:
            where.append("stored_at >= ?")
            params.append(from_ts)
        if to_ts:
            where.append("stored_at <= ?")
            params.append(to_ts)
        sql = (
            "SELECT receipt_id, tenant_id, factor_id, edition_id, stored_at, "
            "expires_at, payload_json FROM factors_explain_history "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY stored_at DESC LIMIT ? OFFSET ?"
        )
        params += [limit, int(offset)]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: List[ExplainRecord] = []
        for r in rows:
            try:
                payload = json.loads(r[6])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            out.append(
                ExplainRecord(
                    receipt_id=r[0],
                    tenant_id=r[1],
                    factor_id=r[2],
                    edition_id=r[3],
                    stored_at=r[4],
                    expires_at=r[5],
                    payload=payload,
                )
            )
        return out

    def get(self, *, tenant_id: str, receipt_id: str) -> Optional[ExplainRecord]:
        """Read a single receipt; tenant-scoped."""
        if not tenant_id or not receipt_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT receipt_id, tenant_id, factor_id, edition_id, stored_at, "
                "expires_at, payload_json FROM factors_explain_history "
                "WHERE receipt_id = ? AND tenant_id = ?",
                (receipt_id, tenant_id),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[6])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        return ExplainRecord(
            receipt_id=row[0],
            tenant_id=row[1],
            factor_id=row[2],
            edition_id=row[3],
            stored_at=row[4],
            expires_at=row[5],
            payload=payload,
        )

    def purge(self, *, tenant_id: str, before_ts: Optional[str] = None) -> int:
        """Tenant-initiated purge (Enterprise opt-out)."""
        if not tenant_id:
            return 0
        sql = "DELETE FROM factors_explain_history WHERE tenant_id = ?"
        params: List[Any] = [tenant_id]
        if before_ts:
            sql += " AND stored_at <= ?"
            params.append(before_ts)
        with self._lock:
            cur = self._conn.execute(sql, params)
        return int(cur.rowcount or 0)

    def gc_expired(self) -> int:
        """Delete expired rows (run from a cron job). Returns # deleted."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM factors_explain_history "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
        return int(cur.rowcount or 0)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Process-wide default store
# ---------------------------------------------------------------------------


_default_store: Optional[ExplainHistoryStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> ExplainHistoryStore:
    import os
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            path = os.environ.get("GL_FACTORS_EXPLAIN_HISTORY_PATH")
            _default_store = ExplainHistoryStore(path or None)
        return _default_store


def set_default_store(store: ExplainHistoryStore) -> None:
    """Inject a store (used by tests and production wiring)."""
    global _default_store
    with _default_store_lock:
        _default_store = store


def reset_default_store() -> None:
    """Clear the default (test utility)."""
    global _default_store
    with _default_store_lock:
        _default_store = None


def get_explain_by_id(receipt_id: str, *, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """GraphQL/REST helper. Must be called with an authenticated tenant_id."""
    if not tenant_id:
        return None
    store = get_default_store()
    rec = store.get(tenant_id=tenant_id, receipt_id=receipt_id)
    return rec.payload if rec else None


__all__ = [
    "ExplainHistoryStore",
    "ExplainRecord",
    "RETENTION_BY_TIER",
    "can_subscribe",
    "get_default_store",
    "get_explain_by_id",
    "reset_default_store",
    "retention_days_for_tier",
    "set_default_store",
]
