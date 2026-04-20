# -*- coding: utf-8 -*-
"""
Credit-weighted usage metering for the Factors API.

Per the FY27 Factors proposal, API calls burn credits at different rates:

- ``search`` endpoints                         = 1 credit
- ``match`` endpoint                           = 2 credits
- ``export`` endpoint                          = 1 credit per 100 rows
- ``audit-bundle`` endpoint (Enterprise-only)  = 5 credits
- everything else                              = 1 credit

Events are appended to a SQLite sink (optional, gated on
``GL_FACTORS_USAGE_SQLITE``) via :mod:`greenlang.factors.billing.usage_sink`.
A separate :func:`record_usage_event` helper writes the credit cost into
a new ``api_usage_credits`` table so the billing pipeline can aggregate
credits without parsing path strings.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from greenlang.factors.billing import usage_sink

logger = logging.getLogger(__name__)


#: Credit table. Keys are substring-matched against ``endpoint_key``.
#: The first matching entry wins, so put the most specific keys first.
ENDPOINT_CREDITS: dict[str, int] = {
    "/audit-bundle": 5,
    "/match": 2,
    "/diff": 2,
    "/export": 1,  # per 100 rows; see row_count scaling below
    "/search/v2": 1,
    "/search/facets": 1,
    "/search": 1,
    "/coverage": 1,
}


@dataclass
class UsageEvent:
    """Structured record of a billable API call."""

    tier: str
    endpoint: str
    method: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    api_key_id: Optional[str]
    credits: int
    row_count: int = 1
    status_code: int = 200
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat()


_lock = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_usage_credits (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tier           TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    method         TEXT NOT NULL,
    user_id        TEXT,
    tenant_id      TEXT,
    api_key_id     TEXT,
    credits        INTEGER NOT NULL,
    row_count      INTEGER NOT NULL DEFAULT 1,
    status_code    INTEGER NOT NULL DEFAULT 200,
    recorded_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auc_tenant_time
    ON api_usage_credits (tenant_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_auc_endpoint_time
    ON api_usage_credits (endpoint, recorded_at DESC);
"""


# ---------------------------------------------------------------------------
# Credit calculation
# ---------------------------------------------------------------------------


def credits_for(endpoint: str, row_count: int = 1) -> int:
    """Return credit cost for an ``endpoint`` (+ optional ``row_count``).

    ``/export`` is priced per 100 rows with a minimum of 1 credit; other
    endpoints charge a flat rate.  Unknown endpoints default to 1.
    """
    base = 1
    for key, cost in ENDPOINT_CREDITS.items():
        if key in endpoint:
            base = cost
            break
    if "/export" in endpoint:
        scaled = max(1, math.ceil(max(1, row_count) / 100))
        return base * scaled
    return base


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------


def _credits_sqlite_path() -> Optional[Path]:
    """Return the SQLite path or ``None`` if metering is disabled."""
    raw = os.getenv("GL_FACTORS_USAGE_SQLITE", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _write_credit(event: UsageEvent) -> None:
    path = _credits_sqlite_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript(_SCHEMA)
            conn.execute(
                """
                INSERT INTO api_usage_credits (
                    tier, endpoint, method, user_id, tenant_id, api_key_id,
                    credits, row_count, status_code, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.tier,
                    event.endpoint,
                    event.method,
                    event.user_id,
                    event.tenant_id,
                    event.api_key_id,
                    event.credits,
                    event.row_count,
                    event.status_code,
                    event.recorded_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def record_usage_event(
    *,
    tier: str,
    endpoint: str,
    method: str = "GET",
    user: Optional[dict] = None,
    row_count: int = 1,
    status_code: int = 200,
    api_key: Optional[str] = None,
) -> UsageEvent:
    """Record a billable API event.

    Writes to two sinks:

    1. ``usage_sink.record_path_hit`` — legacy hit log.
    2. ``api_usage_credits`` — credit-weighted log (this module).

    The second sink is what the aggregator consumes when computing
    monthly invoices; the first is kept for backwards compatibility with
    dashboards that already exist.
    """
    user = user or {}
    credits = credits_for(endpoint, row_count=row_count)
    event = UsageEvent(
        tier=tier,
        endpoint=endpoint,
        method=method,
        user_id=user.get("user_id"),
        tenant_id=user.get("tenant_id"),
        api_key_id=user.get("api_key_id"),
        credits=credits,
        row_count=max(1, int(row_count)),
        status_code=int(status_code),
    )
    _write_credit(event)
    # Legacy hit-log.  The sink hashes the API key internally.
    try:
        usage_sink.record_path_hit(
            path=endpoint,
            api_key=api_key,
            tier=tier,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("usage_sink.record_path_hit failed: %s", exc)
    logger.debug(
        "metering: endpoint=%s tier=%s credits=%d rows=%d",
        endpoint, tier, credits, event.row_count,
    )
    return event


def totals_by_tenant(
    *,
    sqlite_path: Optional[Path] = None,
) -> dict[str, int]:
    """Return credit totals grouped by tenant (for smoke tests + dashboards)."""
    path = sqlite_path or _credits_sqlite_path()
    if path is None or not path.exists():
        return {}
    with _lock:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                "SELECT tenant_id, SUM(credits) FROM api_usage_credits GROUP BY tenant_id"
            ).fetchall()
        finally:
            conn.close()
    return {tenant or "": int(total) for tenant, total in rows}
