# -*- coding: utf-8 -*-
"""
Usage aggregation and quota management for Factors API billing.

Reads from the SQLite usage sink populated by middleware and provides
aggregated views for billing periods, plus quota enforcement.

Tier quotas (requests per month):
    - Community: 1,000
    - Pro: 50,000
    - Enterprise: 500,000

Example:
    >>> aggregator = UsageAggregator("/tmp/factors_usage.db")
    >>> summary = aggregator.aggregate_by_period("abc123", period="monthly")
    >>> print(summary.total_requests)
    42
    >>> quota = aggregator.check_quota("abc123", "community")
    >>> print(quota.remaining)
    958
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier quota definitions (requests per month)
# ---------------------------------------------------------------------------

TIER_QUOTAS: Dict[str, int] = {
    "community": 1_000,
    "pro": 50_000,
    "enterprise": 500_000,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageSummary:
    """Aggregated usage for a billing period."""

    api_key_hash: str
    total_requests: int
    by_endpoint: Dict[str, int]
    by_tier: Dict[str, int]
    period_start: datetime
    period_end: datetime
    period_type: str  # "daily" or "monthly"


@dataclass(frozen=True)
class QuotaStatus:
    """Current quota status for an API key."""

    allowed: int
    used: int
    remaining: int
    overage_amount: int
    reset_at: datetime
    tier: str
    within_quota: bool


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class UsageAggregator:
    """
    Reads usage events from the SQLite usage sink and provides
    aggregated billing views and quota enforcement.

    Attributes:
        db_path: Path to the SQLite usage database.
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize the aggregator with a path to the usage SQLite database.

        Args:
            db_path: Absolute or relative path to the SQLite file.
        """
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            logger.warning(
                "Usage database does not exist yet at %s; "
                "queries will return empty results until requests are recorded.",
                self._db_path,
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only connection to the usage database."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Period helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _period_bounds(
        period: Literal["daily", "monthly"],
        reference: Optional[datetime] = None,
    ) -> tuple[datetime, datetime]:
        """
        Return (start, end) datetimes for the requested billing period.

        Args:
            period: Either "daily" or "monthly".
            reference: Reference point (defaults to now UTC).

        Returns:
            Tuple of (period_start, period_end) as UTC datetimes.
        """
        now = reference or datetime.now(timezone.utc)

        if period == "daily":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        else:  # monthly
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # First day of next month
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)

        return start, end

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_period(
        self,
        api_key_hash: str,
        period: Literal["daily", "monthly"] = "monthly",
        reference: Optional[datetime] = None,
    ) -> UsageSummary:
        """
        Aggregate usage events for a given API key hash within a billing period.

        Args:
            api_key_hash: Truncated SHA-256 hash of the API key (first 16 chars).
            period: Billing period granularity ("daily" or "monthly").
            reference: Reference datetime for period calculation (defaults to now UTC).

        Returns:
            UsageSummary with total counts, by-endpoint breakdown, and by-tier breakdown.
        """
        start, end = self._period_bounds(period, reference)
        start_iso = start.strftime("%Y-%m-%d %H:%M:%S")
        end_iso = end.strftime("%Y-%m-%d %H:%M:%S")

        by_endpoint: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        total = 0

        if not self._db_path.exists():
            return UsageSummary(
                api_key_hash=api_key_hash,
                total_requests=0,
                by_endpoint={},
                by_tier={},
                period_start=start,
                period_end=end,
                period_type=period,
            )

        conn = self._connect()
        try:
            # Per-endpoint counts
            cursor = conn.execute(
                """
                SELECT path, COUNT(*) AS cnt
                FROM api_usage_events
                WHERE api_key_hash = ?
                  AND hit_at >= ?
                  AND hit_at < ?
                GROUP BY path
                ORDER BY cnt DESC
                """,
                (api_key_hash, start_iso, end_iso),
            )
            for row in cursor:
                by_endpoint[row["path"]] = row["cnt"]
                total += row["cnt"]

            # Per-tier counts
            cursor = conn.execute(
                """
                SELECT COALESCE(tier, 'unknown') AS tier, COUNT(*) AS cnt
                FROM api_usage_events
                WHERE api_key_hash = ?
                  AND hit_at >= ?
                  AND hit_at < ?
                GROUP BY tier
                """,
                (api_key_hash, start_iso, end_iso),
            )
            for row in cursor:
                by_tier[row["tier"]] = row["cnt"]

        finally:
            conn.close()

        logger.info(
            "Aggregated %d requests for key_hash=%s period=%s [%s, %s)",
            total,
            api_key_hash[:8],
            period,
            start_iso,
            end_iso,
        )

        return UsageSummary(
            api_key_hash=api_key_hash,
            total_requests=total,
            by_endpoint=by_endpoint,
            by_tier=by_tier,
            period_start=start,
            period_end=end,
            period_type=period,
        )

    # ------------------------------------------------------------------
    # Quota checking
    # ------------------------------------------------------------------

    def check_quota(
        self,
        api_key_hash: str,
        tier: str,
        reference: Optional[datetime] = None,
    ) -> QuotaStatus:
        """
        Check whether the API key is within its monthly quota for the given tier.

        Args:
            api_key_hash: Truncated SHA-256 hash of the API key.
            tier: Pricing tier ("community", "pro", or "enterprise").
            reference: Reference datetime (defaults to now UTC).

        Returns:
            QuotaStatus indicating usage, remaining allowance, and overage.
        """
        tier_lower = tier.lower()
        allowed = TIER_QUOTAS.get(tier_lower, TIER_QUOTAS["community"])

        summary = self.aggregate_by_period(
            api_key_hash, period="monthly", reference=reference
        )
        used = summary.total_requests
        remaining = max(0, allowed - used)
        overage = max(0, used - allowed)

        # Reset at end of current billing period
        _, reset_at = self._period_bounds("monthly", reference)

        status = QuotaStatus(
            allowed=allowed,
            used=used,
            remaining=remaining,
            overage_amount=overage,
            reset_at=reset_at,
            tier=tier_lower,
            within_quota=used <= allowed,
        )

        if not status.within_quota:
            logger.warning(
                "Quota exceeded for key_hash=%s tier=%s: used=%d allowed=%d overage=%d",
                api_key_hash[:8],
                tier_lower,
                used,
                allowed,
                overage,
            )

        return status

    # ------------------------------------------------------------------
    # Listing / admin helpers
    # ------------------------------------------------------------------

    def list_active_keys(
        self,
        period: Literal["daily", "monthly"] = "monthly",
        reference: Optional[datetime] = None,
    ) -> List[Dict[str, object]]:
        """
        List all API key hashes that have made requests in the given period.

        Args:
            period: Billing period granularity.
            reference: Reference datetime.

        Returns:
            List of dicts with api_key_hash, request_count, and latest_hit.
        """
        start, end = self._period_bounds(period, reference)
        start_iso = start.strftime("%Y-%m-%d %H:%M:%S")
        end_iso = end.strftime("%Y-%m-%d %H:%M:%S")

        if not self._db_path.exists():
            return []

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT api_key_hash,
                       COUNT(*) AS request_count,
                       MAX(hit_at) AS latest_hit
                FROM api_usage_events
                WHERE api_key_hash IS NOT NULL
                  AND hit_at >= ?
                  AND hit_at < ?
                GROUP BY api_key_hash
                ORDER BY request_count DESC
                """,
                (start_iso, end_iso),
            )
            return [
                {
                    "api_key_hash": row["api_key_hash"],
                    "request_count": row["request_count"],
                    "latest_hit": row["latest_hit"],
                }
                for row in cursor
            ]
        finally:
            conn.close()

    def total_requests_in_period(
        self,
        period: Literal["daily", "monthly"] = "monthly",
        reference: Optional[datetime] = None,
    ) -> int:
        """
        Count total requests across all keys in a billing period.

        Args:
            period: Billing period granularity.
            reference: Reference datetime.

        Returns:
            Total request count.
        """
        start, end = self._period_bounds(period, reference)
        start_iso = start.strftime("%Y-%m-%d %H:%M:%S")
        end_iso = end.strftime("%Y-%m-%d %H:%M:%S")

        if not self._db_path.exists():
            return 0

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM api_usage_events
                WHERE hit_at >= ? AND hit_at < ?
                """,
                (start_iso, end_iso),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()
