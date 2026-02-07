"""
Cost Tracker - Agent Factory Metering (INFRA-010)

Tracks per-agent costs across categories: compute, tokens, storage.
Accumulates cost entries in memory and flushes them to PostgreSQL
in batches every N seconds for efficient persistence.

Classes:
    - CostCategory: Enumeration of cost categories.
    - CostRates: Configurable rates per resource unit.
    - CostEntry: Single cost record.
    - CostSummary: Aggregated cost summary for queries.
    - CostTracker: Singleton cost tracking service.

Example:
    >>> tracker = CostTracker(db_pool, rates=CostRates())
    >>> await tracker.record_cost(CostEntry(
    ...     agent_key="intake-agent",
    ...     tenant_id="tenant-acme",
    ...     category=CostCategory.COMPUTE,
    ...     amount_usd=0.0042,
    ... ))
    >>> summary = await tracker.query_by_agent("intake-agent")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost Categories
# ---------------------------------------------------------------------------


class CostCategory(str, Enum):
    """Categories of agent execution cost."""

    COMPUTE = "compute"
    """CPU and memory usage costs."""

    TOKENS = "tokens"
    """LLM token consumption costs."""

    STORAGE = "storage"
    """Data storage costs."""

    NETWORK = "network"
    """Network transfer costs."""

    OTHER = "other"
    """Miscellaneous costs."""


# ---------------------------------------------------------------------------
# Cost Rates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRates:
    """Configurable rates per resource unit for cost calculation.

    Attributes:
        usd_per_cpu_hour: Cost per CPU-hour.
        usd_per_1k_tokens: Cost per 1,000 LLM tokens.
        usd_per_gb_month: Cost per GB of storage per month.
        usd_per_gb_transfer: Cost per GB of network transfer.
    """

    usd_per_cpu_hour: float = 0.048
    usd_per_1k_tokens: float = 0.002
    usd_per_gb_month: float = 0.023
    usd_per_gb_transfer: float = 0.09


# ---------------------------------------------------------------------------
# Cost Entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEntry:
    """Single cost record for a billable event.

    Attributes:
        entry_id: Unique identifier for this cost entry.
        agent_key: The agent that incurred the cost.
        tenant_id: Tenant identifier for billing attribution.
        category: Cost category.
        amount_usd: Cost amount in USD.
        quantity: Raw quantity of the resource consumed.
        unit: Unit of the quantity (e.g., 'cpu-seconds', 'tokens', 'gb').
        metadata: Additional context for this cost entry.
        timestamp: When the cost was incurred (UTC).
    """

    entry_id: str = field(default_factory=lambda: str(uuid4()))
    agent_key: str = ""
    tenant_id: str = ""
    category: CostCategory = CostCategory.OTHER
    amount_usd: float = 0.0
    quantity: float = 0.0
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Cost Summary
# ---------------------------------------------------------------------------


@dataclass
class CostSummary:
    """Aggregated cost summary.

    Attributes:
        agent_key: Agent identifier (empty for cross-agent summaries).
        tenant_id: Tenant identifier (empty for cross-tenant summaries).
        total_usd: Total cost across all categories.
        by_category: Breakdown by category.
        entry_count: Number of cost entries.
        period_start: Start of the summary period.
        period_end: End of the summary period.
    """

    agent_key: str = ""
    tenant_id: str = ""
    total_usd: float = 0.0
    by_category: Dict[str, float] = field(default_factory=dict)
    entry_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


# ---------------------------------------------------------------------------
# SQL Statements
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS agent_cost_entries (
    entry_id    TEXT PRIMARY KEY,
    agent_key   TEXT NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT '',
    category    TEXT NOT NULL,
    amount_usd  DOUBLE PRECISION NOT NULL DEFAULT 0,
    quantity    DOUBLE PRECISION NOT NULL DEFAULT 0,
    unit        TEXT NOT NULL DEFAULT '',
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cost_agent
    ON agent_cost_entries (agent_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_tenant
    ON agent_cost_entries (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_category
    ON agent_cost_entries (category, created_at DESC);
"""

_INSERT_SQL = """\
INSERT INTO agent_cost_entries (
    entry_id, agent_key, tenant_id, category,
    amount_usd, quantity, unit, metadata, created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
"""

_SUM_BY_AGENT_SQL = """\
SELECT category, SUM(amount_usd) as total, COUNT(*) as cnt
FROM agent_cost_entries
WHERE agent_key = $1 AND created_at >= $2 AND created_at <= $3
GROUP BY category
"""

_SUM_BY_TENANT_SQL = """\
SELECT category, SUM(amount_usd) as total, COUNT(*) as cnt
FROM agent_cost_entries
WHERE tenant_id = $1 AND created_at >= $2 AND created_at <= $3
GROUP BY category
"""


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Singleton cost tracking service with background batch flushing.

    Accumulates cost entries in an in-memory buffer and flushes them
    to PostgreSQL in batches every flush_interval_s seconds for
    efficient write amplification.

    Attributes:
        rates: Configurable cost rates.
        flush_interval_s: Seconds between background flushes.
    """

    _instance: Optional[CostTracker] = None

    def __init__(
        self,
        db_pool: Any,
        rates: Optional[CostRates] = None,
        flush_interval_s: float = 5.0,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            db_pool: Async PostgreSQL connection pool.
            rates: Cost rates configuration. Uses defaults if None.
            flush_interval_s: Seconds between background flushes.
        """
        self._pool = db_pool
        self.rates = rates or CostRates()
        self.flush_interval_s = flush_interval_s
        self._buffer: List[CostEntry] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        logger.info(
            "CostTracker initialised (flush_interval=%.1fs)", flush_interval_s,
        )

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        db_pool: Optional[Any] = None,
        rates: Optional[CostRates] = None,
    ) -> CostTracker:
        """Get or create the singleton CostTracker instance.

        Args:
            db_pool: Database pool (required on first call).
            rates: Optional cost rates.

        Returns:
            The CostTracker singleton.
        """
        if cls._instance is None:
            if db_pool is None:
                raise RuntimeError(
                    "CostTracker.get_instance() requires db_pool on first call"
                )
            cls._instance = cls(db_pool, rates)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton. Used for testing."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("CostTracker: background flush started")

    async def stop(self) -> None:
        """Stop the background flush task and flush remaining entries."""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush()
        logger.info("CostTracker: stopped and flushed")

    async def initialize_table(self) -> None:
        """Create the cost entries table if it does not exist."""
        conn = await self._acquire_connection()
        try:
            await conn.execute(_CREATE_TABLE_SQL)
        finally:
            await self._release_connection(conn)
        logger.info("CostTracker: table initialised")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record_cost(self, entry: CostEntry) -> None:
        """Record a cost entry to the buffer.

        The entry is held in memory until the next flush cycle.

        Args:
            entry: The cost entry to record.
        """
        async with self._lock:
            self._buffer.append(entry)
        logger.debug(
            "CostTracker: buffered cost entry (agent=%s, $%.6f)",
            entry.agent_key, entry.amount_usd,
        )

    def calculate_compute_cost(
        self, cpu_seconds: float, memory_mb: float = 0,
    ) -> float:
        """Calculate compute cost from resource usage.

        Args:
            cpu_seconds: CPU time consumed in seconds.
            memory_mb: Memory consumed in MB (currently unused in formula).

        Returns:
            Cost in USD.
        """
        cpu_hours = cpu_seconds / 3600.0
        return cpu_hours * self.rates.usd_per_cpu_hour

    def calculate_token_cost(self, token_count: int) -> float:
        """Calculate LLM token cost.

        Args:
            token_count: Number of tokens consumed.

        Returns:
            Cost in USD.
        """
        return (token_count / 1000.0) * self.rates.usd_per_1k_tokens

    def calculate_storage_cost(
        self, gb: float, days: int = 30,
    ) -> float:
        """Calculate storage cost.

        Args:
            gb: Gigabytes stored.
            days: Number of days stored (prorated against monthly rate).

        Returns:
            Cost in USD.
        """
        monthly_fraction = days / 30.0
        return gb * self.rates.usd_per_gb_month * monthly_fraction

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def query_by_agent(
        self,
        agent_key: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> CostSummary:
        """Query cost summary for an agent.

        Args:
            agent_key: Agent key to query.
            start: Period start. Defaults to 30 days ago.
            end: Period end. Defaults to now.

        Returns:
            CostSummary for the agent.
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = datetime(end.year, end.month, 1, tzinfo=timezone.utc)

        conn = await self._acquire_connection()
        try:
            rows = await conn.fetch(_SUM_BY_AGENT_SQL, agent_key, start, end)
        finally:
            await self._release_connection(conn)

        by_cat: Dict[str, float] = {}
        total = 0.0
        count = 0
        for row in rows:
            cat = row["category"]
            amt = float(row["total"])
            by_cat[cat] = amt
            total += amt
            count += int(row["cnt"])

        return CostSummary(
            agent_key=agent_key,
            total_usd=total,
            by_category=by_cat,
            entry_count=count,
            period_start=start,
            period_end=end,
        )

    async def query_by_tenant(
        self,
        tenant_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> CostSummary:
        """Query cost summary for a tenant.

        Args:
            tenant_id: Tenant to query.
            start: Period start. Defaults to start of current month.
            end: Period end. Defaults to now.

        Returns:
            CostSummary for the tenant.
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = datetime(end.year, end.month, 1, tzinfo=timezone.utc)

        conn = await self._acquire_connection()
        try:
            rows = await conn.fetch(_SUM_BY_TENANT_SQL, tenant_id, start, end)
        finally:
            await self._release_connection(conn)

        by_cat: Dict[str, float] = {}
        total = 0.0
        count = 0
        for row in rows:
            cat = row["category"]
            amt = float(row["total"])
            by_cat[cat] = amt
            total += amt
            count += int(row["cnt"])

        return CostSummary(
            tenant_id=tenant_id,
            total_usd=total,
            by_category=by_cat,
            entry_count=count,
            period_start=start,
            period_end=end,
        )

    # ------------------------------------------------------------------
    # Background Flush
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Background task that flushes the buffer periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_s)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("CostTracker: flush loop error: %s", exc)

    async def _flush(self) -> None:
        """Flush all buffered entries to the database."""
        async with self._lock:
            if not self._buffer:
                return
            entries = list(self._buffer)
            self._buffer.clear()

        if not entries:
            return

        try:
            conn = await self._acquire_connection()
            try:
                for entry in entries:
                    await conn.execute(
                        _INSERT_SQL,
                        entry.entry_id,
                        entry.agent_key,
                        entry.tenant_id,
                        entry.category.value,
                        entry.amount_usd,
                        entry.quantity,
                        entry.unit,
                        json.dumps(entry.metadata, default=str),
                        entry.timestamp,
                    )
            finally:
                await self._release_connection(conn)
            logger.info("CostTracker: flushed %d entries", len(entries))
        except Exception as exc:
            logger.error(
                "CostTracker: flush failed for %d entries: %s",
                len(entries), exc,
            )
            # Re-buffer on failure
            async with self._lock:
                self._buffer = entries + self._buffer

    # ------------------------------------------------------------------
    # Connection Helpers
    # ------------------------------------------------------------------

    async def _acquire_connection(self) -> Any:
        """Acquire a connection from the pool."""
        if hasattr(self._pool, "connection"):
            ctx = self._pool.connection()
            conn = await ctx.__aenter__()
            conn._ctx = ctx  # type: ignore[attr-defined]
            return conn
        elif hasattr(self._pool, "acquire"):
            return await self._pool.acquire()
        raise TypeError(f"Unsupported pool type: {type(self._pool).__name__}")

    async def _release_connection(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        if hasattr(conn, "_ctx"):
            await conn._ctx.__aexit__(None, None, None)
        elif hasattr(self._pool, "release"):
            await self._pool.release(conn)


__all__ = [
    "CostCategory",
    "CostEntry",
    "CostRates",
    "CostSummary",
    "CostTracker",
]
